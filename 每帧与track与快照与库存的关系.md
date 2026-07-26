# 每帧、Track、快照与库存的关系

每帧都会将拍到的照片输入到 Yolov5n 中获取识别结果

## 1. 四种数据分别做什么

```text
Detection（单帧 YOLO 结果）
→ 这一帧看到了什么。
→ 只有 cls_id、box、score；没有 item_id。

Track（本次开门的临时轨迹）
→ 记录手操作期间，某个物品可能如何移动。
→ 只用于证明“整理”，不传后台。

current_stable_snapshot（本次稳定快照）
→ 连续 N 帧无手、稳定结果融合后的当前画面。
→ 只有当前看到的物品，不拥有 item_id。

InventoryDB（库存表）
→ 上一次确认后的正式状态。
→ 唯一拥有 item_id；保存 anchor_box、anchor_valid、last_seen_box、status、计数器。
```

它们的关系是：

```text
逐帧 Detection → 更新本次操作的 Track
连续无手帧 Detection → 合成 current_stable_snapshot
current_stable_snapshot + InventoryDB + Track → 得出本次库存变化
```

## 2. 三种比较，作用完全不同

| 比较对象 | 目的 | 是否写库存 |
| --- | --- | --- |
| 当前帧 Detection 与上一帧对应 Detection / Track | 计算本帧移动了多少 | 否 |
| 当前帧 Detection 与 InventoryItem | 给新建 Track 找到它对应的库存 `item_id` | 否 |
| 当前稳定快照与 InventoryDB | 最终判断未变、整理、遮挡、拿出、放入 | 最后统一写入 |

因此不是“只做两帧比较”或“只做库存比较”：

```text
两帧比较（或当前帧对 Track）负责记录过程；
库存比较负责取得原有 item_id；
稳定快照对库存负责最终确认。
```

## 3. 怎么知道要不要创建 Track

不能等到已经确认物品移动后才创建 Track。

原因是：手可能刚进入就完全挡住苹果。此时若还没有 Track，之后 YOLO 看不到苹果，就不知道手拿的是库存中的哪一颗苹果。

因此规则是：

```text
手出现，并且某个库存物品可能受这只手影响
→ 立即为它创建“候选 Track”。
```

“可能受影响”不需要精确判断手真的碰到了物品。只需满足下列任一情况：

```text
1. 手框与物品当前 Detection 框相交或足够接近；
2. 手框与库存物品的 anchor_box / last_seen_box 相交或足够接近；
3. 当前 Detection 能关联到某个 VISIBLE 库存物品，且它在手附近。
```

如果手附近有多个同类物品，可以建立多个候选 Track。它们只是暂时证据，不会立刻导致库存变化；后面只有“Track 路径与稳定快照唯一对应”时，才确认整理。

换句话说：

```text
创建 Track 不等于确认物品移动；
创建 Track 只是避免错过可能的移动过程。
```

## 4. Track 怎么获得库存 item_id

`item_id` 只属于库存表。Track 只保存一个引用：

```cpp
struct Track {
    int track_id;                         // 本次开门的临时编号
    std::optional<int> bound_item_id;     // 引用库存 item_id；可为空
    int cls_id;

    BBox start_box;                       // 从库存 anchor_box 复制的起点
    BBox latest_box;                      // 当前完整物品的估计框
    BBox last_detected_box;               // 最近 YOLO 原始框，可为局部小框
    BBox last_hand_box;
    std::vector<TrackPoint> path;         // 保存 latest_box 的历史点
};
```

创建 Track 时，先用当前 Detection 或手附近的库存位置，尝试关联库存中的同类物品：

```text
当前苹果 Detection / 手框附近位置
→ 匹配到 InventoryItem(item_id=1)
→ Track.bound_item_id = 1
```

匹配时可参考：

```text
cls_id 必须相同；
当前 Detection 与 item.last_seen_box 的位置关系；
anchor_valid=true 时，当前 Detection 与 item.anchor_box 的包含关系；
物品是否位于手框附近。
```

若没有任何库存项能匹配：

```text
Track.bound_item_id = 空
```

它可能是新放入物品。此时只能记录路径，不能提前创建库存 `item_id`。

若 `anchor_valid=false`，则不能读取 `anchor_box`，只能用 `last_seen_box` 做临时关联。

## 5. 每一帧的总流程

每一帧都执行：

```text
1. 摄像头取图；
2. 图像输入 YOLO，得到 Detection 列表；
3. 得到手框 hand_box（若有手）；
4. 判断本帧是否有手；
5. 进入“有手”或“无手”的分支。
```

### 5.1 有手：更新 Track，不生成稳定快照

```text
operation_pending = true；
清空 no_hand_buffer；
本帧不参与稳定快照；
InventoryDB 不修改。
```

处理每个物品 Detection：

```text
1. 先尝试匹配活动 Track；
2. 匹配成功 → 更新该 Track；
3. 没有 Track 匹配 → 尝试关联手附近的库存物品；
4. 关联成功 → 创建绑定 item_id 的候选 Track；
5. 无法关联库存、但物品在手附近 → 创建 bound_item_id 为空的 Track。
```

### 5.2 Detection 仍看得到物品：使用物品框位移

不管 YOLO 给的是完整框还是被手挡住后的局部小框，只要还检测得到物品，统一使用两个相邻检测框的中心位移：

```cpp
object_delta = Center(current_detection.box)
             - Center(track.last_detected_box);

track.latest_box = Move(track.latest_box, object_delta);
track.last_detected_box = current_detection.box;
track.last_hand_box = current_hand_box;
track.path.push_back(track.latest_box);
```

其中：

```text
current_detection.box：只告诉程序“本帧移动了多少”；
latest_box：始终保存完整物品的估计框。
```

所以局部小框不会把 `latest_box` 压缩成小框。

若位移小于抖动阈值，不必追加 `path` 点，直接保留 `latest_box`。这就是路径剪枝。

### 5.3 Detection 完全看不到物品：使用手框位移

若 Track 对应物品没有 Detection、手仍在：

```cpp
hand_delta = Center(current_hand_box)
           - Center(track.last_hand_box);

track.latest_box = Move(track.latest_box, hand_delta);
track.last_hand_box = current_hand_box;
track.path.push_back(track.latest_box);
```

此时 `bound_item_id` 不会丢失。例如苹果被手完全挡住后，`Track.bound_item_id` 仍然是库存苹果的 `item_id=1`。

### 5.4 从完全遮挡重新变成可见

刚从“完全遮挡”变回“重新检测到物品”时，不能用当前小框减去很久以前的小框。第一帧只做：

```cpp
track.last_detected_box = current_detection.box;
```

下一帧再开始使用物品框位移。这样不会让轨迹在“手代理”与“检测框代理”切换时突然跳动。

### 5.5 无手：生成稳定快照，但冻结 Track

手消失后，每帧仍做 YOLO，但处理方式变为：

```text
将本帧 Detection 加入 no_hand_buffer；
不更新任何 Track.latest_box；
不追加任何 Track.path；
不更新 InventoryDB。
```

原因是：手已经离开，Track 记录的物品移动过程已经结束。若继续用手离开的位移更新 Track，就会把“手离开冰箱”误认为“物品继续被带走”。

连续 N 帧无手、检测结果稳定后：

```text
no_hand_buffer → current_stable_snapshot
```

这 N 帧里的 Track 不是删除，而是冻结为证据，等待快照比较时使用。

## 6. 逐帧例子：苹果从左边整理到右边

开门前，InventoryDB 已确认：

```text
item_id=1：苹果，VISIBLE，anchor_box=L，last_seen_box=L
item_id=2：牛奶，VISIBLE，anchor_box=R，last_seen_box=R
```

### 第 1 帧：刚开门，无手

```text
YOLO：苹果 L、牛奶 R
手：无
Track：无
```

处理：本帧加入 `no_hand_buffer`；不创建 Track；不改库存。

### 第 2 帧：仍无手

```text
YOLO：苹果 L、牛奶 R
手：无
Track：无
```

处理：继续填充 `no_hand_buffer`。若达到 N 帧，可生成快照直接对库存；结果是未变化。

### 第 100 帧：手来到苹果附近

```text
YOLO：苹果 L_part、牛奶 R
手：在苹果附近
```

处理：

```text
1. 手出现 → 清空 no_hand_buffer；
2. 苹果 L_part 与库存 item_id=1 关联；
3. 创建 Track#1：
   track_id=1；
   bound_item_id=1；
   start_box=L；
   latest_box=L；
   last_detected_box=L_part；
   path=[L]；
4. InventoryDB 不修改。
```

此时尚未确认苹果真的移动，只是开始记录证据。

### 第 101 帧：苹果被带向中间，仍检测到局部

```text
YOLO：苹果 M_part、牛奶 R
手：仍在
```

处理：

```text
苹果 M_part → 匹配 Track#1；
object_delta = Center(M_part) - Center(L_part)；
latest_box = 将完整估计框 L 平移 object_delta；
path 追加新的完整估计框。
```

这里使用小框之间的位移，但不会把 `M_part` 直接当成完整苹果框。

### 第 102 帧：苹果被手完全遮住

```text
YOLO：没有苹果 Detection，只看到牛奶 R
手：继续向右
```

处理：

```text
Track#1 仍在；
用 hand_delta 平移 Track#1.latest_box；
path 追加新的完整估计框；
InventoryDB 仍不修改。
```

### 第 103 帧：苹果在右侧重新露出局部

```text
YOLO：苹果 R_part、牛奶 R
手：仍在
```

处理：苹果 R_part 匹配 Track#1；但因为刚从完全遮挡恢复，本帧只记录：

```text
Track#1.last_detected_box = R_part
```

下一帧再继续用检测框位移更新。

### 第 104 帧：手离开

```text
YOLO：苹果 R、牛奶 R
手：无
```

处理：

```text
Track#1.path、latest_box 冻结；
本帧作为 no_hand_buffer 的第一帧；
不更新库存。
```

### 第 105 ～ 第 109 帧：连续无手稳定帧

```text
YOLO：苹果 R、牛奶 R
手：无
Track#1：仍保留，但不更新
```

处理：

```text
这些帧只用于投票、融合为 current_stable_snapshot；
Track#1.path 不追加；
Track#1.latest_box 不改变；
InventoryDB 不改变。
```

因此：

> 生成稳定快照的 N 个无手帧中，Track 保留为证据，但不更新。

### 生成稳定快照后：与库存比较

稳定快照中有：

```text
SnapshotItem：苹果 B 在 R
```

库存中仍有：

```text
item_id=1：苹果在 L
```

比较：

```text
1. B 在 R，无法与 item_id=1 的旧位置 L 直接匹配；
2. 查询绑定 item_id=1 的冻结 Track#1；
3. B.box 与 Track#1.path 中右侧的估计框匹配；
4. 唯一对应 → 确认 item_id=1 从 L 整理到 R。
```

最后只写 `planned_changes`：

```text
item_id=1：
    last_seen_box = B.box；
    anchor_box = Track#1 中与 B 匹配的完整估计框；
    anchor_valid = true；
    status = VISIBLE；
```

全部判断结束后，再一次性提交给 InventoryDB 和后台，并清空本次 Track。

## 7. 其他情况

### 7.1 手进来但没有移动物品

可能创建候选 Track，但最终稳定快照中的物品仍与库存原位置直接匹配：

```text
不发生整理；
Track 不影响库存；
提交后清空 Track。
```

### 7.2 手拿走苹果

Track#1 记录了 item_id=1 的过程，但手离开后的稳定快照没有苹果 B 能匹配其路径：

```text
不属于整理；
后续库存比较会得到 item_id=1 → OUT 或 OCCLUDED。
```

### 7.3 放入新的苹果

手中苹果可能有一条：

```text
Track#2：bound_item_id=空
```

稳定快照中出现苹果 B，但 B 不能关联任何已有库存项，也不匹配任何已绑定旧物品的 Track：

```text
确认新放入；
此时才创建新的 item_id；
写入 InventoryDB。
```

## 8. 最简总流程

```text
手出现
→ 为手附近可能受影响的物品创建候选 Track
→ 有 Detection：用物品框位移更新 Track
→ 无 Detection：用手框位移更新 Track
→ 手离开：冻结 Track
→ 连续 N 帧无手：生成稳定快照
→ 稳定快照直接对比 InventoryDB
→ 需要时用冻结 Track 证明“整理”
→ 统一提交库存并清空 Track
```
