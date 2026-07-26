首先要明白:
    一个物体是不可能凭空移动的, 必然是有外物导致他移动的, 而在我们的这个任务中(冰箱中) 这个外物就是手
    物体进来是通过手拿进来的, 进来的时候有一个轨迹 track
    物体出去是通过手拿出去的, 出去的时候有一个轨迹 track
    物体在冰箱中移动(整理)的时候也是通过手来实现的, 这个时候也有一个轨迹 track
但是:
    物体的进入与拿出我们可以通过快照与库存的对比来判断, 因此他们的 track 我们不需要保存 (这两种情况下我们根本不会去使用他们track, 所以这两种情况下的track有没有正确存在无所谓)
    物体的移动(整理) 是我们的track唯一排上用场的时候, 所以我们只需要 (必须要) 保证在整理的时候 track 是正确的

这样就有一个很好的前提点:
    物体要整理的话, 他必然是首先就已经纯在于冰箱中的, 而且这个物品要像移动的话必然首先是物其他物品堆积在他正上方的 (而且无手遮挡)
    这说明他的框一开始其实是不会被手遮挡的, 他的 anchor_box 理应是很完美的 (应该大概率是完整的物体框)
    这样的话他们的 track 的 start_box = item#1.anchor_box 在一开始是比较好的, 不需要担心他的 start_box 一开始会被手遮挡
    而他的 start_box 只要一开始是完整的、没会被手遮挡的话, 后续的操作就不怕了


---

## 手漏检时的兜底规则

正常情况下，物品移动时应该检测到手：

```text
检测到手
→ 为手附近、可能被操作的库存物品创建候选 Track；
→ Track 才有资格在后面证明“整理”。
```

但可能发生：手实际进来了，YOLO 却没有识别到手。

这时不能因为“没有手”就把明显移动中的画面塞进稳定快照；但也不能只靠物品框变化就创建正式 Track、认定整理。

正确的兜底是：

```text
无手 + 前后帧物品明显移动 (是明显移动才行, 如果只是小抖动就不算)
→ suspicious_motion（可疑移动）；
→ 清空快照缓冲；
→ operation_pending = true；
→ 不创建可证明整理的正式 Track。
```

这里“明显移动”只用于判断当前画面不稳定。例如同类物品在两帧之间，中心距离、宽高变化超过正常 YOLO 抖动阈值。

它的作用是：

```text
延迟生成快照，等待画面稳定；
```

它不能证明：

```text
库存 item_id=1 一定移动到了某个新位置。
```

因为无手时的框变化也可能来自 YOLO 抖动、漏检后重现，或两个同类物品对应错误。

## 例子 1：没有手、没有明显移动

```text
第 1 帧：无手；苹果框 L；加入快照缓冲。
第 2 帧：无手；苹果框仍在 L；与第 1 帧差异很小；继续加入缓冲。
第 3 帧：无手；苹果框仍在 L；继续加入缓冲。
...
第 N 帧：连续 N 帧稳定；生成稳定快照。
```

结果：

```text
不创建 Track；
稳定快照直接与库存比较；
苹果仍在原位，库存不发生业务变化。
```

## 例子 2：手被正常识别，苹果被整理

```text
库存：item_id=1 的苹果在左侧 L，anchor_box=L。

第 1 帧：无手；苹果 L；加入快照缓冲。

第 2 帧：检测到手靠近苹果；苹果变成局部框 L_part。
→ 清空快照缓冲；
→ 创建 Track#1，bound_item_id=1；
→ start_box=anchor_box=L；latest_box=L。

第 3 帧：检测到手；苹果局部框移动到 M_part。
→ 用 L_part 到 M_part 的位移更新 Track#1.latest_box；

第 4 帧：检测到手；YOLO 看不到苹果。
→ 用手框位移更新 Track#1.latest_box；

第 5 帧：手离开；苹果出现在右侧 R。
→ 冻结 Track#1；开始新的无手快照缓冲。

第 6 ～ N 帧：连续无手、苹果稳定在 R。
→ 生成稳定快照。
```

最后：

```text
快照中的苹果 R 与库存 item_id=1 的旧位置 L 不直接匹配；
但 R 与 Track#1.path 匹配；
→ 确认是整理；
→ item_id=1 保持不变，只更新位置。
```

## 例子 3：手实际进入，但 YOLO 没识别到手；苹果仍能被看到

```text
第 1 帧：无手；苹果框 L；加入快照缓冲。

第 2 帧：YOLO 仍显示无手；苹果框移动到 M。
→ M 与上一帧 L 的差异明显；
→ 标记 suspicious_motion；
→ 清空快照缓冲；
→ 不创建正式 Track。

第 3 帧：YOLO 仍显示无手；苹果框移动到 R。
→ 仍是 suspicious_motion；
→ 不生成稳定快照；

第 4 帧以后：苹果稳定在 R，且连续 N 帧无手、无明显移动。
→ 生成稳定快照。
```

最后：

```text
没有可靠 Track，不能证明这是整理；
库存比较会把它保守处理为“苹果从 L 消失、苹果在 R 出现”；
通常结果是旧 item → OUT，新物品 → IN。
```

这不是最理想，但库存仍然能继续工作；只是失去了“同一个 item_id 的整理”这一层判断。

## 最终规则

```text
检测到手 + 手附近物品
→ 创建候选 Track；后续可证明整理。

未检测到手 + 物品无明显移动
→ 填充稳定快照缓冲。

未检测到手 + 物品明显移动
→ suspicious_motion，重置稳定快照缓冲；
  不创建正式 Track，不直接判整理。
```


---

## Track 的完整逐帧操作流程

这一节只回答一个问题：从每一帧 YOLO 结果开始，Track 到底怎样创建、怎样更新、怎样在最后帮助判断整理。

### 1. 先区分三个阶段

```text
阶段 A：没有手
→ 不更新 Track；只收集稳定快照的候选帧。

阶段 B：有手
→ 创建或更新 Track；不生成稳定快照；不修改库存。

阶段 C：手离开后
→ Track 冻结；连续 N 帧无手后生成稳定快照；
  再用冻结 Track 判断是否整理。
```

### 2. 每帧开始时，程序手里有什么

```cpp
CurrentFrame {
    std::vector<Detection> item_detections; // YOLO 物品框
    std::optional<BBox> hand_box;           // 本帧手框；无手则为空
}

ProcessStore {
    FrameObservation previous_frame;        // 上一帧 YOLO 结果
    std::vector<Track> active_tracks;       // 本次开门正在保留的 Track
    no_hand_buffer;                         // 连续无手帧，用来合成快照
}
```

其中 `previous_frame` 和 `no_hand_buffer` 不是同一个东西：

```text
previous_frame：只保留一帧，用于计算本帧相对上一帧的位移。
no_hand_buffer：保留 N 帧，用于生成稳定快照。
```

每帧固定先做：

```text
1. 摄像头取图；
2. YOLO 输出本帧物品 Detection 和手 Detection；
3. 坐标转换到统一坐标系；
4. 根据“本帧有没有手”进入不同分支；
5. 本帧结束时 previous_frame = 当前帧。
```

## 3. 阶段 A：本帧没有手

### 3.1 前一帧也没有手 (前后两帧都没手)

此时没有正在进行的手部操作：

```text
当前物品 Detection 与 previous_frame 的物品 Detection 做稳定性比较；
```

比较目的只是判断：

```text
这一帧是否适合进入 no_hand_buffer。
```

不是创建 Track，也不是修改库存。

```text
若同类物品的位置、宽高都在正常抖动范围内
→ 当前帧加入 no_hand_buffer。

若突然消失、突然出现、或位置明显跳变
→ 当前帧视为不稳定；
→ 清空或重新开始 no_hand_buffer；
→ 不创建 Track。
```

`active_tracks` 此时若有，也是上一轮手操作留下的冻结证据；本帧不能修改它们的 `latest_box` 或 `path`。

### 3.2 前一帧有手、本帧没有手

这是从“手操作”切换到“等待稳定快照”的时刻：

```text
1. 所有 active_tracks 立即冻结；
2. 不再用手框或物品框更新它们；
3. 清空旧 no_hand_buffer；
4. 把当前帧作为新的 no_hand_buffer 第 1 帧；
```

特别重要：手离开后不能继续更新 Track。否则手离开冰箱的路线会被错误地加入物品路径。

## 4. 阶段 B：本帧检测到手

一旦本帧有手：

```text
operation_pending = true；
清空 no_hand_buffer；
本帧不参与快照；
InventoryDB 不修改。
```

之后分成两件事：

```text
第一件：当前 Detection 能否匹配已有 Track？
第二件：剩余 Detection 或手附近库存物品，是否需要创建新 Track？
```

### 4.1 第一步：当前 Detection 先匹配已有 Track

假设本帧检测到：

```text
Detection D：apple，box=M_part
```

程序先在 `active_tracks` 中找：

```text
cls_id 同为 apple 的 Track；
```

然后按 Track 上一次的状态选择比较参考框：

```text
Track 上一帧仍检测得到物品 (完全检测到或部分检测到)
→ D.box 与 Track.last_detected_box 比较。

Track 上一帧物品完全被挡住
→ D.box 与 Track.latest_box 比较。
```

为什么参考框不同：

```text
last_detected_box 是同一种“YOLO 原始框”，
适合和当前 YOLO 原始框比较。

latest_box 是完整物品的估计位置，
适合在物品重新出现时，判断它是否在预期位置附近。
```

比较仍使用同一个 `match_box` 思路：

```text
1. cls_id 必须相同；
2. 中心距离必须足够近；
3. 宽度差、长度差不能过大；
4. 多个候选时，选择匹配分数最好的一个；
5. 若最好的两个太接近、无法唯一确定，则本帧不强行匹配。
```

这里只是 Track 的帧间匹配，因此框大小阈值可以比“稳定快照与库存的原地匹配”宽一些：手部分遮挡会让 YOLO 框缩小，但不应轻易断开 Track。

#### 若 D 匹配到 Track#1，怎么办？

**不创建新 Track。** 只更新 Track#1。

```text
D → Track#1
→ Track#1.bound_item_id 保持原样；
→ Track#1 继续记录同一个物品的轨迹；
```

若上一帧 Track#1 也有物品 Detection：

```cpp
object_delta = Center(D.box) - Center(track.last_detected_box);

track.latest_box = Move(track.latest_box, object_delta);
track.last_detected_box = D.box;
track.last_hand_box = current_hand_box;
```

然后：

```cpp
if (Distance(track.latest_box, track.path.back()) > path_step_threshold) {
    track.path.push_back(track.latest_box);
}
```

`latest_box` 始终是完整物品的估计框；D 的小框只用于计算位移，不能直接覆盖 `latest_box`。

若 Track#1 上一帧是“完全遮挡”：

```text
本帧 D 重新出现；
先令 track.last_detected_box = D.box；
本帧不使用很久以前的旧检测框计算 object_delta；
下一帧再恢复 Detection 位移更新。
```

### 4.2 第二步：当前 Detection 没有匹配到 Track

未匹配 Detection 不代表它一定是新物品。它可能是：

```text
1. 手刚碰到的原有库存物品，尚未创建 Track；
2. 手新放入的物品；
3. 同类物品太多，本帧无法确定它属于哪条 Track。
```

因此，对每个“未匹配的 D”，只在 D 位于手附近时，再查询库存：

```text
从 InventoryDB 取同 cls_id、status=VISIBLE 的库存物品；
比较 D.box 与 item.last_seen_box；
若 anchor_valid=true，也参考 item.anchor_box；
同时要求 item 的位置在手附近。
```

结果分三种：

```text
只有一个库存 item 匹配
→ 创建绑定该 item_id 的新 Track。

一个都不匹配
→ D 可能是新放入物品；创建 bound_item_id=空 的临时 Track，
  或仅保留 Detection，等稳定快照再决定是否入库。

多个库存 item 都匹配
→ 身份不确定；不强行绑定到某一个 item_id。
  可以创建不绑定的候选 Track，但它之后不能单独证明整理。
```

### 4.3 手附近没有物品 Detection：手一开始就完全遮挡

这是最容易漏掉的情况。

本帧只有手框，没有苹果 Detection：

```text
手框 H；
InventoryDB 中 item_id=1 的 anchor_box 正好在 H 附近。
```

此时仍可创建：

```cpp
Track#1 {
    bound_item_id = 1;
    start_box = item#1.anchor_box;
    latest_box = item#1.anchor_box;
    last_hand_box = H;
    path = [item#1.anchor_box];
}
```

第一帧没有上一只手框可比较，因此不移动 `latest_box`；从下一帧开始，若手继续移动，才使用手框位移。

若手附近有两个同类库存物品：

```text
可以创建两个候选 Track；
后面只有某一条路径能与稳定快照中的新位置唯一对应时，
才会确认它是整理。
```

## 5. 当前帧没有 Detection，但已有 Track 怎么办

遍历所有本帧没有被 Detection 匹配到的活动 Track：

```text
若手仍靠近 Track.latest_box，且该 Track 在本次手操作中已经激活
→ 认为物品可能被手完全挡住；
→ 使用手框位移更新它。

若手已经明显远离 Track.latest_box
→ 不再让该 Track 跟随手；
→ 暂时冻结它已有的路径，等待物品重新出现或手离开。
```

使用手框位移时：

```cpp
hand_delta = Center(current_hand_box) - Center(track.last_hand_box);
track.latest_box = Move(track.latest_box, hand_delta);
track.last_hand_box = current_hand_box;

if (Distance(track.latest_box, track.path.back()) > path_step_threshold) {
    track.path.push_back(track.latest_box);
}
```

这个“手仍靠近 Track.latest_box”的条件很重要：物品放下后，手继续在冰箱内移动去拿另一个物品时，不应继续把第一件物品拖着走。

若物品放下后重新被 YOLO 检测到，后续 Track 会自动改回“使用物品 Detection 位移”；此后即使手继续移动，物品 Detection 没有移动，Track 也不会再向手的方向移动。

## 6. 如何知道“物品被移动了”

Track 层面只需要判断：它的完整估计框是否离开起点。

```cpp
move_distance = Distance(Center(track.latest_box), Center(track.start_box));

track.has_significant_motion =
    move_distance > move_threshold;
```

`move_threshold` 应大于正常 YOLO 抖动。例如用物品起始框对角线的一小部分作为阈值；具体数值之后用实际视频调。

这个标记只表示：

```text
本次手操作中，这条 Track 看起来确实移动过。
```

它**不等于**库存已经发生整理。

真正确认整理仍必须满足：

```text
1. 库存 item A 在当前稳定快照中，没有在旧位置直接匹配到；
2. 当前稳定快照中存在新位置物品 B；
3. A 绑定的 Track.path 中，至少一个完整估计框匹配 B.box；
4. A、B、Track 的对应关系唯一。
```

满足后才：

```text
保留 A.item_id；
把 A.anchor_box 更新到与 B 匹配的 Track 路径点；
把 A.last_seen_box 更新为 B.box；
```

## 7. 快照在什么时候生成（简要）

Track 只在有手阶段更新。手离开后：

```text
Track 冻结；
连续 N 帧无手、Detection 稳定；
这些 Detection 投票融合为 current_stable_snapshot。
```

快照生成的 N 帧中：

```text
不创建新 Track；
不更新旧 Track；
不修改库存；
```

快照生成后才：

```text
current_stable_snapshot vs InventoryDB；
若出现“旧位置 A 没有、新位置 B 出现”，
再查询 A 的冻结 Track.path，判断是否整理。
```

## 8. 一个完整的逐帧例子

库存初始状态：

```text
item_id=1：苹果，anchor_box=L，last_seen_box=L，VISIBLE
item_id=2：牛奶，anchor_box=R，last_seen_box=R，VISIBLE
active_tracks：空
```

### 第 1 帧：无手

```text
YOLO：苹果 L，牛奶 R；手：无。
→ 加入 no_hand_buffer；
→ 不创建 Track；
→ previous_frame = 本帧。
```

### 第 2 帧：手进入苹果附近，苹果仍部分可见

```text
YOLO：苹果 L_part，牛奶 R；手：H。
```

处理：

```text
1. 清空 no_hand_buffer；
2. active_tracks 为空，因此苹果 L_part 匹配不到已有 Track；
3. 查询库存：同类苹果 item_id=1 在手 H 附近；
4. 创建 Track#1：
   bound_item_id=1；
   start_box=L；latest_box=L；
   last_detected_box=第 1 帧苹果框 L；
   last_hand_box=H；path=[L]；
5. 用第 1 帧 L 到第 2 帧 L_part 的位移，更新 Track#1.latest_box；
6. previous_frame = 第 2 帧。
```

### 第 3 帧：苹果继续向右，仍有局部框

```text
YOLO：苹果 M_part，牛奶 R；手：H2。
```

处理：

```text
1. 苹果 M_part 先匹配 Track#1；
2. 匹配成功，不创建新 Track；
3. object_delta = Center(M_part) - Center(L_part)；
4. latest_box 按 object_delta 平移；
5. path 追加新的完整估计框；
6. previous_frame = 第 3 帧。
```

### 第 4 帧：苹果完全被挡住

```text
YOLO：只有牛奶 R；手：H3。
```

处理：

```text
1. 本帧没有 Detection 匹配 Track#1；
2. 手 H3 仍在 Track#1.latest_box 附近；
3. hand_delta = Center(H3) - Center(H2)；
4. latest_box 按 hand_delta 平移；
5. path 追加新的完整估计框；
```

### 第 5 帧：苹果在右侧露出，手仍在

```text
YOLO：苹果 R_part，牛奶 R；手：H4。
```

处理：

```text
1. 苹果 R_part 匹配 Track#1 的 latest_box；
2. 因为上一帧完全遮挡，本帧先保存 last_detected_box=R_part；
3. 不拿 R_part 和第 3 帧 L_part 直接算位移；
4. Track#1 保留在右侧的 latest_box。
```

### 第 6 帧：手离开

```text
YOLO：苹果 R，牛奶 R；手：无。
```

处理：

```text
1. 冻结 Track#1；
2. 清空 no_hand_buffer；
3. 第 6 帧作为新的无手缓冲第 1 帧；
4. 不更新 InventoryDB。
```

### 第 7 ～ N 帧：无手且稳定

```text
YOLO：苹果 R，牛奶 R；手：无。
```

处理：

```text
持续填充 no_hand_buffer；
Track#1 不更新；
连续 N 帧后生成稳定快照。
```

### 稳定快照比较库存

```text
库存 item_id=1 仍在 L；
快照中苹果 B 在 R；
苹果 B 与 item_id=1 的旧位置 L 不直接匹配；
但 B 与冻结 Track#1.path 中右侧估计框匹配；
→ 确认 item_id=1 整理到 R；
```

最后一次性提交库存，再清空 `active_tracks`。

# -------------------------------------------------------------------
# -------------------------------------------------------------------
# -------------------------------------------------------------------
