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
→ start_box=anchor_box=L；proxy_box=L。

第 3 帧：检测到手；苹果局部框移动到 M_part。
→ 用 L_part 到 M_part 的位移更新 Track#1.proxy_box；

第 4 帧：检测到手；YOLO 看不到苹果。
→ 用手框位移更新 Track#1.proxy_box；

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
→ 创建或更新 Track；不生成快照；不修改库存。

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

一条 Track 中的关键框和字段不能混淆：

```text
last_yolo_box
→ 上一次真正由 YOLO 识别到的物品框；
→ 可以是局部小框；
→ 用来计算“当前 YOLO 框相对上一帧 YOLO 框移动了多少”。
→ 物品被手完全挡住、YOLO 看不到物品时，它为空。

proxy_box
→ 当前完整物品的位置代理框；
→ 从 anchor_box（或 last_seen_box）开始；
→ 被完全遮挡时，仍可按手位移继续移动。

path
→ 本次手操作中每次更新后的 proxy_box 副本列表；
→ 记录完整经过路径，便于查看和作为未确认放下时的后备证据。

release_box
→ 已确认“物品已经放下”时的 proxy_box；
→ 为空表示尚未确认最终放下位置。
```

它们的关系可以写成：

```text
每帧需要当前位置 → 读写 proxy_box。

Track 每次更新 → 把当前 proxy_box 复制一份追加到 path。

确认放下后 → 记录 release_box。

最后判断整理 → 优先使用 release_box；没有 release_box 时才查看完整 path。
```

因此 `proxy_box` 和 `path` 不是重复：

```text
proxy_box = 当前这一刻的位置；
path = 本次手操作的完整历史位置。
```

Track 的主要状态只有三个：

```text
TRACKING_VISIBLE
→ 当前仍能由 YOLO 看见物品，可以完整可见或部分被手遮挡；last_yolo_box 必须有值；
→ 物品框的连续位移可以更新 proxy_box。

FULL_HAND_OCCLUDED
→ 当前看不到物品；last_yolo_box 必须为空；
→ proxy_box 只能跟随手位移移动。

PLACED
→ 已经有足够证据认为物品放下；release_box 有值；
→ 不再让该 Track 跟随手移动。
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

`active_tracks` 此时若有，也是上一轮手操作留下的冻结证据；本帧不能修改它们的 `proxy_box` 或 `path`。

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

本帧一旦有手，固定先做：

```cpp
operation_pending = true;
no_hand_buffer.clear();
// 本帧不参与快照，InventoryDB 不修改
```

然后依次执行下面三步。

### 4.1 所有当前物品 Detection 先更新已有 Track

这里要看所有当前物品 Detection，不只看手附近的物品。因为手可能已放下苹果、转去拿牛奶；苹果虽不在手附近，仍必须接回它原来的 Track。

对本帧每个物品 Detection `D`，按下面的分支处理：

```text
1. 先从 active_tracks 中找 cls_id 与 D 相同的 Track。

2. 对每一条候选 Track：(尝试匹配)

   如果 track.state == TRACKING_VISIBLE： (物品仍可见，可能完整可见或部分被手挡)
       用 D.box 与 track.last_yolo_box 比较。
       主要看中心位置；宽高阈值可以宽松。

   如果 track.state == FULL_HAND_OCCLUDED： (完全被手挡)
       用 D.box 与 track.proxy_box 比较。
       只要求 D 的中心在 proxy_box 内部或附近；
       不要求局部小框与完整 proxy_box 宽高相同。

   如果 track.state == PLACED：(手放开)
       用 D.box 与 track.last_yolo_box 比较，确认这还是已放下的物品。
       正常情况下不移动 proxy_box，也不让它重新跟随手。

3. 如果没有任何 Track 匹配 D：(匹配不上)
       D 暂时标记为“未匹配 Track”；
       后面交给 4.2 查询库存、决定是否创建新 Track。

4. 如果只有一条 Track 匹配 D：(匹配上, 一对一)
       D 绑定这条 Track；
       更新这条 Track；
       不查库存，不创建新 Track。

5. 如果多条 Track 都能匹配 D，且无法选出唯一最佳：(匹配上, 多对一)
       不强行绑定；
       D 暂时标记为“未匹配 Track”。
```

其中每个 Detection、每条 Track 在本帧最多只能匹配一次；“唯一最佳”表示多个候选中有明显最接近的一条。

当第 4 步匹配成功时，更新方式按状态分三种：

```text
如果 track.state == TRACKING_VISIBLE：(之前【部分被手挡, 部分可见】, 现在可能【部分被手挡】, 也可能【手放开】)
    object_delta = Center(D.box) - Center(track.last_yolo_box)
    last_yolo_box = D.box
    last_hand_box = 当前 hand_box

    如果物品连续两帧位移很小，
    且手已不再覆盖物品，
    且 anchor_valid=true 时，D.box 的宽高与 anchor_box 大致相近：
        state = PLACED      (之前【部分被手挡, 部分可见】, 现在可以确认【手放开】)
        proxy_box 的中心校正为 D.box 的中心
        proxy_box 的宽高保持原样（通常就是 anchor_box 的宽高）
        release_box = proxy_box
    否则：
        proxy_box 按 object_delta 平移

    把当前 proxy_box 追加到 path

如果 track.state == FULL_HAND_OCCLUDED：(之前【被手挡完全挡住】, 现在可能【部分被手挡】, 也可能【手放开】)
    说明物品刚从完全遮挡中重新出现
    last_yolo_box = D.box
    last_hand_box = 当前 hand_box
    state = TRACKING_VISIBLE
    本帧不计算 object_delta
    不修改 proxy_box
    把当前 proxy_box 追加到 path
    下一帧再用相邻 YOLO 框的位移更新 proxy_box

如果 track.state == PLACED：
    object_delta = Center(D.box) - Center(track.last_yolo_box)

    如果手重新覆盖该物品，且 object_delta 明显：
        state = TRACKING_VISIBLE
        release_box 置空
        proxy_box 按 object_delta 平移
        last_hand_box = 当前 hand_box
        后续重新记录这一次移动和新的放下位置

    last_yolo_box = D.box
    把当前 proxy_box 追加到 path
```

原因：刚重新出现的 `D.box` 仍可能是局部小框，不能直接拿它改写完整物品的 `proxy_box`；且上一帧没有物品 YOLO 框，无法计算可靠的 `object_delta`。完全遮挡期间的位置变化，已经由 4.3 的手位移更新并记录到了 `path`。

`state` 是 4.1 选择比较对象的主要依据；`last_yolo_box` 是否为空只用于保证状态没有写错：

```text
TRACKING_VISIBLE / PLACED → last_yolo_box 应有值；
FULL_HAND_OCCLUDED        → last_yolo_box 应为空。
```

#### 放下是怎样确认的（简短例子）

```text
第 5 帧：手还在冰箱；苹果刚重新出现于右侧 R_part。
→ state 从 FULL_HAND_OCCLUDED 变为 TRACKING_VISIBLE；
→ 只恢复 last_yolo_box，不修改 proxy_box，也不立刻认定已经放下。

第 6 帧：手已移向牛奶；苹果仍在 R，且与第 5 帧几乎没动。
→ 苹果不再被手覆盖；
→ 若 D.box 的宽高也接近 anchor_box：
  state = PLACED；
  proxy_box 的中心校正为 D.box 中心，宽高保持原样；
  release_box = 校正后的 proxy_box（右侧 R）。

第 7 帧：手继续移动或遮住牛奶。
→ 苹果的 Track 保持 PLACED；不再跟随手移动。
```

### 4.2 剩余 Detection 再尝试创建新 Track

只处理没有匹配到旧 Track、且位于手附近的 Detection。

从库存中筛选：

```text
status = VISIBLE；
cls_id 与 D 相同；
还没有被 active Track 绑定；
位置也在 hand_box 附近。
```

比较 D 与库存物品时：

```text
优先参考 last_seen_box；
anchor_valid=true 时也参考 anchor_box；
D 可能是局部小框，可用“交集 / 较小框面积”很高的包含关系。
```

```text
唯一匹配一个库存 item
→ 创建 Track，并绑定该 item_id。

没有匹配或多个都无法区分
→ 不创建整理用的正式 Track，先略过。
```

创建时：

```cpp
track.bound_item_id = item.item_id;
track.start_box = item.anchor_valid ? item.anchor_box : item.last_seen_box;
track.proxy_box = track.start_box;
track.state = TRACKING_VISIBLE;
track.last_yolo_box = D.box;
track.last_hand_box = current_hand_box;
track.path = [track.start_box];
track.release_box = 空;
```

若 `previous_frame` 中有对应物品框，可立即用“上一帧框 → 当前 D”的位移补上手刚进入的第一段移动，再把新的 `proxy_box` 追加到 `path`；没有则把当前 `proxy_box` 再追加一次，从下一帧开始计算物品位移。

### 4.3 没有 Detection 接上的 Track：按完全遮挡处理

本帧可能看得到牛奶，却看不到已被手遮住的苹果。因此，对每条本帧没有被 Detection 接上的旧 Track：

```text
若手仍靠近该 Track 的 proxy_box
→ 认为物品可能被手完全遮挡；
→ 用手位移更新 proxy_box。
```

```cpp
if (track.state == PLACED) {
    track.release_box.reset();  // 已再次被手操作，旧放下位置失效
    track.last_hand_box = current_hand_box;
    // 刚重新拿起的第一帧没有可靠 hand_delta，不移动 proxy_box
} else {
    hand_delta = Center(current_hand_box) - Center(track.last_hand_box);
    track.proxy_box = Move(track.proxy_box, hand_delta);
    track.last_hand_box = current_hand_box;
}

track.state = FULL_HAND_OCCLUDED;
track.last_yolo_box.reset();  // 已没有可连续比较的物品 YOLO 框
track.path.push_back(track.proxy_box);

// last_yolo_box 只能保存物品框，绝不能写成手框
```

这里不再按“距离是否足够远”筛掉路径点：一次手操作的帧数有限，`path` 直接保留每次更新后的完整轨迹，方便查看。手离开后 Track 会冻结，不会继续追加手离开冰箱的路线。

然后检查库存中手附近、尚未有 active Track 的 VISIBLE 物品：

```text
若该物品当前没有 Detection，也没有 Track
→ 创建候选 Track。
```

这就是“手一进来就完全盖住物品”的情况。此类候选 Track 初始化为：

```text
state = FULL_HAND_OCCLUDED；
proxy_box = anchor_box（没有 anchor_box 时用 last_seen_box）；
last_yolo_box = 空；
path = [proxy_box]；
release_box = 空。
```

第一帧只记录 `last_hand_box`，下一帧才有手位移可计算。

手附近有多个可见库存物品时，可以都建立候选 Track；以后只有路径与稳定快照新位置唯一对应时，才确认哪一个整理。

`OCCLUDED` 库存物品通常不参与 Track 创建；它们主要在稳定快照阶段处理“重新可见”。

## 5. 物品是否移动、快照如何使用 Track

Track 自己只记录“看起来移动过”：

```cpp
Distance(Center(track.proxy_box), Center(track.start_box)) > move_threshold
```

这不等于已经确认整理。

手离开后，Track 冻结。连续 N 帧无手、稳定 Detection 生成 `current_stable_snapshot`。快照与库存比较时：

```text
库存中的旧物品 A 没有在旧位置直接匹配到；
当前快照出现新位置物品 B；

如果 A 所绑定 Track 有 release_box：
    若 B.box 与 release_box 唯一匹配：
        确认 A 整理到了 B。

    若 B.box 不匹配 release_box，或无法唯一匹配：
        不再退回使用 path；
        说明“已确认的放下位置”与当前快照互相矛盾，
        按普通拿出、放入或遮挡流程处理。

如果没有 release_box：
    说明手离开得太快，尚未来得及确认放下；
    若 B.box 与 Track.path 中某个 proxy_box 历史点唯一匹配：
        仍可确认 A 整理到了 B。
```

确认后：

```text
A.item_id 保持不变；
A.anchor_box 更新为与 B 匹配的 release_box 或 Track 路径点；
A.last_seen_box 更新为 B.box。
```

若没有唯一 Track 证据，就按遮挡、拿出或放入的普通库存流程处理。

## 6. 简短逐帧例子：苹果从左边移到右边

```text
库存：item_id=1 的苹果在 L；没有 active Track。

第 1 帧，无手：苹果 L；进入快照缓冲。

第 2 帧，有手：苹果 L_part。
→ 没有旧 Track；L_part 在手附近；
→ 与库存 item_id=1 唯一对应；创建 Track#1。

第 3 帧，有手：苹果 M_part。
→ M_part 先匹配 Track#1.last_yolo_box；
→ 匹配成功，更新 Track#1.proxy_box；把当前位置追加到 path；不新建 Track。

第 4 帧，有手：看不到苹果。
→ state=FULL_HAND_OCCLUDED；
→ 用手位移更新 Track#1.proxy_box；last_yolo_box 置空；当前位置追加到 path。

第 5 帧，有手：苹果 R_part 重新出现。
→ 用 R_part 与 Track#1.proxy_box 接回 Track；
→ state=TRACKING_VISIBLE；last_yolo_box=R_part；本帧不算物品位移；当前位置追加到 path。

第 6 帧，无手：冻结 Track#1，开始无手快照缓冲。

后续连续 N 帧无手稳定：生成快照；
此时尚未来得及确认 release_box；
快照苹果在 R 与 Track#1.path 唯一对应；
→ 确认 item_id=1 从 L 整理到 R。
```
