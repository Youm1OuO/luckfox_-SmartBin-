# ByteTrack 与稳定快照：多物品的一次完整运行模拟

本文模拟一次真实些的开门操作：冰箱里同时有苹果、牛奶、苏打罐；用户只把苹果从左边移到右边。

目的不是记住每个函数，而是看清两套编号怎样出现、怎样配合：

```text
item_id  = 库存物品编号（库存系统自己从 1 开始编号）
track_id = 视频轨迹编号（ByteTrack 自己从 1 开始编号）
```

说明：`TrackEvidence`、`HeldProxyEvidence` 和快照对比已经在当前代码中实现。本文使用“先连续稳定 N 帧，再生成快照”的改进方案；当前代码还没有把这层稳定窗口放到快照前面。

假设：10 FPS、稳定窗口 `N=4`、手被正常检测到。YOLO 在每帧按“苹果、牛奶、苏打罐、手”的顺序输出检测结果。

---

## 0. 本例中的位置和目标

```text
L = 左边            C = 中间            R = 右边

苹果   初始在 L，之后被移动到 R
牛奶   一直在 C，不移动
苏打罐 一直在 R 的下层/另一位置，不移动
```

最后会形成：

```text
库存： item#1 苹果、item#2 牛奶、item#3 苏打罐
轨迹： Track#1 苹果、Track#2 牛奶、Track#3 苏打罐、Track#4 手
```

注意：两个编号刚好都从 1 开始只是巧合，不能认为 `item#1` 永远等于 `Track#1`。

---

## 阶段 1：第一帧，ByteTrack 先产生多条轨迹

### 第 1 帧：YOLO 看到三个物品

```cpp
std::vector<Detection> detections = {
    { .box = L, .score = 0.92, .cls_id = apple    },
    { .box = C, .score = 0.89, .cls_id = milk_box },
    { .box = R, .score = 0.94, .cls_id = soda_can },
};
```

此时还没有任何旧 Track。`tracker.update(detections, 1)` 会依次创建：

```cpp
Track#1 = { .track_id=1, .cls_id=apple,    .box=L, .state=NEW };
Track#2 = { .track_id=2, .cls_id=milk_box, .box=C, .state=NEW };
Track#3 = { .track_id=3, .cls_id=soda_can, .box=R, .state=NEW };
```

可以理解为 ByteTrack 说：

```text
“我第一次在视频中看见这三个目标；先临时叫它们 1、2、3。”
```

它们还是 `NEW`，当前代码不会把 `NEW` 轨迹输出给业务层。

稳定窗口：无手，刚开始计数。

```text
stable_count = 1
```

这里还没有 `item_id`，因为系统尚未确认画面是稳定库存。

---

## 阶段 2：第 2 帧，三条轨迹变成已确认轨迹

### 第 2 帧：三个物品还在原位置

YOLO 框略有抖动：

```text
apple    at L'   （仍接近 L）
milk_box at C'   （仍接近 C）
soda_can at R'   （仍接近 R）
```

ByteTrack 会逐项比较“当前 Detection”与“上一帧 Track”。以苹果为例：

```text
Detection(apple, L') 与 Track#1(apple, L)

类别相同             → 可以尝试匹配
两个框的 IoU 足够高  → 匹配成功
```

因此不创建 `Track#4`，而是更新原来的 `Track#1`：

```cpp
Track#1.box = L';
Track#1.hit_count = 2;
Track#1.state = TRACKED;
```

牛奶、苏打罐同理：

```cpp
Track#2.state = TRACKED;
Track#3.state = TRACKED;
```

这时 `tracker.update()` 返回给业务层的 `tracks` 是：

```cpp
tracks = [ Track#1(苹果), Track#2(牛奶), Track#3(苏打罐) ];
```

稳定窗口：

```text
stable_count = 2
```

---

## 阶段 3：第 3、4 帧，快照确认“当前库存是什么”

第 3、4 帧中，三个物品都没有明显移动，也没有手：

```text
stable_count = 3 → 4
```

达到 `N=4` 后，程序才把这 4 帧的 YOLO `Detection` 放入 `SnapshotBuffer` 投票，生成：

```cpp
Snapshot snap1_ = {
    .items = {
        { .cls_id=apple,    .box=L, .best_score=0.93, .count=4 },
        { .cls_id=milk_box, .box=C, .best_score=0.90, .count=4 },
        { .cls_id=soda_can, .box=R, .best_score=0.95, .count=4 },
    },
    .has_hand = false,
};
```

注意：快照中仍没有 `track_id`。它只说“稳定画面中看到了什么”。

### 此时库存系统才开始分配 `item_id`

首次快照要初始化库存，代码按快照内物品的顺序调用：

```cpp
inventory_.add_item(-1, apple,    L, ...);  // 返回 1
inventory_.add_item(-1, milk_box, C, ...);  // 返回 2
inventory_.add_item(-1, soda_can, R, ...);  // 返回 3
```

于是库存表是：

```text
item#1 = 苹果    box=L  status=VISIBLE
item#2 = 牛奶    box=C  status=VISIBLE
item#3 = 苏打罐  box=R  status=VISIBLE
```

此时两套编号的关系看起来是：

```text
item#1 苹果  ← 当前画面里恰好是 Track#1 苹果
item#2 牛奶  ← 当前画面里恰好是 Track#2 牛奶
item#3 苏打罐 ← 当前画面里恰好是 Track#3 苏打罐
```

但这是因为它们都从 1 开始、物品顺序也碰巧一致。程序不能依赖这种巧合。

库存初始化完成后：

```text
baseline_snapshot = snap1_
operation_context_ = 空
```

从这一刻开始，到下一份稳定快照产生前，ByteTrack 每帧产生的过程信息都写入同一个 `operation_context_`。

---

## 阶段 4：用户开始移动苹果，快照暂停，ByteTrack 继续

现在用户只移动苹果；牛奶和苏打罐始终静止。

### 第 5 帧：手碰到左边苹果

YOLO 看到：

```text
apple at L
milk_box at C
soda_can at R
hand at L
```

ByteTrack：

```text
苹果 Detection    → 继续匹配 Track#1
牛奶 Detection    → 继续匹配 Track#2
苏打罐 Detection  → 继续匹配 Track#3
手的 Detection    → 新建 Track#4（第一次看到手，状态 NEW）
```

这正好说明：`track_id` 是“谁首次出现，就取下一个可用编号”；它不会因为库存已有 3 件物品就自动等于 item_id。

此帧业务层调用：

```cpp
session.update_hand(hand_boxes, tracks, frame_id, time_ms);
```

对三个非手轨迹，`update_operation_context()` 会按位置找对应库存记录，因此建立三条过程证据：

```cpp
active_track_evidences[1] = {
    .track_id=1, .associated_item_id=1,
    .start_box=L, .end_box=L
};

active_track_evidences[2] = {
    .track_id=2, .associated_item_id=2,
    .start_box=C, .end_box=C
};

active_track_evidences[3] = {
    .track_id=3, .associated_item_id=3,
    .start_box=R, .end_box=R
};
```

以后我们只关注苹果这条：

```text
Track#1（短期视频轨迹） ↔ item#1（库存苹果）
```

与此同时，手框与 `item#1.visible_box` 有明显重叠：

```cpp
candidate_held_items[1] = 1;
```

意思是“手可能碰到了 item#1”，不是“item#1 已出库”。

由于检测到手：

```text
stable_count = 0
不生成库存快照
```

牛奶和苏打罐虽然一直稳定，但画面里有手、苹果正在操作，所以这一时段整体不生成库存快照。

---

## 阶段 5：手完全挡住苹果，苹果 Track 暂时丢失

### 第 6 帧：只看到手、牛奶、苏打罐

YOLO：

```text
hand at L
milk_box at C
soda_can at R
没有 apple
```

ByteTrack 的结果：

```text
Track#1（苹果）本帧匹配不到 detection
    → state: TRACKED 改为 LOST
    → 不立即删除，最多保留 60 帧

Track#2（牛奶）继续 TRACKED
Track#3（苏打罐）继续 TRACKED
Track#4（手）第二次匹配成功，变为 TRACKED
```

此时业务层看到：

```text
item#1 曾被手覆盖；
item#1 现在不在当前非手 tracks 中；
这种情况连续达到 2 帧。
```

于是建立：

```cpp
HeldProxyEvidence {
    .item_id = 1,                  // 被挡住的库存苹果
    .original_object_track_id = 1, // 遮挡前的苹果轨迹
    .held_by_hand_track_id = 4,    // 遮挡它的手轨迹
    .last_visible_box = L,
    .proxy_boxes = [ 手在L的框 ],
};
```

请特别看清楚：这里 `item_id=1` 与 `original_object_track_id=1` 数字相同，仍只是本例的巧合。

例如若用户先开门、手先入画，手可能先得到 `Track#1`，苹果才得到 `Track#2`；
而库存初始化的快照没有手，则苹果仍可能是 `item#1`。

```text
item#1（苹果） ↔ Track#2（苹果） ↔ Track#1（手）
```

所以代码必须分别保存 `item_id`、物品 `track_id`、手 `track_id`。

库存此时仍然没有改：

```text
item#1 仍是 VISIBLE，主位置仍是 L
item#2 仍是 VISIBLE，位置 C
item#3 仍是 VISIBLE，位置 R
```

稳定窗口：

```text
stable_count = 0
不生成快照
```

---

## 阶段 6：手把苹果带到右边，记录手部代理路径

### 第 7 帧

YOLO：

```text
hand at M
milk_box at C
soda_can at R
没有 apple
```

ByteTrack：

```text
Track#4.box 从 L 更新到 M
Track#1（苹果）仍是 LOST
Track#2、Track#3 继续 TRACKED
```

程序往同一条证据追加：

```cpp
HeldProxyEvidence.proxy_boxes = [ 手在L的框, 手在M的框 ];
```

### 第 8 帧

YOLO：

```text
hand at R 附近
milk_box at C
soda_can at R 的另一位置
没有 apple
```

ByteTrack：

```text
Track#4.box 从 M 更新到 R 附近
```

代理路径变成：

```cpp
HeldProxyEvidence.proxy_boxes = [ 手在L的框, 手在M的框, 手在R的框 ];
```

移动期间，真正持续工作的事情是：

```text
YOLO 每帧检测
ByteTrack 每帧更新 Track#1/#2/#3/#4
OperationContext 每帧更新轨迹和手持代理证据
```

但没有稳定库存快照，也没有修改库存。

---

## 阶段 7：手离开，苹果在新位置重新出现

### 第 9 帧：苹果在右边出现

YOLO：

```text
apple at R_new
milk_box at C
soda_can at R_old
没有 hand
```

ByteTrack 尝试将 `apple at R_new` 匹配到 LOST 的 `Track#1(苹果 at L)`：

```text
类别相同，但 L 与 R_new 距离太远，IoU 太低
→ 匹配失败
→ 苹果创建新轨迹 Track#5（状态 NEW）
```

这不是库存出错，只是 ByteTrack 的正常限制：它依据相邻框的重叠匹配，远距离瞬移/完全遮挡后不一定能续上旧轨迹。

稳定窗口重新开始：

```text
stable_count = 1
```

### 第 10、11、12 帧：苹果在右边稳定

YOLO 一直看到：苹果 R_new、牛奶 C、苏打罐 R_old；没有手。

ByteTrack：

```text
Track#5（苹果）第二次成功匹配后变为 TRACKED
Track#2（牛奶）一直保持 TRACKED
Track#3（苏打罐）一直保持 TRACKED
Track#4（手）因消失变 LOST，之后超时会移除
```

稳定窗口：

```text
stable_count = 2 → 3 → 4
```

此时 4 帧都无手、物品没有明显变化，因此用这 4 帧生成第二份稳定快照：

```cpp
Snapshot snap2 = {
    .items = {
        { .cls_id=apple,    .box=R_new, .count=4 },
        { .cls_id=milk_box, .box=C,     .count=4 },
        { .cls_id=soda_can, .box=R_old, .count=4 },
    },
    .has_hand = false,
};
```

又一次强调：`snap2` 里没有 `Track#5`。快照只保留最终稳定画面。

现在才执行：

```cpp
compare_snapshots(snap1_, snap2, operation_context_);
```

---

## 阶段 8：快照提出变化，ByteTrack 证据解释变化

### 8.1 只看快照，会得到什么？

```text
苹果：snap1 在 L，snap2 在 R_new
      → 左边苹果 A 消失，右边苹果 B 出现

牛奶：snap1 在 C，snap2 仍在 C
      → 原位置匹配，什么都不做

苏打罐：snap1 在 R_old，snap2 仍在 R_old
      → 原位置匹配，什么都不做
```

因此业务层只对苹果产生疑问：

```text
item#1 的苹果：
    是被拿走后，右边放进了另一颗苹果？
    还是原来的 item#1 从 L 整理到了 R_new？
```

### 8.2 再读取 ByteTrack 过程证据

代码对苹果调用：

```cpp
relocation_evidence_score(
    item_id = 1,
    from = L,
    to = R_new
);
```

它查看本轮 `operation_context_`，找到：

```text
TrackEvidence：
    Track#1 在操作刚开始属于 item#1，起点是 L

HeldProxyEvidence：
    item_id = 1
    original_object_track_id = 1
    held_by_hand_track_id = 4
    proxy_boxes 最后一个手框在 R 附近
```

于是代码检查：

```text
item#1 是否先在 L 被手遮住？                    是
这只手最后的位置是否靠近新苹果 B 的 R_new？     是
A 与 B 是否同类？                                是，都是 apple
A 与 B 的 bbox 面积、宽高比是否接近？           是
有没有另一个同样像的“新苹果”可选？              没有
```

证据足够强，得到：

```cpp
RelocationDecision::CONFIRMED
```

于是业务层只更新库存 `item#1`：

```cpp
inventory_.update_anchor_item(
    item_id = 1,
    track_id = -1, // 当前实现不把新的 Track#5 固化进库存
    box = R_new,
    ...
);
```

最终库存：

```text
item#1 = 苹果，位置从 L 更新到 R_new，状态 VISIBLE
item#2 = 牛奶，仍在 C，状态 VISIBLE
item#3 = 苏打罐，仍在 R_old，状态 VISIBLE

事件 = MOVED（整理）
```

没有发生：

```text
item#1 OUT（苹果被删除）
item#4 IN（右边新建一颗苹果）
```

对比结束后：

```text
baseline_snapshot = snap2
operation_context_ 清空并重建
```

下一次操作会重新积累新的 ByteTrack 过程证据。

---

## 对照：用户真的拿走苹果时

前面第 1～6 阶段相同；但手离开后，苹果没有在任何新位置出现。

第二份稳定快照是：

```text
snap2：只有牛奶 C、苏打罐 R_old
```

快照对比结果：

```text
item#1 苹果在 L 消失；没有任何新苹果 B 可与它组成“整理候选”。
```

因此不会执行 `relocation_match(A, B)`，而是按普通规则：

```text
item#1：VISIBLE → OUT
事件：OUT（取出）
```

中间的 ByteTrack / HeldProxyEvidence 不能单独让物品出库；它们只说明“苹果可能曾被手拿过”。真正判 OUT 的原因仍是：最终稳定快照中苹果确实不见了。

---

## 最后用一句话区分它们

```text
Snapshot：稳定前后，冰箱里有什么不同？
item_id ：库存系统要更新哪一件物品？
track_id：操作过程中，哪个连续画面目标在移动？

快照发现 item#1 的旧位置少了苹果、某处多了苹果；
ByteTrack 提供中间路径；
业务层据此决定是否更新 item#1 的位置。
```
