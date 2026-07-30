# Track 的说明（2.0）

## 1. Track 的作用

> Track 只负责确认：库存中原本【可见】的物品 A，是否被整理到了稳定快照中的物品 B。

```text
逐帧 YOLO 结果
→ 只是 Track 的临时观察；
→ 不直接修改库存。

稳定快照
→ 才用于库存结算。

Track 证据足够且 A、B 唯一对应
→ 输出 “A 整理到 B”。

Track 证据不足或对应不唯一
→ 不判整理，交给普通的遮挡 / 出库 / 入库流程。
```

`hold_and_move` 是 Track 内部的证据，不在库存对比主流程中单独使用。


## 2. 一个 A 只能有一条 Track

```python
track_buffer[A.item_id] = A 当前正在进行的唯一 Track
```

一条 Track 只绑定一个库存物品 A：

```python
track.bound_item_id = A.item_id
```

同一 A 已经存在 Track 时，本帧应更新这条旧 Track，不能再创建新的临时 Track。

“可以有多个候选 Track”的意思只是：一只手可能同时碰到 A1、A2；此时可以分别存在：

```python
track_buffer[A1.item_id]
track_buffer[A2.item_id]
```

它们都是弱候选，初始 `hold_and_move=False`。最后只有满足双向唯一关系的 Track 才能确认整理；不是“一条 Track 绑定多个 A”。

注意：当前帧的 YOLO Detection 没有库存 `item_id`。`track.bound_item_id` 只是 Track 记住的起始库存 A；当前 Detection 必须依靠类别、位置和上一帧信息与 Track 做一对一关联，不能直接靠 `item_id` 匹配。


## 3. Track 保存的数据

```python
Track:
    bound_item_id       # 起始库存物品 A.item_id
    cls_id

    start_box           # 创建时的 A.base_box
    proxy_box           # 最近一次已确认的“完整物品位置”
    last_item_box       # 上一次匹配到的物品 YOLO 框；完全被手挡住时为 None
    last_hand_box       # 上一次手框

    state               # VISIBLE / HAND_OCCLUDED / PLACED / INVALID
    hold_and_move       # 初始 False；本次操作结束后清除
    seen_hand_contact   # 是否已有有效接触证据
    seen_effective_move # 是否已有有效移动证据
    still_at_start_count        # 手移动后，A 仍在原位的连续帧数
    missing_without_hand_count  # A 不见且手也无法继续携带它的连续帧数

    release_box         # 已确认放下的位置；没有则为 None
    hand_path           # 本次操作的手框轨迹，仅用于调试或后续扩展
    proxy_path          # 已确认 proxy_box 的轨迹，仅用于调试或后续扩展
```

`proxy_box` 保持 `start_box` 的宽高。只有验证“手确实带着 A 移动”后，才允许按物品或手的位移平移它；物品完全看不见时，绝不能把手框当成物品框。


## 4. 每帧处理顺序

```python
def process_frame(item_detections, hand_detections):
    if 手的数量 == 0:
        process_no_hand_frame(item_detections)
        return

    if 手的数量 > 1:
        # 本版本不处理多手或交叉手。
        no_hand_buffer.clear()
        track_session_is_ambiguous = True
        return

    hand = 唯一的手框
    no_hand_buffer.clear()
    operation_pending = True

    # 第一步：先更新旧 Track；绝不为已有 Track 的 A 再创建 Track。
    detection_track_pairs = 当前 Detection 与现有 Track 的双向唯一关联(
        item_detections, track_buffer
    )

    for D, track in detection_track_pairs:
        update_track_by_visible_item(track, D, hand)

    for track in 本帧没有匹配到 Detection 的旧 Track:
        update_track_by_hand_or_mark_invalid(track, item_detections, hand)

    删除 state == INVALID 的 Track

    # 第二步：只为“尚无 Track”的 A 创建候选 Track。
    for A in 库存物品:
        if A.status != 【可见】:
            continue

        if A.item_id in track_buffer:
            continue

        if 手与 A 有有效接触证据:
            create_candidate_track(A, hand, 当前与 A 对应的 Detection 或 None)
```

无手时，不更新 Track：

```python
def process_no_hand_frame(item_detections):
    if 上一帧有手:
        freeze_all_tracks()
        no_hand_buffer.clear()
        no_hand_buffer.append(当前帧)
        return

    if 物品明显移动、突然出现或突然消失:
        # 手可能漏检；此时不能生成稳定快照，
        # 但也不能创建可确认整理的正式 Track。
        no_hand_buffer.clear()
        operation_pending = True
        return

    no_hand_buffer.append(当前帧)
```


## 5. 创建候选 Track

创建条件只能来自原本【可见】的 A：

```text
情况 1：当前仍看得见 A，且手与 A 的 Detection 有相邻 / 重叠等接触证据；

情况 2：当前看不见 A，但手对 A.base_box 有足够高的覆盖，
        可以认为手把 A 完全盖住。
```

```python
def create_candidate_track(A, hand, D_or_None):
    track = Track()
    track.bound_item_id = A.item_id
    track.cls_id = A.cls_id

    track.start_box = A.base_box
    track.proxy_box = A.base_box
    track.last_hand_box = hand.box

    track.hold_and_move = False
    track.seen_hand_contact = True
    track.seen_effective_move = False
    track.still_at_start_count = 0
    track.missing_without_hand_count = 0
    track.release_box = None

    track.hand_path = [hand.box]
    track.proxy_path = [track.proxy_box]

    if D_or_None is not None:
        track.state = VISIBLE
        track.last_item_box = D_or_None.box
    else:
        track.state = HAND_OCCLUDED
        track.last_item_box = None

    track_buffer[A.item_id] = track
```

创建 Track 只表示“手可能在操作 A”，不表示 A 已经被拿住或已经整理。


## 6. 先验证，再更新 `proxy_box`

`hold_and_move=True` 必须同时有：

```text
1. 接触证据：手确实接触 / 覆盖 A；
2. 有效移动证据：A 或携带 A 的手确实移动。
```

每一帧都遵守同一个顺序：

```text
先计算 object_delta 与 hand_delta
→ 判断它们是否能证明“手带着 A 移动”
→ 证明成功后，才更新 proxy_box、proxy_path 与 hold_and_move。
```

`last_item_box` 和 `last_hand_box` 可以记录最新观察结果；但证据不足时，`proxy_box` 必须停在上一次已确认的位置。

### 6.1 A 仍能看见

```python
def update_track_by_visible_item(track, D, hand):
    # D 与 track 已在本帧双向唯一关联。

    if track.state == PLACED and 手没有重新接触 D:
        # 手可能已经转去操作别的物品；已放下的 A 不应继续跟手移动。
        track.last_item_box = D.box
        track.last_hand_box = hand.box
        return

    if track.state == PLACED and 手重新接触 D:
        # 同一次操作中又拿起 A，旧的放下位置失效。
        track.state = VISIBLE
        track.release_box = None

    hand_delta = center(hand.box) - center(track.last_hand_box)

    if track.last_item_box is not None:
        object_delta = center(D.box) - center(track.last_item_box)

        if 手与 D 有接触证据:
            track.seen_hand_contact = True

        hand_and_item_move_together = (
            hand_delta 的长度 > hand_move_eps
            and object_delta 的长度 > move_eps
            and object_delta 与 hand_delta 的方向、幅度大致一致
        )

        if hand_and_item_move_together:
            # 现在才确认：手正在带着 A 移动。
            track.proxy_box = move_box(track.proxy_box, object_delta)
            track.proxy_path.append(track.proxy_box)
            track.seen_effective_move = True
            track.still_at_start_count = 0

        elif hand_delta 的长度 > hand_move_eps \
             and D 位于 track.start_box 附近:
            track.still_at_start_count += 1

            if track.still_at_start_count >= still_at_start_frame_limit:
                # 手已经明显移动，A 却连续留在原处；候选 Track 不成立。
                track.state = INVALID
                return
        else:
            # 本帧没有足够证据，不改变已确认的 proxy_box。
            track.still_at_start_count = 0

    track.last_item_box = D.box
    track.last_hand_box = hand.box
    track.hand_path.append(hand.box)
    track.state = VISIBLE

    if track.seen_hand_contact and track.seen_effective_move:
        track.hold_and_move = True

    if D 已稳定且手不再覆盖 D:
        track.state = PLACED
        track.release_box = track.proxy_box
```

### 6.2 A 被手完全遮挡

```python
def update_track_by_hand_or_mark_invalid(track, item_detections, hand):
    A = item_by_id[track.bound_item_id]
    hand_delta = center(hand.box) - center(track.last_hand_box)

    if hand_delta 的长度 > hand_move_eps \
       and 当前 Detection 中存在与 A 对应、且位于 track.start_box 附近的物品:
        # 手已经移动，A 又明确出现在原位，说明没有被手带走。
        track.state = INVALID
        return

    candidate_proxy_box = move_box(track.proxy_box, hand_delta)

    hand_can_continue_carrying_A = (
        手对 candidate_proxy_box 仍有足够覆盖
        or 手与 candidate_proxy_box 有可靠的携带关联
    )

    start_position_is_still_empty = (
        当前 Detection 中不存在与 A 对应、且位于 track.start_box 附近的物品
    )

    if hand_delta 的长度 > hand_move_eps \
       and hand_can_continue_carrying_A \
       and start_position_is_still_empty:
        # 现在才确认：手带着被完全遮挡的 A 移动。
        track.proxy_box = candidate_proxy_box
        track.proxy_path.append(track.proxy_box)
        track.seen_hand_contact = True
        track.seen_effective_move = True
        track.hold_and_move = True
        track.missing_without_hand_count = 0
        track.state = HAND_OCCLUDED
    else:
        # 单帧漏检不能证明 A 被移动；不改 proxy_box，短暂保留候选。
        track.missing_without_hand_count += 1
        if track.missing_without_hand_count > lost_frame_limit:
            track.state = INVALID

    track.last_hand_box = hand.box
    track.last_item_box = None
    track.hand_path.append(hand.box)
```

“连续若干帧”用于抵抗 YOLO 单帧漏检或小抖动，不能只凭一帧就把候选 Track 判真或判假。


## 7. 稳定快照后，输出唯一整理配对

手离开后，Track 冻结。连续 N 帧无手且稳定得到快照后，Track 只处理仍未匹配的 A、B。

```python
def get_unique_move_pairs(库存物品, 快照物品, frozen_tracks):
    if track_session_is_ambiguous:
        return []

    track_A_to_B = {}
    track_B_to_A = {}

    for track in frozen_tracks:
        if track.state == INVALID:
            continue

        if track.hold_and_move != True:
            continue

        A = item_by_id[track.bound_item_id]
        if A.status != 【可见】 or A.affirm:
            continue

        expected_end_box = track.release_box
        if expected_end_box is None:
            expected_end_box = track.proxy_box

        for B in 快照物品:
            if B.item_id != -1:
                continue

            if B.cls_id != A.cls_id:
                continue

            if abs(B.box.width - A.box.width) > eps_w:
                continue

            if abs(B.box.height - A.box.height) > eps_h:
                continue

            if B.box 与 expected_end_box 不足够接近:
                continue

            track_A_to_B[A.item_id].add(B.temporary_id)
            track_B_to_A[B.temporary_id].add(A.item_id)

    unique_move_pairs = []

    for A_item_id in track_A_to_B:
        if track_A_to_B[A_item_id].size != 1:
            continue

        B_id = track_A_to_B[A_item_id][0]

        if track_B_to_A[B_id].size != 1:
            continue

        unique_move_pairs.append((A_item_id, B_id))

    return unique_move_pairs
```

库存对比主流程只使用 `unique_move_pairs`：确认 A 与 B 为整理后，再更新 A 的 `box`、`base_box` 和其他物品的 `block_ids`。


## 8. 边界与取舍

```text
多手、手交叉、手身份丢失
→ 本版本不使用 Track 确认整理。

用户拿走 A，又放入同类同尺寸 B，且 B 落在 Track 终点附近
→ 仍可能被判为 A 整理；这是已接受的业务取舍。

A 或 B 的候选不唯一
→ Track 放弃确认整理，不能强行分配 item_id。
```
