# Track 的说明（2.0）

## 1. Track 的作用

> Track 先确认：库存中原本【可见】的物品 A，是否真的被手移动过。
> Track 不逐帧修改库存；稳定快照生成后，才把“已移动”的证据和最终画面一起用于结算。

```text
逐帧 YOLO 结果
→ 只是 Track 的临时观察；
→ 不直接修改库存。

稳定快照
→ 才用于库存结算。

hold_and_move=True 的 A，若有唯一终点 B
→ 输出 “A 整理到 B”。

hold_and_move=True 的 A，若没有任何终点 B
→ 按 2.0 的单次结果约定，输出 “A 出库”。

候选终点不唯一 / Track 会话 ambiguous
→ 本轮不强行修改该 A。
```

这里的“没有终点 B 就出库”是 2.0 的业务约定：一次结算只处理一个最终结果，
不支持“把 A 移动后又放到不可见处”这类复合动作。若以后要支持该类动作，不能沿用这条规则。

Track 从创建起一直可更新，直到连续 N 帧无手已经生成稳定快照。手在凑满 N 帧前再次出现时，说明还没有得到可结算的静态结果；此时 Track 不冻结，仍可继续更新。最终结算仍只接受一个【整理】或【出库】结果。


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

它们都是弱候选，初始 `hold_and_move=False`。稳定快照生成前会删除没有确认移动的候选；
保留下来的 Track 才参与“整理 / 出库”判断。不是“一条 Track 绑定多个 A”。

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
    shelter_or_hold     # 已有手接触或手完全遮挡 A 的候选证据
    hold_and_move       # 初始 False；本次操作结束后清除
    seen_hand_contact   # 是否已有有效接触证据
    seen_effective_move # 是否已有有效移动证据
    still_at_start_count        # 手移动后，A 仍在原位的连续帧数
    missing_without_hand_count  # A 不见且手也无法继续携带它的连续帧数

    release_box         # 已确认放下的位置；没有则为 None
    hand_path           # 本次操作的手框轨迹，仅用于调试或后续扩展
    proxy_path          # 已确认 proxy_box 的轨迹，仅用于调试或后续扩展

    frozen              # 初始 False；稳定快照生成后设为 True，结算期间禁止更新
```

`proxy_box` 保持 `start_box` 的宽高。只有验证“手确实带着 A 移动”后，才允许按物品或手的位移平移它；物品完全看不见时，绝不能把手框当成物品框。


## 4. 每帧处理顺序

Track 还需要保留上一帧的完整 YOLO 输出（物品检测、手检测以及“上一帧是否有手”）。
它不是 `no_hand_buffer`：无论当前是否需要收集快照，都要在**本帧 Track 处理结束后**把当前结果保存为下一帧的上一帧结果。
因此，第一次出现手时可以用“当前手框 + 上一帧物品框”建立候选，而不会因当前物品已经被手遮住而漏掉 A。

```python
def process_frame(item_detections, hand_detections):
    if 手的数量 == 0:
        return process_no_hand_frame(item_detections)

    if 手的数量 > 1:
        # 本版本不处理多手或交叉手。
        clear_snapshot_buffer()
        operation_pending = True
        track_session_is_ambiguous = True
        return None

    hand = 唯一的手框
    clear_snapshot_buffer()
    operation_pending = True

    # 正常情况下冻结只会发生在快照生成后的同步结算期间。
    # 这里仍显式排除 frozen Track，保证任何更新函数都不会修改它们。
    active_track_buffer = track_buffer 中所有 track.frozen == False 的 Track

    # 第一步：先更新旧 Track；绝不为已有 Track 的 A 再创建 Track。
    detection_track_pairs = 当前 Detection 与现有 Track 的双向唯一关联(
        item_detections, active_track_buffer
    )

    for D, track in detection_track_pairs:
        update_track_by_visible_item(track, D, hand)

    for track in active_track_buffer 中本帧没有匹配到 Detection 的旧 Track:
        update_track_by_hand_or_mark_invalid(track, item_detections, hand)

    删除 state == INVALID 的 Track

    # 第二步：2.0 的单次结果约定只允许本段连续手操作有一个候选 Track。
    # 若已有 Track，不能再因手框靠近相邻物品继续新建。
    if track_buffer 为空:
        candidate_tracks = []
        for A in 库存物品:
            if A.status != 【可见】:
                continue
            if 手与 A 有有效接触证据:
                candidate_tracks.append(A, 当前与 A 对应的 Detection 或 None)

        # 完整被手覆盖 > 当前帧实际重叠 > 仅附近；分数相同按稳定 item_id。
        A, D = candidate_tracks 中接触证据最强的一条
        if A 存在:
            create_candidate_track(A, hand, D)
```

补充两个关联保护：

1. “最多一条候选”的限制会避免实际只拿 A、再把 A 放到 B 前方时，B 因手框靠近或临时消失又生成假 Track；B 应留给稳定快照和 `block_ids` 处理。这个限制正是 2.0 “单次只处理一个最终结果”的实现边界。
2. 若 Track 对应的前景物品原本挡住了一件 `OCCLUDED` 库存物品，手带着该前景物品移动时，优先尝试 `proxy_box + hand_delta` 附近、且仍与手接触的 Detection。这样前景物品移开后，原位置重新露出的同类后景 Detection 不会错误接管前景 Track。

当前帧第一次检测到手、而前一帧没有手时，要保留前一帧的 YOLO 结果：
用**当前手框**与前一帧中 A 的框（或库存中 A 的最近 box）判断接触 / 完全遮挡，
再创建候选 Track。不能只依赖当前帧的物品 Detection，因为 A 可能已经被手遮住。

无手时不更新 Track。只有当前确实需要生成快照时，才把这一帧交给快照模块：

```python
def process_no_hand_frame(item_detections):
    # 调用者已确认当前帧无手。
    # 一次结算完成后，should_collect_snapshot() 为 False；此时是待机，
    # 但外层仍会继续以正常帧率运行 YOLO 和手检测。
    if not should_collect_snapshot():
        return None

    # collect_no_hand_snapshot_frame 是 no_hand_buffer 唯一的写入者。
    return collect_no_hand_snapshot_frame(item_detections)
```

有手时，`process_frame()` 会立即令 `operation_pending = True` 并调用 `clear_snapshot_buffer()`；因此手在凑满 N 帧前再次出现时，已累计的无手帧会被清空，但 Track 保持可更新状态。手再次离开后，`should_collect_snapshot()` 会重新为 True，开始累计新的连续 N 帧。


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

    track.shelter_or_hold = True
    track.hold_and_move = False
    track.frozen = False
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
    if track.frozen:
        return

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
            手与 D 有接触证据
            and hand_delta 的长度 > hand_move_eps
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
    if track.frozen:
        return

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


## 7. 稳定快照生成后先删除未移动候选，再输出整理 / 出库结果

Track 不在手离开时冻结；只有连续 N 帧无手已经生成稳定快照时，才冻结本次尚未结算的所有 Track。这样手在 N 帧前再次出现时，Track 仍可继续记录；最终输出仍受“单次只处理一个结果”的限制。

```python
def freeze_all_tracks():
    # 先删除“只碰到 / 只遮挡、但没有确认移动”的候选。
    # 因此冻结后 frozen_tracks 中每条 Track 都满足 hold_and_move == True。
    frozen_tracks.clear()
    track_ids_to_delete = []

    for item_id, track in track_buffer:
        if track.state == INVALID or track.hold_and_move != True:
            track_ids_to_delete.append(item_id)
            continue

        track.frozen = True
        frozen_tracks.append(track)

    for item_id in track_ids_to_delete:
        track_buffer.erase(item_id)

    return frozen_tracks


def clear_tracks_after_settlement():
    # 结算完成后这批 Track 不再有用途。
    # 先恢复 frozen 标记，再同时清空活动与冻结集合。
    for track in frozen_tracks:
        track.frozen = False

    frozen_tracks.clear()
    track_buffer.clear()


def discard_tracks_and_mark_ambiguous():
    # CLOSING 误判造成的暂停，或库存结算未提交，都会让 Track 的连续证据失效。
    # 只丢弃本开门 Session 的 Track，不修改 inventory，也不重置 operation_pending。
    for track in frozen_tracks:
        track.frozen = False

    frozen_tracks.clear()
    track_buffer.clear()
    track_session_is_ambiguous = True
```

库存结算期间，所有更新 Track 的函数都会先检查 `track.frozen`；被冻结的 Track 不允许再被修改。

`discard_tracks_and_mark_ambiguous()` 用于 Track 证据已经断开、却还不能修改库存的情况。
之后仍可收集新的稳定快照并按库存对比结算，但本次开门 Session 不再用 Track 确认“整理”或“出库”；
直到一次库存结算成功提交后，才把 `track_session_is_ambiguous` 清回 `False`。

```python
TrackSettlementResult:
    move_pairs = []         # [(A.item_id, B.temporary_id)]
    out_item_ids = set()    # 已移动、但没有终点 B 的 A.item_id
    ambiguous_item_ids = set()  # 终点不唯一，不能本轮修改的 A.item_id


def get_track_settlement_result(库存物品, 快照物品, frozen_tracks, item_by_id):
    result = TrackSettlementResult()

    if track_session_is_ambiguous:
        return result

    track_A_to_B = {}
    track_B_to_A = {}
    moved_item_ids = []

    for track in frozen_tracks:
        # freeze_all_tracks() 已过滤过；这里仍作防御性检查。
        if track.state == INVALID or track.frozen != True:
            continue
        if track.hold_and_move != True:
            continue

        A = item_by_id[track.bound_item_id]
        if A.status != 【可见】 or A.affirm:
            continue

        moved_item_ids.append(A.item_id)
        track_A_to_B[A.item_id] = set()
        expected_end_box = track.release_box
        if expected_end_box is None:
            expected_end_box = track.proxy_box

        for B in 快照物品:
            if B.item_id != -1 or B.cls_id != A.cls_id:
                continue
            if abs(B.box.width - expected_end_box.width) > eps_w:
                continue
            if abs(B.box.height - expected_end_box.height) > eps_h:
                continue
            if B.box 与 expected_end_box 不足够接近:
                continue

            track_A_to_B[A.item_id].add(B.temporary_id)
            track_B_to_A.setdefault(B.temporary_id, set()).add(A.item_id)

    for A_item_id in moved_item_ids:
        candidate_B_ids = track_A_to_B[A_item_id]

        if candidate_B_ids.size == 0:
            # 2.0 约定：A 已确认被移动，稳定快照却没有终点 B，即出库。
            result.out_item_ids.add(A_item_id)
            continue

        if candidate_B_ids.size != 1:
            result.ambiguous_item_ids.add(A_item_id)
            continue

        B_id = candidate_B_ids 的唯一元素
        if track_B_to_A[B_id].size != 1:
            result.ambiguous_item_ids.add(A_item_id)
            continue

        result.move_pairs.append((A_item_id, B_id))

    # 2.0 的业务前提是一轮只处理一个最终结果。若同一稳定快照中
    # 出现多个已确认的“整理 / 出库”结果，不能批量套用，应等待下一轮。
    confirmed_item_ids = [A_item_id for A_item_id, _ in result.move_pairs]
    confirmed_item_ids.extend(result.out_item_ids)
    if confirmed_item_ids.size > 1:
        for A_item_id in confirmed_item_ids:
            result.ambiguous_item_ids.add(A_item_id)
        result.move_pairs.clear()
        result.out_item_ids.clear()

    return result
```

库存对比主流程只使用 `TrackSettlementResult`：唯一 A-B 对确认整理；
零个终点 B 对确认出库；不唯一，或同一轮出现多个已确认结果，则本轮不强行修改。


## 8. 边界与取舍

```text
多手、手交叉、手身份丢失
→ 本版本不使用 Track 确认整理或出库。

用户拿走 A，又放入同类同尺寸 B，且 B 落在 Track 终点附近
→ 仍可能被判为 A 整理；这是已接受的业务取舍。

A 或 B 的候选不唯一
→ Track 放弃确认整理或出库，不能强行分配 item_id 或删除 A。

一次操作把 A 移到不可见处
→ 这属于 2.0 明确不处理的复合结果；当前规则会把它视为出库。
```
