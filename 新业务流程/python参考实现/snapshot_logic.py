"""无手帧的稳定性判断与 N 帧稳定快照生成。"""

from __future__ import annotations

from statistics import mean

from geometry import stable_snapshot_match
from models import (
    BBox,
    Detection,
    EngineConfig,
    FrameObservation,
    ProcessStore,
    SnapshotItem,
    StableSnapshot,
)


def _greedy_detection_pairs(
    left: tuple[Detection, ...],
    right: tuple[Detection, ...],
    config: EngineConfig,
) -> list[tuple[int, int]]:
    """稳定性判断用的一对一 Detection 匹配。"""
    candidates: list[tuple[float, int, int]] = []
    for left_index, left_detection in enumerate(left):
        for right_index, right_detection in enumerate(right):
            if left_detection.cls_id != right_detection.cls_id:
                continue
            result = stable_snapshot_match(left_detection.box, right_detection.box, config)
            if result.matched:
                candidates.append((result.score, left_index, right_index))

    pairs: list[tuple[int, int]] = []
    used_left: set[int] = set()
    used_right: set[int] = set()
    for _, left_index, right_index in sorted(candidates):
        if left_index in used_left or right_index in used_right:
            continue
        pairs.append((left_index, right_index))
        used_left.add(left_index)
        used_right.add(right_index)
    return pairs


def frames_are_stable(
    previous: FrameObservation,
    current: FrameObservation,
    config: EngineConfig,
) -> bool:
    """前后两帧都无手时，判断是否只是正常 YOLO 抖动。"""
    if previous.hand_box is not None or current.hand_box is not None:
        return False
    pairs = _greedy_detection_pairs(previous.item_detections, current.item_detections, config)
    # 突然新增/消失物品，或相同类别的位置明显跳变，都视为不稳定。
    return len(pairs) == len(previous.item_detections) == len(current.item_detections)


def _fuse_boxes(boxes: list[BBox]) -> BBox:
    """最简单的融合方式：对同一物品的 N 个框逐坐标取平均。"""
    return BBox(
        left=mean(box.left for box in boxes),
        top=mean(box.top for box in boxes),
        right=mean(box.right for box in boxes),
        bottom=mean(box.bottom for box in boxes),
    )


def build_stable_snapshot(
    frames: list[FrameObservation],
    config: EngineConfig,
) -> StableSnapshot:
    """把连续 N 帧无手稳定 Detection 融合为一份 Snapshot。

    本参考版本以“最后一帧的每个 Detection”为种子，回查前面的每帧。
    只有在每一帧都找到同类稳定匹配的物品，才进入最终快照。
    """
    if not frames:
        raise ValueError("不能从空 no_hand_buffer 生成稳定快照")

    latest = frames[-1]
    snapshot_items: list[SnapshotItem] = []
    for latest_detection in latest.item_detections:
        observations = [latest_detection]
        complete = True
        for frame in frames[:-1]:
            candidates: list[tuple[float, Detection]] = []
            for detection in frame.item_detections:
                if detection.cls_id != latest_detection.cls_id:
                    continue
                result = stable_snapshot_match(detection.box, latest_detection.box, config)
                if result.matched:
                    candidates.append((result.score, detection))
            if len(candidates) != 1:
                complete = False
                break
            observations.append(candidates[0][1])

        if not complete:
            continue
        snapshot_items.append(
            SnapshotItem(
                snapshot_id=len(snapshot_items),
                cls_id=latest_detection.cls_id,
                box=_fuse_boxes([detection.box for detection in observations]),
                score=mean(detection.score for detection in observations),
                vote_count=len(observations),
            )
        )

    return StableSnapshot(frame_index=latest.frame_index, items=tuple(snapshot_items))


def accept_no_hand_frame(
    frame: FrameObservation,
    process: ProcessStore,
    config: EngineConfig,
) -> tuple[StableSnapshot | None, bool]:
    """处理一帧无手画面。

    返回 `(snapshot, suspicious_motion)`：
    - snapshot 不为空：已经攒够 N 帧，可开始库存比较。
    - suspicious_motion=True：前后都无手却明显跳变，只重置缓冲，不创建 Track。
    """
    if frame.hand_box is not None:
        raise ValueError("accept_no_hand_frame 只能处理无手帧")

    previous = process.previous_frame
    suspicious_motion = False
    if previous is None or previous.hand_box is not None:
        # 手刚离开，旧 Track 从此刻起冻结；当前帧是新快照的第 1 帧。
        process.no_hand_buffer = [frame]
    elif frames_are_stable(previous, frame, config):
        process.no_hand_buffer.append(frame)
    else:
        # 手漏检或 YOLO 突然异常时，不让这一帧直接形成稳定快照。
        process.no_hand_buffer = [frame]
        suspicious_motion = True

    if len(process.no_hand_buffer) < config.stable_snapshot_frames:
        return None, suspicious_motion

    snapshot = build_stable_snapshot(process.no_hand_buffer, config)
    process.current_stable_snapshot = snapshot
    return snapshot, suspicious_motion
