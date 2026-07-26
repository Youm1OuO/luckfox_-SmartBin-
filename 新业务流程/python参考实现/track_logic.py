"""有手帧中的 Track 创建、匹配与状态切换。

本文件对应文档的“7.3 手在画面中：更新 Track，不生成快照”。
它不修改正式 InventoryDB，只修改 ProcessStore 中的临时 Track。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from geometry import (
    center_distance,
    containment_ratio,
    hand_near_box,
    hand_overlaps_item,
    move_box_by_center_delta,
    move_box_center_to,
    normal_track_match,
    reappear_track_match,
    release_geometry_reliable,
)
from models import (
    BBox,
    Detection,
    EngineConfig,
    FrameObservation,
    InventoryDB,
    InventoryItem,
    InventoryStatus,
    ProcessStore,
    Track,
    TrackState,
)


@dataclass(frozen=True)
class TrackAssociation:
    """一条 Detection 与一条已有 Track 的唯一关联。"""

    detection_index: int
    track_id: int
    score: float


def _track_match_score(
    detection: Detection,
    track: Track,
    config: EngineConfig,
) -> Optional[float]:
    """根据 Track 的状态，选择正确的比较对象。

    TRACKING_VISIBLE / PLACED：两个都是 YOLO 原始框，比较 last_yolo_box。
    FULL_HAND_OCCLUDED：物品上一帧不可见，用 proxy_box 找重新出现的物品。
    """
    if detection.cls_id != track.cls_id:
        return None

    if track.state == TrackState.FULL_HAND_OCCLUDED:
        result = reappear_track_match(detection.box, track.proxy_box, config)
    else:
        if track.last_yolo_box is None:
            # 这是内部状态错误，正常情况下不会发生；参考版选择不强行匹配。
            return None
        result = normal_track_match(detection.box, track.last_yolo_box, config)

    return result.score if result.matched else None


def associate_detections_to_existing_tracks(
    detections: tuple[Detection, ...],
    tracks: dict[int, Track],
    config: EngineConfig,
) -> list[TrackAssociation]:
    """为已有 Track 做一对一的、保守的匹配。

    如果某 Detection 或某 Track 的两个最佳候选太接近，就不强行关联。
    这比“随便挑一个最近的”更安全；模糊情况留给后面的快照逻辑处理。
    """
    candidates: list[TrackAssociation] = []
    by_detection: dict[int, list[TrackAssociation]] = {}
    by_track: dict[int, list[TrackAssociation]] = {}

    for detection_index, detection in enumerate(detections):
        for track_id, track in tracks.items():
            score = _track_match_score(detection, track, config)
            if score is None:
                continue
            candidate = TrackAssociation(detection_index, track_id, score)
            candidates.append(candidate)
            by_detection.setdefault(detection_index, []).append(candidate)
            by_track.setdefault(track_id, []).append(candidate)

    def has_clear_best(items: list[TrackAssociation]) -> bool:
        items.sort(key=lambda item: item.score)
        if len(items) <= 1:
            return bool(items)
        # 分数差很小代表“两个候选几乎一样好”，本帧不作身份决定。
        return items[1].score - items[0].score > 0.15

    allowed_detection = {
        index for index, items in by_detection.items() if has_clear_best(items)
    }
    allowed_track = {
        track_id for track_id, items in by_track.items() if has_clear_best(items)
    }

    selected: list[TrackAssociation] = []
    used_detections: set[int] = set()
    used_tracks: set[int] = set()
    for candidate in sorted(candidates, key=lambda item: item.score):
        if candidate.detection_index not in allowed_detection:
            continue
        if candidate.track_id not in allowed_track:
            continue
        if candidate.detection_index in used_detections or candidate.track_id in used_tracks:
            continue
        selected.append(candidate)
        used_detections.add(candidate.detection_index)
        used_tracks.add(candidate.track_id)
    return selected


def _track_start_box(item: InventoryItem) -> BBox:
    return item.anchor_box if item.anchor_valid else item.last_seen_box


def _new_track(
    process: ProcessStore,
    item: InventoryItem,
    hand_box: BBox,
    detection: Optional[Detection],
) -> Track:
    """为库存中的旧物品创建 Track。"""
    start_box = _track_start_box(item)
    track = Track(
        track_id=process.next_track_id,
        bound_item_id=item.item_id,
        cls_id=item.cls_id,
        start_box=start_box,
        proxy_box=start_box,
        last_yolo_box=detection.box if detection else None,
        last_hand_box=hand_box,
        path=[start_box],
        release_box=None,
        state=(TrackState.TRACKING_VISIBLE if detection else TrackState.FULL_HAND_OCCLUDED),
    )
    process.next_track_id += 1
    process.tracks[track.track_id] = track
    return track


def _find_unique_inventory_item_for_detection(
    detection: Detection,
    hand_box: BBox,
    inventory: InventoryDB,
    process: ProcessStore,
    config: EngineConfig,
) -> Optional[InventoryItem]:
    """当前 Detection 不属于旧 Track 时，尝试把它绑定到一个旧库存物品。"""
    already_bound = {track.bound_item_id for track in process.tracks.values()}
    candidates: list[tuple[float, InventoryItem]] = []

    for item in inventory.visible_items():
        if item.item_id in already_bound or item.cls_id != detection.cls_id:
            continue
        reference_boxes = [item.last_seen_box]
        if item.anchor_valid:
            reference_boxes.append(item.anchor_box)

        if not hand_near_box(hand_box, _track_start_box(item), config):
            continue
        best_containment = max(containment_ratio(detection.box, box) for box in reference_boxes)
        if best_containment >= config.inventory_containment_ratio:
            candidates.append((best_containment, item))

    candidates.sort(key=lambda pair: pair[0], reverse=True)
    if len(candidates) == 1:
        return candidates[0][1]
    if len(candidates) >= 2 and candidates[0][0] - candidates[1][0] > 0.10:
        return candidates[0][1]
    return None


def _previous_detection_for_item(
    previous_frame: Optional[FrameObservation],
    item: InventoryItem,
    config: EngineConfig,
) -> Optional[Detection]:
    """可选优化：补上手刚进入时的第一段位移。"""
    if previous_frame is None:
        return None
    candidates = [
        detection
        for detection in previous_frame.item_detections
        if detection.cls_id == item.cls_id
        and containment_ratio(detection.box, _track_start_box(item))
        >= config.inventory_containment_ratio
    ]
    return candidates[0] if len(candidates) == 1 else None


def _update_track_from_detection(
    track: Track,
    detection: Detection,
    hand_box: BBox,
    inventory_item: InventoryItem,
    config: EngineConfig,
) -> None:
    """已有 Track 成功接到当前物品 Detection 后的状态更新。"""
    if track.state == TrackState.FULL_HAND_OCCLUDED:
        # 当前框可能才刚从手后露出，不能拿很久前的框计算 object_delta。
        track.last_yolo_box = detection.box
        track.last_hand_box = hand_box
        track.state = TrackState.TRACKING_VISIBLE
        track.path.append(track.proxy_box)
        return

    if track.last_yolo_box is None:
        # 防御性分支：状态不一致时，重新建立 YOLO 原始框基准。
        track.last_yolo_box = detection.box
        track.last_hand_box = hand_box
        track.path.append(track.proxy_box)
        return

    old_center = track.last_yolo_box.center
    new_center = detection.box.center
    dx = new_center[0] - old_center[0]
    dy = new_center[1] - old_center[1]
    object_motion = hypot2(dx, dy)

    if track.state == TrackState.PLACED:
        # 已放下的物品正常只刷新观察，不应跟着别处的手继续走。
        if hand_overlaps_item(hand_box, detection.box, config) and object_motion > config.release_motion_distance:
            track.state = TrackState.TRACKING_VISIBLE
            track.release_box = None
            track.proxy_box = move_box_by_center_delta(track.proxy_box, dx, dy)
            track.last_hand_box = hand_box
        track.last_yolo_box = detection.box
        track.path.append(track.proxy_box)
        return

    # TRACKING_VISIBLE：连续两个 YOLO 框可以用于估计物品位移。
    track.last_yolo_box = detection.box
    track.last_hand_box = hand_box

    can_confirm_release = (
        object_motion <= config.release_motion_distance
        and not hand_overlaps_item(hand_box, detection.box, config)
        and (
            not inventory_item.anchor_valid
            or release_geometry_reliable(detection.box, inventory_item.anchor_box, config)
        )
    )
    if can_confirm_release:
        track.state = TrackState.PLACED
        track.proxy_box = move_box_center_to(track.proxy_box, detection.box.center)
        track.release_box = track.proxy_box
    else:
        track.proxy_box = move_box_by_center_delta(track.proxy_box, dx, dy)

    # 用户希望保留完整路径：每次 Track 更新都追加，即使位置没有变化。
    track.path.append(track.proxy_box)


def hypot2(dx: float, dy: float) -> float:
    """避免本文件额外暴露 math.hypot 的导入细节。"""
    return (dx * dx + dy * dy) ** 0.5


def _update_missing_track(track: Track, hand_box: BBox, config: EngineConfig) -> None:
    """本帧没有 Detection 接到该 Track 时，尝试按完全遮挡处理。"""
    if not hand_near_box(hand_box, track.proxy_box, config):
        # 手已去操作别处物品，不能再把这条 Track 跟着手拖走。
        return

    if track.state == TrackState.PLACED:
        # 已放下物品又被拿起的第一帧：还没有可靠的 hand_delta。
        track.release_box = None
        track.last_hand_box = hand_box
    else:
        old_hand_center = track.last_hand_box.center
        new_hand_center = hand_box.center
        track.proxy_box = move_box_by_center_delta(
            track.proxy_box,
            new_hand_center[0] - old_hand_center[0],
            new_hand_center[1] - old_hand_center[1],
        )
        track.last_hand_box = hand_box

    track.state = TrackState.FULL_HAND_OCCLUDED
    track.last_yolo_box = None
    track.path.append(track.proxy_box)


def _create_tracks_for_unmatched_detections(
    detections: tuple[Detection, ...],
    unmatched_detection_indices: set[int],
    hand_box: BBox,
    inventory: InventoryDB,
    process: ProcessStore,
    config: EngineConfig,
) -> list[str]:
    events: list[str] = []
    for index in sorted(unmatched_detection_indices):
        detection = detections[index]
        if not hand_near_box(hand_box, detection.box, config):
            continue
        item = _find_unique_inventory_item_for_detection(
            detection, hand_box, inventory, process, config
        )
        if item is None:
            # 新放入物品不创建“整理证明 Track”。
            continue

        track = _new_track(process, item, hand_box, detection)
        previous = _previous_detection_for_item(process.previous_frame, item, config)
        if previous is not None:
            old_center = previous.box.center
            new_center = detection.box.center
            track.proxy_box = move_box_by_center_delta(
                track.proxy_box,
                new_center[0] - old_center[0],
                new_center[1] - old_center[1],
            )
        track.path.append(track.proxy_box)
        events.append(f"创建 Track#{track.track_id}，绑定 item_id={item.item_id}")
    return events


def _create_fully_occluded_candidate_tracks(
    hand_box: BBox,
    inventory: InventoryDB,
    process: ProcessStore,
    config: EngineConfig,
) -> list[str]:
    """处理“手第一帧进来就完全盖住物品”的情况。"""
    events: list[str] = []
    already_bound = {track.bound_item_id for track in process.tracks.values()}
    for item in inventory.visible_items():
        if item.item_id in already_bound:
            continue
        if not hand_near_box(hand_box, _track_start_box(item), config):
            continue
        track = _new_track(process, item, hand_box, detection=None)
        events.append(f"创建全遮挡候选 Track#{track.track_id}，绑定 item_id={item.item_id}")
    return events


def update_tracks_for_hand_frame(
    frame: FrameObservation,
    inventory: InventoryDB,
    process: ProcessStore,
    config: EngineConfig,
) -> list[str]:
    """处理一帧“检测到手”的结果。

    返回日志字符串，便于用户逐帧检查 Python 参考实现的行为。
    """
    if frame.hand_box is None:
        raise ValueError("update_tracks_for_hand_frame 只能处理有手的帧")

    hand_box = frame.hand_box
    events: list[str] = []
    process.operation_pending = True
    process.no_hand_buffer.clear()

    # 第一阶段：所有 Detection 先尝试接回已有 Track。
    existing_track_ids = set(process.tracks)
    associations = associate_detections_to_existing_tracks(
        frame.item_detections, process.tracks, config
    )
    matched_detection_indices = {association.detection_index for association in associations}
    matched_track_ids = {association.track_id for association in associations}

    for association in associations:
        track = process.tracks[association.track_id]
        inventory_item = inventory.items[track.bound_item_id]
        _update_track_from_detection(
            track,
            frame.item_detections[association.detection_index],
            hand_box,
            inventory_item,
            config,
        )
        events.append(
            f"Detection#{association.detection_index} 接回 Track#{track.track_id}，state={track.state.name}"
        )

    # 第二阶段：没有接到旧 Track 的 Detection，才尝试创建绑定库存的 Track。
    unmatched_detection_indices = set(range(len(frame.item_detections))) - matched_detection_indices
    events.extend(
        _create_tracks_for_unmatched_detections(
            frame.item_detections,
            unmatched_detection_indices,
            hand_box,
            inventory,
            process,
            config,
        )
    )

    # 第三阶段：已有 Track 本帧没有 Detection，手还在附近时按完全遮挡处理。
    for track_id in existing_track_ids:
        track = process.tracks[track_id]
        if track_id not in matched_track_ids:
            _update_missing_track(track, hand_box, config)

    # 最后补上“本帧根本没有物品 Detection，但手盖住库存物品”的候选 Track。
    events.extend(_create_fully_occluded_candidate_tracks(hand_box, inventory, process, config))
    return events
