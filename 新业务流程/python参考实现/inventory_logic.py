"""稳定快照直接与 InventoryDB 比较，生成 planned_changes。

这是 Python 参考版最重要的“提交前逻辑”：所有修改都先保存在
ComparisonResult 中，最后由 engine 决定是否一次性提交到 InventoryDB。
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Optional

from geometry import center_distance, stable_snapshot_match
from models import (
    BBox,
    ComparisonResult,
    EngineConfig,
    InventoryDB,
    InventoryItem,
    InventoryStatus,
    ProcessStore,
    SnapshotItem,
    StableSnapshot,
    Track,
)


def _copy_working_items(inventory: InventoryDB) -> dict[int, InventoryItem]:
    """本轮比较中的“临时库存视图”，不修改正式 InventoryDB。"""
    return {item_id: item.copy() for item_id, item in inventory.items.items()}


def _record(result: ComparisonResult, message: str) -> None:
    result.events.append(message)


def _match_snapshot_to_items(
    snapshot: StableSnapshot,
    item_ids: set[int],
    working_items: dict[int, InventoryItem],
    config: EngineConfig,
) -> list[tuple[int, int, float]]:
    """SnapshotItem 与库存项目之间的一对一、保守匹配。

    返回 `(snapshot_id, item_id, score)`。当候选不唯一时，不返回该对象。
    """
    candidates: list[tuple[float, int, int]] = []
    by_snapshot: dict[int, list[tuple[float, int, int]]] = {}
    by_item: dict[int, list[tuple[float, int, int]]] = {}

    for snapshot_item in snapshot.items:
        for item_id in item_ids:
            item = working_items[item_id]
            if item.cls_id != snapshot_item.cls_id:
                continue
            # 原位置直接匹配优先 last_seen；anchor_valid 时额外试 anchor。
            matches = [stable_snapshot_match(snapshot_item.box, item.last_seen_box, config)]
            if item.anchor_valid:
                matches.append(stable_snapshot_match(snapshot_item.box, item.anchor_box, config))
            valid = [match for match in matches if match.matched]
            if not valid:
                continue
            score = min(match.score for match in valid)
            candidate = (score, snapshot_item.snapshot_id, item_id)
            candidates.append(candidate)
            by_snapshot.setdefault(snapshot_item.snapshot_id, []).append(candidate)
            by_item.setdefault(item_id, []).append(candidate)

    def clear_best(items: list[tuple[float, int, int]]) -> bool:
        items.sort()
        return len(items) == 1 or items[1][0] - items[0][0] > 0.15

    allowed_snapshot = {
        snapshot_id for snapshot_id, items in by_snapshot.items() if clear_best(items)
    }
    allowed_item = {item_id for item_id, items in by_item.items() if clear_best(items)}

    selected: list[tuple[int, int, float]] = []
    used_snapshot: set[int] = set()
    used_item: set[int] = set()
    for score, snapshot_id, item_id in sorted(candidates):
        if snapshot_id not in allowed_snapshot or item_id not in allowed_item:
            continue
        if snapshot_id in used_snapshot or item_id in used_item:
            continue
        selected.append((snapshot_id, item_id, score))
        used_snapshot.add(snapshot_id)
        used_item.add(item_id)
    return selected


def _snapshot_by_id(snapshot: StableSnapshot) -> dict[int, SnapshotItem]:
    return {item.snapshot_id: item for item in snapshot.items}


def _track_for_item(process: ProcessStore, item_id: int) -> Optional[Track]:
    tracks = [track for track in process.tracks.values() if track.bound_item_id == item_id]
    return tracks[0] if len(tracks) == 1 else None


def _best_track_evidence(
    snapshot_item: SnapshotItem,
    track: Track,
    config: EngineConfig,
) -> Optional[tuple[float, BBox, str]]:
    """给“整理”寻找证据框。

    release_box 存在时只能使用它；它不匹配就不能降级为 path。
    release_box 为空时，才允许从完整路径的历史点中找证据。
    """
    if track.release_box is not None:
        result = stable_snapshot_match(snapshot_item.box, track.release_box, config)
        if not result.matched:
            return None
        return result.score, track.release_box, "release_box"

    candidates: list[tuple[float, BBox]] = []
    for point in track.path:
        result = stable_snapshot_match(snapshot_item.box, point, config)
        if result.matched:
            candidates.append((result.score, point))
    if not candidates:
        return None
    score, point = min(candidates, key=lambda item: item[0])
    return score, point, "path"


def _select_unique_organize_pairs(
    unmatched_visible_item_ids: set[int],
    unmatched_snapshot_ids: set[int],
    snapshot_by_id: dict[int, SnapshotItem],
    process: ProcessStore,
    config: EngineConfig,
) -> list[tuple[int, int, BBox, str]]:
    """从旧库存 A、新位置快照 B、Track 三者中选择唯一整理关系。"""
    candidates: list[tuple[float, int, int, BBox, str]] = []
    by_old_item: dict[int, list[tuple[float, int, int, BBox, str]]] = {}
    by_snapshot: dict[int, list[tuple[float, int, int, BBox, str]]] = {}

    for item_id in unmatched_visible_item_ids:
        track = _track_for_item(process, item_id)
        if track is None:
            continue
        for snapshot_id in unmatched_snapshot_ids:
            evidence = _best_track_evidence(snapshot_by_id[snapshot_id], track, config)
            if evidence is None:
                continue
            score, evidence_box, source = evidence
            candidate = (score, item_id, snapshot_id, evidence_box, source)
            candidates.append(candidate)
            by_old_item.setdefault(item_id, []).append(candidate)
            by_snapshot.setdefault(snapshot_id, []).append(candidate)

    def clear_best(items: list[tuple[float, int, int, BBox, str]]) -> bool:
        items.sort(key=lambda item: item[0])
        return len(items) == 1 or items[1][0] - items[0][0] > 0.15

    allowed_old = {item_id for item_id, items in by_old_item.items() if clear_best(items)}
    allowed_snapshot = {
        snapshot_id for snapshot_id, items in by_snapshot.items() if clear_best(items)
    }

    selected: list[tuple[int, int, BBox, str]] = []
    used_old: set[int] = set()
    used_snapshot: set[int] = set()
    for _, item_id, snapshot_id, evidence_box, source in sorted(candidates, key=lambda item: item[0]):
        if item_id not in allowed_old or snapshot_id not in allowed_snapshot:
            continue
        if item_id in used_old or snapshot_id in used_snapshot:
            continue
        selected.append((item_id, snapshot_id, evidence_box, source))
        used_old.add(item_id)
        used_snapshot.add(snapshot_id)
    return selected


def _near_old_position(box: BBox, old_position: BBox, config: EngineConfig) -> bool:
    """遮挡处理只需要“附近”，不要求类别或宽高完全一样。"""
    return center_distance(box, old_position) <= config.reappear_center_distance


def _mark_item_visible(
    item: InventoryItem,
    box: BBox,
    *,
    anchor_box: Optional[BBox] = None,
) -> InventoryItem:
    updated = item.copy()
    updated.status = InventoryStatus.VISIBLE
    updated.last_seen_box = box
    if anchor_box is not None:
        updated.anchor_box = anchor_box
        updated.anchor_valid = True
    return updated


def _apply_organize_occlusion_adjustments(
    old_item: InventoryItem,
    new_box: BBox,
    working_items: dict[int, InventoryItem],
    snapshot_by_id: dict[int, SnapshotItem],
    unprocessed_snapshot_ids: set[int],
    unprocessed_visible_item_ids: set[int],
    result: ComparisonResult,
    config: EngineConfig,
) -> None:
    """整理确认后，在旧位置处理“露出”，在新位置处理“被遮住”。

    这一步是文档中的保护逻辑：它只使用库存 anchor_box 和当前快照，
    不会猜测库存物品曾经移动到哪里。
    """
    # 旧位置：A 离开后，原来 OCCLUDED 的 C 可能重新露出。
    for snapshot_id in list(unprocessed_snapshot_ids):
        snapshot_item = snapshot_by_id[snapshot_id]
        if not _near_old_position(snapshot_item.box, old_item.anchor_box, config):
            continue
        occluded_candidates = [
            item
            for item in working_items.values()
            if item.status == InventoryStatus.OCCLUDED
            and item.cls_id == snapshot_item.cls_id
            and stable_snapshot_match(snapshot_item.box, item.last_seen_box, config).matched
        ]
        if len(occluded_candidates) == 1:
            revealed = _mark_item_visible(occluded_candidates[0], snapshot_item.box)
            working_items[revealed.item_id] = revealed
            result.set_item(revealed)
            unprocessed_snapshot_ids.remove(snapshot_id)
            _record(result, f"整理后旧位置露出 item_id={revealed.item_id}")
            break

    # 新位置：库存中原来可见的 D 若其旧 anchor 在 B 附近、现在又没匹配到，可能被 B 遮挡。
    for item_id in list(unprocessed_visible_item_ids):
        item = working_items[item_id]
        if item.item_id == old_item.item_id or item.status != InventoryStatus.VISIBLE:
            continue
        if _near_old_position(item.anchor_box, new_box, config):
            hidden = item.copy()
            hidden.status = InventoryStatus.OCCLUDED
            working_items[hidden.item_id] = hidden
            result.set_item(hidden)
            unprocessed_visible_item_ids.remove(hidden.item_id)
            _record(result, f"整理后新位置遮住 item_id={hidden.item_id}")


def _apply_occlusion_counters(
    working_items: dict[int, InventoryItem],
    snapshot: StableSnapshot,
    operation_pending: bool,
    result: ComparisonResult,
    config: EngineConfig,
) -> None:
    """一个开门会话最多更新一次遮挡计数器。"""
    if not operation_pending:
        return

    for item in working_items.values():
        if item.status != InventoryStatus.OCCLUDED:
            continue
        nearby_visible_count = sum(
            1
            for snapshot_item in snapshot.items
            if _near_old_position(snapshot_item.box, item.anchor_box, config)
        )
        updated = item.copy()
        if nearby_visible_count == 0:
            updated.no_occluder_count += 1
            updated.duplicate_count = 0
        elif nearby_visible_count == 1:
            updated.duplicate_count += 1
            updated.no_occluder_count = 0
        else:
            updated.no_occluder_count = 0
            updated.duplicate_count = 0

        if updated.no_occluder_count >= config.occlusion_counter_threshold:
            updated.status = InventoryStatus.OUT
        elif updated.duplicate_count >= config.occlusion_counter_threshold:
            updated.status = InventoryStatus.VISIBLE
        working_items[updated.item_id] = updated
        result.set_item(updated)


def compare_snapshot_to_inventory(
    snapshot: StableSnapshot,
    inventory: InventoryDB,
    process: ProcessStore,
    config: EngineConfig,
) -> ComparisonResult:
    """按文档顺序比较“当前稳定快照”与 InventoryDB。

    顺序：原位置直接匹配 → 整理 → 剩余消失物品 → 剩余新出现物品 → 遮挡计数器。
    """
    result = ComparisonResult()
    working_items = _copy_working_items(inventory)
    snapshot_by_id = _snapshot_by_id(snapshot)
    unprocessed_snapshot_ids = set(snapshot_by_id)
    unprocessed_visible_item_ids = {
        item_id
        for item_id, item in working_items.items()
        if item.status == InventoryStatus.VISIBLE
    }

    # 第一步：原位置未移动、仍可见的库存物品。
    direct_pairs = _match_snapshot_to_items(
        snapshot, unprocessed_visible_item_ids, working_items, config
    )
    for snapshot_id, item_id, _ in direct_pairs:
        snapshot_item = snapshot_by_id[snapshot_id]
        item = working_items[item_id].copy()
        item.status = InventoryStatus.VISIBLE
        item.last_seen_box = snapshot_item.box
        # 当前完整框包含旧 anchor 时，可补强 anchor。
        if item.anchor_valid and stable_snapshot_match(snapshot_item.box, item.anchor_box, config).matched:
            item.anchor_box = snapshot_item.box
            item.anchor_valid = True
        working_items[item_id] = item
        result.set_item(item)
        unprocessed_snapshot_ids.remove(snapshot_id)
        unprocessed_visible_item_ids.remove(item_id)
        _record(result, f"item_id={item_id} 原位置直接匹配")

    # 第二步：原位置消失 A + 新位置 B + Track 证据 = 整理。
    organize_pairs = _select_unique_organize_pairs(
        unprocessed_visible_item_ids,
        unprocessed_snapshot_ids,
        snapshot_by_id,
        process,
        config,
    )
    for item_id, snapshot_id, evidence_box, evidence_source in organize_pairs:
        old_item = working_items[item_id]
        snapshot_item = snapshot_by_id[snapshot_id]
        organized = _mark_item_visible(
            old_item,
            snapshot_item.box,
            anchor_box=evidence_box,
        )
        working_items[item_id] = organized
        result.set_item(organized)
        unprocessed_visible_item_ids.remove(item_id)
        unprocessed_snapshot_ids.remove(snapshot_id)
        _record(
            result,
            f"item_id={item_id} 通过 {evidence_source} 确认整理到 snapshot_id={snapshot_id}",
        )
        _apply_organize_occlusion_adjustments(
            old_item,
            snapshot_item.box,
            working_items,
            snapshot_by_id,
            unprocessed_snapshot_ids,
            unprocessed_visible_item_ids,
            result,
            config,
        )

    # 第三步：原来可见、现在仍未匹配的 A，判断拿出还是被遮挡。
    for item_id in list(unprocessed_visible_item_ids):
        item = working_items[item_id]
        nearby_snapshot_ids = [
            snapshot_id
            for snapshot_id in unprocessed_snapshot_ids
            if _near_old_position(snapshot_by_id[snapshot_id].box, item.anchor_box, config)
        ]
        if not nearby_snapshot_ids:
            updated = item.copy()
            updated.status = InventoryStatus.OUT
            working_items[item_id] = updated
            result.set_item(updated)
            _record(result, f"item_id={item_id} 旧位置附近无物品，计划 OUT")
        else:
            # 优先检查出现的 C 是否是库存中原来 OCCLUDED 的物品重新露出。
            revealed = False
            for snapshot_id in nearby_snapshot_ids:
                snapshot_item = snapshot_by_id[snapshot_id]
                candidates = [
                    candidate
                    for candidate in working_items.values()
                    if candidate.status == InventoryStatus.OCCLUDED
                    and candidate.cls_id == snapshot_item.cls_id
                    and stable_snapshot_match(snapshot_item.box, candidate.last_seen_box, config).matched
                ]
                if len(candidates) == 1:
                    candidate = _mark_item_visible(candidates[0], snapshot_item.box)
                    working_items[candidate.item_id] = candidate
                    result.set_item(candidate)
                    unprocessed_snapshot_ids.remove(snapshot_id)
                    revealed = True
                    _record(result, f"item_id={candidate.item_id} 在旧位置重新可见")
                    break

            updated = item.copy()
            updated.status = InventoryStatus.OUT if revealed else InventoryStatus.OCCLUDED
            working_items[item_id] = updated
            result.set_item(updated)
            _record(
                result,
                f"item_id={item_id} 旧位置有其他物品，计划 {updated.status.name}",
            )
        unprocessed_visible_item_ids.remove(item_id)

    # 第四步：当前新出现、仍未关联 item_id 的 B。
    next_item_id = inventory.next_item_id
    for snapshot_id in list(unprocessed_snapshot_ids):
        snapshot_item = snapshot_by_id[snapshot_id]
        candidate_ids = {
            item_id
            for item_id, item in working_items.items()
            if item.status in (InventoryStatus.OCCLUDED, InventoryStatus.OUT)
            and item.cls_id == snapshot_item.cls_id
        }
        pairs = _match_snapshot_to_items(
            StableSnapshot(snapshot.frame_index, (snapshot_item,)),
            candidate_ids,
            working_items,
            config,
        )
        if len(pairs) == 1:
            _, item_id, _ = pairs[0]
            restored = _mark_item_visible(working_items[item_id], snapshot_item.box)
            working_items[item_id] = restored
            result.set_item(restored)
            _record(result, f"snapshot_id={snapshot_id} 恢复 item_id={item_id}")
        else:
            new_item = InventoryItem(
                item_id=next_item_id,
                cls_id=snapshot_item.cls_id,
                status=InventoryStatus.VISIBLE,
                anchor_box=snapshot_item.box,
                anchor_valid=True,
                last_seen_box=snapshot_item.box,
            )
            result.planned_new_items.append(new_item)
            _record(result, f"snapshot_id={snapshot_id} 确认新放入，分配 item_id={next_item_id}")
            next_item_id += 1
        unprocessed_snapshot_ids.remove(snapshot_id)

    _apply_occlusion_counters(
        working_items,
        snapshot,
        process.operation_pending,
        result,
        config,
    )
    return result
