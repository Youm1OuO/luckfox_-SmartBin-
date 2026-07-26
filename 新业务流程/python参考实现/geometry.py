"""BBox 的几何计算与统一匹配函数。

业务代码不直接手写中心距离、宽高比例等公式；统一调用本文件，
这样以后把 Python 翻为 C++ 时只需对应实现这一组小函数。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot, inf

from models import BBox, EngineConfig


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.center
    bx, by = b.center
    return hypot(ax - bx, ay - by)


def width_ratio(a: BBox, b: BBox) -> float:
    """返回 >= 1 的对称宽度比例；1.0 表示同宽。"""
    if a.width <= 0.0 or b.width <= 0.0:
        return inf
    return max(a.width / b.width, b.width / a.width)


def height_ratio(a: BBox, b: BBox) -> float:
    """返回 >= 1 的对称高度比例；1.0 表示同高。"""
    if a.height <= 0.0 or b.height <= 0.0:
        return inf
    return max(a.height / b.height, b.height / a.height)


def intersection_area(a: BBox, b: BBox) -> float:
    left = max(a.left, b.left)
    top = max(a.top, b.top)
    right = min(a.right, b.right)
    bottom = min(a.bottom, b.bottom)
    return max(0.0, right - left) * max(0.0, bottom - top)


def containment_ratio(a: BBox, b: BBox) -> float:
    """交集 / 较小框面积。

    创建 Track 时，D.box 可能是被手遮住后的局部框；此时 IoU 会偏小，
    但“局部框几乎被完整库存框包含”仍然是有价值的证据。
    """
    smaller_area = min(a.area, b.area)
    if smaller_area <= 0.0:
        return 0.0
    return intersection_area(a, b) / smaller_area


def boxes_overlap(a: BBox, b: BBox, minimum_ratio: float = 0.0) -> bool:
    """判断两个框是否相交；minimum_ratio 使用交集/较小框面积。"""
    return containment_ratio(a, b) >= minimum_ratio


def move_box_by_center_delta(box: BBox, dx: float, dy: float) -> BBox:
    return box.moved(dx, dy)


def move_box_center_to(box: BBox, target_center: tuple[float, float]) -> BBox:
    """放下确认时：位置采用当前 Detection，宽高保留完整 proxy_box。"""
    return box.with_center(*target_center)


@dataclass(frozen=True)
class BoxMatch:
    matched: bool
    score: float
    center_distance: float
    width_ratio: float
    height_ratio: float


def match_boxes(
    a: BBox,
    b: BBox,
    *,
    max_center_distance: float,
    max_width_ratio: float = inf,
    max_height_ratio: float = inf,
) -> BoxMatch:
    """统一的框匹配方法。

    所有地方都比较“中心距离、框宽、框高”。不同业务场景只传不同阈值：
    - 原位置稳定快照：宽高阈值较严格；
    - Track 的连续局部框：宽高阈值较宽松；
    - 完全遮挡后的重现：宽高阈值为无限大，只检查中心附近。
    """
    distance = center_distance(a, b)
    w_ratio = width_ratio(a, b)
    h_ratio = height_ratio(a, b)
    matched = (
        distance <= max_center_distance
        and w_ratio <= max_width_ratio
        and h_ratio <= max_height_ratio
    )

    # 数值越小代表越接近。它只用于候选排序，真正是否匹配仍由上面的阈值决定。
    score = (
        distance / max(max_center_distance, 1.0)
        + (w_ratio - 1.0 if max_width_ratio != inf else 0.0)
        + (h_ratio - 1.0 if max_height_ratio != inf else 0.0)
    )
    return BoxMatch(matched, score, distance, w_ratio, h_ratio)


def normal_track_match(a: BBox, b: BBox, config: EngineConfig) -> BoxMatch:
    return match_boxes(
        a,
        b,
        max_center_distance=config.normal_center_distance,
        max_width_ratio=config.normal_max_width_ratio,
        max_height_ratio=config.normal_max_height_ratio,
    )


def reappear_track_match(detection_box: BBox, proxy_box: BBox, config: EngineConfig) -> BoxMatch:
    return match_boxes(
        detection_box,
        proxy_box,
        max_center_distance=config.reappear_center_distance,
    )


def stable_snapshot_match(a: BBox, b: BBox, config: EngineConfig) -> BoxMatch:
    return match_boxes(
        a,
        b,
        max_center_distance=config.snapshot_center_distance,
        max_width_ratio=config.snapshot_max_width_ratio,
        max_height_ratio=config.snapshot_max_height_ratio,
    )


def release_geometry_reliable(detection_box: BBox, anchor_box: BBox, config: EngineConfig) -> bool:
    """当前框是否足够接近完整 anchor_box 的尺寸。"""
    return (
        width_ratio(detection_box, anchor_box) <= config.release_max_width_ratio
        and height_ratio(detection_box, anchor_box) <= config.release_max_height_ratio
    )


def hand_near_box(hand_box: BBox, item_box: BBox, config: EngineConfig) -> bool:
    """手在物品附近：中心足够近，或二者有交集。"""
    return (
        center_distance(hand_box, item_box) <= config.hand_near_distance
        or intersection_area(hand_box, item_box) > 0.0
    )


def hand_overlaps_item(hand_box: BBox, item_box: BBox, config: EngineConfig) -> bool:
    """手是否仍覆盖物品的一部分。"""
    return containment_ratio(hand_box, item_box) >= config.hand_overlap_ratio
