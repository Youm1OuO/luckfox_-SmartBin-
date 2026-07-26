"""本参考实现中所有数据结构的定义。

这些类只描述业务数据，不调用摄像头、YOLO 或后台接口。
以后改写 C++ 时，建议保持相同的字段名称，便于逐项对照。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, auto
from math import hypot
from typing import Optional


@dataclass(frozen=True)
class BBox:
    """统一坐标系中的矩形框。left/top/right/bottom 均为像素坐标。"""

    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return max(0.0, self.right - self.left)

    @property
    def height(self) -> float:
        return max(0.0, self.bottom - self.top)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2.0, (self.top + self.bottom) / 2.0)

    @property
    def diagonal(self) -> float:
        return hypot(self.width, self.height)

    def moved(self, dx: float, dy: float) -> "BBox":
        """保留宽高，仅把框平移。"""
        return BBox(self.left + dx, self.top + dy, self.right + dx, self.bottom + dy)

    def with_center(self, center_x: float, center_y: float) -> "BBox":
        """保留宽高，只校正中心位置。确认放下时会用到。"""
        return BBox(
            center_x - self.width / 2.0,
            center_y - self.height / 2.0,
            center_x + self.width / 2.0,
            center_y + self.height / 2.0,
        )


@dataclass(frozen=True)
class Detection:
    """一帧 YOLO 的一个物品检测结果。它本身没有 item_id。"""

    cls_id: int
    box: BBox
    score: float = 1.0


@dataclass(frozen=True)
class FrameObservation:
    """已经完成坐标统一、类别筛选后的单帧结果。"""

    item_detections: tuple[Detection, ...]
    hand_box: Optional[BBox]
    frame_index: int


class InventoryStatus(Enum):
    VISIBLE = auto()
    OCCLUDED = auto()
    OUT = auto()


@dataclass
class InventoryItem:
    """库存表中的正式物品。只有稳定快照比较结束后才能修改它。"""

    item_id: int
    cls_id: int
    status: InventoryStatus
    anchor_box: BBox
    anchor_valid: bool
    last_seen_box: BBox
    no_occluder_count: int = 0
    duplicate_count: int = 0

    def copy(self) -> "InventoryItem":
        """planned_changes 使用副本，避免处理中提前修改正式库存。"""
        return replace(self)


class TrackState(Enum):
    """Track 的三个主要状态。

    TRACKING_VISIBLE：YOLO 还能看见物品，可以是完整框或局部框。
    FULL_HAND_OCCLUDED：YOLO 看不到物品，proxy_box 暂时跟随手移动。
    PLACED：已确认物品放下，release_box 是最终位置证据。
    """

    TRACKING_VISIBLE = auto()
    FULL_HAND_OCCLUDED = auto()
    PLACED = auto()


@dataclass
class Track:
    """仅用来证明“库存中已有物品被整理”的临时轨迹。"""

    track_id: int
    bound_item_id: int
    cls_id: int
    start_box: BBox
    proxy_box: BBox
    last_yolo_box: Optional[BBox]
    last_hand_box: BBox
    path: list[BBox] = field(default_factory=list)
    release_box: Optional[BBox] = None
    state: TrackState = TrackState.TRACKING_VISIBLE


@dataclass(frozen=True)
class SnapshotItem:
    """N 帧无手稳定结果融合出的一个物品，不保存 item_id。"""

    snapshot_id: int
    cls_id: int
    box: BBox
    score: float
    vote_count: int


@dataclass(frozen=True)
class StableSnapshot:
    frame_index: int
    items: tuple[SnapshotItem, ...]


@dataclass
class ProcessStore:
    """一次开门会话的临时状态；会话结束后全部清空。"""

    previous_frame: Optional[FrameObservation] = None
    no_hand_buffer: list[FrameObservation] = field(default_factory=list)
    current_stable_snapshot: Optional[StableSnapshot] = None
    tracks: dict[int, Track] = field(default_factory=dict)
    operation_pending: bool = False
    next_track_id: int = 1

    def reset_after_commit(self) -> None:
        """稳定快照已经提交库存后，旧 Track 不得带入下一轮比较。"""
        self.no_hand_buffer.clear()
        self.current_stable_snapshot = None
        self.tracks.clear()
        self.operation_pending = False


@dataclass
class InventoryDB:
    """Python 参考版的库存表。真实项目中它对应后台读取/写入的数据。"""

    items: dict[int, InventoryItem]
    next_item_id: int = 1

    def __post_init__(self) -> None:
        if self.items:
            self.next_item_id = max(self.next_item_id, max(self.items) + 1)

    def visible_items(self) -> list[InventoryItem]:
        return [item for item in self.items.values() if item.status == InventoryStatus.VISIBLE]


@dataclass
class ComparisonResult:
    """稳定快照与库存比较后的计划，不会直接改写 InventoryDB。"""

    planned_updates: dict[int, InventoryItem] = field(default_factory=dict)
    planned_new_items: list[InventoryItem] = field(default_factory=list)
    events: list[str] = field(default_factory=list)

    def effective_item(self, original: InventoryItem) -> InventoryItem:
        """返回本轮 planned_changes 中的版本；没有计划时返回原始库存副本。"""
        return self.planned_updates.get(original.item_id, original)

    def set_item(self, item: InventoryItem) -> None:
        self.planned_updates[item.item_id] = item

    def commit_to(self, inventory: InventoryDB) -> None:
        """在所有步骤成功结束后，一次性提交到正式库存。"""
        for item_id, updated in self.planned_updates.items():
            inventory.items[item_id] = updated.copy()
        for new_item in self.planned_new_items:
            inventory.items[new_item.item_id] = new_item.copy()
        if inventory.items:
            inventory.next_item_id = max(inventory.next_item_id, max(inventory.items) + 1)


@dataclass(frozen=True)
class EngineConfig:
    """所有阈值集中在此处。真实视频调参时优先改这里，而不是改业务流程。"""

    stable_snapshot_frames: int = 3

    # 常规“同一物品”框匹配：中心距离 + 宽高比例。
    normal_center_distance: float = 70.0
    normal_max_width_ratio: float = 1.8
    normal_max_height_ratio: float = 1.8

    # 全遮挡后重新出现：局部框与完整 proxy_box 的宽高可能不同，只看中心。
    reappear_center_distance: float = 110.0

    # 库存/快照的原位置直接匹配，比 Track 帧间匹配更严格。
    snapshot_center_distance: float = 45.0
    snapshot_max_width_ratio: float = 1.35
    snapshot_max_height_ratio: float = 1.35

    # 手和物品的接近/覆盖判断。
    hand_near_distance: float = 130.0
    hand_overlap_ratio: float = 0.05

    # 创建 Track 时，局部 YOLO 框与库存框的包含程度。
    inventory_containment_ratio: float = 0.65

    # 放下确认与路径记录。
    release_motion_distance: float = 18.0
    release_max_width_ratio: float = 1.35
    release_max_height_ratio: float = 1.35

    # 遮挡计数器的阈值；实际项目可按需要调整。
    occlusion_counter_threshold: int = 2
