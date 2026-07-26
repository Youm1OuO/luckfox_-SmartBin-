"""把每帧 Track、快照、库存比较串起来的参考业务引擎。

外部只需要把 YOLO 的结果转换为 `Detection` 列表，并把手框传入本类。
摄像头、RKNN、后台通信都不属于此 Python 参考实现的职责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from inventory_logic import compare_snapshot_to_inventory
from models import (
    BBox,
    ComparisonResult,
    Detection,
    EngineConfig,
    FrameObservation,
    InventoryDB,
    ProcessStore,
    StableSnapshot,
)
from snapshot_logic import accept_no_hand_frame
from track_logic import update_tracks_for_hand_frame


@dataclass
class FrameResult:
    """处理一帧后的可读结果，demo.py 会把它打印出来。"""

    frame_index: int
    snapshot: Optional[StableSnapshot] = None
    comparison: Optional[ComparisonResult] = None
    events: list[str] = field(default_factory=list)


class FridgeBusinessEngine:
    """单次开门会话的业务流程入口。"""

    def __init__(self, inventory: InventoryDB, config: EngineConfig | None = None) -> None:
        self.inventory = inventory
        self.config = config or EngineConfig()
        self.process = ProcessStore()

    def open_door(self) -> None:
        """开始一次新的开门会话：库存不清空，只清理临时 ProcessStore。"""
        self.process = ProcessStore()

    def process_frame(
        self,
        frame_index: int,
        item_detections: list[Detection],
        hand_box: Optional[BBox],
    ) -> FrameResult:
        """输入一帧已经处理好的 YOLO 结果。"""
        frame = FrameObservation(tuple(item_detections), hand_box, frame_index)
        result = FrameResult(frame_index=frame_index)

        if frame.hand_box is not None:
            result.events.extend(
                update_tracks_for_hand_frame(frame, self.inventory, self.process, self.config)
            )
        else:
            snapshot, suspicious_motion = accept_no_hand_frame(frame, self.process, self.config)
            if suspicious_motion:
                # 没检测到手但画面明显移动，只阻止错误快照，不创建正式整理 Track。
                self.process.operation_pending = True
                result.events.append("无手但明显移动：suspicious_motion，重置快照缓冲")

            if snapshot is not None:
                result.snapshot = snapshot
                comparison = compare_snapshot_to_inventory(
                    snapshot, self.inventory, self.process, self.config
                )
                comparison.commit_to(self.inventory)
                result.comparison = comparison
                result.events.extend(comparison.events)
                result.events.append("planned_changes 已一次性提交 InventoryDB")
                self.process.reset_after_commit()

        # 无论有手无手，本帧结束才写 previous_frame。
        self.process.previous_frame = frame
        return result
