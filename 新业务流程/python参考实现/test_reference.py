"""Python 参考实现的基础回归测试。

运行：
    python3 -m unittest -v test_reference.py

这些测试不验证 YOLO；它们只验证输入 Detection 后，业务层是否得到预期库存结果。
"""

from __future__ import annotations

import unittest

from engine import FridgeBusinessEngine
from models import BBox, Detection, EngineConfig, InventoryDB, InventoryItem, InventoryStatus


APPLE = 0


def make_box(left: float, top: float, right: float, bottom: float) -> BBox:
    return BBox(left, top, right, bottom)


def apple(item_box: BBox) -> Detection:
    return Detection(cls_id=APPLE, box=item_box, score=0.9)


def run_stable_no_hand(
    engine: FridgeBusinessEngine,
    start_frame: int,
    detections: list[Detection],
) -> None:
    """连续输入 N 帧无手稳定画面，触发一份稳定快照。"""
    for offset in range(engine.config.stable_snapshot_frames):
        engine.process_frame(start_frame + offset, detections, None)


class BusinessReferenceTests(unittest.TestCase):
    def test_unchanged_item_keeps_item_id(self) -> None:
        left = make_box(100, 100, 200, 200)
        inventory = InventoryDB(
            {1: InventoryItem(1, APPLE, InventoryStatus.VISIBLE, left, True, left)}
        )
        engine = FridgeBusinessEngine(inventory)

        run_stable_no_hand(engine, 1, [apple(left)])

        self.assertEqual(engine.inventory.items[1].status, InventoryStatus.VISIBLE)
        self.assertEqual(engine.inventory.items[1].last_seen_box, left)
        self.assertEqual(set(engine.inventory.items), {1})

    def test_missing_visible_item_becomes_out(self) -> None:
        left = make_box(100, 100, 200, 200)
        inventory = InventoryDB(
            {1: InventoryItem(1, APPLE, InventoryStatus.VISIBLE, left, True, left)}
        )
        engine = FridgeBusinessEngine(inventory)

        run_stable_no_hand(engine, 1, [])

        self.assertEqual(engine.inventory.items[1].status, InventoryStatus.OUT)

    def test_new_item_gets_new_item_id(self) -> None:
        right = make_box(300, 100, 400, 200)
        engine = FridgeBusinessEngine(InventoryDB({}))

        run_stable_no_hand(engine, 1, [apple(right)])

        self.assertEqual(len(engine.inventory.items), 1)
        new_item = next(iter(engine.inventory.items.values()))
        self.assertEqual(new_item.item_id, 1)
        self.assertEqual(new_item.status, InventoryStatus.VISIBLE)
        self.assertEqual(new_item.anchor_box, right)

    def test_release_box_proves_organize(self) -> None:
        left = make_box(100, 100, 200, 200)
        middle_part = make_box(205, 115, 280, 190)
        right_part = make_box(305, 110, 385, 190)
        right_full = make_box(300, 100, 400, 200)
        inventory = InventoryDB(
            {1: InventoryItem(1, APPLE, InventoryStatus.VISIBLE, left, True, left)}
        )
        engine = FridgeBusinessEngine(
            inventory,
            EngineConfig(
                stable_snapshot_frames=3,
                normal_center_distance=130.0,
                reappear_center_distance=150.0,
            ),
        )

        engine.process_frame(1, [apple(left)], None)
        engine.process_frame(2, [apple(make_box(110, 110, 185, 185))], make_box(80, 80, 220, 220))
        engine.process_frame(3, [apple(middle_part)], make_box(180, 80, 320, 220))
        engine.process_frame(4, [], make_box(280, 80, 420, 220))
        engine.process_frame(5, [apple(right_part)], make_box(285, 80, 425, 220))
        engine.process_frame(6, [apple(right_full)], make_box(470, 80, 610, 220))
        run_stable_no_hand(engine, 7, [apple(right_full)])

        self.assertEqual(set(engine.inventory.items), {1})
        self.assertEqual(engine.inventory.items[1].status, InventoryStatus.VISIBLE)
        self.assertEqual(engine.inventory.items[1].anchor_box, right_full)
        self.assertEqual(engine.inventory.items[1].last_seen_box, right_full)


if __name__ == "__main__":
    unittest.main(verbosity=2)
