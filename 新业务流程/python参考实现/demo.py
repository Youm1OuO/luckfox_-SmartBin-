"""可直接运行的逐帧小例子。

运行：
    python3 demo.py

例子：库存里的苹果从左侧 L 被手移动到右侧 R；手还在画面中时，
苹果连续稳定两帧，因而生成 release_box。最终稳定快照用 release_box
证明这是整理，而不是“拿出旧苹果 + 放入新苹果”。
"""

from __future__ import annotations

from engine import FridgeBusinessEngine
from models import BBox, Detection, EngineConfig, InventoryDB, InventoryItem, InventoryStatus


APPLE = 0


def box(left: float, top: float, right: float, bottom: float) -> BBox:
    return BBox(left, top, right, bottom)


def apple(item_box: BBox) -> Detection:
    return Detection(cls_id=APPLE, box=item_box, score=0.95)


def print_state(engine: FridgeBusinessEngine) -> None:
    print("  Track:")
    if not engine.process.tracks:
        print("    (无)")
    for track in engine.process.tracks.values():
        print(
            "   ",
            f"#{track.track_id} item={track.bound_item_id} state={track.state.name}",
            f"proxy={track.proxy_box}",
            f"path_points={len(track.path)}",
            f"release={track.release_box}",
        )
    print("  Inventory:")
    for item in engine.inventory.items.values():
        print(
            "   ",
            f"item={item.item_id} status={item.status.name}",
            f"anchor={item.anchor_box}",
            f"last_seen={item.last_seen_box}",
        )


def feed(
    engine: FridgeBusinessEngine,
    frame_index: int,
    detections: list[Detection],
    hand: BBox | None,
) -> None:
    print(f"\n===== 第 {frame_index} 帧：{'有手' if hand else '无手'} =====")
    result = engine.process_frame(frame_index, detections, hand)
    for event in result.events:
        print("  -", event)
    if result.snapshot:
        print("  生成稳定快照：", result.snapshot.items)
    print_state(engine)


def main() -> None:
    left = box(100, 100, 200, 200)
    middle_part = box(205, 115, 280, 190)  # 手部分遮挡后的 YOLO 局部框
    right_part = box(305, 110, 385, 190)
    right_full = box(300, 100, 400, 200)

    inventory = InventoryDB(
        items={
            1: InventoryItem(
                item_id=1,
                cls_id=APPLE,
                status=InventoryStatus.VISIBLE,
                anchor_box=left,
                anchor_valid=True,
                last_seen_box=left,
            )
        }
    )
    config = EngineConfig(
        stable_snapshot_frames=3,
        # demo 的框变化较大，因此适当放宽连续 Track 框的中心距离。
        normal_center_distance=130.0,
        reappear_center_distance=150.0,
    )
    engine = FridgeBusinessEngine(inventory, config)
    engine.open_door()

    # 1. 无手稳定：只是作为开门后的普通画面。
    feed(engine, 1, [apple(left)], None)

    # 2. 手接近苹果，苹果仍部分可见：创建绑定 item_id=1 的 Track。
    feed(engine, 2, [apple(box(110, 110, 185, 185))], box(80, 80, 220, 220))

    # 3. 手拿着苹果移动，局部 YOLO 框一起向右移动。
    feed(engine, 3, [apple(middle_part)], box(180, 80, 320, 220))

    # 4. 苹果被手完全挡住：proxy_box 改为跟随手位移。
    feed(engine, 4, [], box(280, 80, 420, 220))

    # 5. 苹果在右侧重新出现：本帧只接回 last_yolo_box，不算 object_delta。
    feed(engine, 5, [apple(right_part)], box(285, 80, 425, 220))

    # 6. 手转去别处；苹果完整可见且稳定，确认 PLACED + release_box。
    feed(engine, 6, [apple(right_full)], box(470, 80, 610, 220))

    # 7. 手离开后连续 N 帧稳定，生成快照并提交库存。
    feed(engine, 7, [apple(right_full)], None)
    feed(engine, 8, [apple(right_full)], None)
    feed(engine, 9, [apple(right_full)], None)


if __name__ == "__main__":
    main()
