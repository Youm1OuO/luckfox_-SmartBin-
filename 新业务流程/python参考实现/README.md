# Python 参考实现

这个目录把以下两份设计文档写成了可运行的 Python 参考代码：

- `../track的说明.md`
- `../单快照直接对库存的例子.md`

它只实现业务逻辑，不包含摄像头、RKNN、YOLO、RTSP 或后台通信。真实项目中只需把每帧 YOLO 输出转换成 `Detection`，再交给 `FridgeBusinessEngine`。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `models.py` | 所有数据结构：`InventoryItem`、`Track`、`SnapshotItem`、`ProcessStore` 等。 |
| `geometry.py` | BBox 的中心距离、包含关系、统一匹配函数。 |
| `track_logic.py` | 有手时创建和更新 Track，以及三种状态切换。 |
| `snapshot_logic.py` | 连续无手稳定帧的判断与稳定快照融合。 |
| `inventory_logic.py` | 稳定快照与库存比较，输出 `planned_changes`。 |
| `engine.py` | 串联每帧 Track、快照、库存提交的入口。 |
| `demo.py` | 苹果从左边整理到右边的完整逐帧例子。 |

## 先看哪里

建议按这个顺序阅读：

```text
models.py
→ geometry.py
→ demo.py
→ track_logic.py
→ snapshot_logic.py
→ inventory_logic.py
→ engine.py
```

## 运行示例

在本目录执行：

```bash
python3 demo.py
```

输出会逐帧打印：

```text
当前 Track 的 state
proxy_box / path / release_box
稳定快照
planned_changes 提交后的库存
```

基础回归测试：

```bash
python3 -m unittest -v test_reference.py
```

## 与未来 C++ 代码的对应

Python 与未来 C++ 尽量保持同名字段：

```text
InventoryItem.anchor_box / last_seen_box
Track.last_yolo_box / proxy_box / path / release_box
TrackState.TRACKING_VISIBLE / FULL_HAND_OCCLUDED / PLACED
ProcessStore.no_hand_buffer / tracks / operation_pending
ComparisonResult.planned_updates
```

阈值全部放在 `EngineConfig`。先用真实视频观察日志，再调阈值；不要先改主流程。
