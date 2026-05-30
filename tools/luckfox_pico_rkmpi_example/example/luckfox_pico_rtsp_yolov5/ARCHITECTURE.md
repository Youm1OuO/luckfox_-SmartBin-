# 业务层架构（骨架版）

> 这一版只搭"稳态切片 + track_id diff"的最小可演示骨架。
> 部分取出 / 再认领 / 后台上报等扩展尚未实现，留作后续。

## 模块依赖

```
   ┌─────────────────┐
   │  fridge_config  │   全局阈值与类别 ID
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │    geometry     │   BBox / IoU / 中心距
   └────────┬────────┘
            │
   ┌────────▼────────┐
   │     tracker     │   ByteTrack-Lite
   └────────┬────────┘
            │
   ┌────────▼────────┐    ┌─────────────────┐
   │    snapshot     │    │    stability    │
   │  (Snapshot,     │    │  (StabilityMon) │
   │   diff()）      │    └────────┬────────┘
   └────────┬────────┘             │
            │   ┌─────────────────┐│
            └──►│    inventory    │◄┘
                │  (track_id 主键)│
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │     session     │  编排
                │ (SessionManager)│
                └────────┬────────┘
                         │
                ┌────────▼────────┐
                │     main.cc     │  RKNN + 跟踪 + 会话 + RTSP
                └─────────────────┘
```

`hand_state.{h,cc}` 仍存在，但当前业务路径不再调用，留作后续扩展（例如"主动确认"或"OCR 触发"）。

## 每帧的处理流程

1. 从 VI 取 YUV → BGR → 翻 180°
2. letterbox → RKNN 推理 → 后处理
3. 把 detection 坐标映射回原图、转成 `fridge::Detection`
4. `tracker.update()` → `tracks` (含 track_id)
5. **`session.update(tracks, frame_id)`** ← 业务核心
   1. `StabilityMonitor` 推进，返回本帧的 transition
   2. 若 transition = STABLE→DISTURBED：冻结 `before_` 快照
   3. 若 transition = DISTURBED→STABLE：
      - 取 `after_` 快照
      - 第一次进入稳态：用 after seed 库存（baseline）
      - 之后：`diff_snapshots(before_, after_)` → 一组 ChangeEvent → 应用到库存
   4. 顺手 `evict_long_occluded()` 把超过 grace 期的 OCCLUDED 项升级为 GONE 并移除
6. 在画面上画 track 框 + 系统状态 OSD
7. 编码 H264 → RTSP 推流

## 关键参数（fridge_config.h）

| 常量 | 默认值 | 含义 |
|---|---|---|
| `STABLE_CONFIRM_FRAMES` | 6 | 进入 STABLE 需连续 N 帧"看起来稳"|
| `DISTURBED_CONFIRM_FRAMES` | 1 | 进入 DISTURBED 立刻生效 |
| `STABLE_MOTION_WINDOW` | 5 | 稳态判定看每个 track 最近多少帧的位移 |
| `STABLE_MOVE_PIX` | 8.0 | 稳态判定的位移阈值（像素）|
| `RELOCATE_MOVE_PIX` | 30.0 | 整理事件的位移阈值（像素）|
| `SNAPSHOT_MIN_SCORE` | 0.3 | 进入快照的最小检测分数 |

`SessionManager` 构造函数的 `occluded_grace_frames` 默认 300（10 FPS 下 ≈ 30 秒）。

## 事件三种类型

| 类型 | 含义 | 库存动作 |
|---|---|---|
| IN | track_id 在 after 但不在 before | 新增条目（VISIBLE） |
| OUT | track_id 在 before 但不在 after | **不删**，状态降为 OCCLUDED；超时后才升 GONE 并移除 |
| RELOCATE | 两边都有，位置差 > 阈值 | 更新位置 |

## 未实现的扩展（待补）

- **再认领 / ID 飘移兜底**：long 遮挡导致 ByteTrack 断 ID 时，把 OUT+IN 候选合并为 MOVE
- **部分取出**：bbox 面积比 + 离散等级（100% / 75% / 50% / 25%）
- **后台 JSON 上报**：参考 `端侧返回数据格式.txt`，session 结束时一次性 PUSH
- **门事件检测**：全图灰度跃升触发"开门 → 拉后台库存"
- **持久化**：SQLite / JSON 文件，跨重启保留状态
