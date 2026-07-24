# 新业务流程5 — 全面重写实现计划

## 概述

按照 `新业务流程5.md` 的设计，重写冰箱视觉系统的业务逻辑层。底层基础设施（ByteTrack-Lite、RKNN推理、RTSP推流、云端上传）保持不变，业务层（快照、库存、会话管理器、主循环）完全重写。

---

## 架构总览

```
main.cc 每帧循环：
  ├─ 1. 摄像头采集 + YOLO推理 + 坐标映射（不变）
  ├─ 2. ByteTrack-Lite 每帧更新（不变）
  ├─ 3. 手检测：遍历当前帧tracks，手bbox vs 库存物品bbox重叠 → HELD
  ├─ 4. 非手时：推入 SnapshotBuffer（3帧投票缓冲区）
  │     └─ 攒满3帧 → 生成 Snapshot → 送入 SessionManager
  ├─ 5. SessionManager：快照状态机 + 3步对比 + 库存更新
  └─ 6. 事件上报 + 画面绘制 + RTSP推流（基本不变）
```

---

## 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `include/fridge_config.h` | 修改 | 更新阈值，增加新常量 |
| `include/geometry.h` | 修改 | 增加 `diagonal()` 和 `normalized_nearby_distance()` |
| `include/inventory.h` | 重写 | 三态模型：可见/遮挡/出库 |
| `src/inventory.cc` | 重写 | 匹配新接口 |
| `include/snapshot.h` | 重写 | 多帧投票 Snapshot + SnapshotBuffer |
| `src/snapshot.cc` | 重写 | 多帧投票算法 |
| `include/session.h` | 重写 | 快照状态机 + HELD + 对比逻辑 |
| `src/session.cc` | 重写 | 完整业务逻辑 |
| `src/main.cc` | 重写 | 新主循环流水线 |
| `include/tracker.h` | 不变 | ByteTrack-Lite 接口不变 |
| `src/tracker.cc` | 不变 | ByteTrack-Lite 实现不变 |
| `include/cloud_uploader.h` | 不变 | 云端上传器不变 |
| `src/cloud_uploader.cc` | 不变 | 云端上传器不变 |
| `include/stability.h` | 删除 | 被快照状态机替代 |
| `src/stability.cc` | 删除 | 被快照状态机替代 |
| `include/hand_state.h` | 删除 | 不再需要 |
| `src/hand_state.cc` | 删除 | 不再需要 |

---

## 详细设计

### 1. `include/fridge_config.h` — 更新阈值

保留 ByteTrack-Lite 的阈值不变（HIGH_SCORE_THRESH 等），更新/新增以下：

```
// 严格身份匹配（5条件，状态变化时使用）
constexpr float IDENTITY_CENTER_DIST  = 15.0f;   // ε1: 中心距离阈值（像素）
constexpr float IDENTITY_AREA_RATIO   = 0.2f;    // ε2: 面积比差异阈值 (|A-B|/max(A,B))
constexpr float IDENTITY_IOU_THRESH   = 0.7f;     // ε3: IoU 阈值
constexpr float IDENTITY_COLOR_DIFF   = 20.0f;    // 像素颜色差异阈值（每通道 0~255）

// 多帧快照投票
constexpr int   SNAPSHOT_N            = 3;        // N帧为一个快照
constexpr float SNAPSHOT_S            = 0.6f;     // 投票阈值百分比（60%）
constexpr float SNAPSHOT_MIN_SCORE    = 0.3f;     // 最低检测分数

// 自适应"附近"距离阈值
constexpr float NEARBY_DISTANCE_THRESH = 1.2f;    // normalized_distance < 此值 = "附近"

// 手离开确认
constexpr int HAND_LEAVE_FRAMES       = 5;        // 连续N帧无手 → 确认手离开
constexpr int HAND_STABLE_FRAMES      = 3;        // 手离开后等N帧再拍快照

// 出库物品过期时间（毫秒）
constexpr long long OUT_ITEM_EXPIRE_MS = 600000LL; // 10分钟

// 帧尺寸
constexpr float FRAME_W = 1280.0f;
constexpr float FRAME_H = 720.0f;
```

### 2. `include/geometry.h` — 增加自适应距离

新增函数：

```cpp
// 物体对角线长度
inline float diagonal(const BBox& b) {
    return std::sqrt(b.w() * b.w() + b.h() * b.h());
}

// 自适应归一化中心距离（用于"附近"判定）
inline float normalized_nearby_distance(const BBox& a, const BBox& b) {
    float d_min = std::min(diagonal(a), diagonal(b));
    if (d_min <= 0.0f) return 999.0f;  // 退化情况
    return center_distance(a, b) / d_min;
}

// 面积比差异（用于严格匹配第3条件）
inline float area_ratio_diff(const BBox& a, const BBox& b) {
    float aa = a.area(), ab = b.area();
    float larger = std::max(aa, ab);
    if (larger <= 0.0f) return 0.0f;
    return std::abs(aa - ab) / larger;
}
```

### 3. `include/inventory.h` — 三态模型

```cpp
enum class ItemStatus {
    VISIBLE,    // 【可见】— YOLO能识别到，在冰箱中
    OCCLUDED,   // 【遮挡】— 被其他物品/手遮挡，仍在冰箱中
    OUT,        // 【出库】— 不在冰箱中（临时保留，用于恢复匹配）
};

struct InventoryItem {
    int item_id;              // 稳定身份ID，永不变
    int track_id;             // 当前绑定的ByteTrack ID（-1表示未绑定）
    int cls_id;               // 物品类别
    BBox box;                 // 当前位置（最新检测到的bbox）
    float score;              // 最新检测分数
    ItemStatus status;        // 当前状态
    int created_frame;        // 入库帧号
    int updated_frame;        // 最近更新帧号
    long long created_time_ms;// 入库时间戳
    long long out_time_ms;    // 出库时间戳（仅OUT状态有效，用于过期清理）
};
```

InventoryDB 新增方法：
- `add_item()` — 新增物品（默认VISIBLE）
- `find_by_item(item_id)` — 按item_id查找
- `find_similar(cls_id, box, frame)` — 按类别+位置在库存中查找相似物品（严格匹配用）
- `set_status(item_id, status)` — 更新状态
- `update_item(item_id, track_id, box, score, frame)` — 更新位置和分数
- `remove_item(item_id)` — 永久删除
- `cleanup_expired(now_ms)` — 清理过期的OUT物品
- `items()` — 返回所有物品（只读）
- `count_by_status(status)` — 按状态计数
- `to_json()` — JSON序列化
- `print()` — 调试打印

### 4. `include/snapshot.h` — 多帧投票

```cpp
// 投票缓冲区中的一个候选物品
struct VotingItem {
    int cls_id;
    BBox box;           // 平均位置（或最高分帧的位置）
    float best_score;   // 最高分数
    int count;          // 出现帧数
};

// 一份快照（投票过滤后的最终结果）
struct Snapshot {
    std::vector<VotingItem> items;  // 通过投票阈值的物品
    int frame_id;                   // 快照帧号（最后一帧）
    bool valid;                     // 是否有效
    bool has_hand;                  // 是否包含手（包含手的快照不用于对比）
};

// 多帧投票缓冲区
class SnapshotBuffer {
public:
    SnapshotBuffer(int N = 3, float s = 0.6f);

    // 每帧调用：推入一帧的检测结果
    void push(const std::vector<Detection>& detections, int frame_id, bool has_hand);

    // 缓冲区是否满（攒够N帧）
    bool full() const;

    // 取出快照并重置缓冲区
    Snapshot take_snapshot();

private:
    int N_;
    float s_;
    std::vector<std::vector<Detection>> frames_;
    std::vector<int> frame_ids_;
    std::vector<bool> hand_flags_;
};
```

`make_snapshot_voting()` 内部算法：
1. 遍历N帧，每帧的detection与投票表（VotingItem列表）做严格身份匹配
2. 匹配上 → count++，位置取加权平均
3. 没匹配上 → 新增到投票表
4. N帧结束后，保留 count >= N*s 的物品
5. 如果任何一帧有手，标记 has_hand=true

### 5. `include/session.h` — 会话管理器

```cpp
// 快照状态机
enum class SnapState {
    IDLE,        // 初始化阶段（等待第一份有效快照）
    COMPARE,     // 正常对比阶段
};

class SessionManager {
public:
    SessionManager();

    // ===== 主接口 =====

    // 初始化（开门时从后台获取库存）
    void init_from_backend(const std::vector<InventoryItem>& items);

    // 每帧调用：手检测（手bbox vs 库存物品bbox重叠 → HELD）
    // hand_boxes: 当前帧所有手的bbox
    // frame_id: 当前帧号
    // 返回：是否有HELD状态变化
    bool update_hand(const std::vector<BBox>& hand_boxes, int frame_id);

    // 推入一份快照（由SnapshotBuffer生成）
    // 返回：是否产生了事件
    SettlementResult push_snapshot(const Snapshot& snap, const cv::Mat& frame);

    // ===== 查询接口 =====
    bool has_backend() const;
    const InventoryDB& inventory() const;
    InventoryDB& inventory();

    // ===== 手状态信息 =====
    bool hand_present() const;    // 当前是否有手在画面中
    int  no_hand_streak() const;  // 连续无手帧数

private:
    // ---- 快照状态机 ----
    SnapState snap_state_;
    Snapshot snap1_;          // 基准快照
    Snapshot contrast_;       // 对比基准
    int snapshot_counter_;    // 快照计数

    // ---- 手部状态 ----
    bool hand_present_;
    int no_hand_streak_;
    std::map<int, int> candidate_held_;  // item_id → 连续被手重叠的帧数

    // ---- 库存 ----
    InventoryDB inventory_;
    bool backend_initialized_;
    std::vector<InventoryItem> backend_items_;

    // ---- 辅助方法 ----
    // 初始化时拍第一份快照
    void init_snapshot(const Snapshot& snap, const cv::Mat& frame);

    // 判断两份快照是否"差不多"（宽松匹配）
    bool snapshots_similar(const Snapshot& a, const Snapshot& b, const cv::Mat& frame);

    // step 6 三步对比
    SettlementResult compare_snapshots(const Snapshot& snap2, const cv::Mat& frame);

    // 严格身份匹配（5条件）
    bool match_strict(const BBox& a, int cls_a, const BBox& b, int cls_b, const cv::Mat& frame);

    // 颜色差异（16x16 resize + 平均RGB差）
    float color_diff(const cv::Mat& frame, const BBox& a, const BBox& b);

    // 打印库存
    void print_inventory();

    // 清理过期OUT物品
    void cleanup_expired(long long now_ms);
};
```

### 6. `src/session.cc` — 核心业务逻辑

#### 6.1 手检测（`update_hand`）

```
每帧调用：
  if (当前帧有手) {
      hand_present_ = true
      no_hand_streak_ = 0
      for (每个库存物品 item) {
          if (item.status == OUT) continue
          if (手的bbox与item的bbox有重叠 overlap_ratio_of_smaller > 0) {
              candidate_held_[item_id]++
              if (candidate_held_[item_id] >= 1) {  // 重叠即标CANDIDATE
                  item.status = OCCLUDED  // 标为遮挡（被手挡住）
              }
          } else {
              candidate_held_.erase(item_id)  // 不重叠了，取消候选
          }
      }
  } else {
      hand_present_ = false
      no_hand_streak_++
      candidate_held_.clear()
  }
  返回 has_changes
```

#### 6.2 快照状态机（`push_snapshot`）

```
if (snap.has_hand) {
    // 有手的快照不用于对比，直接丢弃
    return SettlementResult()
}

switch (snap_state_) {
    case IDLE:
        // 等待初始化
        if (!backend_initialized_) {
            // 没有后台库存 → 用第一份快照初始化
            init_snapshot(snap, frame)
            snap_state_ = COMPARE
        } else {
            // 有后台库存 → 初始化时匹配
            init_from_backend_snapshot(snap, frame)
            snap_state_ = COMPARE
        }
        snap1_ = snap
        contrast_ = snap
        snapshot_counter_ = 1
        return SettlementResult()（触发库存打印）

    case COMPARE:
        snapshot_counter_++
        current_ = snap

        if (snapshots_similar(current_, contrast_, frame)) {
            // 稳定了 → snap2 = current
            snap2_ = current_
            // 执行 step 6 三步对比
            result = compare_snapshots(snap2_, frame)
            // 交换：snap1 = snap2, 重置
            snap1_ = snap2_
            contrast_ = snap2_
            snap2_ = invalid
            return result
        } else {
            // 不稳定 → contrast = current, 继续等
            contrast_ = current_
            return SettlementResult()
        }
}
```

#### 6.3 快照相似性判断（`snapshots_similar`）

用严格身份匹配（5条件）对比两份快照：
- 对 snap1 的每个物品，在 snap2 中找匹配
- 匹配率 >= 80%（大部分物品都匹配上了）→ 相似
- 同时检查 snap2 中是否有 snap1 没有的新物品（数量差异 < 20%）

#### 6.4 step 6 三步对比（`compare_snapshots`）

**第一步：消失的物品（候选"被拿走"）**
```
for (每个库存中 VISIBLE 的物品 A) {
    if (A 在 snap2 中有严格匹配) continue  // 还在，没变化

    // A 消失了 → 检查 snap2 中是否有物品 C 在 A 的"附近"
    found_C = false
    for (snap2 中的每个物品 C) {
        if (normalized_nearby_distance(A.box, C.box) < NEARBY_DISTANCE_THRESH) {
            // C 在 A 附近
            occluded_item = 在库存中找与 C 严格匹配的 OCCLUDED 物品
            if (occluded_item) {
                // C 是旧物品露出来 → A 被拿走
                A.status = OUT
                occluded_item.status = VISIBLE
                occluded_item 更新为 C 的 bbox/score
                emit OUT event for A
            } else if (C 不在库存中) {
                // C 是新物品压住了 A
                A.status = OCCLUDED
            }
            found_C = true
            break
        }
    }
    if (!found_C) {
        // 什么都没出现 → A 被拿走
        A.status = OUT
        emit OUT event for A
    }
}
```

**第二步：snap2 中所有物品与库存匹配**
```
for (snap2 中的每个物品 C) {
    // 先与 OUT 物品做严格匹配
    out_item = 在 OUT 状态物品中找严格匹配
    if (out_item) {
        out_item.status = VISIBLE
        out_item 更新为 C 的 bbox/score/track_id
        continue
    }

    // 再与全部库存做严格匹配
    inv_item = 在库存中找严格匹配（OCCLUDED + VISIBLE）
    if (inv_item) {
        if (inv_item.status == OCCLUDED) {
            inv_item.status = VISIBLE
            inv_item 更新为 C 的 bbox/score
        }
        // VISIBLE 的已经处理过了，跳过
        continue
    }

    // 没匹配上 → 新物品入库
    inventory_.add_item(C.cls_id, C.box, C.score, ...)
    emit IN event
}
```

**第三步：例行检查**
```
// 清理冗余 OUT
for (每个 OUT 物品 A) {
    inv_match = 在 VISIBLE/OCCLUDED 物品中找严格匹配
    if (inv_match) {
        // 库存中已经有了，OUT记录是多余的
        inventory_.remove_item(A.item_id)
    }
}

// 清理冗余 OCCLUDED
for (每个 OCCLUDED 物品 A) {
    vis_match = 在 VISIBLE 物品中找严格匹配
    if (vis_match) {
        // 已经VISIBLE了，OCCLUDED记录是多余的
        inventory_.remove_item(A.item_id)
    }
}
```

#### 6.5 严格身份匹配（5条件）

```
match_strict(A_box, A_cls, B_box, B_cls, frame):
    1. A_cls == B_cls                                          → 类别相同
    2. center_distance(A_box, B_box) < IDENTITY_CENTER_DIST    → 中心距离近
    3. area_ratio_diff(A_box, B_box) < IDENTITY_AREA_RATIO     → 面积接近
    4. iou(A_box, B_box) > IDENTITY_IOU_THRESH                 → IoU高
    5. color_diff(frame, A_box, B_box) < IDENTITY_COLOR_DIFF   → 颜色相似
    全部满足 → true
```

### 7. `src/main.cc` — 新主循环

```
main() {
    // === 硬件初始化（不变）===
    init RKNN model, camera, encoder, RTSP

    // === 业务模块初始化 ===
    ByteTrackLite tracker;          // 不变
    SessionManager session;         // 新的
    SnapshotBuffer snap_buffer(3, 0.6f);  // 3帧投票缓冲区
    CloudUploader cloud;            // 不变
    cloud.start();

    // === 开关门检测（不变）===
    door_open/close detection via brightness

    while(1) {
        // --- 摄像头采集 + YOLO推理 + 坐标映射（不变）---
        capture frame, letterbox, inference, map coordinates

        // --- ByteTrack 每帧更新（不变）---
        tracks = tracker.update(detections, frame_id)

        // --- 分类手和食物 ---
        hand_boxes = [t.box for t in tracks if is_hand(t.cls_id)]
        food_dets = [d for d in detections if is_food(d.cls_id)]

        // --- 手检测：手bbox vs 库存物品bbox ---
        session.update_hand(hand_boxes, frame_id)

        // --- 快照缓冲 ---
        bool has_hand = !hand_boxes.empty()
        snap_buffer.push(food_dets, frame_id, has_hand)

        if (snap_buffer.full()) {
            Snapshot snap = snap_buffer.take_snapshot()
            if (!snap.has_hand) {
                SettlementResult res = session.push_snapshot(snap, frame)
                // 处理事件（上报云端、截图、打印库存）
                for (ev : res.events) { ... }
            }
        }

        // --- 画面绘制（基本不变）---
        draw bounding boxes + system state OSD

        // --- RTSP推流（不变）---
        encode H264, push RTSP
    }
}
```

---

## 实现顺序

1. **geometry.h** — 增加 `diagonal()`、`normalized_nearby_distance()`、`area_ratio_diff()`
2. **fridge_config.h** — 更新所有阈值常量
3. **inventory.h + inventory.cc** — 三态模型 + CRUD + 过期清理
4. **snapshot.h + snapshot.cc** — SnapshotBuffer + 多帧投票算法
5. **session.h + session.cc** — 快照状态机 + HELD + 3步对比
6. **main.cc** — 新主循环
7. **删除** stability.h/cc、hand_state.h/cc

---

## 不变的文件

- `tracker.h` / `tracker.cc` — ByteTrack-Lite 跟踪器
- `cloud_uploader.h` / `cloud_uploader.cc` — 云端上传器
- `luckfox_mpi.h` / `luckfox_mpi.cc` — 硬件抽象
- `yolov5.h` / `yolov5.cc` / `postprocess.cc` — RKNN推理
