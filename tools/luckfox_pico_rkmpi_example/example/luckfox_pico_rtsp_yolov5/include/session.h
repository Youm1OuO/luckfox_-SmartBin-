// ============================================================================
//  session.h
//  会话管理器 — 新业务流程5：纯快照对比
//
//  核心思路：
//    - 时间段划分只靠快照对比（不依赖手识别）
//    - 手识别保留用于 HELD 和整理（手快就用，手慢不影响主流程）
//    - 对比相邻两个时间段的快照 → 判断放入/拿走/遮挡
//    - 物品状态：VISIBLE（可见）、OCCLUDED（被其他物品遮挡）、TAKEN（被拿走）
//    - 身份匹配分两种：严格（5条件，状态变更用）和宽松（2条件，N帧计数用）
//    - 一个时间段至少维持 MIN_SEGMENT_SNAPSHOTS 个微快照（防止过度分割）
// ============================================================================
#ifndef __FRIDGE_SESSION_H
#define __FRIDGE_SESSION_H

#include <vector>
#include <map>
#include <set>
#include <opencv2/core.hpp>
#include "tracker.h"
#include "snapshot.h"
#include "stability.h"
#include "inventory.h"

namespace fridge {

// 出入库事件类型
enum class EventKind { IN, OUT, MOVED, IMAGE_UPDATE };

// 一条出入库事件
struct InventoryEvent {
    EventKind kind;
    int       item_id;
    int       cls_id;
    BBox      box;
    float     score;
    bool      crop_valid;
};

struct SettlementResult {
    bool happened = false;
    std::vector<InventoryEvent> events;
};

// 被手拿着的物品
struct HeldItem {
    int item_id;
    int cls_id;
    BBox original_pos;
    float score;
    int hand_track_id;
    int pickup_frame;
};

// 时间段内聚合的一个物品快照
struct SegmentItem {
    int track_id;
    int cls_id;
    BBox box;           // 最佳位置（取 score 最高的那帧）
    float best_score;
    int seen_frames;    // 该时间段内被检测到的帧数
};

// 时间段快照
struct TimeSegment {
    int start_frame;
    int end_frame;
    std::map<int, SegmentItem> items;  // track_id -> 聚合结果
    bool valid;
};

// 会话管理器状态
enum class SessionPhase {
    IDLE,           // 开机初始化（收集初始快照）
    COLLECTING,     // 正在收集时间段快照（手在时也继续收集，HELD在内部处理）
    COMPARING,      // 时间段结束，做快照对比
};

class SessionManager {
public:
    explicit SessionManager(int segment_frames = 10);

    // 旧接口（无 frame，走兜底逻辑）
    SettlementResult update(const std::vector<Track>& tracks, int frame_id);

    // 新接口（带 frame + 时间戳）
    SettlementResult update(const std::vector<Track>& tracks, int frame_id,
                            const cv::Mat& frame, long long time_ms);

    // 开门时用后台库存初始化
    void init_from_backend(const std::vector<InventoryItem>& backend_items,
                           const cv::Mat& frame);

    bool has_backend() const { return backend_initialized_; }
    SystemState system_state() const { return current_state_; }
    bool has_baseline() const { return baseline_initialized_; }
    const InventoryDB& inventory() const { return inventory_; }
    InventoryDB& inventory() { return inventory_; }

private:
    InventoryDB inventory_;
    bool baseline_initialized_;
    SystemState current_state_;
    bool backend_initialized_;
    std::vector<InventoryItem> backend_inventory_;

    // ===== 时间段快照 =====
    SessionPhase phase_;
    TimeSegment current_segment_;
    TimeSegment prev_snapshot_;
    bool has_prev_snapshot_;
    int no_hand_streak_;
    int segment_frames_;
    int snapshot_count_;    // 当前时间段内已收集的微快照数（用于最小持续时间判断）

    // ===== HELD 状态机（手在时） =====
    std::map<int, HeldItem> held_items_;
    int last_hand_frame_;

    // ===== 图片更新 =====
    std::set<int> image_updated_;
    long long current_time_ms_;
    std::map<int, BBox> item_last_box_;  // 上一帧位置（用于图片更新的稳定检测）

    // ===== N帧微快照（方法B用） =====
    int n_frame_counter_;                   // 当前N帧内已处理的帧数
    std::vector<SegmentItem> n_frame_items_;// 当前N帧内的候选物品（用宽松匹配聚合）
    TimeSegment last_n_snapshot_;            // 上一个N帧微快照
    bool has_last_n_snapshot_;

    // ===== 辅助函数 =====
    static float color_diff_crop(const cv::Mat& frame, const BBox& a, const BBox& b);
    static bool is_same_item_strict(const BBox& box_a, int cls_a,
                                     const BBox& box_b, int cls_b,
                                     const cv::Mat& frame);

    // 宽松匹配：只需类别相同 + 位置大致相同（基于bbox对角线）
    static bool is_same_item_loose(const BBox& box_a, int cls_a,
                                    const BBox& box_b, int cls_b);

    // 计算宽松匹配的距离阈值（bbox对角线 * IDENTITY_LOOSE_DIAG_RATIO）
    static float loose_dist_thresh(const BBox& box);

    // 将当前帧的检测结果聚合到时间段中
    void aggregate_to_segment(const std::vector<const Track*>& foods, int frame_id);

    // 检测新 HELD 物品（手盖住了哪些库存物品）
    void detect_held_items(const std::vector<const Track*>& hands,
                           const std::vector<const Track*>& foods,
                           int frame_id);

    // 处理 held 物品（手在时的逻辑：整理 + 取出确认）
    SettlementResult process_held_items(const std::vector<const Track*>& hands,
                                         const std::vector<const Track*>& foods,
                                         bool hand_effective,
                                         int frame_id,
                                         const cv::Mat& frame);

    // 三步处理：时间段快照对比（先减后加 + 例行检查）
    SettlementResult process_segment_comparison(
        const TimeSegment& new_seg,
        int frame_id, const cv::Mat& frame);

    // 方法B：检查N帧微快照是否触发时间段边界
    bool check_method_b_boundary(const std::vector<const Track*>& foods,
                                  int frame_id);

    // 过滤时间段快照中低于N帧阈值的物品
    void filter_segment_by_threshold(TimeSegment& seg);

    // 打印库存状态
    void print_inventory();

    // ===== 可调参数 =====
    static constexpr float HAND_COVER_RATIO = 0.5f;
    static constexpr int HAND_INERTIA_FRAMES = 10;
    static constexpr float SMOOTH_RELOCATE_PIX = 60.0f;
    static constexpr float SMOOTH_SETTLE_PIX = 8.0f;
    static constexpr float FRAME_W = 1280.0f;
    static constexpr float FRAME_H = 720.0f;
    static constexpr int IMAGE_UPDATE_DELAY_FRAMES = 5;
};

}  // namespace fridge

#endif  // __FRIDGE_SESSION_H
