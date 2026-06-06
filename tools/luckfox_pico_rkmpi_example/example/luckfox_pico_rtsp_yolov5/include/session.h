// ============================================================================
//  session.h
//  会话管理器 — 新业务流程5：逐帧对比库存（track_id 优先匹配）
//
//  核心思路：
//    - ByteTrack 的 track_id 是最可靠的匹配信号（同一 track_id = 同一物体）
//    - track_id 匹配不到时，用放宽的几何匹配兜底
//    - 先减后加：先处理消失的物品，再处理新出现的物品
//    - 整理检测：HELD（手拿着）+ 平滑移动（不依赖手）两种方式并行
// ============================================================================
#ifndef __FRIDGE_SESSION_H
#define __FRIDGE_SESSION_H

#include <vector>
#include <map>
#include <set>
#include <opencv2/core.hpp>
#include "tracker.h"
#include "stability.h"
#include "inventory.h"

namespace fridge {

enum class EventKind { IN, OUT, MOVED, IMAGE_UPDATE };

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

struct HeldItem {
    int item_id;
    int cls_id;
    BBox original_pos;
    float score;
    int hand_track_id;
    int pickup_frame;
};

enum class SessionPhase {
    IDLE,           // 开机初始化（等待手首次出现）
    ACTIVE,         // 正常运行（无手，逐帧对比库存）
    HAND_ACTIVE,    // 手在画面中
};

class SessionManager {
public:
    explicit SessionManager(int stable_frames = 10);

    SettlementResult update(const std::vector<Track>& tracks, int frame_id);
    SettlementResult update(const std::vector<Track>& tracks, int frame_id,
                            const cv::Mat& frame, long long time_ms);

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

    // ===== 阶段管理 =====
    SessionPhase phase_;
    int stable_streak_;
    int stable_frames_;
    int current_frame_id_;

    // ===== HELD 状态机 =====
    std::map<int, HeldItem> held_items_;
    int last_hand_frame_;
    static constexpr int HAND_CONFIRM_LEAVE = 3;

    // ===== 平滑移动整理检测 =====
    std::map<int, BBox> item_anchor_;
    std::map<int, BBox> item_last_box_;

    // ===== 图片更新 =====
    std::set<int> image_updated_;
    long long current_time_ms_;

    // ===== 辅助函数 =====
    static float color_diff_crop(const cv::Mat& frame, const BBox& a, const BBox& b);
    static bool is_same_item_strict(const BBox& box_a, int cls_a,
                                     const BBox& box_b, int cls_b,
                                     const cv::Mat& frame);

    // 核心：逐帧对比库存（先减后加 + 恢复检查）
    SettlementResult compare_frame_to_inventory(const std::vector<const Track*>& foods,
                                                 const std::vector<const Track*>& hands,
                                                 const cv::Mat& frame,
                                                 int frame_id);

    // 遮挡恢复检查：OCCLUDED 物品被 YOLO 重新检测到 → 恢复 INVENTORY
    void occlusion_check(const std::vector<const Track*>& foods,
                          const cv::Mat& frame,
                          int frame_id);

    // HELD 机制
    void detect_held_items(const std::vector<const Track*>& hands,
                           const std::vector<const Track*>& foods,
                           int frame_id);
    SettlementResult process_held_items(const std::vector<const Track*>& hands,
                                         const std::vector<const Track*>& foods,
                                         bool hand_effective,
                                         int frame_id,
                                         const cv::Mat& frame);

    // 平滑移动整理检测
    void detect_smooth_relocate(const std::vector<const Track*>& foods,
                                int frame_id,
                                const cv::Mat& frame,
                                SettlementResult& res);

    void print_inventory();

    // ===== 可调参数 =====
    static constexpr float HAND_COVER_RATIO = 0.5f;
    static constexpr int HAND_INERTIA_FRAMES = 10;
    static constexpr float SMOOTH_RELOCATE_PIX = 60.0f;
    static constexpr float SMOOTH_SETTLE_PIX = 8.0f;
    static constexpr float FRAME_W = 1280.0f;
    static constexpr float FRAME_H = 720.0f;
    static constexpr int IMAGE_UPDATE_DELAY_FRAMES = 5;
    // TAKEN 恢复窗口：物品被标为 TAKEN 后，在多少帧内仍允许通过宽松匹配恢复
    static constexpr int TAKEN_RECOVERY_FRAMES = 5;
    // 帧跳过：每隔几帧做一次对比（1=每帧，3=每3帧）
    static constexpr int COMPARE_INTERVAL = 3;
    int compare_counter_ = 0;
};

}  // namespace fridge

#endif  // __FRIDGE_SESSION_H
