// ============================================================================
//  session.h
//  会话管理器 — item_id 身份追踪 + 抗抖动 + 整理识别
//
//  核心目标（按优先级）：
//    1) 关门时库存准确（抗 YOLO 抖动、抗手遮挡）
//    2) 整理时追踪"是哪个 item 动了，从哪到哪"（item_id 身份保持）
//    3) 过程中实时显示 IN/OUT/RELOCATE（给人看，可以不完美）
//
//  关键策略：
//    - 物品消失不立刻删：有手→不删；没手但短暂消失→不删（抖动）；
//      没手且长时间消失→才删（OUT）
//    - 整理：物品被手盖住→记为 held，手移动放下→把同一个 item_id
//      重新绑定到新 track + 新位置（身份不变）
// ============================================================================
#ifndef __FRIDGE_SESSION_H
#define __FRIDGE_SESSION_H

#include <vector>
#include <map>
#include "tracker.h"
#include "snapshot.h"
#include "stability.h"
#include "inventory.h"

namespace fridge {

struct SettlementResult {
    bool happened = false;
};

// 被手拿着的物品（held）
struct HeldItem {
    int item_id;            // 库存里的稳定身份
    int cls_id;
    BBox original_pos;      // 被拿起前的位置
    float score;
    int hand_track_id;      // 哪只手拿的
    int pickup_frame;       // 被拿起的帧号
};

// 候选新物品（连续几帧稳定出现才确认为 IN，抗抖动）
struct PendingNew {
    int track_id;
    int cls_id;
    BBox box;
    float score;
    int first_seen_frame;
    int seen_count;
};

class SessionManager {
public:
    explicit SessionManager(int occluded_grace_frames = 30);

    SettlementResult update(const std::vector<Track>& tracks, int frame_id);

    SystemState system_state() const { return current_state_; }
    bool has_baseline() const { return baseline_initialized_; }
    const InventoryDB& inventory() const { return inventory_; }
    InventoryDB& inventory() { return inventory_; }

private:
    InventoryDB inventory_;
    bool baseline_initialized_;
    int occluded_grace_frames_;     // 物品消失多少帧后才判 OUT（无手时）
    SystemState current_state_;

    // 被手拿着的物品 (key = item_id)
    std::map<int, HeldItem> held_items_;

    // 候选新物品 (key = track_id)
    std::map<int, PendingNew> pending_news_;

    // ===== 方案 B：手消失惯性 =====
    // 最后一次画面里有手的帧号。手消失后的 HAND_INERTIA_FRAMES 帧内
    // 仍然认为"可能有手"，物品不轻易判拿走（抗手识别不稳）
    int last_hand_frame_;

    // ===== 可调参数 =====
    // 新物品要连续稳定出现多少帧才确认为 IN（抗抖动）
    static constexpr int NEW_CONFIRM_FRAMES = 3;
    // 手盖住物品的判定：物品被手覆盖比例阈值
    static constexpr float HAND_COVER_RATIO = 0.5f;
    // 手消失惯性：手 track 消失后多少帧内仍认为"可能有手"
    static constexpr int HAND_INERTIA_FRAMES = 10;
    // 画面尺寸（用于方案 A 手臂延伸到边缘的计算）
    static constexpr float FRAME_W = 720.0f;
    static constexpr float FRAME_H = 480.0f;
};

}  // namespace fridge

#endif  // __FRIDGE_SESSION_H
