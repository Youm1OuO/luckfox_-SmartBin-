// ============================================================================
//  session.h
//  2.0 会话业务层：持久库存 + 手后 Track + 稳定快照原子结算
// ============================================================================
#ifndef __FRIDGE_SESSION_H
#define __FRIDGE_SESSION_H

#include <map>
#include <vector>

#include "inventory.h"
#include "snapshot.h"

namespace fridge {

// IN / OUT / MOVED 会上报云端；OCCLUDED / REVEALED 只表示本地可见状态变化。
enum class EventKind { IN, OUT, MOVED, OCCLUDED, REVEALED };

struct InventoryEvent {
    EventKind kind;
    int item_id;
    int cls_id;
    BBox box;          // IN / OUT 的位置；MOVED 时等于 after_box
    BBox before_box;   // 仅 MOVED 使用
    BBox after_box;    // 仅 MOVED 使用
    float score;
};

struct SettlementResult {
    // committed 表示 working_inventory 已完整替换正式库存。失败时 events 必为空。
    bool committed = false;
    bool happened = false;
    std::vector<InventoryEvent> events;
};

struct FrameProcessResult {
    bool stable_snapshot_generated = false;
    SettlementResult settlement;
};

// 冷启动首次快照与正式手后结算严格分开，不能借 operation_pending 混用。
enum class InitialCheckState {
    NONE,
    WAITING,
    // 仅后台不可用的本地测试兜底：首次快照建立本地库存，不是只读校验。
    BOOTSTRAP_FROM_SNAPSHOT,
    DONE,
    SKIPPED,
    NOT_NEEDED,
};

enum class BackendStatus {
    UNKNOWN,
    TRUSTED,
    NO_TRUSTED_BACKEND,
};

enum class OperationTrackState {
    VISIBLE,
    HAND_OCCLUDED,
    PLACED,
    INVALID,
};

struct OperationTrack {
    int bound_item_id = -1;
    int cls_id = -1;

    BBox start_box;                 // 从 A.base_box 复制，表示可靠起点
    BBox proxy_box;                 // 仅有已验证移动时才更新的完整物品代理框
    BBox last_item_box;
    bool has_last_item_box = false;
    BBox last_hand_box;
    bool has_last_hand_box = false;

    OperationTrackState state = OperationTrackState::VISIBLE;
    bool shelter_or_hold = false;   // 已有手接触 / 遮挡候选证据
    bool hold_and_move = false;     // 已同时确认接触和有效移动
    bool seen_hand_contact = false;
    bool seen_effective_move = false;
    int still_at_start_count = 0;
    int missing_without_hand_count = 0;

    BBox release_box;
    bool has_release_box = false;
    std::vector<BBox> hand_path;
    std::vector<BBox> proxy_path;
    bool frozen = false;
};

class SessionManager {
public:
    SessionManager();

    // 新 Session 只清理会话数据，绝不清空跨关门的 inventory / next_item_id。
    void start_new_session(long long time_ms = 0);
    void init_from_backend(const std::vector<InventoryItem>& items,
                           bool authoritative_empty = false);
    void mark_backend_unavailable();
    void finish_session(long long time_ms = 0);

    // 门进入 CLOSING 时暂停证据链；若误判恢复 OPEN，按文档有条件丢弃 Track。
    void begin_closing_guard();
    void resume_after_false_closing();

    FrameProcessResult process_frame(const std::vector<Detection>& food_detections,
                                     const std::vector<BBox>& hand_boxes,
                                     int frame_id, long long time_ms);

    bool needs_backend_inventory() const {
        return session_active_ && !has_local_inventory_ &&
               backend_status_ == BackendStatus::UNKNOWN;
    }
    bool has_local_inventory() const { return has_local_inventory_; }
    bool ready() const {
        return session_active_ && has_local_inventory_ &&
               initial_check_state_ != InitialCheckState::WAITING;
    }
    bool hand_present() const { return hand_present_; }
    bool operation_pending() const { return operation_pending_; }
    int no_hand_streak() const { return no_hand_streak_; }
    InitialCheckState initial_check_state() const { return initial_check_state_; }
    const InventoryDB& inventory() const { return inventory_; }
    InventoryDB& inventory() { return inventory_; }
    const std::map<int, OperationTrack>& operation_tracks() const {
        return track_buffer_;
    }

    void print_inventory() const;

private:
    bool should_collect_snapshot_() const;
    void save_previous_yolo_result_(const std::vector<Detection>& food_detections,
                                    bool current_has_hand);
    void rebuild_persistent_item_index_();

    void finalize_initial_check_before_hand_();
    void validate_initial_snapshot_(const Snapshot& snapshot) const;
    void initialize_from_bootstrap_snapshot_(const Snapshot& snapshot);

    void update_tracks_while_hand_present_(const std::vector<Detection>& detections,
                                           const BBox& hand_box, int frame_id);
    void update_track_by_visible_item_(OperationTrack& track, const Detection& detection,
                                       const BBox& hand_box);
    void update_track_by_hand_or_mark_invalid_(OperationTrack& track,
                                                const std::vector<Detection>& detections,
                                                const BBox& hand_box);
    void create_candidate_tracks_(const std::vector<Detection>& detections,
                                  const BBox& hand_box, bool first_hand_frame);
    void freeze_all_tracks_();
    void clear_tracks_after_settlement_();
    void discard_tracks_and_mark_ambiguous_();

    SettlementResult settle_snapshot_(const Snapshot& snapshot);

    InventoryDB inventory_;
    // 持久索引只引用 inventory_ 的 map 元素；replace_all 后必须重建。
    std::map<int, InventoryItem*> item_by_id_;

    SnapshotBuffer no_hand_buffer_;
    std::map<int, OperationTrack> track_buffer_;
    std::vector<OperationTrack> frozen_tracks_;

    std::vector<Detection> previous_food_detections_;
    bool previous_had_hand_ = false;
    bool track_session_is_ambiguous_ = false;
    bool operation_pending_ = false;
    bool hand_present_ = false;
    bool has_local_inventory_ = false;
    bool session_active_ = false;
    int no_hand_streak_ = 0;
    long long current_time_ms_ = 0;
    InitialCheckState initial_check_state_ = InitialCheckState::NONE;
    BackendStatus backend_status_ = BackendStatus::UNKNOWN;
};

}  // namespace fridge

#endif  // __FRIDGE_SESSION_H
