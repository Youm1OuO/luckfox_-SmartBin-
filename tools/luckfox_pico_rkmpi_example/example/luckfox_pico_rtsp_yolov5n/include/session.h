// ============================================================================
//  session.h
//  3.0 会话业务层：正式库存 + 手操作工作库存 + 逐帧候选状态机
//
//  正式 InventoryDB 只在一段连续手操作收尾时提交。手在画面中时，所有
//  HAND_*、疑似新物品 D、轨迹、遮挡和出库判断都保存在工作副本中。
// ============================================================================
#ifndef __FRIDGE_SESSION_H
#define __FRIDGE_SESSION_H

#include <map>
#include <set>
#include <vector>

#include "inventory.h"
#include "snapshot.h"

namespace fridge {

enum class EventKind { IN, OUT, MOVED, OCCLUDED, REVEALED };

struct InventoryEvent {
    EventKind kind = EventKind::IN;
    int item_id = -1;
    int cls_id = -1;
    BBox box;
    BBox before_box;
    BBox after_box;
    float score = 0.0f;
};

struct SettlementResult {
    bool committed = false;
    bool happened = false;
    std::vector<InventoryEvent> events;
};

struct FrameProcessResult {
    bool stable_snapshot_generated = false;
    SettlementResult settlement;
};

enum class InitialCheckState {
    NONE,
    WAITING,
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

// 只存在于一段连续手操作内的状态；InventoryItem 不保存 HAND_* 状态。
enum class OperationTrackState {
    NORMAL,
    HAND_PARTIAL_BLOCKED,
    HAND_FULL_BLOCKED,
    PLACED,
};

// 疑似新物品 D 的首次发现来源。来源只决定后续需要哪一条确认链路；
// 无论哪一种，正式 IN 都仍要等无手稳定快照提交。
enum class SuspectSource {
    NONE,
    HAND_VISIBLE_D,
    C_POSITION_REPLACEMENT_D,
    POST_HAND_REVEAL_D,
};

struct MoveValue {
    float dx = 0.0f;
    float dy = 0.0f;
};

// 一条运行时记录既可代表已有库存物品，也可代表 suspect_id < 0 的疑似 D。
// item_id 为正式工作库存 id；未提升的 D 只有 suspect_id。
struct OperationTrack {
    int item_id = -1;
    int suspect_id = 0;
    bool is_suspect_new = false;
    bool promoted_to_working_inventory = false;
    SuspectSource suspect_source = SuspectSource::NONE;
    int cls_id = -1;

    // original_box 在进入 HAND_* 时固定；主轨迹只从它叠加 move_values 得到。
    BBox original_box;
    BBox last_seen_box;
    bool has_last_seen_box = false;
    BBox first_hand_block_box;
    bool has_first_hand_block_box = false;
    BBox last_hand_block_box;
    bool has_last_hand_block_box = false;
    // C 曾暂时不可见后重新出现的同类 B。它先只是候选，必须连续自匹配
    // 或由无手稳定快照支持，不能在第一帧直接把它当成 C 或新 D。
    BBox reappear_candidate_box;
    bool has_reappear_candidate_box = false;
    BBox placed_box;
    bool has_placed_box = false;

    OperationTrackState state = OperationTrackState::NORMAL;
    bool hold_and_move = false;
    bool shelter_or_hold = false;
    bool drop_confirmed = false;
    int hold_evidence_count = 0;
    int not_hold_evidence_count = 0;
    int self_match_count = 0;
    int reappear_candidate_match_count = 0;
    int drop_evidence_count = 0;
    // 只用于已有 C：上一有效有手帧没有可靠看到 C 时，下一次同类 B 即使
    // 正好落在预计轨迹上，也先走重新出现候选的二次确认。
    bool reappearance_pending = false;
    bool reappear_candidate_started_touching_hand = false;
    int hand_track_start_index = -1;
    // 仅 POST_HAND_REVEAL_D 使用：它在第几张无手帧首次出现。
    // 后续一张有效无手帧若不能自匹配，就丢弃该候选而不产生事件。
    int post_hand_reveal_no_hand_streak = -1;

    std::vector<MoveValue> move_values;
    std::vector<BBox> track;       // 每个点均为完整物品坐标系的估计框
};

class SessionManager {
public:
    SessionManager();

    void start_new_session(long long time_ms = 0);
    void init_from_backend(const std::vector<InventoryItem>& items,
                           bool authoritative_empty = false);
    void mark_backend_unavailable();
    void finish_session(long long time_ms = 0);

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
    bool operation_pending() const { return working_inventory_active_; }
    int no_hand_streak() const { return no_hand_streak_; }
    InitialCheckState initial_check_state() const { return initial_check_state_; }
    const InventoryDB& inventory() const { return inventory_; }
    InventoryDB& inventory() { return inventory_; }
    const std::map<int, OperationTrack>& operation_tracks() const {
        return track_buffer_;
    }

    void print_inventory() const;

private:
    // 会话/初始化
    void rebuild_persistent_item_index_();
    void reset_operation_runtime_();
    void begin_working_operation_(const BBox& hand_box,
                                  const std::vector<Detection>& detections);
    void finalize_initial_check_before_hand_();
    void validate_initial_snapshot_(const Snapshot& snapshot) const;
    void initialize_from_bootstrap_snapshot_(const Snapshot& snapshot);

    // 有手逐帧处理
    void process_effective_hand_frame_(const BBox& hand_box,
                                       const std::vector<Detection>& detections,
                                       bool first_hand_frame);
    void append_move_to_existing_hand_tracks_(const MoveValue& delta);
    void update_existing_hand_tracks_(const BBox& hand_box,
                                      const std::vector<Detection>& detections,
                                      const MoveValue& delta,
                                      std::set<int>* claimed_detection_indices,
                                      std::map<int, int>* known_item_owner);
    void scan_or_update_suspects_(const BBox& hand_box,
                                  const std::vector<Detection>& detections,
                                  std::set<int>* claimed_detection_indices,
                                  const std::map<int, int>& known_item_owner,
                                  bool first_hand_frame);
    void mark_newly_hand_blocked_items_(const BBox& hand_box,
                                        const std::vector<Detection>& detections,
                                        std::set<int>* claimed_detection_indices,
                                        std::map<int, int>* known_item_owner);
    // 在逐帧 D 预扫描前，为本帧仍处于普通可见状态的旧库存保留其唯一的
    // 严格匹配框。这样同类新 D 即使贴着旧物品出现，也不会被旧物品的
    // “局部可能匹配”吞掉。
    void reserve_visible_known_detections_(
        const BBox& hand_box,
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner);
    void apply_suspect_cover_evidence_(const BBox& hand_box,
                                       const std::vector<Detection>& detections);

    // 手离开后的收尾与提交
    void observe_no_hand_frame_(const std::vector<Detection>& detections);
    void register_post_hand_reveal_suspects_(
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices);
    SettlementResult settle_stable_snapshot_(const Snapshot& snapshot);
    void clear_runtime_after_commit_();

    // 运行时对象与工作库存操作
    int runtime_key_for_item_(int item_id) const { return item_id; }
    int new_suspect_id_();
    OperationTrack* find_runtime_for_item_(int item_id);
    const OperationTrack* find_runtime_for_item_(int item_id) const;
    void promote_suspect_(int runtime_key, const Detection& detection,
                          int frame_id);
    void confirm_rearrange_(OperationTrack& track, const BBox& release_box,
                            float score, int frame_id);
    void release_not_held_(OperationTrack& track, bool occluded);
    void mark_pending_out_(int item_id);
    void refresh_confirmed_blockers_(const std::set<int>& observed_working_ids);

    InventoryDB inventory_;  // 正式库存
    std::map<int, InventoryItem*> item_by_id_;

    // 手操作期间唯一可写的库存副本。
    std::map<int, InventoryItem> working_inventory_;
    std::map<int, InventoryItem> operation_start_inventory_;
    int working_next_item_id_ = 1;
    bool working_inventory_active_ = false;

    // key 为 item_id（>=1）或 suspect_id（<0）。
    std::map<int, OperationTrack> track_buffer_;
    std::set<int> pending_in_ids_;
    std::set<int> pending_out_ids_;
    std::set<int> confirmed_moved_ids_;
    std::set<int> released_hand_candidate_ids_;
    std::vector<BBox> hand_track_;
    BBox old_hand_box_;
    bool has_old_hand_box_ = false;
    int next_suspect_id_ = -1;

    SnapshotBuffer no_hand_buffer_;
    bool hand_present_ = false;
    bool session_active_ = false;
    bool has_local_inventory_ = false;
    int no_hand_streak_ = 0;
    long long current_time_ms_ = 0;
    InitialCheckState initial_check_state_ = InitialCheckState::NONE;
    BackendStatus backend_status_ = BackendStatus::UNKNOWN;
};

}  // namespace fridge

#endif  // __FRIDGE_SESSION_H
