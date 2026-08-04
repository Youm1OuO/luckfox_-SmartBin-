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
#include "tracker.h"

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
    // 3.0 对每张无手帧直接执行收尾状态机；这里不代表多帧快照或投票。
    bool no_hand_frame_processed = false;
    SettlementResult settlement;
};

enum class InitialCheckState {
    NONE,
    WAITING,
    BOOTSTRAP_FROM_FRAME,
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

// 与 HAND_* 正交的低覆盖率接触状态。它只存在于一次手操作的运行时轨迹中，
// 不写入 InventoryItem；普通可见/遮挡物品的默认值为 NONE。
enum class ContactState {
    NONE,
    CONTACT_CANDIDATE,
    CONTACT_MOVING,
};

// 疑似新物品 D 的首次发现来源。来源只决定后续需要哪一条确认链路；
// 无论哪一种，正式 IN 都仍要等后续无手帧完成直接确认并提交。
enum class SuspectSource {
    NONE,
    HAND_VISIBLE_D,
    C_POSITION_REPLACEMENT_D,
    POST_HAND_REVEAL_D,
};

// 旧库存 C 在一次手操作中的结算结论。它与 HAND_* / CONTACT_* 状态分离：
// state=NORMAL 只表示当前没有活动遮挡状态，不能单独证明 C 已经结案。
enum class ExistingItemResolution {
    NONE,
    STATIC_CONFIRMED,
    MOVED_CONFIRMED,
    OUT_CONFIRMED,
    OCCLUDED_CONFIRMED,
};

// 最近一次将旧 C 退出活动 HAND/CONTACT 轨迹的直接原因。用于约束释放
// 语义并生成可追溯日志，不能用模糊的“没匹配到”伪造 STATIC_CONFIRMED。
enum class ReleaseReason {
    NONE,
    ORIGINAL_DETECTION,
    CONTACT_RETURNED_ORIGINAL,
    FULLY_OCCLUDED,
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

    // original_box 在本次手操作开始影响物品时固定。
    BBox original_box;
    BBox last_seen_box;
    bool has_last_seen_box = false;
    BBox first_hand_block_box;
    bool has_first_hand_block_box = false;
    BBox last_hand_block_box;
    bool has_last_hand_block_box = false;
    // C 曾暂时不可见后重新出现的同类 B。它先只是候选，必须连续自匹配
    // 或由后续无手帧支持，不能在第一帧直接把它当成 C 或新 D。
    BBox reappear_candidate_box;
    bool has_reappear_candidate_box = false;
    BBox placed_box;
    bool has_placed_box = false;
    // CONTACT_* 转为 HAND_* 时，不能再从 original_box（旧位置）开始推算。
    // 这里保存接触阶段最后一次可靠真实框；之后 move_values 仅累计转入
    // HAND_* 之后的手位移，original_box 仍保留给最终“是否移动”比较。
    BBox hand_estimate_anchor_box;
    bool has_hand_estimate_anchor_box = false;

    OperationTrackState state = OperationTrackState::NORMAL;
    ContactState contact_state = ContactState::NONE;
    // 旧 C 从首次被手影响到本轮提交前都必须有明确结论。NORMAL 不会清掉
    // 这个义务；只有 resolution 被确认后才允许结束无手结算。
    ExistingItemResolution resolution = ExistingItemResolution::NONE;
    ReleaseReason release_reason = ReleaseReason::NONE;
    bool needs_no_hand_settlement = false;
    bool hold_and_move = false;
    bool shelter_or_hold = false;
    bool drop_confirmed = false;
    int hold_evidence_count = 0;
    int not_hold_evidence_count = 0;
    int self_match_count = 0;
    int reappear_candidate_match_count = 0;
    // 新建的旧库存 CONTACT_* / HAND_* 轨迹保护期。保护期内仍可把
    // 同类 B 记录为 tentative_b，但不能将 B 写入本帧排他认领表。
    int claim_grace_remaining = 0;
    BBox tentative_b_box;
    bool has_tentative_b_box = false;
    int tentative_b_match_count = 0;
    bool tentative_b_started_touching_hand = false;
    int drop_evidence_count = 0;
    // 只用于已有 C：上一有效有手帧没有可靠看到 C 时，下一次同类 B 即使
    // 正好落在预计轨迹上，也先走重新出现候选的二次确认。
    bool reappearance_pending = false;
    bool reappear_candidate_started_touching_hand = false;
    int hand_track_start_index = -1;
    // 仅 POST_HAND_REVEAL_D 使用：它在第几张无手帧首次出现。
    // 后续一张有效无手帧若不能自匹配，就丢弃该候选而不产生事件。
    int post_hand_reveal_no_hand_streak = -1;
    // 无手阶段的同一对象直接观测次数。它是时序自匹配证据，不做框平均或投票。
    int no_hand_self_match_count = 0;
    // 旧 C 在无手阶段未被直接认领、也没有完整遮挡解释的连续次数。
    int no_hand_missing_count = 0;

    // HAND_* 的手位移估计路径。CONTACT_* 不使用它来推算物品位置。
    std::vector<MoveValue> move_values;
    std::vector<BBox> track;       // HAND_* 的完整物品坐标系估计框
    // CONTACT_* 的实际物品观测路径；它不受手腕/手臂位移方向影响。
    std::vector<MoveValue> observed_move_values;
    std::vector<BBox> observed_track;
    // 可选的手部辅助记录。它只描述手，不参与 CONTACT_* 的身份/终点判断。
    std::vector<MoveValue> hand_move_values;
    bool contact_started_touching_hand = false;
    // 当前帧存在可解释但不唯一的 CONTACT B；此时手离开后不能直接 OUT。
    bool contact_path_ambiguous = false;
    // 同类 HAND_* C 对同一个 B 存在多个成熟解释时，稳定结算也不能按
    // item_id 顺序强行绑定；直到后续帧给出唯一解释前保持未决。
    bool b_claim_ambiguous = false;
    // 当前直接无手帧里存在一条可接上本 C 路径、但无法唯一归属的同类框。
    // 它是逐帧证据，下一张无手帧重新计算；有它时不能累计 OUT。
    bool no_hand_candidate_ambiguous = false;
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
    void validate_initial_no_hand_frame_(const std::vector<Detection>& detections) const;
    void initialize_from_bootstrap_no_hand_frame_(const std::vector<Detection>& detections,
                                                  int frame_id);

    // 有手逐帧处理
    void process_effective_hand_frame_(const BBox& hand_box,
                                       const std::vector<Detection>& detections,
                                       bool first_hand_frame);
    void append_move_to_existing_hand_tracks_(const MoveValue& delta);
    void update_existing_contact_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner);
    void mark_new_contact_candidates_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        std::set<int>* new_existing_track_ids);
    void update_existing_hand_tracks_(const BBox& hand_box,
                                      const std::vector<Detection>& detections,
                                      const MoveValue& delta,
                                      std::set<int>* claimed_detection_indices,
                                      std::map<int, int>* known_item_owner);
    void reopen_released_static_tracks_(const BBox& hand_box,
                                        const std::vector<Detection>& detections,
                                        const MoveValue& delta);
    void scan_or_update_suspects_(const BBox& hand_box,
                                  const std::vector<Detection>& detections,
                                  std::set<int>* claimed_detection_indices,
                                  const std::map<int, int>& known_item_owner,
                                  bool first_hand_frame);
    void mark_newly_hand_blocked_items_(const BBox& hand_box,
                                        const std::vector<Detection>& detections,
                                        std::set<int>* claimed_detection_indices,
                                        std::map<int, int>* known_item_owner,
                                        std::set<int>* new_existing_track_ids);
    // 在逐帧 D 预扫描前，为本帧仍处于普通可见状态的旧库存保留其唯一的
    // 严格匹配框。这样同类新 D 即使贴着旧物品出现，也不会被旧物品的
    // “局部可能匹配”吞掉。
    void reserve_visible_known_detections_(
        const BBox& hand_box,
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner);
    void apply_suspect_cover_evidence_(const BBox& hand_box,
                                       const std::vector<Detection>& detections,
                                       bool hand_moved);
    void advance_claim_grace_(const std::set<int>& new_existing_track_ids);

    // 手离开后的收尾与提交
    void observe_no_hand_frame_(const std::vector<Detection>& detections);
    void register_post_hand_reveal_suspects_(
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices);
    SettlementResult settle_no_hand_frame_(const std::vector<Detection>& detections,
                                           int frame_id);
    bool has_unresolved_no_hand_state_(const std::set<int>& observed_item_ids,
                                       const std::set<int>& fully_occluded_item_ids);
    // 单摄像头下同类旧物品的最终可见实例结算。它只在连续无手帧中、
    // 同类可见框数量持续少于旧库存时启用；有手阶段和 D 证据链不使用它。
    void prepare_visible_count_settlement_(const std::vector<Detection>& detections);
    void clear_visible_count_settlement_(bool restore_uncommitted_outs);
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
    void release_not_held_(OperationTrack& track, bool occluded,
                           ReleaseReason reason,
                           int evidence_detection_index = -1,
                           const BBox* evidence_box = nullptr,
                           const char* caller = nullptr);
    void mark_pending_out_(int item_id);
    void refresh_confirmed_blockers_(const std::set<int>& observed_working_ids);
    void trace_(const char* tag, const char* format, ...) const;
    void trace_track_(const char* tag, const OperationTrack& track,
                      const char* reason) const;

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
    // 每张无手帧的同类“一框一物品”保留结果。key 是 detection index，
    // value 是唯一保留该框的旧 item_id；同一框绝不再同时阻止其他 C 的 OUT。
    std::map<int, int> visible_count_detection_owner_;
    std::set<int> visible_count_survivor_ids_;
    std::set<int> visible_count_out_candidate_ids_;
    // 这两个容器跨连续无手帧保存；手重新出现或可见数量恢复时必须撤销。
    std::map<int, int> visible_count_missing_counts_;
    std::set<int> visible_count_confirmed_out_ids_;
    std::map<int, std::set<int> > visible_count_prior_survivors_by_cls_;
    // 上一张直接无手帧中各保留实例的真实框。只有框仍连续时，才能沿用
    // 对应的 survivor / OUT 缺失计数，避免一次框跳变被误当作连续缺失。
    std::map<int, std::map<int, BBox> > visible_count_prior_survivor_boxes_by_cls_;
    std::vector<BBox> hand_track_;
    BBox old_hand_box_;
    bool has_old_hand_box_ = false;
    int next_suspect_id_ = -1;
    int next_operation_id_ = 1;
    int active_operation_id_ = 0;
    int trace_frame_id_ = -1;
    bool trace_hand_phase_ = false;

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
