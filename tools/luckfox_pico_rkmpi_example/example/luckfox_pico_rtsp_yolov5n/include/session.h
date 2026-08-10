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
#include <string>
#include <vector>

#include "inventory.h"
#include "tracker.h"

namespace fridge {

namespace session_internal {
struct BlockerTransitionPlan;

// 同一物理物品被重复检测时的运行时诊断。它不代表库存对象，也不拥有
// detection；仅在当前操作中说明弱框为何暂时不进入 C/D 身份仲裁。
struct DetectionShadowHint {
    int detection_index = -1;
    int owner_detection_index = -1;
    int owner_item_id = -1;
    int owner_runtime_key = 0;
    int frame_id = -1;
    float score = 0.0f;
    float owner_score = 0.0f;
    float iom_value = 0.0f;
    float iou_value = 0.0f;
    float center_norm = 0.0f;
    float width_ratio = 0.0f;
    float height_ratio = 0.0f;

    bool valid() const {
        return detection_index >= 0 && owner_detection_index >= 0;
    }
};
}

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

// 手操作中的实时观察状态。它与最终库存事件分离：provisional=true 的状态
// 可以在后续帧被修正，不能单独产生 IN / OUT / MOVED。
enum class LiveObservationState {
    NONE,
    HAND_CONTACT,
    POSSIBLE_MOVED,
    POSSIBLE_OCCLUDED,
    PROVISIONAL_D,
    C_D_ALIAS,
    POST_HAND_REVEAL_D,
    PLACED,
};

// 最近一次将旧 C 退出活动 HAND/CONTACT 轨迹的直接原因。用于约束释放
// 语义并生成可追溯日志，不能用模糊的“没匹配到”伪造 STATIC_CONFIRMED。
enum class ReleaseReason {
    NONE,
    ORIGINAL_DETECTION,
    CONTACT_RETURNED_ORIGINAL,
    FULLY_OCCLUDED,
    // 有手阶段已临时静态确认的旧 C，在无手阶段连续两帧得到唯一、
    // 尺度一致的近原位框后才使用；它不是普通原位检测。
    STABLE_NEAR_ORIGINAL_NO_HAND,
};

struct MoveValue {
    float dx = 0.0f;
    float dy = 0.0f;
};

// 手的跨帧身份由业务层维护，不能使用当前 YOLO hand_boxes 的数组下标。
// 手框短暂合并、漏检或对应关系不唯一时，相关物品暂停使用手位移，避免把
// 另一只手的 delta 写入自己的 move_values。
enum class HandTrackState {
    OBSERVED,
    TEMP_LOST,
    MERGED_OR_AMBIGUOUS,
};

struct HandTrack {
    int hand_id = -1;
    BBox last_box;
    BBox previous_box;
    bool has_previous_box = false;
    MoveValue last_delta;
    bool has_last_delta = false;
    std::vector<BBox> history;
    int last_seen_frame = -1;
    int missing_frame_count = 0;
    HandTrackState state = HandTrackState::OBSERVED;
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
    // 该物品在 carrier_hand_id 自己的 HandTrack::history 中的起始位置。
    // 它不再引用整个操作共用的 hand_boxes 输出序列。
    int hand_track_start_index = -1;
    // HAND_* 的位置推算只能使用这一条内部手轨迹。-1 表示当前没有可靠
    // 手来源；ambiguous 时保留已有路径但不追加任意手的 delta。
    int carrier_hand_id = -1;
    bool carrier_hand_ambiguous = false;
    // carrier_hand_ambiguous 只描述当前帧能否安全使用 hand delta；一旦某条
    // 旧 C 因 hand_id 暂失、合并或接手歧义而跳过 delta，本标记在本轮事务内
    // 保留该历史，以便无手阶段只凭严格的物品直接证据恢复，而不借用别的手。
    bool hand_delta_interrupted = false;
    // 没有最终可见框的 OUT 不能从 hand_delta_interrupted 本身推导。这里只
    // 保存该旧 C 在有手阶段唯一、贴手且已离开原位置的最后直接候选；无手
    // 阶段仍必须经过原有连续缺失、alias 和遮挡仲裁才能使用它。
    bool has_direct_exit_evidence = false;
    BBox direct_exit_box;
    int direct_exit_frame = -1;
    // 物品自身的局部可见路径。它与 hand delta 分开保存：即使 hand_id
    // 因合并/漏检而暂停位移，只要当前 B 在全局一对一归属中连续属于本 C，
    // 仍可作为后续无手缺失链的受限入口，绝不单帧直接 OUT。
    bool has_direct_object_exit_evidence = false;
    int direct_object_path_streak = 0;
    int direct_object_last_frame = -1;
    BBox direct_object_last_box;
    bool has_direct_object_last_box = false;

    // 完全不可见时的物品级手组离场见证。候选 hand_id 只来自实际接触/覆盖
    // 该 C 的唯一内部手轨迹；任何 hand 合并、漏检或不安全重绑都会使本轮
    // witness 失效。它也只能开启既有无手缺失确认，不能直接生成 OUT/MOVED。
    std::set<int> possible_carrier_hand_ids;
    std::map<int, BBox> possible_carrier_last_boxes;
    bool carrier_capture_context = false;
    bool capture_was_fully_hidden = false;
    bool hand_group_identity_invalid = false;
    bool hand_group_exit_witness = false;
    int hand_group_exit_frame = -1;
    // 仅 POST_HAND_REVEAL_D 使用：它在第几张无手帧首次出现。
    // 后续一张有效无手帧若不能自匹配，就丢弃该候选而不产生事件。
    int post_hand_reveal_no_hand_streak = -1;
    // 无手阶段的同一对象直接观测次数。它是时序自匹配证据，不做框平均或投票。
    int no_hand_self_match_count = 0;
    // 旧 C 在无手阶段未被直接认领、也没有完整遮挡解释的连续次数。
    int no_hand_missing_count = 0;

    // r16：仅用于 operation-start 旧 C 的无手静态灰区收尾。它不能与
    // OUT 缺失、alias D 缺失或 D 的直接匹配计数混用。两张连续、唯一且
    // 尺度一致的近原位直接框，才允许旧 C 走既有 release 语义。
    int stable_near_original_no_hand_count = 0;
    bool has_stable_near_original_box = false;
    BBox stable_near_original_box;

    // C-D alias 只存在于当前手操作的运行时层。被隔离的 D 仍逐帧更新，
    // 但在有手阶段不得写入 working_inventory_ 成为正式/排他所有者。
    bool pending_d_quarantined_by_old_c = false;
    std::set<int> conflicting_old_item_ids;     // D -> operation-start old C
    std::set<int> conflicting_suspect_keys;     // old C -> runtime D key
    int alias_no_hand_match_count = 0;
    // 所有关联旧 C 已独立结算后，隔离 D 连续没有任何直接无手证据的次数。
    // 它不能复用旧 C 的 no_hand_missing_count，避免影响 OUT/遮挡语义。
    int alias_no_hand_missing_count = 0;
    bool alias_no_hand_matched_this_frame = false;
    int no_hand_detection_index = -1;

    // 对外/调试可追溯的实时观察层；它不替代 state、resolution 或最终事件。
    LiveObservationState live_state = LiveObservationState::NONE;
    bool live_state_provisional = true;
    int live_state_last_changed_frame = -1;
    std::string live_state_reason;

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
    // 当前候选被证据更强的另一个旧 C 暂时预约。它不等于直接观察到本 C，
    // 也不允许立刻开始 OUT；在本帧没有自己的可靠终点时仍保持未决。
    bool no_hand_candidate_reserved_by_stronger_owner = false;
};

// HAND 期“手 + 已提升 D 可能遮住旧 C”的临时几何证据。它不属于库存关系图，
// 也不能直接改变 status/block_ids/proof；只有无手阶段完成 D 的直接身份确认后，
// blocker lifecycle 才能把相同事实写成正式状态。
struct PendingFrontEvidence {
    int candidate_front_runtime_key = 0;
    int candidate_front_item_id = -1;
    BBox hand_box;
    BBox front_box;
    bool hand_and_front_cover_target = false;
    bool front_alone_covers_target = false;
    int frame_id = -1;
};

// 只在当前手操作尚未提交时保存的因果遮挡证明。它不是正式库存状态，也
// 不替代 OperationTrack::resolution：用途是在别的轨迹仍需无手结算时，
// 让已经成立的 CAUSAL_FRONT_MISSING 能逐帧续验，而不是被弱候选预约重置。
struct ProvisionalCausalOcclusion {
    int target_item_id = -1;
    OcclusionProof proof;
    BBox target_operation_start_box;
    // witness 的上一张已验证直接无手框；下一张必须与它连续。
    std::map<int, BBox> last_witness_boxes;
    // 建立 proof 时沿用的既有 causal 覆盖下限，不新增全局阈值。
    float required_coverage_ratio = 0.0f;
    int witness_confirmed_frame = -1;
    int last_validated_frame = -1;
};

// 一张有手帧中，operation-start 旧 C 对检测框的直接原位归属计划。它仅描述
// 本帧事实，不写 claimed、working_inventory 或任何证据计数；调用方在路径
// 匹配、D 扫描前依据该计划排除别的 C/D 对同一框的借用。
enum class HandDirectOldOwnerStrength {
    STRICT,
    LOCAL_CONTINUOUS,
    LOCAL_WEAK,
};

struct HandDirectOldOwnerPlan {
    // detection index -> operation-start old item_id。
    std::map<int, int> owner_by_detection;
    // item_id -> detection index，便于当前 owner 自己继续使用 LOCAL_CONTINUOUS。
    std::map<int, int> detection_by_item;
    std::map<int, HandDirectOldOwnerStrength> strength_by_detection;
    // 无法唯一裁决的原位候选。所有 C/D 都只能保持未决，不能把它当路径证据。
    std::set<int> ambiguous_detection_indices;
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
    void begin_working_operation_(const std::vector<BBox>& hand_boxes,
                                  const std::vector<Detection>& detections);
    void finalize_initial_check_before_hand_();
    void validate_initial_no_hand_frame_(const std::vector<Detection>& detections) const;
    void initialize_from_bootstrap_no_hand_frame_(const std::vector<Detection>& detections,
                                                  int frame_id);

    // 有手逐帧处理
    void process_effective_hand_frame_(const std::vector<BBox>& hand_boxes,
                                       const std::vector<Detection>& detections,
                                       bool first_hand_frame);
    void initialize_hand_tracks_(const std::vector<BBox>& hand_boxes);
    void update_hand_tracks_(const std::vector<BBox>& hand_boxes);
    int unique_current_hand_id_for_box_(const BBox& box,
                                        bool require_cover) const;
    void associate_track_with_hand_(OperationTrack* track, int hand_id,
                                    bool ambiguous);
    bool current_hand_box_for_id_(int hand_id, BBox* hand_box) const;
    bool current_hand_delta_for_id_(int hand_id, MoveValue* delta) const;
    bool current_hand_box_for_track_(const OperationTrack& track,
                                     BBox* hand_box) const;
    bool current_hand_delta_for_track_(const OperationTrack& track,
                                       MoveValue* delta) const;
    int hand_history_size_(int hand_id) const;
    bool any_current_hand_moved_() const;
    void mark_hand_delta_interrupted_(OperationTrack* track, const char* reason);
    void record_direct_exit_evidence_from_reappear_candidate_(
        OperationTrack* track, const std::vector<BBox>& hand_boxes,
        const char* source);
    void record_direct_object_exit_evidence_(
        const std::vector<BBox>& hand_boxes,
        const std::vector<Detection>& detections,
        const std::map<int, int>& known_item_owner);
    void update_hand_group_exit_witnesses_(
        const std::vector<Detection>& detections,
        const std::map<int, int>& known_item_owner);
    void append_move_to_existing_hand_tracks_();
    void update_existing_contact_tracks_(
        const std::vector<BBox>& hand_boxes,
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        const HandDirectOldOwnerPlan& direct_old_owner_plan);
    void mark_new_contact_candidates_(
        const std::vector<BBox>& hand_boxes,
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        std::set<int>* new_existing_track_ids);
    void update_existing_hand_tracks_(const std::vector<BBox>& hand_boxes,
                                      const std::vector<Detection>& detections,
                                      std::set<int>* claimed_detection_indices,
                                      std::map<int, int>* known_item_owner,
                                      const HandDirectOldOwnerPlan& direct_old_owner_plan);
    void reopen_released_static_tracks_(const std::vector<BBox>& hand_boxes,
                                        const std::vector<Detection>& detections);
    void scan_or_update_suspects_(const std::vector<BBox>& hand_boxes,
                                  const std::vector<Detection>& detections,
                                  std::set<int>* claimed_detection_indices,
                                  const std::map<int, int>& known_item_owner,
                                  const HandDirectOldOwnerPlan& direct_old_owner_plan,
                                  bool first_hand_frame);
    void mark_newly_hand_blocked_items_(const std::vector<BBox>& hand_boxes,
                                        const std::vector<Detection>& detections,
                                        std::set<int>* claimed_detection_indices,
                                        std::map<int, int>* known_item_owner,
                                        std::set<int>* new_existing_track_ids);
    // 在逐帧 D 预扫描前，为本帧仍处于普通可见状态的旧库存保留其唯一的
    // 严格匹配框。这样同类新 D 即使贴着旧物品出现，也不会被旧物品的
    // “局部可能匹配”吞掉。
    void reserve_visible_known_detections_(
        const std::vector<BBox>& hand_boxes,
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner);
    // 使用 reserve_visible_known_detections_ 的相同语义，在副本上计算本帧
    // 可由普通静态库存唯一认领的 detection -> item 映射，不写入任何状态。
    std::map<int, int> build_mutually_unique_hand_static_owner_by_detection_(
        const std::vector<BBox>& hand_boxes,
        const std::vector<Detection>& detections,
        const std::set<int>& claimed_seed,
        const std::map<int, int>& known_item_owner_seed) const;
    HandDirectOldOwnerPlan build_mutually_unique_hand_direct_old_owner_by_detection_(
        const std::vector<Detection>& detections,
        const std::set<int>& claimed_seed,
        const std::map<int, int>& known_item_owner_seed) const;
    void apply_suspect_cover_evidence_(const std::vector<BBox>& hand_boxes,
                                       const std::vector<Detection>& detections,
                                       bool any_hand_moved);
    void record_pending_front_evidence_(int target_item_id,
                                        int candidate_front_runtime_key,
                                        int candidate_front_item_id,
                                        const BBox& hand_box,
                                        const BBox& front_box,
                                        bool front_alone_covers_target);
    void clear_pending_front_evidence_for_target_(int target_item_id,
                                                  const char* reason);
    void clear_pending_front_evidence_for_suspect_(int runtime_key,
                                                   const char* reason);
    void advance_claim_grace_(const std::set<int>& new_existing_track_ids);

    // 手离开后的收尾与提交
    void observe_no_hand_frame_(const std::vector<Detection>& detections);
    void register_post_hand_reveal_suspects_(
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices);
    SettlementResult settle_no_hand_frame_(const std::vector<Detection>& detections,
                                           int frame_id);
    // 细节31：无手期后手矫正（补登记）。分两步，都只在 settle_no_hand_frame_ 里调用，
    // 有手期逻辑完全不变，纯增量兜底：
    //  (1) update：每张无手帧更新补 IN/补 OUT 的连续计数（落在手活动区域内 + 严格准入
    //      的候选才计数）。若存在"已在累积、但还没到 FLOW3_NOHAND_CORRECT_FRAMES"的候选，
    //      返回 true，表示应再等一帧（defer-commit）让它攒够证据。
    //  (2) apply：在最终提交那一帧，对连续计数已达标的候选执行补 IN / 补 OUT，
    //      事件追加进 *events、库存改动写进 *final_items。
    // 判定候选是否"落在本轮手活动区域内"复用 box_in_hand_activity_region_。
    bool update_no_hand_correction_streaks_(
        const std::vector<Detection>& detections,
        const std::vector<int>& observation_owner,
        const std::set<int>& observed_ids,
        const std::map<int, InventoryItem>& final_items,
        const std::map<int, session_internal::BlockerTransitionPlan>&
            transition_plans);
    void apply_no_hand_correction_(
        const std::vector<Detection>& detections,
        const std::vector<int>& observation_owner,
        const std::map<int, session_internal::BlockerTransitionPlan>&
            transition_plans,
        std::map<int, InventoryItem>* final_items,
        std::vector<InventoryEvent>* events);
    // 候选框/物品框是否落在"本轮手活动区域"内：hand_track_ 里任一手框按比例外扩后，
    // 与 box 的 IoM >= FLOW3_NOHAND_CORRECT_REGION_IOM 即算命中。
    bool box_in_hand_activity_region_(const BBox& box) const;
    // 一个未绑定的无手框是否是"补 IN 的合法候选"（不含帧数条件）：落在手活动区域内、
    // 高分、无法归属任何已有物品/D、不与任何已登记框或更优候选高度重叠。
    bool is_correction_in_candidate_(
        int detection_index, const std::vector<Detection>& detections,
        const std::vector<int>& observation_owner,
        const std::map<int, InventoryItem>& final_items) const;
    // 一个旧 C 是否是"补 OUT 的合法候选"（不含帧数条件）：本帧未被观察到、消失位置在
    // 手活动区域内、不在既有 OUT/MOVED 流程内，且遮挡判定三分派通过（细节31 §2.4 第3条）：
    // OCCLUDED→否；普通 VISIBLE→是；HOLD_FOR_PENDING_OCCLUSION→额外看"老位置在快照里是否
    // 被未匹配库存的框覆盖"（被覆盖→否，空了→是）。detections/observation_owner 供 HOLD
    // 分支查"未匹配框"用。
    bool is_correction_out_candidate_(
        int item_id, const std::map<int, InventoryItem>& final_items,
        const std::set<int>& observed_ids,
        const std::vector<Detection>& detections,
        const std::vector<int>& observation_owner,
        const std::map<int, session_internal::BlockerTransitionPlan>&
            transition_plans) const;
    // 只在无手阶段全局一对一归属完成后使用。它不是普通 MOVED 门槛的替代，
    // 仅补偿 hand_id 保护性中断造成的 delta 空档。
    bool can_confirm_direct_recovered_move_(const OperationTrack& track,
                                            const InventoryItem& original,
                                            const Detection& endpoint,
                                            int detection_index) const;
    bool has_unresolved_no_hand_state_(
        const std::vector<Detection>& detections,
        const std::set<int>& observed_item_ids,
        const std::set<int>& fully_occluded_item_ids,
        const std::map<int, session_internal::BlockerTransitionPlan>&
            transition_plans);
    // 普通 direct-missing OUT 即将达到门槛时的窄事务保护：只有另一个
    // operation-start 旧物已有真实移动证据，且本帧自己的候选框确实压在
    // 目标可靠 base_box 上，才暂缓这一轮 OUT。它不写 blocker/事件。
    bool defer_direct_missing_out_for_possible_occlusion_(
        int target_item_id, const std::vector<Detection>& detections,
        int* blocker_item_id, int* detection_index) const;
    // 单摄像头下同类旧物品的最终可见实例结算。它只在连续无手帧中、
    // 同类可见框数量持续少于旧库存时启用；有手阶段和 D 证据链不使用它。
    void prepare_visible_count_settlement_(const std::vector<Detection>& detections);
    // prepare_visible_count_settlement_ 只预约当前帧的 detection；缺失计数
    // 和 pending OUT 必须等待遮挡 lifecycle plan 完成后再应用。
    void apply_visible_count_missing_evidence_(
        const std::map<int, session_internal::BlockerTransitionPlan>&
            transition_plans);
    void update_pending_occlusion_evidence_(
        const std::map<int, session_internal::BlockerTransitionPlan>&
            transition_plans);
    // 只撤回当前操作中已不符合最终 lifecycle plan 的 OUT 候选；不推进任何
    // 帧计数，因此可在同一帧的单调固定点循环中安全重复调用。
    bool reconcile_pending_out_with_occlusion_plan_(
        const std::map<int, session_internal::BlockerTransitionPlan>&
            transition_plans);
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
    void rollback_provisional_moved_to_direct_original_(
        OperationTrack* track, int detection_index, const Detection& detection);
    void release_not_held_(OperationTrack& track, bool occluded,
                           ReleaseReason reason,
                           int evidence_detection_index = -1,
                           const BBox* evidence_box = nullptr,
                           const char* caller = nullptr);
    // r16 的窄收尾：只处理有手后已暂时静态、但框抖动落在 CONTACT 原位
    // 门槛与正式 MOVED 门槛之间的 operation-start 旧 C。
    bool try_release_stable_near_original_no_hand_(
        OperationTrack* track, int detection_index, const Detection& detection,
        const std::map<int, int>& independent_static_owner_by_detection,
        const char* source);
    void reset_stable_near_original_no_hand_evidence_(OperationTrack* track,
                                                       const char* reason);
    void mark_pending_out_(int item_id);
    void refresh_confirmed_blockers_(std::map<int, InventoryItem>* final_items,
                                     const std::set<int>& observed_working_ids,
                                     const std::set<int>& confirmed_front_ids,
                                     std::set<int>* fully_occluded_item_ids,
                                     std::map<int, session_internal::BlockerTransitionPlan>*
                                         transition_plans);
    bool advance_occlusion_loss_out_evidence_(
        const std::map<int, InventoryItem>& final_items,
        const std::set<int>& observed_working_ids,
        const std::set<int>& fully_occluded_item_ids);
    void link_suspect_to_conflicting_old_items_(int runtime_key,
                                                const std::set<int>& old_item_ids,
                                                const char* phase,
                                                int detection_index);
    void set_live_state_(OperationTrack* track, LiveObservationState state,
                         bool provisional, const char* reason);
    void update_hand_live_states_();
    void unlink_quarantined_suspect_(int runtime_key, const char* action);
    bool old_track_has_unresolved_alias_(const OperationTrack& track) const;
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
    // key 为被遮挡旧 C 的正式 item_id；仅用于 HAND 期可追溯证据，不参与
    // formal blocker graph 的写入。
    std::map<int, PendingFrontEvidence> pending_front_evidence_by_target_;
    // 在本轮尚未提交时保留已经成立的 CAUSAL_FRONT_MISSING。它必须在每张
    // 后续无手帧重新验证，手重新进入或操作结束时清空。
    std::map<int, ProvisionalCausalOcclusion> provisional_causal_occlusions_;
    // 当前无手帧的身份计划：key 为旧 item_id，value 为本帧仅在该旧物
    // 原位仲裁中暂时排除的跨类别低分重复 detection index。它不跨帧、
    // 不写入 InventoryItem，也不影响 OSD/原始 detection。
    std::map<int, std::set<int> > cross_class_duplicate_identity_exclusions_;
    // 每张有手/无手帧的通用重复观测诊断。shadow 不删除 YOLO 输出，也不
    // 建立 D、alias 或库存关系；它只让弱重复框暂时不参加当前帧身份仲裁。
    std::set<int> shadow_detection_indices_;
    std::map<int, int> shadow_owner_by_detection_;
    std::map<int, session_internal::DetectionShadowHint>
        shadow_hint_by_detection_;
    // 每张无手帧的同类“一框一物品”保留结果。key 是 detection index，
    // value 是唯一保留该框的旧 item_id；同一框绝不再同时阻止其他 C 的 OUT。
    std::map<int, int> visible_count_detection_owner_;
    std::set<int> visible_count_survivor_ids_;
    std::set<int> visible_count_out_candidate_ids_;
    // 当前无手帧中，同类数量缺额已经被一对一分配解决的旧 C。它与
    // visible_count_out_candidate_ids_ 的 OUT 计数生命周期分离，遮挡计划
    // 的同一帧固定点重算仍须保留这份身份结论。
    std::set<int> visible_count_identity_relaxed_ids_;
    // 这两个容器跨连续无手帧保存；手重新出现或可见数量恢复时必须撤销。
    std::map<int, int> visible_count_missing_counts_;
    std::set<int> visible_count_confirmed_out_ids_;
    std::set<int> visible_count_continuity_reset_item_ids_;
    std::map<int, std::set<int> > visible_count_prior_survivors_by_cls_;
    // 上一张直接无手帧中各保留实例的真实框。只有框仍连续时，才能沿用
    // 对应的 survivor / OUT 缺失计数，避免一次框跳变被误当作连续缺失。
    std::map<int, std::map<int, BBox> > visible_count_prior_survivor_boxes_by_cls_;
    // 历史完整遮挡解释失效后，且旧 C 本轮没有自己的运行时轨迹时使用的
    // 连续缺失计数。它只存在于当前事务，不能复用 D 或可见数量计数。
    std::map<int, int> occlusion_loss_missing_counts_;
    // 当前操作内 DISAPPEARANCE_SUPPORTED 的连续无手候选；它与普通 OUT、
    // visible-count 和遮挡失效后的 OUT 计数相互独立。
    std::map<int, int> pending_occlusion_missing_counts_;
    std::map<int, std::set<int> > pending_occlusion_witness_ids_;
    // 同一 witness 身份还必须保持可比的无手直接框；不能把前后位置明显
    // 不连续的前景检测误拼成一份“连续消失支持”证据。
    std::map<int, std::map<int, BBox> > pending_occlusion_witness_boxes_;
    // 所有手框的扁平历史只用于“任意手是否曾接触过 D”这类上下文；它不再
    // 承担某件物品的 move_value 来源。
    std::vector<BBox> hand_track_;
    // key 为稳定内部 hand_id。current_hand_* 只描述正在处理的一张有手帧。
    std::map<int, HandTrack> old_hands_;
    std::vector<BBox> current_hand_boxes_;
    std::map<int, int> current_hand_id_by_detection_;
    std::map<int, MoveValue> current_hand_delta_by_id_;
    std::set<int> current_reliable_hand_delta_ids_;
    int next_hand_id_ = 1;
    int next_suspect_id_ = -1;
    int next_operation_id_ = 1;
    int active_operation_id_ = 0;
    int trace_frame_id_ = -1;
    bool trace_hand_phase_ = false;

    // 细节31：无手期补登记的跨帧连续计数（仅本 operation 内有效，
    // reset_operation_runtime_ 处清空）。补 IN 候选按"上一帧框 + 类别"连续匹配累计；
    // 补 OUT 按 item_id 累计连续缺失帧数。手活动区域直接复用已有的 hand_track_。
    struct NoHandCorrectInStreak {
        BBox box;        // 最近一帧的候选框
        int cls_id = -1;
        int frames = 0;  // 连续稳定帧数
    };
    std::vector<NoHandCorrectInStreak> nohand_correct_in_streak_;
    std::map<int, int> nohand_correct_out_streak_;  // key=item_id, value=连续缺失帧数

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
