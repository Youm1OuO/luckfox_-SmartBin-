#ifndef __FRIDGE_SESSION_INTERNAL_H
#define __FRIDGE_SESSION_INTERNAL_H

#include <cstddef>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "fridge_config.h"
#include "session.h"

namespace fridge {
namespace session_internal {

// 一次遮挡计划中的纯几何分类结果。它不携带身份或状态机结论，也不会写入库存。
struct CoverageEvaluation {
    bool strict_full = false;
    bool edge_residual_full = false;
    bool full = false;
    bool residual_is_outer_boundary_only = false;
    float covered_ratio = 0.0f;
    float uncovered_ratio = 1.0f;
    std::size_t cover_box_count = 0;
};

enum class VisibilityDecision {
    KEEP_VISIBLE,
    KEEP_OCCLUDED,
    ENTER_OCCLUDED,
    REVEAL_VISIBLE,
    PENDING_OCCLUSION_EVIDENCE,
    PENDING_REVEAL_OR_OUT,
};

enum class OutDisposition {
    NORMAL_OUT_EVIDENCE,
    HOLD_FOR_PENDING_OCCLUSION,
    BLOCKED_BY_CONFIRMED_OCCLUSION,
    START_AFTER_OCCLUSION_LOSS,
    NOT_APPLICABLE,
};

// Event-driven relation graph for one no-hand transaction.  It is calculated
// from operation-start relations plus confirmed enter/move/exit deltas; it is
// never a per-frame intersection cache.
struct BlockerRelationGraph {
    std::map<int, std::set<int> > effective_by_target;
    std::map<int, std::set<int> > added_by_target;
    std::map<int, std::set<int> > removed_by_target;
    std::map<int, std::set<int> > moved_by_target;
};

struct OcclusionDecisionInput {
    ItemStatus previous_status = ItemStatus::VISIBLE;
    OcclusionProof previous_proof;
    CoverageEvaluation before_geometry;
    CoverageEvaluation after_geometry;
    bool previous_proof_valid = false;
    bool previous_proof_unchanged = false;
    bool relation_changed_by_confirmed_front = false;
    bool current_confirmed_front = false;
    bool valid_direct_observation = false;
    bool observation_conflict = false;
    bool target_has_independent_exit_evidence = false;
    // Narrow, confirmed target-side conclusion used by the causal route.
    // HAND/POSSIBLE_MOVED clues deliberately do not set this flag.
    bool target_has_confirmed_independent_exit = false;
    bool causal_front_missing_candidate = false;
    int prior_disappearance_missing_frames = 0;
    std::set<int> after_witness_blocker_ids;
    std::set<int> causal_witness_blocker_ids;
};

struct OcclusionDecisionResult {
    VisibilityDecision visibility = VisibilityDecision::KEEP_VISIBLE;
    OutDisposition out = OutDisposition::NORMAL_OUT_EVIDENCE;
    OcclusionProof proposed_proof;
    bool disappearance_candidate = false;
    bool causal_front_missing_candidate = false;
    int matching_missing_frames = 0;
    bool allow_occluded_transition = false;
    bool allow_revealed_transition = false;
};

// 只存在于一次无手结算调用的 blocker 因果计划。它不写入 InventoryItem，
// 也不跨操作保存；正式状态/事件只能读取通过门控的字段。
struct BlockerTransitionPlan {
    std::set<int> historical_blocker_ids;
    std::set<int> effective_blocker_ids;
    std::set<int> added_blocker_ids;
    std::set<int> removed_blocker_ids;
    std::set<int> moved_blocker_ids;
    CoverageEvaluation coverage_before;
    CoverageEvaluation coverage_after;
    bool coverage_changed_by_confirmed_front = false;
    bool valid_target_observed = false;
    bool observation_conflict = false;
    bool allow_occluded_transition = false;
    bool allow_revealed_transition = false;
    OcclusionProof before_proof;
    OcclusionProof proposed_proof;
    VisibilityDecision visibility = VisibilityDecision::KEEP_VISIBLE;
    OutDisposition out = OutDisposition::NORMAL_OUT_EVIDENCE;
    bool target_has_independent_exit_evidence = false;
    bool target_has_confirmed_independent_exit = false;
    bool causal_front_missing_candidate = false;
    bool disappearance_candidate = false;
    int matching_missing_frames = 0;
    std::set<int> disappearance_witness_ids;
    std::map<int, BBox> disappearance_witness_boxes;
};

CoverageEvaluation evaluate_coverage_facts(
        const BBox& target_box, const std::vector<BBox>& strict_cover_boxes,
        const std::vector<BBox>& edge_cover_boxes, bool allow_edge_residual);
BlockerRelationGraph build_event_driven_blocker_graph(
        const std::map<int, InventoryItem>& operation_start,
        const std::map<int, InventoryItem>& working,
        const std::set<int>& confirmed_front_ids,
        const std::set<int>& confirmed_moved_ids,
        const std::set<int>& confirmed_out_ids);
// Fixed-point helper: lifecycle can only retract a pending OUT candidate.
// It never adds one or advances per-frame evidence counters.
std::set<int> retain_pending_out_candidates(
        const std::set<int>& pending_out_ids,
        const std::map<int, BlockerTransitionPlan>& transition_plans);
bool occlusion_proof_witnesses_valid(const OcclusionProof& proof,
                                     const std::set<int>& effective_blocker_ids);
bool front_witness_boxes_are_continuous(
        const std::map<int, InventoryItem>& items,
        const std::map<int, BBox>& previous_boxes,
        const std::map<int, BBox>& current_boxes);
OcclusionDecisionResult decide_occlusion_lifecycle(
        const OcclusionDecisionInput& input);

float ratio_difference(float a, float b);
float box_edge_distance(const BBox& a, const BBox& b);
bool strict_match_box(int cls_id, const BBox& reference,
                      int observed_cls_id, const BBox& observed);
bool strict_match(const InventoryItem& item, const Detection& observed);
bool partial_match_box(int cls_id, const BBox& reference,
                       int observed_cls_id, const BBox& observed,
                       float iom_threshold = INVENTORY_PARTIAL_IOM);
bool partial_match(const InventoryItem& item, const Detection& observed);
bool hand_partial_match_box(int cls_id, const BBox& reference,
                            int observed_cls_id, const BBox& observed);
bool track_match_box(int cls_id, const BBox& reference,
                     int observed_cls_id, const BBox& observed);
bool hand_is_near(const BBox& hand, const BBox& object);
float hand_cover_ratio(const BBox& hand, const BBox& object);
bool hand_fully_covers(const BBox& hand, const BBox& object);
bool hand_affects(const BBox& hand, const BBox& object);
bool hand_touches_detection(const BBox& hand, const BBox& detection_box);
const char* suspect_source_name(SuspectSource source);
const char* operation_track_state_name(OperationTrackState state);
const char* contact_state_name(ContactState state);
const char* existing_resolution_name(ExistingItemResolution resolution);
const char* release_reason_name(ReleaseReason reason);
const char* live_observation_state_name(LiveObservationState state);
const char* runtime_owner_kind_name(const OperationTrack& track);
const char* event_kind_name(EventKind kind);
bool existing_item_needs_settlement(const OperationTrack& track);
bool existing_item_resolved_without_current_detection(const OperationTrack* track);
bool existing_track_is_terminal(const OperationTrack& track);
bool hand_track_touches_detection(const std::vector<BBox>& hand_track,
                                  const Detection& detection);
BBox move_box(const BBox& box, const MoveValue& delta);
MoveValue total_move(const OperationTrack& track);
BBox estimated_box(const OperationTrack& track);
bool is_active_existing_hand_track(const OperationTrack& track);
bool is_active_contact_track(const OperationTrack& track);
bool is_active_runtime_track(const OperationTrack& track);
bool is_unresolved_operation_start_old_track(const OperationTrack& track);
bool is_claim_protected(const OperationTrack& track);
bool is_claim_mature(const OperationTrack& track);
void record_tentative_b(OperationTrack* track, const Detection& detection,
                        bool touching_hand);
void seed_reappear_from_tentative_b(OperationTrack* track);
bool reappear_candidate_is_confirmed(const OperationTrack& track);
bool reappear_candidate_path_matches(const OperationTrack& track,
                                     const Detection& observed);
void start_reappear_candidate(OperationTrack* track, const Detection& detection,
                              bool started_touching_hand);
bool update_reappear_candidate(OperationTrack* track, const Detection& detection,
                               bool started_touching_hand);
float complete_box_size_difference(const BBox& complete, const BBox& observed);
bool becomes_more_like_complete_box(const OperationTrack& track,
                                    const BBox& previous,
                                    const BBox& current);
float move_length(const MoveValue& delta);
bool has_meaningful_hand_move(const OperationTrack& track);
InventoryItem make_inventory_item(int item_id, const Detection& detection,
                                  int frame_id, long long time_ms);
void update_seen(InventoryItem& item, const Detection& detection, int frame_id);
InventoryEvent make_event(EventKind kind, const InventoryItem& item,
                          const BBox& before = BBox(),
                          const BBox& after = BBox());
void subtract_cover(const BBox& piece, const BBox& cover,
                    std::vector<BBox>* output);
bool fully_covered_by(const BBox& target, const std::vector<BBox>& covers);
// 严格差集失败后的窄边缘残余检查。调用方必须先完成身份、confirmed
// blocker 和“target 未直接可见”的业务门控；本 helper 只判断残余几何。
bool edge_residual_within_target_border(const BBox& target,
                                        const std::vector<BBox>& covers,
                                        float edge_px);

int unique_detection_for_box(const std::vector<Detection>& detections,
                             const std::set<int>& claimed, int cls_id,
                             const BBox& reference, bool allow_partial,
                             bool allow_track);
int best_detection_for_box(const std::vector<Detection>& detections,
                           const std::set<int>& claimed, int cls_id,
                           const BBox& reference, bool allow_partial,
                           bool allow_track);
bool matches_hand_affected_reference(int cls_id, const BBox& reference,
                                     const Detection& detection,
                                     const BBox& hand);
int unique_hand_affected_detection_for_box(const std::vector<Detection>& detections,
                                           const std::set<int>& claimed,
                                           int cls_id, const BBox& reference,
                                           const BBox& hand);
int best_hand_affected_detection_for_box(const std::vector<Detection>& detections,
                                         const std::set<int>& claimed,
                                         int cls_id, const BBox& reference,
                                         const BBox& hand);
int hand_affected_existing_candidate_count(
        const std::map<int, InventoryItem>& working, const Detection& detection,
        const BBox& hand);
int unique_detection_at_old_position(const std::vector<Detection>& detections,
                                     const std::set<int>& claimed,
                                     const OperationTrack& track);
bool any_detection_at_old_position(const std::vector<Detection>& detections,
                                   const OperationTrack& track);
bool old_position_is_clean(const std::vector<Detection>& detections,
                           const OperationTrack& track,
                           const std::map<int, InventoryItem>& working);
bool detection_strictly_matches_other_item(const Detection& detection,
                                           int excluded_item_id,
                                           const std::map<int, InventoryItem>& working);

enum class StrictOwnerKind {
    NONE,
    START_OLD_C,
    PENDING_D,
    QUARANTINED_PENDING_D,
    CONFIRMED_D,
};

struct StrictDetectionOwner {
    StrictOwnerKind kind = StrictOwnerKind::NONE;
    int item_id = -1;
    int runtime_key = 0;
};

// 同类候选框在当前事务内的身份竞争关系。它不写正式库存，也不替代已有的
// 严格 owner；仅用于确保临时 D 不会遗漏一个仍可合理拥有该框的旧 C。
struct SameClassCandidateContext {
    std::set<int> direct_old_item_ids;
    std::set<int> viable_unresolved_old_item_ids;
    std::set<int> matching_suspect_runtime_keys;
};

// 当前帧跨类别重复检测的只读诊断。它只降低可疑框在身份仲裁中的权威，
// 不删除 detection，也不直接改变库存、block_ids 或事件。
struct CrossClassDuplicateHint {
    int detection_index = -1;
    int competing_index = -1;
    float score = 0.0f;
    float competing_score = 0.0f;
    float iom_value = 0.0f;
    float iou_value = 0.0f;
    float center_norm = 0.0f;
    float width_ratio = 0.0f;
    float height_ratio = 0.0f;

    bool valid() const { return detection_index >= 0 && competing_index >= 0; }
};

CrossClassDuplicateHint find_cross_class_duplicate_hint(
        const std::vector<Detection>& detections, int detection_index);

const char* strict_owner_kind_name(StrictOwnerKind kind);
const OperationTrack* runtime_for_working_item(
        int item_id, const std::map<int, OperationTrack>& tracks,
        int* runtime_key);
StrictDetectionOwner strict_owner_for_detection(
        const Detection& detection, int excluded_item_id,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks);
bool strict_owner_blocks_old_c(StrictOwnerKind kind);
bool strict_owner_is_quarantined_alias_of_old_c(
        const StrictDetectionOwner& owner, int old_item_id,
        const std::map<int, OperationTrack>& tracks);
SameClassCandidateContext build_same_class_candidate_context(
        const Detection& detection, int detection_index,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& known_item_owner);

float contact_reference_cost(const OperationTrack& track,
                             const BBox& reference,
                             const Detection& observed);
float contact_path_match_cost(const OperationTrack& track,
                              const Detection& observed);
void append_contact_observation(OperationTrack* track, const Detection& detection,
                                bool touching_hand);
bool contact_detection_is_at_original(const OperationTrack& track,
                                      const Detection& detection);
int unique_contact_original_detection(const std::vector<Detection>& detections,
                                      const std::set<int>& claimed,
                                      const OperationTrack& track,
                                      const std::map<int, InventoryItem>& working);
int unique_contact_detection_for_track(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        const OperationTrack& track, const std::map<int, InventoryItem>& working,
        const std::map<int, OperationTrack>& tracks);
bool has_contact_path_candidate(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        const OperationTrack& track, const std::map<int, InventoryItem>& working);
int unique_c_reappear_owner_for_detection(
        const Detection& detection, const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& known_item_owner);
// 当前同类 detection 对一个成熟旧 C 的临时仲裁强度。仅用于同帧预约，
// 不改变既有路径匹配、确认门槛或正式库存。
int reappear_owner_evidence_strength(const OperationTrack& track,
                                     const Detection& detection);
void mark_mature_hand_b_ambiguity(
        const Detection& detection, std::map<int, OperationTrack>* tracks,
        const std::map<int, int>& known_item_owner);
float suspect_d_reappearance_path_cost(const OperationTrack& track,
                                       const Detection& observed);
int unique_no_hand_reappear_detection_for_track(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        int runtime_key, const OperationTrack& track,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks);
bool has_ambiguous_no_hand_reappear_candidate(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        int runtime_key, const OperationTrack& track,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& independent_static_owner_by_detection,
        bool* reserved_by_stronger_owner = 0);
bool confirmed_blocker_covers_old_c(
        const OperationTrack& c, const std::vector<Detection>& detections,
        const std::map<int, int>& known_item_owner,
        const std::map<int, OperationTrack>& tracks);
int unique_c_replacement_owner_for_detection(
        const Detection& detection, const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& known_item_owner);
bool quarantined_suspect_matches_detection(const OperationTrack& suspect,
                                           const Detection& detection);
bool old_c_has_independently_settled_identity(const OperationTrack& old);
bool all_conflicting_old_c_independently_settled(
        const OperationTrack& suspect,
        const std::map<int, OperationTrack>& tracks);
bool detection_is_duplicate_of_settled_old_c(const OperationTrack& old,
                                             int detection_index,
                                             const Detection& detection);
bool quarantined_suspect_detection_is_duplicate_of_settled_old_c(
        const OperationTrack& suspect,
        const std::map<int, OperationTrack>& tracks,
        int suspect_detection_index, const Detection& detection);
bool quarantined_suspect_has_distinct_old_c_detections(
        const OperationTrack& suspect,
        const std::map<int, OperationTrack>& tracks,
        int suspect_detection_index, const Detection& suspect_detection);
int unique_d_reappearance_detection_for_track(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        int runtime_key, const OperationTrack& track,
        const std::map<int, InventoryItem>& working,
        const std::map<int, OperationTrack>& tracks);
bool detection_can_belong_to_active_track(const Detection& detection,
                                          const OperationTrack& track);
bool detection_matches_old_working_inventory(
        const Detection& detection, const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start);
bool has_unique_operation_start_owner(
        const Detection& detection, int item_id,
        const std::map<int, InventoryItem>& operation_start);
bool detection_conflicts_with_active_track(
        const Detection& detection, const std::map<int, OperationTrack>& tracks);
bool protected_existing_track_blocks_post_hand_d(
        const Detection& detection, const std::map<int, OperationTrack>& tracks);
bool has_active_runtime_for_item(const std::map<int, OperationTrack>& tracks,
                                 int item_id);
void reserve_unique_no_hand_static_inventory_detections(
        const std::vector<Detection>& detections,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::map<int, OperationTrack>& tracks,
        std::set<int>* claimed_detection_indices);

float final_motion_reference_diagonal(const BBox& before);
float normalized_final_motion_distance(const BBox& before, const BBox& after);
bool boxes_differ_as_move(const BBox& before, const BBox& after);
float strict_match_cost(int cls_id, const BBox& reference,
                        const Detection& observed);

struct HandDirectOldOwnerCandidate {
    int item_id = -1;
    int detection_index = -1;
    HandDirectOldOwnerStrength strength = HandDirectOldOwnerStrength::STRICT;
    enum class LocalContinuity {
        NONE,
        STRICT_LAST_HAND_BOX,
        TRACK_LAST_HAND_BOX,
        PARTIAL_LAST_HAND_BOX,
    } local_continuity = LocalContinuity::NONE;
    float cost = std::numeric_limits<float>::infinity();
};

int hand_direct_old_owner_strength_rank(HandDirectOldOwnerStrength strength);
int hand_direct_old_owner_local_continuity_rank(
        HandDirectOldOwnerCandidate::LocalContinuity continuity);
bool hand_direct_old_owner_candidate_better(
        const HandDirectOldOwnerCandidate& left,
        const HandDirectOldOwnerCandidate& right);
bool hand_direct_old_owner_candidate_tied(
        const HandDirectOldOwnerCandidate& left,
        const HandDirectOldOwnerCandidate& right);
int direct_old_owner_detection_for_item(const HandDirectOldOwnerPlan& plan,
                                        int item_id);
HandDirectOldOwnerStrength direct_old_owner_strength_for_detection(
        const HandDirectOldOwnerPlan& plan, int detection_index);
std::set<int> claimed_with_other_direct_old_owners(
        const std::set<int>& claimed, const HandDirectOldOwnerPlan& plan,
        int current_item_id);
float visible_count_owner_cost(const InventoryItem& original,
                               const OperationTrack* runtime,
                               const Detection& observed);
bool visible_count_survivor_box_is_continuous(int cls_id, const BBox& previous,
                                               const BBox& current);
bool detection_matches_confirmed_suspect_d(
        const Detection& detection, const std::map<int, OperationTrack>& tracks);
BBox choose_primary_hand(const std::vector<BBox>& hand_boxes);
bool hand_boxes_effectively_same(const BBox& a, const BBox& b);
void bind_mutually_unique(const std::map<int, InventoryItem>& items,
                          const std::set<int>& candidate_item_ids,
                          const std::vector<Detection>& observed,
                          std::map<int, BBox>* references,
                          std::map<int, int>* item_to_observation,
                          std::vector<int>* observation_owner,
                          bool track_mode, bool partial_mode,
                          const std::map<int, std::set<int> >* excluded_by_item = 0);
const OperationTrack* find_track_for_item(
        const std::map<int, OperationTrack>& tracks, int item_id);
std::map<int, int> build_independent_no_hand_static_owner_by_detection(
        const std::vector<Detection>& detections,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::map<int, OperationTrack>& tracks);
float track_path_match_cost(const InventoryItem& item, const OperationTrack& track,
                            const Detection& observed);
void bind_mutually_unique_track_paths(
        const std::map<int, InventoryItem>& items,
        const std::set<int>& candidate_item_ids,
        const std::vector<Detection>& observed,
        const std::map<int, OperationTrack>& tracks,
        std::map<int, int>* item_to_observation,
        std::vector<int>* observation_owner,
        const std::map<int, std::set<int> >* excluded_by_item = 0);

}  // namespace session_internal
}  // namespace fridge

#endif  // __FRIDGE_SESSION_INTERNAL_H
