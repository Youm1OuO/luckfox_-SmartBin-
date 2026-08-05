// ============================================================================
//  session.cc
//  3.0：工作库存 + HAND_* 候选 + 疑似新物品 D + 无手逐帧条件提交
// ============================================================================
#include "session.h"
#include "fridge_config.h"

#include <algorithm>
#include <cstdarg>
#include <cmath>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace fridge {
namespace {

float ratio_difference(float a, float b) {
    const float larger = std::max(std::fabs(a), std::fabs(b));
    return larger > 0.001f ? std::fabs(a - b) / larger : 0.0f;
}

float box_edge_distance(const BBox& a, const BBox& b) {
    const float dx = std::max(std::max(a.x1 - b.x2, b.x1 - a.x2), 0.0f);
    const float dy = std::max(std::max(a.y1 - b.y2, b.y1 - a.y2), 0.0f);
    return std::sqrt(dx * dx + dy * dy);
}

bool strict_match_box(int cls_id, const BBox& reference,
                      int observed_cls_id, const BBox& observed) {
    return cls_id == observed_cls_id && reference.area() > 0.0f &&
           observed.area() > 0.0f &&
           normalized_nearby_distance(reference, observed)
               <= INVENTORY_STRICT_CENTER_NORM &&
           ratio_difference(reference.w(), observed.w())
               <= INVENTORY_STRICT_WIDTH_RATIO &&
           ratio_difference(reference.h(), observed.h())
               <= INVENTORY_STRICT_HEIGHT_RATIO;
}

bool strict_match(const InventoryItem& item, const Detection& observed) {
    return strict_match_box(item.cls_id, item.box, observed.cls_id, observed.box);
}

bool partial_match_box(int cls_id, const BBox& reference,
                       int observed_cls_id, const BBox& observed,
                       float iom_threshold = INVENTORY_PARTIAL_IOM) {
    return cls_id == observed_cls_id && reference.area() > 0.0f &&
           observed.area() > 0.0f && iom(reference, observed) >= iom_threshold;
}

bool partial_match(const InventoryItem& item, const Detection& observed) {
    return partial_match_box(item.cls_id, item.box, observed.cls_id, observed.box);
}

// 仅用于“手正在遮挡”的逐帧认领。被手挡住时，YOLO 经常只给出原物品的一部分，
// 用普通 partial_match 的高 IoM 阈值会把已有 C 错当成新的 D。这里仍要求：
// 同类、局部框有足够部分落在旧框内、局部框不会比旧框异常大、中心也不能太远。
// 它不是无手收尾阶段的身份匹配，绝不能在别处滥用。
bool hand_partial_match_box(int cls_id, const BBox& reference,
                            int observed_cls_id, const BBox& observed) {
    if (cls_id != observed_cls_id || reference.area() <= 0.0f ||
        observed.area() <= 0.0f) {
        return false;
    }
    const float observed_covered = intersection_area(reference, observed) / observed.area();
    const float area_ratio = observed.area() / reference.area();
    return observed_covered >= FLOW3_HAND_PARTIAL_MIN_OBSERVED_COVER &&
           area_ratio <= FLOW3_HAND_PARTIAL_MAX_AREA_RATIO &&
           normalized_nearby_distance(reference, observed) <= FLOW3_HAND_PARTIAL_CENTER_NORM;
}

bool track_match_box(int cls_id, const BBox& reference,
                     int observed_cls_id, const BBox& observed) {
    if (cls_id != observed_cls_id || reference.area() <= 0.0f ||
        observed.area() <= 0.0f) {
        return false;
    }
    if (partial_match_box(cls_id, reference, observed_cls_id, observed,
                          FLOW3_TRACK_PARTIAL_IOM)) {
        return true;
    }
    return normalized_nearby_distance(reference, observed) <= FLOW3_TRACK_CENTER_NORM &&
           ratio_difference(reference.w(), observed.w()) <= FLOW3_TRACK_WIDTH_RATIO &&
           ratio_difference(reference.h(), observed.h()) <= FLOW3_TRACK_HEIGHT_RATIO;
}

bool hand_is_near(const BBox& hand, const BBox& object) {
    return box_edge_distance(hand, object) <= FLOW3_HAND_ATTACH_DISTANCE &&
           intersection_area(hand, object) <= FLOW3_HAND_NEAR_MAX_INTERSECTION_AREA;
}

// 已有库存物品进入 HAND_* 的唯一几何依据。这里的 object 必须是该物品
// 被手影响前保存的完整可靠框（或其完整估计框），绝不能是被遮挡后缩小的
// 当前检测框。
float hand_cover_ratio(const BBox& hand, const BBox& object) {
    return cover_ratio(object, hand);
}

bool hand_fully_covers(const BBox& hand, const BBox& object) {
    return hand_cover_ratio(hand, object) >= FLOW3_HAND_FULL_COVER_RATIO;
}

bool hand_affects(const BBox& hand, const BBox& object) {
    return hand_cover_ratio(hand, object) >= FLOW3_HAND_PARTIAL_COVER_RATIO;
}

bool hand_touches_detection(const BBox& hand, const BBox& detection_box) {
    // 新 D 没有“被遮挡前完整框”，所以不能套用 e1/e2。它只需满足手边
    // 相贴或有足够大的检测框交集，作为“跟手出现”的上下文证据。
    return hand_is_near(hand, detection_box) ||
           intersection_area(hand, detection_box) >=
               FLOW3_HAND_DETECTION_OVERLAP_AREA;
}

const char* suspect_source_name(SuspectSource source) {
    switch (source) {
        case SuspectSource::HAND_VISIBLE_D:
            return "HAND_VISIBLE_D";
        case SuspectSource::C_POSITION_REPLACEMENT_D:
            return "C_POSITION_REPLACEMENT_D";
        case SuspectSource::POST_HAND_REVEAL_D:
            return "POST_HAND_REVEAL_D";
        case SuspectSource::NONE:
            break;
    }
    return "NONE";
}

const char* operation_track_state_name(OperationTrackState state) {
    switch (state) {
        case OperationTrackState::NORMAL:
            return "NORMAL";
        case OperationTrackState::HAND_PARTIAL_BLOCKED:
            return "HAND_PARTIAL_BLOCKED";
        case OperationTrackState::HAND_FULL_BLOCKED:
            return "HAND_FULL_BLOCKED";
        case OperationTrackState::PLACED:
            return "PLACED";
    }
    return "UNKNOWN";
}

const char* contact_state_name(ContactState state) {
    switch (state) {
        case ContactState::NONE:
            return "NONE";
        case ContactState::CONTACT_CANDIDATE:
            return "CONTACT_CANDIDATE";
        case ContactState::CONTACT_MOVING:
            return "CONTACT_MOVING";
    }
    return "UNKNOWN";
}

const char* existing_resolution_name(ExistingItemResolution resolution) {
    switch (resolution) {
        case ExistingItemResolution::NONE:
            return "NONE";
        case ExistingItemResolution::STATIC_CONFIRMED:
            return "STATIC_CONFIRMED";
        case ExistingItemResolution::MOVED_CONFIRMED:
            return "MOVED_CONFIRMED";
        case ExistingItemResolution::POSITION_REBASED:
            return "POSITION_REBASED";
        case ExistingItemResolution::OUT_CONFIRMED:
            return "OUT_CONFIRMED";
        case ExistingItemResolution::OCCLUDED_CONFIRMED:
            return "OCCLUDED_CONFIRMED";
    }
    return "UNKNOWN";
}

const char* release_reason_name(ReleaseReason reason) {
    switch (reason) {
        case ReleaseReason::NONE:
            return "NONE";
        case ReleaseReason::ORIGINAL_DETECTION:
            return "ORIGINAL_DETECTION";
        case ReleaseReason::CONTACT_RETURNED_ORIGINAL:
            return "CONTACT_RETURNED_ORIGINAL";
        case ReleaseReason::FULLY_OCCLUDED:
            return "FULLY_OCCLUDED";
        case ReleaseReason::STABLE_NEAR_ORIGINAL_NO_HAND:
            return "STABLE_NEAR_ORIGINAL_NO_HAND";
    }
    return "UNKNOWN";
}

const char* live_observation_state_name(LiveObservationState state) {
    switch (state) {
        case LiveObservationState::NONE:
            return "NONE";
        case LiveObservationState::HAND_CONTACT:
            return "HAND_CONTACT";
        case LiveObservationState::POSSIBLE_MOVED:
            return "POSSIBLE_MOVED";
        case LiveObservationState::POSSIBLE_OCCLUDED:
            return "POSSIBLE_OCCLUDED";
        case LiveObservationState::PROVISIONAL_D:
            return "PROVISIONAL_D";
        case LiveObservationState::C_D_ALIAS:
            return "C_D_ALIAS";
        case LiveObservationState::POST_HAND_REVEAL_D:
            return "POST_HAND_REVEAL_D";
        case LiveObservationState::PLACED:
            return "PLACED";
    }
    return "UNKNOWN";
}

const char* runtime_owner_kind_name(const OperationTrack& track) {
    if (!track.is_suspect_new) return "start-old-c";
    return track.pending_d_quarantined_by_old_c
        ? "quarantined-pending-d" : "pending-d";
}

const char* event_kind_name(EventKind kind) {
    switch (kind) {
        case EventKind::IN:
            return "IN";
        case EventKind::OUT:
            return "OUT";
        case EventKind::MOVED:
            return "MOVED";
        case EventKind::OCCLUDED:
            return "OCCLUDED";
        case EventKind::REVEALED:
            return "REVEALED";
    }
    return "UNKNOWN";
}

bool existing_item_needs_settlement(const OperationTrack& track) {
    return !track.is_suspect_new && track.item_id > 0 &&
           track.needs_no_hand_settlement &&
           track.resolution != ExistingItemResolution::MOVED_CONFIRMED &&
           track.resolution != ExistingItemResolution::POSITION_REBASED &&
           track.resolution != ExistingItemResolution::OUT_CONFIRMED &&
           track.resolution != ExistingItemResolution::OCCLUDED_CONFIRMED;
}

bool existing_item_resolved_without_current_detection(const OperationTrack* track) {
    if (!track || track->is_suspect_new) return false;
    return track->resolution == ExistingItemResolution::MOVED_CONFIRMED ||
           track->resolution == ExistingItemResolution::POSITION_REBASED ||
           track->resolution == ExistingItemResolution::OUT_CONFIRMED ||
           track->resolution == ExistingItemResolution::OCCLUDED_CONFIRMED;
}

bool existing_track_is_terminal(const OperationTrack& track) {
    return !track.is_suspect_new &&
           (track.resolution == ExistingItemResolution::MOVED_CONFIRMED ||
            track.resolution == ExistingItemResolution::POSITION_REBASED ||
            track.resolution == ExistingItemResolution::OUT_CONFIRMED ||
            track.resolution == ExistingItemResolution::OCCLUDED_CONFIRMED);
}

// D 全程被手挡住时，手离开后的完整 B 可能位于手轨迹中段，而不是最后
// 一个手框旁。因此必须检查整个公共 hand_track，而不是只比较 old_hand_box_。
bool hand_track_touches_detection(const std::vector<BBox>& hand_track,
                                  const Detection& detection) {
    for (size_t i = 0; i < hand_track.size(); ++i) {
        if (hand_touches_detection(hand_track[i], detection.box)) return true;
    }
    return false;
}

BBox move_box(const BBox& box, const MoveValue& delta) {
    return BBox(box.x1 + delta.dx, box.y1 + delta.dy,
                box.x2 + delta.dx, box.y2 + delta.dy);
}

MoveValue total_move(const OperationTrack& track) {
    MoveValue result;
    for (size_t i = 0; i < track.move_values.size(); ++i) {
        result.dx += track.move_values[i].dx;
        result.dy += track.move_values[i].dy;
    }
    return result;
}

BBox estimated_box(const OperationTrack& track) {
    const BBox& anchor = track.has_hand_estimate_anchor_box
        ? track.hand_estimate_anchor_box : track.original_box;
    return move_box(anchor, total_move(track));
}

bool is_active_existing_hand_track(const OperationTrack& track) {
    return !track.is_suspect_new && track.item_id > 0 &&
           (track.state == OperationTrackState::HAND_PARTIAL_BLOCKED ||
            track.state == OperationTrackState::HAND_FULL_BLOCKED);
}

bool is_active_contact_track(const OperationTrack& track) {
    return !track.is_suspect_new && track.item_id > 0 &&
           track.contact_state != ContactState::NONE;
}

bool is_active_runtime_track(const OperationTrack& track) {
    return track.state != OperationTrackState::NORMAL ||
           track.contact_state != ContactState::NONE;
}

bool is_unresolved_operation_start_old_track(const OperationTrack& track) {
    return !track.is_suspect_new && track.item_id > 0 &&
           !existing_track_is_terminal(track) &&
           (is_active_runtime_track(track) ||
            existing_item_needs_settlement(track) ||
            track.reappearance_pending || track.needs_no_hand_settlement);
}

// 保护期只针对已有库存 C；疑似新物品 D 不参加“同类旧 C 的 B 仲裁”。
// claim_grace_remaining 的值在一张有效帧处理完后递减，因此新建于 t0 的
// 轨迹在 t0/t1/t2 仍受保护，t3 开始才成熟。
bool is_claim_protected(const OperationTrack& track) {
    return !track.is_suspect_new && track.item_id > 0 &&
           is_active_runtime_track(track) && track.claim_grace_remaining > 0;
}

bool is_claim_mature(const OperationTrack& track) {
    return !track.is_suspect_new && track.item_id > 0 &&
           is_active_runtime_track(track) && track.claim_grace_remaining <= 0;
}

void seed_reappear_from_tentative_b(OperationTrack* track);

void record_tentative_b(OperationTrack* track, const Detection& detection,
                        bool touching_hand) {
    if (!track || track->is_suspect_new || track->cls_id != detection.cls_id) return;
    if (!track->has_tentative_b_box ||
        !track_match_box(track->cls_id, track->tentative_b_box,
                         detection.cls_id, detection.box)) {
        track->tentative_b_box = detection.box;
        track->has_tentative_b_box = true;
        track->tentative_b_match_count = 1;
        track->tentative_b_started_touching_hand = touching_hand;
        seed_reappear_from_tentative_b(track);
        return;
    }
    track->tentative_b_box = detection.box;
    ++track->tentative_b_match_count;
    track->tentative_b_started_touching_hand =
        track->tentative_b_started_touching_hand || touching_hand;
    seed_reappear_from_tentative_b(track);
}

void seed_reappear_from_tentative_b(OperationTrack* track) {
    if (!track || !track->has_tentative_b_box) return;
    if (!track->has_reappear_candidate_box ||
        !track_match_box(track->cls_id, track->reappear_candidate_box,
                         track->cls_id, track->tentative_b_box)) {
        track->reappear_candidate_box = track->tentative_b_box;
        track->has_reappear_candidate_box = true;
        track->reappear_candidate_match_count =
            track->tentative_b_match_count;
    } else {
        track->reappear_candidate_box = track->tentative_b_box;
        track->reappear_candidate_match_count = std::max(
            track->reappear_candidate_match_count,
            track->tentative_b_match_count);
    }
    track->reappear_candidate_started_touching_hand =
        track->reappear_candidate_started_touching_hand ||
        track->tentative_b_started_touching_hand;
}

bool reappear_candidate_is_confirmed(const OperationTrack& track) {
    return track.has_reappear_candidate_box &&
           track.reappear_candidate_match_count >=
               FLOW3_REAPPEAR_CANDIDATE_CONFIRM_FRAMES;
}

// 这不是普通身份匹配，也不直接更新库存：它只回答“同类 B 是否仍在 C 的
// 完整候选路径附近，值得先暂存为 C 的重新出现候选”。相比正常轨迹匹配略宽，
// 是为了抵抗有效帧间的手框/YOLO 跳动；随后仍有二次自匹配和无手收尾约束。
bool reappear_candidate_path_matches(const OperationTrack& track,
                                     const Detection& observed) {
    if (track.cls_id != observed.cls_id || observed.box.area() <= 0.0f) {
        return false;
    }

    if (track.has_reappear_candidate_box &&
        track_match_box(track.cls_id, track.reappear_candidate_box,
                        observed.cls_id, observed.box)) {
        return true;
    }

    const size_t path_size = track.track.empty() ? 1 : track.track.size();
    for (size_t i = 0; i < path_size; ++i) {
        const BBox reference = track.track.empty() ? estimated_box(track) : track.track[i];
        if (reference.area() <= 0.0f) continue;
        if (track_match_box(track.cls_id, reference,
                            observed.cls_id, observed.box)) {
            return true;
        }
        if (normalized_nearby_distance(reference, observed.box) <=
                FLOW3_REAPPEAR_CANDIDATE_CENTER_NORM &&
            ratio_difference(reference.w(), observed.box.w()) <=
                FLOW3_REAPPEAR_CANDIDATE_WIDTH_RATIO &&
            ratio_difference(reference.h(), observed.box.h()) <=
                FLOW3_REAPPEAR_CANDIDATE_HEIGHT_RATIO) {
            return true;
        }
    }
    return false;
}

void start_reappear_candidate(OperationTrack* track, const Detection& detection,
                              bool started_touching_hand) {
    if (!track) return;
    track->reappear_candidate_box = detection.box;
    track->has_reappear_candidate_box = true;
    track->reappear_candidate_match_count = 1;
    track->drop_evidence_count = 0;
    track->reappearance_pending = false;
    track->reappear_candidate_started_touching_hand = started_touching_hand;
}

// 返回本次 B 是否让候选达到“连续自匹配”门槛。若 B 已不再像原来的
// candidate，则把它当作新的首次候选重新开始；不会把两个 B 混成一个 C。
bool update_reappear_candidate(OperationTrack* track, const Detection& detection,
                               bool started_touching_hand) {
    if (!track) return false;
    if (!track->has_reappear_candidate_box ||
        !track_match_box(track->cls_id, track->reappear_candidate_box,
                         detection.cls_id, detection.box)) {
        start_reappear_candidate(track, detection, started_touching_hand);
        return false;
    }
    ++track->reappear_candidate_match_count;
    track->reappear_candidate_box = detection.box;
    // 首帧已贴手是强上下文；后续帧即使手离开 B，也不能把这个事实抹掉。
    track->reappear_candidate_started_touching_hand =
        track->reappear_candidate_started_touching_hand || started_touching_hand;
    return reappear_candidate_is_confirmed(*track);
}

float complete_box_size_difference(const BBox& complete, const BBox& observed) {
    return 0.5f * (ratio_difference(complete.w(), observed.w()) +
                   ratio_difference(complete.h(), observed.h()));
}

bool becomes_more_like_complete_box(const OperationTrack& track,
                                    const BBox& previous,
                                    const BBox& current) {
    if (track.original_box.area() <= 0.0f) return false;
    const float before = complete_box_size_difference(track.original_box, previous);
    const float after = complete_box_size_difference(track.original_box, current);
    return after + FLOW3_DROP_FULL_BOX_IMPROVEMENT <= before;
}

float move_length(const MoveValue& delta) {
    return std::sqrt(delta.dx * delta.dx + delta.dy * delta.dy);
}

bool has_meaningful_hand_move(const OperationTrack& track) {
    for (size_t i = 0; i < track.move_values.size(); ++i) {
        if (move_length(track.move_values[i]) >= TRACK_HAND_MOVE_EPS) return true;
    }
    return false;
}

// 细节15的 A -> B 不是“手的意图”，而是可观察到的物理前驱关系。这里只
// 接受 A 自己的真实检测位置；手的预测轨迹只能帮助已有 HAND_* 收尾，不能
// 单独把某个邻居的框分配成 B/C。
bool direct_track_has_observed_motion(const OperationTrack& track,
                                      BBox* observed_box) {
    if (track.is_suspect_new || track.indirect_move_candidate ||
        track.item_id <= 0 || track.original_box.area() <= 0.0f) {
        return false;
    }

    // MOVED_CONFIRMED 本身不足以证明它是新的直接推动源：间接成员 B 也可能
    // 因自己的位移达到正式门槛而得到该结论。这里必须保留 HAND/CONTACT 的
    // 直接证据，避免已经确认的 B 在后续未决帧继续向外扩散一条伪推动链。
    const bool direct_motion_evidence =
        (track.contact_state == ContactState::CONTACT_MOVING && track.hold_and_move) ||
        ((track.state == OperationTrackState::HAND_PARTIAL_BLOCKED ||
          track.state == OperationTrackState::HAND_FULL_BLOCKED ||
          track.state == OperationTrackState::PLACED) &&
         (track.hold_and_move || has_meaningful_hand_move(track)));
    if (!direct_motion_evidence) return false;

    BBox result;
    bool has_result = false;
    if (track.has_placed_box) {
        result = track.placed_box;
        has_result = true;
    } else if (!track.observed_track.empty()) {
        result = track.observed_track.back();
        has_result = true;
    } else if (track.has_last_seen_box) {
        result = track.last_seen_box;
        has_result = true;
    }
    if (!has_result || result.area() <= 0.0f ||
        center_distance(track.original_box, result) <= 0.0f) {
        return false;
    }
    if (observed_box) *observed_box = result;
    return true;
}

// 不引入新的像素阈值：两个操作开始时的旧物品只要框相交，或按现有
// CONTACT 路径的尺度化“附近”语义相邻，才可能属于同一次碰撞/推挤链。
bool boxes_are_adjacent_for_indirect_chain(const BBox& a, const BBox& b) {
    return a.area() > 0.0f && b.area() > 0.0f &&
           (intersection_area(a, b) > 0.0f ||
            normalized_nearby_distance(a, b) <= FLOW3_CONTACT_PATH_CENTER_NORM);
}

bool indirect_member_matches_detection(const InventoryItem& original,
                                       const Detection& detection) {
    const BBox reference = original.base_box.area() > 0.0f
        ? original.base_box : original.box;
    if (original.cls_id != detection.cls_id || reference.area() <= 0.0f ||
        detection.box.area() <= 0.0f) {
        return false;
    }
    if (track_match_box(original.cls_id, reference, detection.cls_id, detection.box)) {
        return true;
    }
    return normalized_nearby_distance(reference, detection.box) <=
               FLOW3_REAPPEAR_CANDIDATE_CENTER_NORM &&
           ratio_difference(reference.w(), detection.box.w()) <=
               FLOW3_REAPPEAR_CANDIDATE_WIDTH_RATIO &&
           ratio_difference(reference.h(), detection.box.h()) <=
               FLOW3_REAPPEAR_CANDIDATE_HEIGHT_RATIO;
}

float suspect_d_reappearance_path_cost(const OperationTrack& track,
                                       const Detection& observed);

bool detection_conflicts_with_pending_suspect_d(
        const Detection& detection, const std::map<int, OperationTrack>& tracks) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& suspect = it->second;
        if (!suspect.is_suspect_new || suspect.cls_id != detection.cls_id) continue;
        const BBox reference = suspect.has_placed_box ? suspect.placed_box :
            (suspect.has_last_seen_box ? suspect.last_seen_box : estimated_box(suspect));
        if (strict_match_box(suspect.cls_id, reference,
                             detection.cls_id, detection.box) ||
            partial_match_box(suspect.cls_id, reference,
                              detection.cls_id, detection.box) ||
            track_match_box(suspect.cls_id, reference,
                            detection.cls_id, detection.box) ||
            suspect_d_reappearance_path_cost(suspect, detection) <
                std::numeric_limits<float>::infinity()) {
            return true;
        }
    }
    return false;
}

struct IndirectMoveSource {
    int item_id = -1;
    BBox original_box;
    BBox observed_box;
    float axis_x = 0.0f;
    float axis_y = 0.0f;
};

struct IndirectMoveAssignment {
    int item_id = -1;
    int detection_index = -1;
    int predecessor_item_id = -1;
    int cluster_id = -1;
};

float indirect_projection(const BBox& box, float axis_x, float axis_y) {
    return box.cx() * axis_x + box.cy() * axis_y;
}

InventoryItem make_inventory_item(int item_id, const Detection& detection,
                                  int frame_id, long long time_ms) {
    InventoryItem item;
    item.item_id = item_id;
    item.cls_id = detection.cls_id;
    item.box = detection.box;
    item.base_box = detection.box;
    item.score = detection.score;
    item.status = ItemStatus::VISIBLE;
    item.created_frame = frame_id;
    item.updated_frame = frame_id;
    item.created_time_ms = time_ms;
    return item;
}

void update_seen(InventoryItem& item, const Detection& detection, int frame_id) {
    item.box = detection.box;
    item.score = detection.score;
    item.updated_frame = frame_id;
}

InventoryEvent make_event(EventKind kind, const InventoryItem& item,
                          const BBox& before = BBox(),
                          const BBox& after = BBox()) {
    InventoryEvent event;
    event.kind = kind;
    event.item_id = item.item_id;
    event.cls_id = item.cls_id;
    event.box = kind == EventKind::MOVED ? after : item.box;
    event.before_box = before;
    event.after_box = after;
    event.score = item.score;
    return event;
}

// 以矩形差集计算“若干 cover 的并集是否覆盖 target”，避免把两个交集面积相加。
void subtract_cover(const BBox& piece, const BBox& cover,
                    std::vector<BBox>* output) {
    const float ix1 = std::max(piece.x1, cover.x1);
    const float iy1 = std::max(piece.y1, cover.y1);
    const float ix2 = std::min(piece.x2, cover.x2);
    const float iy2 = std::min(piece.y2, cover.y2);
    if (ix2 <= ix1 || iy2 <= iy1) {
        output->push_back(piece);
        return;
    }
    const BBox pieces[] = {
        BBox(piece.x1, piece.y1, piece.x2, iy1),
        BBox(piece.x1, iy2, piece.x2, piece.y2),
        BBox(piece.x1, iy1, ix1, iy2),
        BBox(ix2, iy1, piece.x2, iy2),
    };
    for (size_t i = 0; i < sizeof(pieces) / sizeof(pieces[0]); ++i) {
        if (pieces[i].area() > COVER_REMAINING_AREA_EPS) output->push_back(pieces[i]);
    }
}

bool fully_covered_by(const BBox& target, const std::vector<BBox>& covers) {
    std::vector<BBox> remaining(1, target);
    for (size_t ci = 0; ci < covers.size(); ++ci) {
        std::vector<BBox> next;
        for (size_t ri = 0; ri < remaining.size(); ++ri) {
            subtract_cover(remaining[ri], covers[ci], &next);
        }
        remaining.swap(next);
    }
    float area = 0.0f;
    for (size_t i = 0; i < remaining.size(); ++i) area += remaining[i].area();
    return area <= COVER_REMAINING_AREA_EPS;
}

int unique_detection_for_box(const std::vector<Detection>& detections,
                             const std::set<int>& claimed,
                             int cls_id, const BBox& reference,
                             bool allow_partial, bool allow_track) {
    int result = -1;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) || detections[i].cls_id != cls_id) continue;
        const bool matched = strict_match_box(cls_id, reference,
                                              detections[i].cls_id, detections[i].box) ||
            (allow_partial && partial_match_box(cls_id, reference,
                                                 detections[i].cls_id, detections[i].box)) ||
            (allow_track && track_match_box(cls_id, reference,
                                             detections[i].cls_id, detections[i].box));
        if (!matched) continue;
        if (result >= 0) return -1;
        result = static_cast<int>(i);
    }
    return result;
}

// 与 unique_detection_for_box 不同：当同类相邻导致候选不止一个时，允许
// “唯一最近”的检测框继续归属原 track。完全相同或代价相同的候选仍返回 -1，
// 保持未决，不用遍历顺序硬分配。
int best_detection_for_box(const std::vector<Detection>& detections,
                           const std::set<int>& claimed,
                           int cls_id, const BBox& reference,
                           bool allow_partial, bool allow_track) {
    float best_cost = std::numeric_limits<float>::infinity();
    int result = -1;
    bool tied = false;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) || detections[i].cls_id != cls_id) continue;
        const Detection& detection = detections[i];
        float cost = std::numeric_limits<float>::infinity();
        if (strict_match_box(cls_id, reference, detection.cls_id, detection.box)) {
            cost = normalized_nearby_distance(reference, detection.box) +
                0.35f * ratio_difference(reference.w(), detection.box.w()) +
                0.35f * ratio_difference(reference.h(), detection.box.h());
        } else if (allow_partial &&
                   partial_match_box(cls_id, reference, detection.cls_id, detection.box)) {
            cost = 1.0f + (1.0f - iom(reference, detection.box));
        } else if (allow_track &&
                   track_match_box(cls_id, reference, detection.cls_id, detection.box)) {
            cost = 2.0f + normalized_nearby_distance(reference, detection.box) +
                0.35f * ratio_difference(reference.w(), detection.box.w()) +
                0.35f * ratio_difference(reference.h(), detection.box.h());
        }
        if (!(cost < std::numeric_limits<float>::infinity())) continue;
        if (cost + 0.0001f < best_cost) {
            best_cost = cost;
            result = static_cast<int>(i);
            tied = false;
        } else if (std::fabs(cost - best_cost) <= 0.0001f) {
            tied = true;
        }
    }
    return tied ? -1 : result;
}

// 现有物品刚进入 HAND_* 时的专用唯一认领。它允许局部框，前提是手确实
// 覆盖该物品或当前检测框；这样不会把远处同类物品吸到旧库存上。
bool matches_hand_affected_reference(int cls_id, const BBox& reference,
                                     const Detection& detection, const BBox& hand) {
    if (detection.cls_id != cls_id) return false;
    // 是否属于“被手影响的旧物品”，由手对旧物品完整参考框的覆盖率决定；
    // 不用当前局部检测框的面积或交集代替，否则新 D 很容易被认成旧物品。
    if (!hand_affects(hand, reference)) return false;
    return strict_match_box(cls_id, reference, detection.cls_id, detection.box) ||
        partial_match_box(cls_id, reference, detection.cls_id, detection.box) ||
        hand_partial_match_box(cls_id, reference, detection.cls_id, detection.box);
}

int unique_hand_affected_detection_for_box(const std::vector<Detection>& detections,
                                           const std::set<int>& claimed,
                                           int cls_id, const BBox& reference,
                                           const BBox& hand) {
    int result = -1;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) || detections[i].cls_id != cls_id) continue;
        if (!matches_hand_affected_reference(cls_id, reference, detections[i], hand)) continue;
        if (result >= 0) return -1;
        result = static_cast<int>(i);
    }
    return result;
}

// 当手同时影响同一旧物品附近的多个同类框时，不能简单返回“不唯一”并
// 放弃：例如旧苹果完整框和新苹果框都在手边。这里仅在代价存在唯一最小值
// 时选它；随后调用方仍会检查该检测框不会同样合理地属于另一个旧 item。
int best_hand_affected_detection_for_box(const std::vector<Detection>& detections,
                                         const std::set<int>& claimed,
                                         int cls_id, const BBox& reference,
                                         const BBox& hand) {
    float best_cost = std::numeric_limits<float>::infinity();
    int result = -1;
    bool tied = false;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) || detections[i].cls_id != cls_id) continue;
        if (!matches_hand_affected_reference(cls_id, reference, detections[i], hand)) continue;

        float cost = 0.0f;
        if (strict_match_box(cls_id, reference, detections[i].cls_id, detections[i].box)) {
            cost = normalized_nearby_distance(reference, detections[i].box);
        } else if (partial_match_box(cls_id, reference, detections[i].cls_id,
                                     detections[i].box)) {
            cost = 1.0f + (1.0f - iom(reference, detections[i].box));
        } else {
            const float observed_cover = intersection_area(reference, detections[i].box) /
                std::max(detections[i].box.area(), 1.0f);
            cost = 2.0f + normalized_nearby_distance(reference, detections[i].box) +
                (1.0f - observed_cover);
        }
        if (cost + 0.0001f < best_cost) {
            best_cost = cost;
            result = static_cast<int>(i);
            tied = false;
        } else if (std::fabs(cost - best_cost) <= 0.0001f) {
            tied = true;
        }
    }
    return tied ? -1 : result;
}

int hand_affected_existing_candidate_count(
        const std::map<int, InventoryItem>& working, const Detection& detection,
        const BBox& hand) {
    int count = 0;
    for (std::map<int, InventoryItem>::const_iterator it = working.begin();
         it != working.end(); ++it) {
        if (it->second.status == ItemStatus::OCCLUDED) continue;
        const BBox reference = it->second.base_box.area() > 0.0f
            ? it->second.base_box : it->second.box;
        if (!matches_hand_affected_reference(it->second.cls_id, reference,
                                              detection, hand)) {
            continue;
        }
        ++count;
        if (count > 1) return count;
    }
    return count;
}

int unique_detection_at_old_position(const std::vector<Detection>& detections,
                                     const std::set<int>& claimed,
                                     const OperationTrack& track) {
    int result = -1;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) || detections[i].cls_id != track.cls_id) continue;
        if (intersection_area(track.original_box, detections[i].box) <
            FLOW3_OLD_POSITION_OVERLAP_AREA) {
            continue;
        }
        if (!partial_match_box(track.cls_id, track.original_box,
                               detections[i].cls_id, detections[i].box,
                               FLOW3_TRACK_PARTIAL_IOM)) {
            continue;
        }
        if (result >= 0) return -1;
        result = static_cast<int>(i);
    }
    return result;
}

bool any_detection_at_old_position(const std::vector<Detection>& detections,
                                   const OperationTrack& track) {
    for (size_t i = 0; i < detections.size(); ++i) {
        if (intersection_area(track.original_box, detections[i].box) >=
            FLOW3_OLD_POSITION_OVERLAP_AREA) {
            return true;
        }
    }
    return false;
}

bool old_position_is_clean(const std::vector<Detection>& detections,
                           const OperationTrack& track,
                           const std::map<int, InventoryItem>& working) {
    for (size_t i = 0; i < detections.size(); ++i) {
        const Detection& d = detections[i];
        if (intersection_area(track.original_box, d.box) < FLOW3_OLD_POSITION_OVERLAP_AREA) {
            continue;
        }
        if (d.cls_id == track.cls_id &&
            partial_match_box(track.cls_id, track.original_box, d.cls_id, d.box,
                              FLOW3_TRACK_PARTIAL_IOM)) {
            return false;
        }
        bool known_other = false;
        for (std::map<int, InventoryItem>::const_iterator it = working.begin();
             it != working.end(); ++it) {
            if (it->first == track.item_id) continue;
            if (partial_match(it->second, d)) {
                known_other = true;
                break;
            }
        }
        if (!known_other) return false;
    }
    return true;
}

bool detection_strictly_matches_other_item(const Detection& detection,
                                           int excluded_item_id,
                                           const std::map<int, InventoryItem>& working) {
    for (std::map<int, InventoryItem>::const_iterator it = working.begin();
         it != working.end(); ++it) {
        if (it->first == excluded_item_id) continue;
        if (strict_match(it->second, detection)) return true;
    }
    return false;
}

// working_inventory_ 中既可能有操作开始前的旧库存，也可能有尚未提交的
// staged D。无手收尾不能再把后者打印或处理成 "other old item"。
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

const char* strict_owner_kind_name(StrictOwnerKind kind) {
    switch (kind) {
        case StrictOwnerKind::NONE:
            return "none";
        case StrictOwnerKind::START_OLD_C:
            return "start-old-c";
        case StrictOwnerKind::PENDING_D:
            return "pending-d";
        case StrictOwnerKind::QUARANTINED_PENDING_D:
            return "quarantined-pending-d";
        case StrictOwnerKind::CONFIRMED_D:
            return "confirmed-d";
    }
    return "unknown";
}

const OperationTrack* runtime_for_working_item(
        int item_id, const std::map<int, OperationTrack>& tracks,
        int* runtime_key) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (it->second.item_id == item_id) {
            if (runtime_key) *runtime_key = it->first;
            return &it->second;
        }
    }
    return nullptr;
}

StrictDetectionOwner strict_owner_for_detection(
        const Detection& detection, int excluded_item_id,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks) {
    // 操作开始库存优先：细节10的独立静止旧 B 规则依赖其严格所有权。
    for (std::map<int, InventoryItem>::const_iterator it = working.begin();
         it != working.end(); ++it) {
        if (it->first == excluded_item_id || !strict_match(it->second, detection)) {
            continue;
        }
        if (operation_start.count(it->first)) {
            StrictDetectionOwner owner;
            owner.kind = StrictOwnerKind::START_OLD_C;
            owner.item_id = it->first;
            owner.runtime_key = it->first;
            return owner;
        }
    }

    for (std::map<int, InventoryItem>::const_iterator it = working.begin();
         it != working.end(); ++it) {
        if (it->first == excluded_item_id || !strict_match(it->second, detection)) {
            continue;
        }
        StrictDetectionOwner owner;
        owner.item_id = it->first;
        int runtime_key = 0;
        const OperationTrack* runtime = runtime_for_working_item(
            it->first, tracks, &runtime_key);
        owner.runtime_key = runtime_key;
        if ((runtime && runtime->is_suspect_new &&
             runtime->pending_d_quarantined_by_old_c) ||
            (pending_in_ids.count(it->first) && runtime &&
             runtime->pending_d_quarantined_by_old_c)) {
            owner.kind = StrictOwnerKind::QUARANTINED_PENDING_D;
        } else if ((runtime && runtime->is_suspect_new) ||
                   pending_in_ids.count(it->first)) {
            owner.kind = StrictOwnerKind::PENDING_D;
        } else {
            owner.kind = StrictOwnerKind::CONFIRMED_D;
        }
        return owner;
    }

    // 隔离 D 在有手阶段不会进入 working_inventory_，仍须能作为实时
    // owner 被日志识别，但它没有排除旧 C 的正式权限。
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& track = it->second;
        if (!track.is_suspect_new || !track.pending_d_quarantined_by_old_c ||
            track.cls_id != detection.cls_id) {
            continue;
        }
        const BBox reference = track.has_last_seen_box ? track.last_seen_box : estimated_box(track);
        if (!strict_match_box(track.cls_id, reference,
                              detection.cls_id, detection.box)) {
            continue;
        }
        StrictDetectionOwner owner;
        owner.kind = StrictOwnerKind::QUARANTINED_PENDING_D;
        owner.item_id = track.item_id;
        owner.runtime_key = it->first;
        return owner;
    }
    return StrictDetectionOwner();
}

bool strict_owner_blocks_old_c(StrictOwnerKind kind) {
    return kind == StrictOwnerKind::START_OLD_C ||
           kind == StrictOwnerKind::CONFIRMED_D;
}

bool strict_owner_is_quarantined_alias_of_old_c(
        const StrictDetectionOwner& owner, int old_item_id,
        const std::map<int, OperationTrack>& tracks) {
    if (owner.kind != StrictOwnerKind::QUARANTINED_PENDING_D) return false;
    std::map<int, OperationTrack>::const_iterator suspect = tracks.find(owner.runtime_key);
    return suspect != tracks.end() &&
           suspect->second.conflicting_old_item_ids.count(old_item_id) > 0;
}

// CONTACT_* 只使用物品自己的真实观测路径。手框位移不能替代这里的 BBox
// 位移，因为手腕不动时手指仍可能推动物品。
float contact_reference_cost(const OperationTrack& track,
                             const BBox& reference,
                             const Detection& observed) {
    if (reference.area() <= 0.0f || observed.box.area() <= 0.0f ||
        track.cls_id != observed.cls_id ||
        ratio_difference(reference.w(), observed.box.w()) >
            FLOW3_CONTACT_WIDTH_RATIO ||
        ratio_difference(reference.h(), observed.box.h()) >
            FLOW3_CONTACT_HEIGHT_RATIO) {
        return std::numeric_limits<float>::infinity();
    }
    const float center = normalized_nearby_distance(reference, observed.box);
    if (center > FLOW3_CONTACT_PATH_CENTER_NORM &&
        !track_match_box(track.cls_id, reference, observed.cls_id, observed.box)) {
        return std::numeric_limits<float>::infinity();
    }
    return center + 0.35f * ratio_difference(reference.w(), observed.box.w()) +
        0.35f * ratio_difference(reference.h(), observed.box.h());
}

float contact_path_match_cost(const OperationTrack& track,
                              const Detection& observed) {
    // C 一旦从 CONTACT_* 过渡成 HAND_*，此前真实看见过的物品路径仍然
    // 有价值，不能因为状态转换而丢掉。只有既不是 CONTACT_*、也没有任何
    // 真实观测时，才说明它不存在可用的 contact 路径。
    if ((track.contact_state == ContactState::NONE && track.observed_track.empty()) ||
        track.cls_id != observed.cls_id || observed.box.area() <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }

    float best = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < track.observed_track.size(); ++i) {
        best = std::min(best, contact_reference_cost(track,
                                                      track.observed_track[i], observed));
    }
    if (track.has_last_seen_box) {
        best = std::min(best, contact_reference_cost(track, track.last_seen_box, observed));
    }
    if (track.original_box.area() > 0.0f) {
        best = std::min(best, contact_reference_cost(track, track.original_box, observed));
    }
    return best;
}

void append_contact_observation(OperationTrack* track, const Detection& detection,
                                bool touching_hand) {
    if (!track) return;
    if (track->has_last_seen_box) {
        MoveValue delta;
        delta.dx = detection.box.cx() - track->last_seen_box.cx();
        delta.dy = detection.box.cy() - track->last_seen_box.cy();
        track->observed_move_values.push_back(delta);
    }
    track->observed_track.push_back(detection.box);
    track->last_seen_box = detection.box;
    track->has_last_seen_box = true;
    track->contact_path_ambiguous = false;
    if (touching_hand) {
        track->last_hand_block_box = detection.box;
        track->has_last_hand_block_box = true;
        track->contact_started_touching_hand = true;
    }
}

bool contact_detection_is_at_original(const OperationTrack& track,
                                      const Detection& detection) {
    if (track.original_box.area() <= 0.0f ||
        detection.cls_id != track.cls_id) {
        return false;
    }
    // CONTACT_* 不能使用 HAND_* 的宽松 IoM 单独认定“仍在原位”：物品
    // 已经移动一小段时仍可能与原框重叠超过 0.5。先要求中心移动不超过
    // 物品抖动阈值，再接受严格/局部的形状匹配。
    if (center_distance(track.original_box, detection.box) >
        FLOW3_CONTACT_OBJECT_MOVE_EPS) {
        return false;
    }
    return intersection_area(track.original_box, detection.box) >=
               FLOW3_OLD_POSITION_OVERLAP_AREA &&
        (strict_match_box(track.cls_id, track.original_box,
                          detection.cls_id, detection.box) ||
         partial_match_box(track.cls_id, track.original_box,
                           detection.cls_id, detection.box,
                           FLOW3_TRACK_PARTIAL_IOM));
}

int unique_contact_original_detection(const std::vector<Detection>& detections,
                                      const std::set<int>& claimed,
                                      const OperationTrack& track,
                                      const std::map<int, InventoryItem>& working) {
    int result = -1;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) ||
            detection_strictly_matches_other_item(detections[i], track.item_id, working) ||
            !contact_detection_is_at_original(track, detections[i])) {
            continue;
        }
        if (result >= 0) return -1;
        result = static_cast<int>(i);
    }
    return result;
}

// 为一个 CONTACT_* C 在当前帧找唯一的真实 B。候选必须不严格属于其他库存，
// 且不能同时落在另一条活动 CONTACT_* 轨迹上；否则保持未决，不能按遍历顺序抢框。
int unique_contact_detection_for_track(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        const OperationTrack& track, const std::map<int, InventoryItem>& working,
        const std::map<int, OperationTrack>& tracks) {
    float best_cost = std::numeric_limits<float>::infinity();
    int best_index = -1;
    bool tied = false;
    for (size_t di = 0; di < detections.size(); ++di) {
        const int detection_index = static_cast<int>(di);
        const Detection& detection = detections[di];
        if (claimed.count(detection_index) ||
            detection.cls_id != track.cls_id ||
            detection_strictly_matches_other_item(detection, track.item_id, working)) {
            continue;
        }
        const float cost = contact_path_match_cost(track, detection);
        if (!(cost < std::numeric_limits<float>::infinity())) continue;

        bool competing_contact = false;
        for (std::map<int, OperationTrack>::const_iterator other = tracks.begin();
             other != tracks.end(); ++other) {
            const OperationTrack& other_track = other->second;
            if (other_track.item_id <= 0 || other_track.item_id == track.item_id ||
                other_track.is_suspect_new ||
                other_track.contact_state == ContactState::NONE ||
                !is_claim_mature(other_track)) {
                continue;
            }
            if (contact_path_match_cost(other_track, detection) <
                std::numeric_limits<float>::infinity()) {
                competing_contact = true;
                break;
            }
        }
        if (competing_contact) continue;

        if (cost + 0.0001f < best_cost) {
            best_cost = cost;
            best_index = detection_index;
            tied = false;
        } else if (std::fabs(cost - best_cost) <= 0.0001f) {
            tied = true;
        }
    }
    return tied ? -1 : best_index;
}

bool has_contact_path_candidate(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        const OperationTrack& track, const std::map<int, InventoryItem>& working) {
    for (size_t di = 0; di < detections.size(); ++di) {
        if (claimed.count(static_cast<int>(di)) ||
            detection_strictly_matches_other_item(detections[di], track.item_id, working)) {
            continue;
        }
        if (contact_path_match_cost(track, detections[di]) <
            std::numeric_limits<float>::infinity()) {
            return true;
        }
    }
    return false;
}

// 一张未认领的同类贴手 B，只能在“恰好一个没有自己检测框的 HAND_* C”
// 的完整候选轨迹附近时，先暂存为该 C 的重新出现候选。返回 -2 表示多个 C
// 都同样合理；调用方必须保持未决，不能把 B 改登记为 D。
int unique_c_reappear_owner_for_detection(
        const Detection& detection,
        const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& known_item_owner) {
    int owner = -1;
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& c = it->second;
        if (!is_active_existing_hand_track(c) || !is_claim_mature(c) ||
            c.cls_id != detection.cls_id ||
            known_item_owner.count(c.item_id) ||
            !reappear_candidate_path_matches(c, detection)) {
            continue;
        }
        if (owner >= 0) return -2;
        owner = it->first;
    }
    return owner;
}

void mark_mature_hand_b_ambiguity(
        const Detection& detection, std::map<int, OperationTrack>* tracks,
        const std::map<int, int>& known_item_owner) {
    if (!tracks) return;
    for (std::map<int, OperationTrack>::iterator it = tracks->begin();
         it != tracks->end(); ++it) {
        OperationTrack& c = it->second;
        if (!is_active_existing_hand_track(c) || !is_claim_mature(c) ||
            c.cls_id != detection.cls_id || known_item_owner.count(c.item_id) ||
            !reappear_candidate_path_matches(c, detection)) {
            continue;
        }
        c.b_claim_ambiguous = true;
    }
}

float suspect_d_reappearance_path_cost(const OperationTrack& track,
                                       const Detection& observed);

// 无手首帧可能才第一次看见 B，因此没有“贴手”这一强证据。这里只允许
// 唯一 C 在完整候选路径附近认领它，并排除严格属于另一件旧库存的框；后续仍
// 必须通过 candidate 自匹配和后续无手直接帧，不能在这一帧直接提交移动。
int unique_no_hand_reappear_detection_for_track(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        int runtime_key, const OperationTrack& track,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks) {
    int result = -1;
    const std::map<int, int> no_known_owner;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) ||
            !reappear_candidate_path_matches(track, detections[i])) {
            continue;
        }
        const StrictDetectionOwner strict_owner = strict_owner_for_detection(
            detections[i], track.item_id, working, operation_start,
            pending_in_ids, tracks);
        if (strict_owner_blocks_old_c(strict_owner.kind)) continue;
        bool belongs_to_existing_d = false;
        for (std::map<int, OperationTrack>::const_iterator other = tracks.begin();
             other != tracks.end(); ++other) {
            if (!other->second.is_suspect_new ||
                other->second.state == OperationTrackState::NORMAL ||
                other->second.cls_id != detections[i].cls_id) {
                continue;
            }
            const OperationTrack& d = other->second;
            // C-D alias 是同一个真实物品的暂定双轨迹。它不能反过来让 C
            // 失去唯一的无手候选；C 仍要先拿到直接框，随后再等第二帧仲裁。
            if (d.pending_d_quarantined_by_old_c &&
                d.conflicting_old_item_ids.count(track.item_id)) {
                continue;
            }
            const BBox reference = d.has_placed_box ? d.placed_box : estimated_box(d);
            if (track_match_box(d.cls_id, reference, detections[i].cls_id,
                                detections[i].box) ||
                (d.has_last_seen_box && track_match_box(
                    d.cls_id, d.last_seen_box, detections[i].cls_id, detections[i].box)) ||
                (d.has_last_hand_block_box && track_match_box(
                    d.cls_id, d.last_hand_block_box, detections[i].cls_id,
                    detections[i].box)) ||
                suspect_d_reappearance_path_cost(d, detections[i]) <
                    std::numeric_limits<float>::infinity()) {
                belongs_to_existing_d = true;
                break;
            }
        }
        if (belongs_to_existing_d) continue;
        if (unique_c_reappear_owner_for_detection(detections[i], tracks,
                                                   no_known_owner) != runtime_key) {
            continue;
        }
        if (result >= 0) return -1;
        result = static_cast<int>(i);
    }
    return result;
}

// 与 unique_no_hand_reappear_detection_for_track 配套：它不尝试把歧义框
// 分配给 C，只回答“本帧是否存在本可接上 C 路径、但因同类冲突无法唯一
// 归属的框”。这类框不能作为 C 已缺失的 OUT 证据。
bool has_ambiguous_no_hand_reappear_candidate(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        int runtime_key, const OperationTrack& track,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& independent_static_owner_by_detection) {
    if (!is_claim_mature(track)) return false;
    const std::map<int, int> no_known_owner;
    int viable_count = 0;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) ||
            !reappear_candidate_path_matches(track, detections[i])) {
            continue;
        }
        // 已经在本帧通过双方唯一的严格原位关系保留给另一旧 C 的框，对当前
        // C 只表示“不是我”。它不能再次被解释成路径歧义并永久阻止回退搜索。
        std::map<int, int>::const_iterator static_owner =
            independent_static_owner_by_detection.find(static_cast<int>(i));
        if (static_owner != independent_static_owner_by_detection.end() &&
            static_owner->second != track.item_id) {
            continue;
        }
        const StrictDetectionOwner strict_owner = strict_owner_for_detection(
            detections[i], track.item_id, working, operation_start,
            pending_in_ids, tracks);
        if (strict_owner_blocks_old_c(strict_owner.kind)) {
            return true;
        }
        // Pending D 不是 "other old item"，但在没有可把它归给 C 的直接
        // 路径时仍是需要继续观察的实时候选，不能据此累计 C 的 OUT。
        if (strict_owner.kind == StrictOwnerKind::PENDING_D ||
            strict_owner.kind == StrictOwnerKind::QUARANTINED_PENDING_D) {
            return true;
        }
        const int owner = unique_c_reappear_owner_for_detection(
            detections[i], tracks, no_known_owner);
        if (owner != runtime_key) {
            if (owner >= 0 || owner == -2) return true;
            continue;
        }
        ++viable_count;
        if (viable_count > 1) return true;
    }
    return false;
}

// HAND_FULL + hold_and_move=False 时，只有“已确认”的其他库存/已经放下的 D
// 才能解释 C 仍在旧位置被遮挡。未提升或未放下的 D 仍是模糊帧，不能拿它
// 增加 not_hold 证据。
bool confirmed_blocker_covers_old_c(
        const OperationTrack& c, const std::vector<Detection>& detections,
        const std::map<int, int>& known_item_owner,
        const std::map<int, OperationTrack>& tracks) {
    for (std::map<int, int>::const_iterator owner = known_item_owner.begin();
         owner != known_item_owner.end(); ++owner) {
        if (owner->first == c.item_id || owner->second < 0 ||
            owner->second >= static_cast<int>(detections.size())) {
            continue;
        }
        bool is_unconfirmed_d = false;
        for (std::map<int, OperationTrack>::const_iterator runtime = tracks.begin();
             runtime != tracks.end(); ++runtime) {
            if (runtime->second.item_id == owner->first &&
                runtime->second.is_suspect_new && !runtime->second.drop_confirmed) {
                is_unconfirmed_d = true;
                break;
            }
        }
        if (is_unconfirmed_d) continue;
        if (cover_ratio(c.original_box, detections[owner->second].box) >=
            FLOW3_D_PARTIAL_COVER_RATIO) {
            return true;
        }
    }

    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& other = it->second;
        if (other.item_id <= 0 || other.item_id == c.item_id ||
            (!other.drop_confirmed && other.state != OperationTrackState::PLACED)) {
            continue;
        }
        const BBox box = other.has_placed_box ? other.placed_box : other.last_seen_box;
        if (box.area() > 0.0f &&
            cover_ratio(c.original_box, box) >= FLOW3_D_PARTIAL_COVER_RATIO) {
            return true;
        }
    }
    return false;
}

// 有手阶段的第二种 D 来源：旧 C 已进入 HAND_*，当前又没有自己的检测框，
// 但一个未认领的 D 覆盖了 C 的原位置。这里返回唯一的 C；多个 C 都合理时
// 保持未决，不能按 map 顺序挑一个。
int unique_c_replacement_owner_for_detection(
        const Detection& detection,
        const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& known_item_owner) {
    int owner = -1;
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& c = it->second;
        if (c.is_suspect_new || c.item_id <= 0 ||
            (c.state != OperationTrackState::HAND_PARTIAL_BLOCKED &&
             c.state != OperationTrackState::HAND_FULL_BLOCKED) ||
            // 细节7的保护期只解决“同类 B 可能属于新建 C”的仲裁。不同
            // 类别的新 D 覆盖 C 旧位置仍可立即走 replacement D 链路。
            (is_claim_protected(c) && c.cls_id == detection.cls_id) ||
            known_item_owner.count(c.item_id) || c.original_box.area() <= 0.0f) {
            continue;
        }
        if (cover_ratio(c.original_box, detection.box) <
            FLOW3_C_REPLACEMENT_MIN_COVER_RATIO) {
            continue;
        }
        if (owner >= 0) return -1;
        owner = c.item_id;
    }
    return owner;
}

// D 的初始框可能只是被手露出的一小条，手离开后第一次看到的 B 却是完整框。
// 普通单点轨迹匹配要求尺寸大致相近，无法覆盖“局部 D → 完整 B”；这里沿整条
// D 路径找最合理的重现位置。正常严格/局部/轨迹匹配优先，局部→完整只作为后备。
float suspect_d_reappearance_path_cost(const OperationTrack& track,
                                       const Detection& observed) {
    if (!track.is_suspect_new || track.cls_id != observed.cls_id ||
        observed.box.area() <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }

    const size_t path_size = track.track.empty() ? 1 : track.track.size();
    float best_cost = std::numeric_limits<float>::infinity();
    for (size_t pi = 0; pi < path_size; ++pi) {
        const BBox reference = track.track.empty() ? estimated_box(track) : track.track[pi];
        if (reference.area() <= 0.0f) continue;

        float cost = std::numeric_limits<float>::infinity();
        if (strict_match_box(track.cls_id, reference, observed.cls_id, observed.box)) {
            cost = normalized_center_shift(reference, observed.box);
        } else if (track_match_box(track.cls_id, reference,
                                   observed.cls_id, observed.box)) {
            cost = 1.0f + normalized_center_shift(reference, observed.box);
        } else if (observed.box.area() >= reference.area()) {
            // 完整 B 未必和预测局部框重合到普通 IoM 阈值；允许它覆盖局部框的一部分，
            // 或在更大的完整框尺度内保持足够接近。两个条件都只用于 D，不用于 C。
            const float partial_cover = cover_ratio(reference, observed.box);
            const float center_shift = normalized_center_shift(reference, observed.box);
            if (partial_cover >= FLOW3_D_REAPPEAR_MIN_PARTIAL_COVER ||
                center_shift <= FLOW3_D_REAPPEAR_MAX_CENTER_SHIFT_NORM) {
                cost = 2.0f + (1.0f - partial_cover) + 0.25f * center_shift;
            }
        }
        best_cost = std::min(best_cost, cost);
    }
    return best_cost;
}

bool quarantined_suspect_matches_detection(const OperationTrack& suspect,
                                           const Detection& detection) {
    if (!suspect.is_suspect_new || !suspect.pending_d_quarantined_by_old_c ||
        suspect.cls_id != detection.cls_id) {
        return false;
    }
    const BBox reference = suspect.has_placed_box ? suspect.placed_box : estimated_box(suspect);
    return strict_match_box(suspect.cls_id, reference,
                            detection.cls_id, detection.box) ||
           (suspect.has_last_seen_box && strict_match_box(
               suspect.cls_id, suspect.last_seen_box,
               detection.cls_id, detection.box)) ||
           (suspect.has_last_hand_block_box && strict_match_box(
               suspect.cls_id, suspect.last_hand_block_box,
               detection.cls_id, detection.box)) ||
           suspect_d_reappearance_path_cost(suspect, detection) <
               std::numeric_limits<float>::infinity();
}

bool old_c_has_independently_settled_identity(const OperationTrack& old) {
    return !old.is_suspect_new &&
           (existing_track_is_terminal(old) ||
            (old.resolution == ExistingItemResolution::STATIC_CONFIRMED &&
             !old.needs_no_hand_settlement &&
             old.state == OperationTrackState::NORMAL &&
             old.contact_state == ContactState::NONE));
}

// 细节12：只有 D 关联的每个 operation-start old C 都已由自己的既有无手
// 证据链结算，D 的连续无直接证据才可以作为“过期 alias”的证据。找不到
// 任一 C 或 C 仍未结算都必须继续等待，不能凭 D 的缺失删除真实物品。
bool all_conflicting_old_c_independently_settled(
        const OperationTrack& suspect,
        const std::map<int, OperationTrack>& tracks) {
    if (!suspect.is_suspect_new || !suspect.pending_d_quarantined_by_old_c ||
        suspect.conflicting_old_item_ids.empty()) {
        return false;
    }
    for (std::set<int>::const_iterator old_id = suspect.conflicting_old_item_ids.begin();
         old_id != suspect.conflicting_old_item_ids.end(); ++old_id) {
        std::map<int, OperationTrack>::const_iterator old = tracks.find(*old_id);
        if (old == tracks.end() ||
            !old_c_has_independently_settled_identity(old->second)) {
            return false;
        }
    }
    return true;
}

bool detection_is_duplicate_of_settled_old_c(const OperationTrack& old,
                                              int detection_index,
                                              const Detection& detection) {
    if (!old_c_has_independently_settled_identity(old) ||
        old.cls_id != detection.cls_id) {
        return false;
    }

    // 同一张无手帧已经由旧 C 直接认领的 detection index 是最强的一框一物品
    // 证据。不能再用普通 strict_match 把相邻的同类真实 D 误判成这一个 C。
    if (old.no_hand_detection_index >= 0) {
        return old.no_hand_detection_index == detection_index;
    }

    // 旧 C 是前一帧已经结算、因而本帧没有重新参与认领时，只把近乎同一框的
    // 观察当作 duplicate 的后备证据。partial 的高 IoM 语义比 strict 更窄，
    // 不会吞掉“相邻但略有重叠”的真实同类 D。
    const BBox reference = old.has_placed_box ? old.placed_box :
        (old.has_last_seen_box ? old.last_seen_box : old.original_box);
    return partial_match_box(old.cls_id, reference, detection.cls_id, detection.box);
}

bool quarantined_suspect_detection_is_duplicate_of_settled_old_c(
        const OperationTrack& suspect,
        const std::map<int, OperationTrack>& tracks,
        int suspect_detection_index,
        const Detection& detection) {
    for (std::set<int>::const_iterator old_id = suspect.conflicting_old_item_ids.begin();
         old_id != suspect.conflicting_old_item_ids.end(); ++old_id) {
        std::map<int, OperationTrack>::const_iterator old = tracks.find(*old_id);
        if (old != tracks.end() &&
            detection_is_duplicate_of_settled_old_c(
                old->second, suspect_detection_index, detection)) {
            return true;
        }
    }
    return false;
}

bool quarantined_suspect_has_distinct_old_c_detections(
        const OperationTrack& suspect,
        const std::map<int, OperationTrack>& tracks,
        int suspect_detection_index,
        const Detection& suspect_detection) {
    if (!suspect.pending_d_quarantined_by_old_c ||
        suspect.conflicting_old_item_ids.empty()) {
        return false;
    }
    for (std::set<int>::const_iterator old_id = suspect.conflicting_old_item_ids.begin();
         old_id != suspect.conflicting_old_item_ids.end(); ++old_id) {
        std::map<int, OperationTrack>::const_iterator old = tracks.find(*old_id);
        if (old == tracks.end()) return false;
        if (old_c_has_independently_settled_identity(old->second)) {
            // 已独立结算的 C 只在当前 D 框与 C 的结算框不同的前提下，
            // 才能作为“C、D 各有一个框”的证据。否则仍是一框两身份。
            if (detection_is_duplicate_of_settled_old_c(
                    old->second, suspect_detection_index, suspect_detection)) {
                return false;
            }
            continue;
        }
        if (old->second.no_hand_detection_index < 0 ||
            old->second.no_hand_detection_index == suspect_detection_index) {
            return false;
        }
    }
    return true;
}

// 只在 D 原有的单点匹配全部失败后调用。B 必须不严格属于已有库存，且在
// D 候选路径和多个 D 候选之间都有唯一更优的来源；否则保持未决，不能强认领。
int unique_d_reappearance_detection_for_track(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        int runtime_key, const OperationTrack& track,
        const std::map<int, InventoryItem>& working,
        const std::map<int, OperationTrack>& tracks) {
    const float tie_epsilon = 0.0001f;
    float best_cost = std::numeric_limits<float>::infinity();
    int best_detection = -1;
    bool tied_detection = false;

    for (size_t di = 0; di < detections.size(); ++di) {
        if (claimed.count(static_cast<int>(di))) continue;
        const Detection& detection = detections[di];
        // 未提升 D 要排除全部库存；已提升 D 只允许排除“其他”库存物品。
        if (detection_strictly_matches_other_item(detection, track.item_id, working)) {
            continue;
        }
        const float cost = suspect_d_reappearance_path_cost(track, detection);
        if (!(cost < std::numeric_limits<float>::infinity())) continue;
        if (cost + tie_epsilon < best_cost) {
            best_cost = cost;
            best_detection = static_cast<int>(di);
            tied_detection = false;
        } else if (std::fabs(cost - best_cost) <= tie_epsilon) {
            tied_detection = true;
        }
    }
    if (best_detection < 0 || tied_detection) return -1;

    // 即使当前 D 找到了唯一最近的 B，若另一个疑似 D 对同一个 B 更合理或同样
    // 合理，也不能按 track_buffer_ 的遍历顺序抢占这个检测框。
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (it->first == runtime_key || !it->second.is_suspect_new ||
            it->second.state == OperationTrackState::NORMAL) {
            continue;
        }
        const float other_cost = suspect_d_reappearance_path_cost(
            it->second, detections[best_detection]);
        if (other_cost <= best_cost + tie_epsilon) return -1;
    }
    return best_detection;
}

// 手离开后首次完整出现的 B 不能抢占任何已有 C/D。这里不是正式库存的
// 最终绑定，而是一个保守的“是否仍可能属于旧对象”检查：只要任一活动轨迹
// 能合理解释 B，就不把 B 另建成 POST_HAND_REVEAL_D。
bool detection_can_belong_to_active_track(const Detection& detection,
                                          const OperationTrack& track) {
    if ((!is_active_runtime_track(track) && !track.indirect_move_candidate) ||
        track.cls_id != detection.cls_id) {
        return false;
    }
    if (track.indirect_move_candidate && track.has_indirect_candidate_box &&
        track_match_box(track.cls_id, track.indirect_candidate_box,
                        detection.cls_id, detection.box)) {
        return true;
    }
    // 即便 CONTACT_* 已升级为 HAND_*，也优先尊重此前由真实检测框建立的
    // observed_track；这能避免低覆盖率推/拉后把同一个 B 又预登记成 D。
    if (!track.observed_track.empty() &&
        contact_path_match_cost(track, detection) <
            std::numeric_limits<float>::infinity()) {
        return true;
    }
    const BBox reference = track.has_placed_box ? track.placed_box : estimated_box(track);
    if (strict_match_box(track.cls_id, reference, detection.cls_id, detection.box) ||
        partial_match_box(track.cls_id, reference, detection.cls_id, detection.box) ||
        track_match_box(track.cls_id, reference, detection.cls_id, detection.box)) {
        return true;
    }
    if (track.original_box.area() > 0.0f &&
        (strict_match_box(track.cls_id, track.original_box,
                          detection.cls_id, detection.box) ||
         partial_match_box(track.cls_id, track.original_box,
                           detection.cls_id, detection.box) ||
         track_match_box(track.cls_id, track.original_box,
                         detection.cls_id, detection.box))) {
        return true;
    }
    if (track.has_last_seen_box &&
        track_match_box(track.cls_id, track.last_seen_box,
                        detection.cls_id, detection.box)) {
        return true;
    }
    if (track.has_last_hand_block_box &&
        track_match_box(track.cls_id, track.last_hand_block_box,
                        detection.cls_id, detection.box)) {
        return true;
    }
    if (track.has_reappear_candidate_box &&
        track_match_box(track.cls_id, track.reappear_candidate_box,
                        detection.cls_id, detection.box)) {
        return true;
    }
    for (size_t i = 0; i < track.track.size(); ++i) {
        if (track_match_box(track.cls_id, track.track[i],
                            detection.cls_id, detection.box)) {
            return true;
        }
    }
    if (!track.is_suspect_new &&
        (track.reappearance_pending || track.has_reappear_candidate_box) &&
        reappear_candidate_path_matches(track, detection)) {
        return true;
    }
    return track.is_suspect_new &&
        suspect_d_reappearance_path_cost(track, detection) <
            std::numeric_limits<float>::infinity();
}

bool detection_matches_old_working_inventory(
        const Detection& detection,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start) {
    for (std::map<int, InventoryItem>::const_iterator original =
             operation_start.begin(); original != operation_start.end(); ++original) {
        std::map<int, InventoryItem>::const_iterator current =
            working.find(original->first);
        if (current == working.end()) continue;
        const BBox reference = current->second.base_box.area() > 0.0f
            ? current->second.base_box : current->second.box;
        if (strict_match_box(current->second.cls_id, reference,
                             detection.cls_id, detection.box) ||
            partial_match_box(current->second.cls_id, reference,
                              detection.cls_id, detection.box)) {
            return true;
        }
    }
    return false;
}

// 当前直接检测框即使同时落入新 D 的宽松范围，只要它在本次操作开始前的
// 旧库存中只有一个严格/局部来源，就优先还给该旧 C。多个旧 C 都合理时
// 返回 false，保留同类歧义而不按遍历顺序强行认领。
bool has_unique_operation_start_owner(
        const Detection& detection, int item_id,
        const std::map<int, InventoryItem>& operation_start) {
    int owner = -1;
    for (std::map<int, InventoryItem>::const_iterator it = operation_start.begin();
         it != operation_start.end(); ++it) {
        const BBox reference = it->second.base_box.area() > 0.0f
            ? it->second.base_box : it->second.box;
        if (!strict_match_box(it->second.cls_id, reference,
                              detection.cls_id, detection.box) &&
            !partial_match_box(it->second.cls_id, reference,
                               detection.cls_id, detection.box)) {
            continue;
        }
        if (owner >= 0) return false;
        owner = it->first;
    }
    return owner == item_id;
}

bool detection_conflicts_with_active_track(
        const Detection& detection,
        const std::map<int, OperationTrack>& tracks) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (detection_can_belong_to_active_track(detection, it->second)) return true;
    }
    return false;
}

bool protected_existing_track_blocks_post_hand_d(
        const Detection& detection,
        const std::map<int, OperationTrack>& tracks) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& track = it->second;
        if (is_claim_protected(track) && track.cls_id == detection.cls_id) {
            return true;
        }
    }
    return false;
}

bool has_active_runtime_for_item(const std::map<int, OperationTrack>& tracks,
                                 int item_id) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (it->second.item_id == item_id && is_active_runtime_track(it->second)) {
            return true;
        }
    }
    return false;
}

// 无手阶段，未参与本次 HAND_* 的旧库存也必须先占用自己唯一的严格框。
// 否则一个已有 D 的宽松局部/轨迹匹配可能会先抢走它，进而制造重复 IN。
void reserve_unique_no_hand_static_inventory_detections(
        const std::vector<Detection>& detections,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::map<int, OperationTrack>& tracks,
        std::set<int>* claimed) {
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        std::map<int, int> item_to_detection;
        std::map<int, std::vector<int> > detection_to_items;

        for (std::map<int, InventoryItem>::const_iterator original =
                 operation_start.begin(); original != operation_start.end(); ++original) {
            if (has_active_runtime_for_item(tracks, original->first)) continue;
            std::map<int, InventoryItem>::const_iterator current =
                working.find(original->first);
            if (current == working.end()) continue;
            const BBox reference = current->second.base_box.area() > 0.0f
                ? current->second.base_box : current->second.box;
            bool already_claimed_for_item = false;
            for (std::set<int>::const_iterator claimed_index = claimed->begin();
                 claimed_index != claimed->end(); ++claimed_index) {
                if (*claimed_index < 0 ||
                    static_cast<size_t>(*claimed_index) >= detections.size()) {
                    continue;
                }
                if (strict_match_box(current->second.cls_id, reference,
                                     detections[*claimed_index].cls_id,
                                     detections[*claimed_index].box)) {
                    already_claimed_for_item = true;
                    break;
                }
            }
            if (already_claimed_for_item) continue;
            int only_detection = -1;
            for (size_t di = 0; di < detections.size(); ++di) {
                const int index = static_cast<int>(di);
                if (claimed->count(index) ||
                    !strict_match_box(current->second.cls_id, reference,
                                      detections[di].cls_id, detections[di].box)) {
                    continue;
                }
                if (only_detection >= 0) {
                    only_detection = -1;
                    break;
                }
                only_detection = index;
            }
            if (only_detection >= 0) {
                item_to_detection[original->first] = only_detection;
                detection_to_items[only_detection].push_back(original->first);
            }
        }

        for (std::map<int, int>::const_iterator item = item_to_detection.begin();
             item != item_to_detection.end(); ++item) {
            const int detection_index = item->second;
            if (claimed->count(detection_index) ||
                detection_to_items[detection_index].size() != 1) {
                continue;
            }
            claimed->insert(detection_index);
            made_progress = true;
        }
    }
}

bool boxes_differ_as_move(const BBox& before, const BBox& after) {
    return center_distance(before, after) >= FLOW3_COMMIT_MOVE_CENTER_DISTANCE;
}

// 严格匹配已经成立时，用一个连续代价挑选“最像”的框。逐帧预扫描不能
// 只依赖 map 的遍历顺序：同类物品相邻时，旧物品通常会同时满足多个
// 宽松条件，必须先把它最接近的那个严格框占住，剩下的框才有机会成为 D。
float strict_match_cost(int cls_id, const BBox& reference,
                        const Detection& observed) {
    if (!strict_match_box(cls_id, reference, observed.cls_id, observed.box)) {
        return std::numeric_limits<float>::infinity();
    }
    const float center = normalized_nearby_distance(reference, observed.box);
    const float width = ratio_difference(reference.w(), observed.box.w());
    const float height = ratio_difference(reference.h(), observed.box.h());
    return center + 0.35f * width + 0.35f * height;
}

// 细节9的最终无手同类数量结算只在“一个框代表一个可见实例”时使用。
// 即使几何证据完全重叠，也要给同类旧 C 一个稳定、可复现的保留顺序；
// 这里的回退距离不会单独产生 MOVED，只决定不可区分时保留哪个 item_id。
float visible_count_owner_cost(const InventoryItem& original,
                               const OperationTrack* runtime,
                               const Detection& observed) {
    if (original.cls_id != observed.cls_id || observed.box.area() <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    const BBox base = original.base_box.area() > 0.0f
        ? original.base_box : original.box;
    const float shape_cost =
        0.35f * ratio_difference(base.w(), observed.box.w()) +
        0.35f * ratio_difference(base.h(), observed.box.h());
    float result = 20.0f + normalized_nearby_distance(base, observed.box) + shape_cost;
    if (strict_match_box(original.cls_id, base, observed.cls_id, observed.box)) {
        result = strict_match_cost(original.cls_id, base, observed);
    } else if (partial_match_box(original.cls_id, base,
                                 observed.cls_id, observed.box)) {
        result = 1.0f + (1.0f - iom(base, observed.box));
    }

    if (!runtime || runtime->is_suspect_new) return result;
    const BBox runtime_reference = runtime->has_placed_box ? runtime->placed_box :
        estimated_box(*runtime);
    if (strict_match_box(runtime->cls_id, runtime_reference,
                         observed.cls_id, observed.box)) {
        result = std::min(result,
                          2.0f + strict_match_cost(runtime->cls_id,
                                                   runtime_reference, observed));
    } else if (track_match_box(runtime->cls_id, runtime_reference,
                               observed.cls_id, observed.box)) {
        result = std::min(result,
                          3.0f + normalized_nearby_distance(runtime_reference,
                                                             observed.box) + shape_cost);
    }
    if (runtime->has_last_seen_box &&
        track_match_box(runtime->cls_id, runtime->last_seen_box,
                        observed.cls_id, observed.box)) {
        result = std::min(result,
                          4.0f + normalized_nearby_distance(runtime->last_seen_box,
                                                             observed.box) + shape_cost);
    }
    return result;
}

// 数量不足只能跨连续、可解释的无手框累计。这里复用既有 track 匹配范围，
// 不新增一套数值阈值；它只用于判断“上一张保留的可见实例是否仍是同一框”。
bool visible_count_survivor_box_is_continuous(int cls_id, const BBox& previous,
                                               const BBox& current) {
    return track_match_box(cls_id, previous, cls_id, current);
}

// 已经完成自身连续确认的 D 不能为了填补旧 C 的可见数量而被抢回。未完成
// 证据链的 suspect D 仍不能单独阻断旧 C 的保守处理。
bool detection_matches_confirmed_suspect_d(
        const Detection& detection,
        const std::map<int, OperationTrack>& tracks) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& track = it->second;
        if (!track.is_suspect_new || !track.promoted_to_working_inventory ||
            !track.drop_confirmed || track.cls_id != detection.cls_id) {
            continue;
        }
        const BBox reference = track.has_placed_box ? track.placed_box :
            (track.has_last_seen_box ? track.last_seen_box : estimated_box(track));
        if (strict_match_box(track.cls_id, reference, detection.cls_id, detection.box) ||
            partial_match_box(track.cls_id, reference, detection.cls_id, detection.box) ||
            track_match_box(track.cls_id, reference, detection.cls_id, detection.box)) {
            return true;
        }
    }
    return false;
}

BBox choose_primary_hand(const std::vector<BBox>& hand_boxes) {
    BBox best;
    float best_area = -1.0f;
    for (size_t i = 0; i < hand_boxes.size(); ++i) {
        if (hand_boxes[i].area() > best_area) {
            best = hand_boxes[i];
            best_area = hand_boxes[i].area();
        }
    }
    return best;
}

bool hand_boxes_effectively_same(const BBox& a, const BBox& b) {
    return std::fabs(a.x1 - b.x1) <= HAND_MICRO_MOVE_SKIP_EPS &&
           std::fabs(a.y1 - b.y1) <= HAND_MICRO_MOVE_SKIP_EPS &&
           std::fabs(a.x2 - b.x2) <= HAND_MICRO_MOVE_SKIP_EPS &&
           std::fabs(a.y2 - b.y2) <= HAND_MICRO_MOVE_SKIP_EPS;
}

// 给当前无手帧建立严格的一对一绑定：只有 item 和 detection 两侧都唯一才绑定。
void bind_mutually_unique(const std::map<int, InventoryItem>& items,
                          const std::set<int>& candidate_item_ids,
                          const std::vector<Detection>& observed,
                          std::map<int, BBox>* references,
                          std::map<int, int>* item_to_observation,
                          std::vector<int>* observation_owner,
                          bool track_mode, bool partial_mode) {
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        std::map<int, std::vector<int> > item_candidates;
        std::vector<std::vector<int> > observation_candidates(observed.size());
        for (std::set<int>::const_iterator id = candidate_item_ids.begin();
             id != candidate_item_ids.end(); ++id) {
            if (item_to_observation->count(*id)) continue;
            std::map<int, InventoryItem>::const_iterator item_it = items.find(*id);
            if (item_it == items.end()) continue;
            const BBox reference = references->count(*id) ? (*references)[*id] :
                item_it->second.box;
            for (size_t si = 0; si < observed.size(); ++si) {
                if ((*observation_owner)[si] >= 0 || observed[si].cls_id != item_it->second.cls_id) {
                    continue;
                }
                bool matched = false;
                if (track_mode) {
                    matched = track_match_box(item_it->second.cls_id, reference,
                                              observed[si].cls_id, observed[si].box);
                } else if (partial_mode) {
                    matched = partial_match_box(item_it->second.cls_id, reference,
                                                observed[si].cls_id, observed[si].box);
                } else {
                    matched = strict_match_box(item_it->second.cls_id, reference,
                                               observed[si].cls_id, observed[si].box);
                }
                if (!matched) continue;
                item_candidates[*id].push_back(static_cast<int>(si));
                observation_candidates[si].push_back(*id);
            }
        }
        for (std::map<int, std::vector<int> >::const_iterator it = item_candidates.begin();
             it != item_candidates.end(); ++it) {
            if (it->second.size() != 1) continue;
            const int si = it->second.front();
            if (observation_candidates[si].size() != 1 || (*observation_owner)[si] >= 0) continue;
            (*item_to_observation)[it->first] = si;
            (*observation_owner)[si] = it->first;
            made_progress = true;
        }
    }
}

const OperationTrack* find_track_for_item(
        const std::map<int, OperationTrack>& tracks, int item_id) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (it->second.item_id == item_id) return &it->second;
    }
    return nullptr;
}

// 无手帧的同类回退不能只因为一个框“几何上靠近”邻居就排除它。这里先建立
// 双方都唯一的严格原位所有权：只有旧 B 在当前帧也只有这一个严格原位框，且
// 该框不同时属于另一旧 C，才可作为 A 的排除证据。它不修改 claimed，不改变
// 正常 C->B/D 仲裁，仅供过期 reappear candidate 被证伪后继续 fallback 使用。
std::map<int, int> build_independent_no_hand_static_owner_by_detection(
        const std::vector<Detection>& detections,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::map<int, OperationTrack>& tracks) {
    std::set<int> candidate_item_ids;
    std::map<int, BBox> references;
    for (std::map<int, InventoryItem>::const_iterator original =
             operation_start.begin(); original != operation_start.end(); ++original) {
        std::map<int, InventoryItem>::const_iterator current =
            working.find(original->first);
        if (current == working.end() ||
            original->second.status == ItemStatus::OCCLUDED ||
            current->second.status == ItemStatus::OCCLUDED) {
            continue;
        }
        const OperationTrack* runtime = find_track_for_item(tracks, original->first);
        if (runtime && (runtime->is_suspect_new ||
                        runtime->resolution == ExistingItemResolution::MOVED_CONFIRMED ||
                        runtime->resolution == ExistingItemResolution::POSITION_REBASED ||
                        runtime->resolution == ExistingItemResolution::OUT_CONFIRMED ||
                        runtime->resolution == ExistingItemResolution::OCCLUDED_CONFIRMED)) {
            continue;
        }
        const BBox reference = original->second.base_box.area() > 0.0f
            ? original->second.base_box : original->second.box;
        if (reference.area() <= 0.0f) continue;
        candidate_item_ids.insert(original->first);
        references[original->first] = reference;
    }

    std::map<int, int> item_to_detection;
    std::vector<int> detection_owner(detections.size(), -1);
    bind_mutually_unique(operation_start, candidate_item_ids, detections, &references,
                         &item_to_detection, &detection_owner, false, false);

    std::map<int, int> result;
    for (size_t i = 0; i < detection_owner.size(); ++i) {
        if (detection_owner[i] >= 0) {
            result[static_cast<int>(i)] = detection_owner[i];
        }
    }
    return result;
}

// 返回当前无手检测到该物品“完整候选轨迹”的最小匹配代价。它只在终点严格匹配
// 和原位置回查都没有解决时使用，避免把手持物体在路径中途放下的情况误判 OUT。
float track_path_match_cost(const InventoryItem& item, const OperationTrack& track,
                            const Detection& observed) {
    if (item.cls_id != observed.cls_id || track.cls_id != observed.cls_id) {
        return std::numeric_limits<float>::infinity();
    }
    float best_cost = std::numeric_limits<float>::infinity();
    if (!track.observed_track.empty()) {
        best_cost = contact_path_match_cost(track, observed);
    }
    for (size_t pi = 0; pi < track.track.size(); ++pi) {
        const BBox& reference = track.track[pi];
        float cost = std::numeric_limits<float>::infinity();
        if (strict_match_box(item.cls_id, reference, observed.cls_id, observed.box)) {
            cost = normalized_nearby_distance(reference, observed.box) +
                0.35f * ratio_difference(reference.w(), observed.box.w()) +
                0.35f * ratio_difference(reference.h(), observed.box.h());
        } else if (track_match_box(item.cls_id, reference, observed.cls_id, observed.box)) {
            cost = 2.0f + normalized_nearby_distance(reference, observed.box) +
                0.35f * ratio_difference(reference.w(), observed.box.w()) +
                0.35f * ratio_difference(reference.h(), observed.box.h());
        }
        best_cost = std::min(best_cost, cost);
    }
    // 重新出现 B 已经连续自匹配后，它是实际观测位置，优先级高于纯手位移
    // 推算出来的完整路径点；仍只作为当前无手帧的候选绑定，不单帧提交。
    if (reappear_candidate_is_confirmed(track)) {
        const BBox& reference = track.reappear_candidate_box;
        float cost = std::numeric_limits<float>::infinity();
        if (strict_match_box(item.cls_id, reference, observed.cls_id, observed.box)) {
            cost = normalized_nearby_distance(reference, observed.box) +
                0.35f * ratio_difference(reference.w(), observed.box.w()) +
                0.35f * ratio_difference(reference.h(), observed.box.h());
        } else if (track_match_box(item.cls_id, reference,
                                   observed.cls_id, observed.box)) {
            cost = 2.0f + normalized_nearby_distance(reference, observed.box) +
                0.35f * ratio_difference(reference.w(), observed.box.w()) +
                0.35f * ratio_difference(reference.h(), observed.box.h());
        }
        best_cost = std::min(best_cost, cost);
    }
    return best_cost;
}

// 路径可能有多个点、同类物品也可能有多条路径，所以这里同样只提交双方
// 都是“唯一最近”的 pair。任何平分都保持未决，交给后续的旧位置回查或 OUT。
void bind_mutually_unique_track_paths(
        const std::map<int, InventoryItem>& items,
        const std::set<int>& candidate_item_ids,
        const std::vector<Detection>& observed,
        const std::map<int, OperationTrack>& tracks,
        std::map<int, int>* item_to_observation,
        std::vector<int>* observation_owner) {
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        std::map<int, int> best_observation_for_item;
        std::set<int> tied_items;
        std::map<int, std::vector<std::pair<int, float> > > candidates_for_observation;

        for (std::set<int>::const_iterator id = candidate_item_ids.begin();
             id != candidate_item_ids.end(); ++id) {
            if (item_to_observation->count(*id)) continue;
            std::map<int, InventoryItem>::const_iterator item = items.find(*id);
            const OperationTrack* track = find_track_for_item(tracks, *id);
            if (item == items.end() || !track || track->track.empty()) continue;

            float best_cost = std::numeric_limits<float>::infinity();
            int best_observation = -1;
            bool tied = false;
            for (size_t si = 0; si < observed.size(); ++si) {
                if ((*observation_owner)[si] >= 0) continue;
                const float cost = track_path_match_cost(item->second, *track, observed[si]);
                if (!(cost < std::numeric_limits<float>::infinity())) continue;
                candidates_for_observation[static_cast<int>(si)].push_back(
                    std::make_pair(*id, cost));
                if (cost + 0.0001f < best_cost) {
                    best_cost = cost;
                    best_observation = static_cast<int>(si);
                    tied = false;
                } else if (std::fabs(cost - best_cost) <= 0.0001f) {
                    tied = true;
                }
            }
            if (best_observation >= 0 && !tied) {
                best_observation_for_item[*id] = best_observation;
            } else if (best_observation >= 0) {
                tied_items.insert(*id);
            }
        }

        std::map<int, int> best_item_for_observation;
        std::set<int> tied_observations;
        for (std::map<int, std::vector<std::pair<int, float> > >::const_iterator si =
                 candidates_for_observation.begin();
             si != candidates_for_observation.end(); ++si) {
            float best_cost = std::numeric_limits<float>::infinity();
            int best_item = -1;
            bool tied = false;
            for (size_t ci = 0; ci < si->second.size(); ++ci) {
                const int item_id = si->second[ci].first;
                const float cost = si->second[ci].second;
                if (cost + 0.0001f < best_cost) {
                    best_cost = cost;
                    best_item = item_id;
                    tied = false;
                } else if (std::fabs(cost - best_cost) <= 0.0001f) {
                    tied = true;
                }
            }
            if (best_item >= 0 && !tied) {
                best_item_for_observation[si->first] = best_item;
            } else if (best_item >= 0) {
                tied_observations.insert(si->first);
            }
        }

        for (std::map<int, int>::const_iterator item = best_observation_for_item.begin();
             item != best_observation_for_item.end(); ++item) {
            const int observation_index = item->second;
            if (tied_items.count(item->first) || tied_observations.count(observation_index) ||
                (*observation_owner)[observation_index] >= 0 ||
                !best_item_for_observation.count(observation_index) ||
                best_item_for_observation[observation_index] != item->first) {
                continue;
            }
            (*item_to_observation)[item->first] = observation_index;
            (*observation_owner)[observation_index] = item->first;
            made_progress = true;
        }
    }
}

}  // namespace

SessionManager::SessionManager() {}

void SessionManager::trace_(const char* tag, const char* format, ...) const {
    if (!FLOW3_DEBUG_TRACE_LOG) return;
    printf("[3.0-TRACE][%s][op=%d][frame=%d][%s] ", tag,
           active_operation_id_, trace_frame_id_,
           trace_hand_phase_ ? "HAND" : "NO_HAND");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void SessionManager::set_live_state_(OperationTrack* track,
                                     LiveObservationState state,
                                     bool provisional,
                                     const char* reason) {
    if (!track) return;
    const std::string next_reason = reason ? reason : "NONE";
    if (track->live_state == state &&
        track->live_state_provisional == provisional &&
        track->live_state_reason == next_reason) {
        return;
    }
    track->live_state = state;
    track->live_state_provisional = provisional;
    track->live_state_last_changed_frame = trace_frame_id_;
    track->live_state_reason = next_reason;
    const BBox box = track->has_last_seen_box ? track->last_seen_box : estimated_box(*track);
    trace_("LIVE-STATE",
           "owner-kind=%s runtime=%d item=%d cls=%d state=%s provisional=%d "
           "reason=%s box=(%.1f,%.1f,%.1f,%.1f)",
           runtime_owner_kind_name(*track),
           track->is_suspect_new ? track->suspect_id : track->item_id,
           track->item_id, track->cls_id,
           live_observation_state_name(state), provisional ? 1 : 0,
           track->live_state_reason.c_str(),
           box.x1, box.y1, box.x2, box.y2);
}

void SessionManager::update_hand_live_states_() {
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new) {
            set_live_state_(&track,
                            track.pending_d_quarantined_by_old_c
                                ? LiveObservationState::C_D_ALIAS
                                : LiveObservationState::PROVISIONAL_D,
                            true,
                            track.pending_d_quarantined_by_old_c
                                ? "same-class-unsettled-old-alias"
                                : "hand-visible-provisional-d");
            continue;
        }
        if (!is_unresolved_operation_start_old_track(track)) continue;
        if (!track.conflicting_suspect_keys.empty()) {
            set_live_state_(&track, LiveObservationState::C_D_ALIAS, true,
                            "same-class-provisional-d-alias");
        } else if (track.contact_state != ContactState::NONE) {
            set_live_state_(&track, LiveObservationState::HAND_CONTACT, true,
                            "contact-track-active");
        } else if (track.hold_and_move || has_meaningful_hand_move(track) ||
                   track.reappearance_pending) {
            set_live_state_(&track, LiveObservationState::POSSIBLE_MOVED, true,
                            "hand-path-or-reappearance-pending");
        } else if (track.state == OperationTrackState::HAND_FULL_BLOCKED) {
            set_live_state_(&track, LiveObservationState::POSSIBLE_OCCLUDED, true,
                            "hand-fully-covers-old-c");
        } else {
            set_live_state_(&track, LiveObservationState::HAND_CONTACT, true,
                            "hand-affects-old-c");
        }
    }
}

bool SessionManager::old_track_has_unresolved_alias_(const OperationTrack& track) const {
    if (track.is_suspect_new) return false;
    for (std::set<int>::const_iterator key = track.conflicting_suspect_keys.begin();
         key != track.conflicting_suspect_keys.end(); ++key) {
        std::map<int, OperationTrack>::const_iterator suspect = track_buffer_.find(*key);
        if (suspect != track_buffer_.end() && suspect->second.is_suspect_new &&
            suspect->second.pending_d_quarantined_by_old_c) {
            return true;
        }
    }
    return false;
}

void SessionManager::unlink_quarantined_suspect_(int runtime_key,
                                                  const char* action) {
    std::map<int, OperationTrack>::iterator suspect = track_buffer_.find(runtime_key);
    if (suspect == track_buffer_.end() || !suspect->second.is_suspect_new) return;
    OperationTrack& d = suspect->second;
    for (std::set<int>::const_iterator old_id = d.conflicting_old_item_ids.begin();
         old_id != d.conflicting_old_item_ids.end(); ++old_id) {
        std::map<int, OperationTrack>::iterator old = track_buffer_.find(*old_id);
        if (old != track_buffer_.end()) {
            old->second.conflicting_suspect_keys.erase(runtime_key);
            trace_("C-D-ALIAS",
                   "old-item=%d suspect=%d phase=%s action=%s relation=closed",
                   *old_id, d.suspect_id,
                   trace_hand_phase_ ? "HAND" : "NO_HAND",
                   action ? action : "unlink");
        }
    }
    d.conflicting_old_item_ids.clear();
    d.pending_d_quarantined_by_old_c = false;
    d.alias_no_hand_match_count = 0;
    d.alias_no_hand_missing_count = 0;
}

void SessionManager::trace_track_(const char* tag, const OperationTrack& track,
                                  const char* reason) const {
    const BBox expected = estimated_box(track);
    trace_(tag,
           "item=%d suspect=%d cls=%d state=%s contact=%s resolution=%s "
           "needs_settle=%d static-near-original=%d/%d indirect=%d/%d indirect-pre=%d "
           "indirect-cluster=%d indirect-index=%d grace=%d hold=%d not_hold=%d "
           "missing=%d ambiguous=%d "
           "owner-kind=%s quarantine=%d alias-old=%zu alias-d=%zu alias-no-hand=%d "
           "alias-missing=%d "
           "live=%s provisional=%d "
           "original=(%.1f,%.1f,%.1f,%.1f) expected=(%.1f,%.1f,%.1f,%.1f) "
           "last=%d:(%.1f,%.1f,%.1f,%.1f) reappear=%d/%d:(%.1f,%.1f,%.1f,%.1f) "
           "reason=%s",
           track.item_id, track.suspect_id, track.cls_id,
           operation_track_state_name(track.state),
           contact_state_name(track.contact_state),
           existing_resolution_name(track.resolution),
           track.needs_no_hand_settlement ? 1 : 0,
           track.stable_near_original_no_hand_count,
           track.has_stable_near_original_box ? 1 : 0,
           track.indirect_move_candidate ? 1 : 0,
           track.indirect_no_hand_match_count,
           track.indirect_predecessor_item_id,
           track.indirect_cluster_id,
           track.indirect_no_hand_detection_index,
           track.claim_grace_remaining, track.hold_evidence_count,
           track.not_hold_evidence_count, track.no_hand_missing_count,
           (track.b_claim_ambiguous || track.contact_path_ambiguous ||
            track.no_hand_candidate_ambiguous) ? 1 : 0,
           runtime_owner_kind_name(track),
           track.pending_d_quarantined_by_old_c ? 1 : 0,
           track.conflicting_old_item_ids.size(),
           track.conflicting_suspect_keys.size(),
           track.alias_no_hand_match_count,
           track.alias_no_hand_missing_count,
           live_observation_state_name(track.live_state),
           track.live_state_provisional ? 1 : 0,
           track.original_box.x1, track.original_box.y1,
           track.original_box.x2, track.original_box.y2,
           expected.x1, expected.y1, expected.x2, expected.y2,
           track.has_last_seen_box ? 1 : 0,
           track.last_seen_box.x1, track.last_seen_box.y1,
           track.last_seen_box.x2, track.last_seen_box.y2,
           track.has_reappear_candidate_box ? 1 : 0,
           track.reappear_candidate_match_count,
           track.reappear_candidate_box.x1, track.reappear_candidate_box.y1,
           track.reappear_candidate_box.x2, track.reappear_candidate_box.y2,
           reason ? reason : "NONE");
}

void SessionManager::rebuild_persistent_item_index_() {
    item_by_id_.clear();
    std::map<int, InventoryItem>& items = inventory_.mutable_items();
    for (std::map<int, InventoryItem>::iterator it = items.begin(); it != items.end(); ++it) {
        item_by_id_[it->first] = &it->second;
    }
}

void SessionManager::reset_operation_runtime_() {
    working_inventory_.clear();
    operation_start_inventory_.clear();
    working_next_item_id_ = inventory_.next_item_id();
    working_inventory_active_ = false;
    track_buffer_.clear();
    pending_in_ids_.clear();
    pending_out_ids_.clear();
    confirmed_moved_ids_.clear();
    confirmed_position_update_ids_.clear();
    released_hand_candidate_ids_.clear();
    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    visible_count_out_candidate_ids_.clear();
    visible_count_missing_counts_.clear();
    visible_count_confirmed_out_ids_.clear();
    visible_count_prior_survivors_by_cls_.clear();
    visible_count_prior_survivor_boxes_by_cls_.clear();
    hand_track_.clear();
    has_old_hand_box_ = false;
    next_suspect_id_ = -1;
    no_hand_streak_ = 0;
    active_operation_id_ = 0;
}

void SessionManager::start_new_session(long long time_ms) {
    reset_operation_runtime_();
    current_time_ms_ = time_ms;
    hand_present_ = false;
    session_active_ = true;
    if (has_local_inventory_) {
        initial_check_state_ = InitialCheckState::NOT_NEEDED;
    } else {
        initial_check_state_ = InitialCheckState::NONE;
        backend_status_ = BackendStatus::UNKNOWN;
    }
}

void SessionManager::init_from_backend(const std::vector<InventoryItem>& items,
                                       bool authoritative_empty) {
    if (!session_active_ || has_local_inventory_ ||
        backend_status_ != BackendStatus::UNKNOWN) return;
    if (items.empty() && !authoritative_empty) {
        mark_backend_unavailable();
        return;
    }

    std::map<int, InventoryItem> loaded;
    int next_id = inventory_.next_item_id();
    for (size_t i = 0; i < items.size(); ++i) {
        InventoryItem item = items[i];
        if (item.item_id <= 0 || loaded.count(item.item_id)) item.item_id = next_id++;
        if (item.base_box.area() <= 0.0f) item.base_box = item.box;
        if (item.box.area() <= 0.0f) item.box = item.base_box;
        if (item.base_box.area() <= 0.0f || item.box.area() <= 0.0f) continue;
        item.status = ItemStatus::VISIBLE;
        item.block_ids.clear();
        loaded[item.item_id] = item;
        next_id = std::max(next_id, item.item_id + 1);
    }
    inventory_.replace_all(loaded, next_id);
    rebuild_persistent_item_index_();
    has_local_inventory_ = true;
    backend_status_ = BackendStatus::TRUSTED;
    initial_check_state_ = InitialCheckState::WAITING;
}

void SessionManager::mark_backend_unavailable() {
    if (!session_active_ || has_local_inventory_) return;
    backend_status_ = BackendStatus::NO_TRUSTED_BACKEND;
    initial_check_state_ = ALLOW_INITIAL_FRAME_BOOTSTRAP_WHEN_BACKEND_UNAVAILABLE
        ? InitialCheckState::BOOTSTRAP_FROM_FRAME : InitialCheckState::NONE;
}

void SessionManager::finish_session(long long time_ms) {
    current_time_ms_ = time_ms;
    // 未完成操作不提交；正式库存保持上一次已确认结果。
    reset_operation_runtime_();
    hand_present_ = false;
    initial_check_state_ = InitialCheckState::NONE;
    session_active_ = false;
}

void SessionManager::begin_closing_guard() {
    has_old_hand_box_ = false;
}

void SessionManager::resume_after_false_closing() {
    // CLOSING 打断了连续证据链，最安全的处理是丢弃工作副本，保留正式库存。
    if (working_inventory_active_) reset_operation_runtime_();
}

void SessionManager::validate_initial_no_hand_frame_(
        const std::vector<Detection>& detections) const {
    int matched = 0;
    for (size_t si = 0; si < detections.size(); ++si) {
        int count = 0;
        for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
             it != inventory_.items().end(); ++it) {
            if (strict_match(it->second, detections[si]) ||
                partial_match(it->second, detections[si])) {
                ++count;
            }
        }
        if (count == 1) ++matched;
    }
    printf("[INIT-CHECK] 首张无手帧校验：%d/%zu 个框可唯一对应；库存未修改\n",
           matched, detections.size());
}

void SessionManager::initialize_from_bootstrap_no_hand_frame_(
        const std::vector<Detection>& detections, int frame_id) {
    std::map<int, InventoryItem> loaded;
    int next_id = inventory_.next_item_id();
    for (size_t i = 0; i < detections.size(); ++i) {
        loaded[next_id] = make_inventory_item(next_id, detections[i], frame_id,
                                               current_time_ms_);
        ++next_id;
    }
    inventory_.replace_all(loaded, next_id);
    rebuild_persistent_item_index_();
    has_local_inventory_ = true;
    initial_check_state_ = InitialCheckState::DONE;
    printf("[INIT] 本地首张无手帧建库：%zu 个物品\n", loaded.size());
}

void SessionManager::finalize_initial_check_before_hand_() {
    if (initial_check_state_ != InitialCheckState::WAITING &&
        initial_check_state_ != InitialCheckState::BOOTSTRAP_FROM_FRAME) return;
    // 手在任何无手帧前就进入画面时，没有可用于首次校验/建库的直接检测。
    if (initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_FRAME) {
        initialize_from_bootstrap_no_hand_frame_(std::vector<Detection>(), 0);
    } else {
        initial_check_state_ = InitialCheckState::SKIPPED;
    }
}

int SessionManager::new_suspect_id_() {
    return next_suspect_id_--;
}

OperationTrack* SessionManager::find_runtime_for_item_(int item_id) {
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        if (it->second.item_id == item_id) return &it->second;
    }
    return nullptr;
}

const OperationTrack* SessionManager::find_runtime_for_item_(int item_id) const {
    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        if (it->second.item_id == item_id) return &it->second;
    }
    return nullptr;
}

void SessionManager::begin_working_operation_(const BBox& hand_box,
                                               const std::vector<Detection>& detections) {
    working_inventory_ = inventory_.items();
    operation_start_inventory_ = inventory_.items();
    working_next_item_id_ = inventory_.next_item_id();
    working_inventory_active_ = true;
    track_buffer_.clear();
    pending_in_ids_.clear();
    pending_out_ids_.clear();
    confirmed_moved_ids_.clear();
    confirmed_position_update_ids_.clear();
    released_hand_candidate_ids_.clear();
    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    visible_count_out_candidate_ids_.clear();
    visible_count_missing_counts_.clear();
    visible_count_confirmed_out_ids_.clear();
    visible_count_prior_survivors_by_cls_.clear();
    visible_count_prior_survivor_boxes_by_cls_.clear();
    hand_track_.clear();
    hand_track_.push_back(hand_box);
    old_hand_box_ = hand_box;
    has_old_hand_box_ = true;
    no_hand_streak_ = 0;
    active_operation_id_ = next_operation_id_++;
    trace_("STATE", "begin operation inventory=%zu detections=%zu hand=(%.1f,%.1f,%.1f,%.1f)",
           working_inventory_.size(), detections.size(), hand_box.x1, hand_box.y1,
           hand_box.x2, hand_box.y2);
    process_effective_hand_frame_(hand_box, detections, true);
}

void SessionManager::append_move_to_existing_hand_tracks_(const MoveValue& delta) {
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.contact_state != ContactState::NONE) {
            // 仅作调试/未来扩展记录；CONTACT_* 的物品位置不能由手位移推算。
            track.hand_move_values.push_back(delta);
        }
        if (track.state != OperationTrackState::HAND_PARTIAL_BLOCKED &&
            track.state != OperationTrackState::HAND_FULL_BLOCKED) {
            continue;
        }
        track.move_values.push_back(delta);
        track.track.push_back(estimated_box(track));
    }
}

void SessionManager::update_existing_contact_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
    std::vector<std::pair<int, int> > release_keys;
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (!is_active_contact_track(track)) continue;

        // 若 CONTACT_* 已经用真实 B 看见过物品的新位置，覆盖率必须相对
        // 当前真实位置计算；仍拿最初 A.box 计算会使“先推、后握住”永远
        // 停留在 CONTACT_*。
        const BBox reference = track.has_tentative_b_box ? track.tentative_b_box
            : (track.has_last_seen_box ? track.last_seen_box : track.original_box);
        if (reference.area() <= 0.0f) continue;

        // 覆盖率达到 e2 后，接触候选转入已有 HAND_* 流程；同一帧稍后由
        // update_existing_hand_tracks_ 继续处理，避免两套逻辑同时认领。
        if (hand_fully_covers(hand_box, reference) || hand_affects(hand_box, reference)) {
            // 保留 contact 阶段的真实位置作为之后 HAND_* 估计的原点。
            // original_box 不能改，它还要用于判断最终是否发生整理/出库。
            track.hand_estimate_anchor_box = reference;
            track.has_hand_estimate_anchor_box = true;
            track.move_values.clear();
            track.track.clear();
            track.track.push_back(reference);
            track.contact_state = ContactState::NONE;
            track.state = hand_fully_covers(hand_box, reference)
                ? OperationTrackState::HAND_FULL_BLOCKED
                : OperationTrackState::HAND_PARTIAL_BLOCKED;
            if (track.has_last_seen_box) {
                track.first_hand_block_box = track.last_seen_box;
                track.last_hand_block_box = track.last_seen_box;
                track.has_first_hand_block_box = true;
                track.has_last_hand_block_box = true;
            }
            trace_track_("STATE", track, "contact-to-hand-transition");
            continue;
        }

        const int observed_index = unique_contact_detection_for_track(
            detections, *claimed_detection_indices, track, working_inventory_,
            track_buffer_);
        if (observed_index < 0) {
            // 漏检、多个同类候选或与其他库存冲突都不是“未持有”的证据。
            if (has_contact_path_candidate(detections, *claimed_detection_indices,
                                           track, working_inventory_)) {
                track.contact_path_ambiguous = true;
            }
            continue;
        }

        const Detection& detection = detections[observed_index];
        const bool touching_hand = hand_is_near(hand_box, detection.box);
        const bool had_touch_before = track.contact_started_touching_hand;
        const BBox previous = track.has_last_seen_box
            ? track.last_seen_box : track.original_box;
        const bool at_original = contact_detection_is_at_original(track, detection);
        const float object_move = center_distance(previous, detection.box);
        const int matching_tentative_count =
            track.has_tentative_b_box &&
            track_match_box(track.cls_id, track.tentative_b_box,
                            detection.cls_id, detection.box)
                ? track.tentative_b_match_count : 0;

        if (is_claim_protected(track) && !at_original) {
            // 保护期内照常检查这条 C→B 本地路径，但不得把 B 写入 claimed /
            // known_item_owner；同一 B 仍必须对成熟 C 和 D 链路开放仲裁。
            record_tentative_b(&track, detection, touching_hand);
            // observed_track 是 C 自己的本地真实观测，不等于本帧的排他
            // B 认领。保存它可避免 CONTACT→HAND 时丢掉最新可靠物品框。
            append_contact_observation(&track, detection, touching_hand);
            if (object_move >= FLOW3_CONTACT_OBJECT_MOVE_EPS &&
                (had_touch_before || touching_hand)) {
                ++track.hold_evidence_count;
                track.not_hold_evidence_count = 0;
            }
            continue;
        }

        claimed_detection_indices->insert(observed_index);
        if (track.item_id > 0) (*known_item_owner)[track.item_id] = observed_index;
        append_contact_observation(&track, detection, touching_hand);

        if (!at_original) {
            track.has_tentative_b_box = false;
            track.tentative_b_match_count = 0;
            track.tentative_b_started_touching_hand = false;
        }

        if (track.contact_state == ContactState::CONTACT_MOVING) {
            std::map<int, InventoryItem>::iterator item =
                working_inventory_.find(track.item_id);
            if (item != working_inventory_.end()) {
                update_seen(item->second, detection, trace_frame_id_);
            }
            continue;
        }

        if (at_original) {
            track.has_tentative_b_box = false;
            track.tentative_b_match_count = 0;
            track.tentative_b_started_touching_hand = false;
            track.has_reappear_candidate_box = false;
            track.reappear_candidate_match_count = 0;
            track.reappear_candidate_started_touching_hand = false;
            ++track.not_hold_evidence_count;
            track.hold_evidence_count = 0;
        } else if (object_move >= FLOW3_CONTACT_OBJECT_MOVE_EPS &&
                   (had_touch_before || touching_hand)) {
            // 只要第一次有效 B 已经与手相贴，后续 B 可在手离开后继续沿
            // observed_track 确认，不把手框方向当作物品方向。
            ++track.hold_evidence_count;
            if (matching_tentative_count > 0) {
                // t3 的唯一 B 接上 t0~t2 已记录的本地证据时，直接继承
                // 连续性；它仍是在成熟后才变成正式归属。
                track.hold_evidence_count = std::max(
                    track.hold_evidence_count, matching_tentative_count);
            }
            track.not_hold_evidence_count = 0;
        }

        if (track.not_hold_evidence_count >= FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
            release_keys.push_back(std::make_pair(it->first, observed_index));
            continue;
        }
        if (track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
            track.contact_state = ContactState::CONTACT_MOVING;
            track.hold_and_move = true;
            track.hold_evidence_count = 0;
            track.not_hold_evidence_count = 0;
            track.shelter_or_hold = true;
            printf("[3.0] item#%d 确认低覆盖率推/拉，进入 CONTACT_MOVING\n",
                   track.item_id);
        }
    }

    for (size_t i = 0; i < release_keys.size(); ++i) {
        std::map<int, OperationTrack>::iterator it = track_buffer_.find(release_keys[i].first);
        if (it != track_buffer_.end() &&
            it->second.contact_state == ContactState::CONTACT_CANDIDATE) {
            release_not_held_(it->second, false,
                              ReleaseReason::CONTACT_RETURNED_ORIGINAL,
                              release_keys[i].second,
                              &detections[release_keys[i].second].box,
                              "hand-contact-returned-original");
        }
    }
}

void SessionManager::mark_new_contact_candidates_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        std::set<int>* new_existing_track_ids) {
    // 已释放的候选在手仍停留附近时不反复创建；手离开该物品后才允许下一次
    // 接触重新建立候选。
    for (std::set<int>::iterator released = released_hand_candidate_ids_.begin();
         released != released_hand_candidate_ids_.end();) {
        std::map<int, InventoryItem>::const_iterator item =
            working_inventory_.find(*released);
        if (item == working_inventory_.end() ||
            !hand_is_near(hand_box, item->second.base_box.area() > 0.0f
                                      ? item->second.base_box : item->second.box)) {
            released = released_hand_candidate_ids_.erase(released);
        } else {
            ++released;
        }
    }

    for (std::map<int, InventoryItem>::const_iterator it = working_inventory_.begin();
         it != working_inventory_.end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status == ItemStatus::OCCLUDED ||
            released_hand_candidate_ids_.count(item.item_id)) {
            continue;
        }
        OperationTrack* existing = find_runtime_for_item_(item.item_id);
        if (existing && is_active_runtime_track(*existing)) continue;

        const BBox reference = item.base_box.area() > 0.0f
            ? item.base_box : item.box;
        if (reference.area() <= 0.0f || hand_cover_ratio(hand_box, reference) >=
                FLOW3_HAND_PARTIAL_COVER_RATIO ||
            !hand_is_near(hand_box, reference)) {
            continue;
        }

        OperationTrack track;
        track.item_id = item.item_id;
        track.cls_id = item.cls_id;
        track.original_box = reference;
        track.contact_state = ContactState::CONTACT_CANDIDATE;
        track.needs_no_hand_settlement = true;
        track.shelter_or_hold = true;
        track.claim_grace_remaining = FLOW3_NEW_TRACK_CLAIM_GRACE_FRAMES;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track_buffer_[item.item_id] = track;
        OperationTrack& created = track_buffer_[item.item_id];
        if (new_existing_track_ids) new_existing_track_ids->insert(item.item_id);

        int observed_index = -1;
        std::map<int, int>::const_iterator owner = known_item_owner->find(item.item_id);
        if (owner != known_item_owner->end() &&
            owner->second >= 0 &&
            owner->second < static_cast<int>(detections.size()) &&
            contact_path_match_cost(created, detections[owner->second]) <
                std::numeric_limits<float>::infinity()) {
            observed_index = owner->second;
        } else {
            observed_index = unique_contact_detection_for_track(
                detections, *claimed_detection_indices, created,
                working_inventory_, track_buffer_);
        }
        if (observed_index >= 0) {
            const bool touching = hand_is_near(hand_box, detections[observed_index].box);
            const bool at_original = contact_detection_is_at_original(
                created, detections[observed_index]);
            // 保护期只禁止“移动后的同类 B”的排他归属。C 若确实仍在旧
            // 位置，保留自己的严格框是安全的；否则撤销 reserve 阶段可能
            // 做出的预占，只把它保存为本地 tentative B。
            if (at_original || !is_claim_protected(created)) {
                if (!claimed_detection_indices->count(observed_index)) {
                    claimed_detection_indices->insert(observed_index);
                }
                (*known_item_owner)[item.item_id] = observed_index;
                append_contact_observation(&created, detections[observed_index], touching);
            } else {
                claimed_detection_indices->erase(observed_index);
                known_item_owner->erase(item.item_id);
                record_tentative_b(&created, detections[observed_index], touching);
                append_contact_observation(&created, detections[observed_index], touching);
            }
        }
        printf("[3.0] item#%d 进入 CONTACT_CANDIDATE（低覆盖率且手相贴）\n",
               item.item_id);
        trace_track_("STATE", created, "create-contact-candidate");
    }
}

void SessionManager::promote_suspect_(int runtime_key, const Detection& detection,
                                      int frame_id) {
    std::map<int, OperationTrack>::iterator it = track_buffer_.find(runtime_key);
    if (it == track_buffer_.end()) return;
    OperationTrack& track = it->second;
    if (!track.is_suspect_new || track.promoted_to_working_inventory ||
        track.self_match_count < NEW_ITEM_CONFIRM_FRAMES) {
        return;
    }
    if (track.pending_d_quarantined_by_old_c) {
        trace_("D-GUARD",
               "suspect=%d cls=%d action=keep-quarantined-runtime-only "
               "reason=unresolved-c-d-alias self-match=%d formal-owner-authority=0",
               track.suspect_id, track.cls_id, track.self_match_count);
        return;
    }
    const int item_id = working_next_item_id_++;
    working_inventory_[item_id] = make_inventory_item(item_id, detection,
                                                       frame_id, current_time_ms_);
    // 初次检测可能只是局部框；它只作为临时位置，后续无手直接帧会刷新。
    track.item_id = item_id;
    track.promoted_to_working_inventory = true;
    pending_in_ids_.insert(item_id);
    printf("[3.0] suspect#%d 已提升为工作库存 item#%d（尚未正式 IN）\n",
           track.suspect_id, item_id);
    trace_track_("STATE", track, "promote-suspect-to-working-inventory");
}

std::map<int, int>
SessionManager::build_mutually_unique_hand_static_owner_by_detection_(
        const BBox& hand_box,
        const std::vector<Detection>& detections,
        const std::set<int>& claimed_seed,
        const std::map<int, int>& known_item_owner_seed) const {
    std::set<int> planned_claimed_detection_indices(claimed_seed);
    std::map<int, int> planned_known_item_owner(known_item_owner_seed);
    std::map<int, int> owner_by_detection;

    // 只处理没有正在进行 HAND_* / PLACED 轨迹的普通静态物品。循环中的
    // claimed / known owner 仅是副本，因此该计划可在 D 更新前安全预读。
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        std::map<int, int> best_detection_for_item;
        std::set<int> tied_items;
        std::map<int, std::vector<std::pair<int, float> > > candidates_for_detection;

        for (std::map<int, InventoryItem>::const_iterator item =
                 working_inventory_.begin();
             item != working_inventory_.end(); ++item) {
            if (item->second.status == ItemStatus::OCCLUDED ||
                planned_known_item_owner.count(item->first)) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(item->first);
            if (runtime && is_active_runtime_track(*runtime)) continue;

            const BBox reference = item->second.base_box.area() > 0.0f
                ? item->second.base_box : item->second.box;
            float best_cost = std::numeric_limits<float>::infinity();
            int best_detection = -1;
            bool tied = false;
            for (size_t di = 0; di < detections.size(); ++di) {
                const int detection_index = static_cast<int>(di);
                if (planned_claimed_detection_indices.count(detection_index)) continue;
                const float cost = strict_match_cost(item->second.cls_id, reference,
                                                     detections[di]);
                if (!(cost < std::numeric_limits<float>::infinity())) continue;
                candidates_for_detection[detection_index].push_back(
                    std::make_pair(item->first, cost));
                if (cost + 0.0001f < best_cost) {
                    best_cost = cost;
                    best_detection = detection_index;
                    tied = false;
                } else if (std::fabs(cost - best_cost) <= 0.0001f) {
                    tied = true;
                }
            }
            // 手对该旧物品完整框的覆盖率已经达到 e2 时，必须交给 HAND_*
            // 分支建立候选；不能根据缩小后的当前检测框来决定它是否静态。
            if (best_detection >= 0 && !tied &&
                !hand_affects(hand_box, reference)) {
                best_detection_for_item[item->first] = best_detection;
            } else if (best_detection >= 0) {
                tied_items.insert(item->first);
            }
        }

        std::map<int, int> best_item_for_detection;
        std::set<int> tied_detections;
        for (std::map<int, std::vector<std::pair<int, float> > >::const_iterator di =
                 candidates_for_detection.begin();
             di != candidates_for_detection.end(); ++di) {
            float best_cost = std::numeric_limits<float>::infinity();
            int best_item = -1;
            bool tied = false;
            for (size_t ci = 0; ci < di->second.size(); ++ci) {
                const int item_id = di->second[ci].first;
                const float cost = di->second[ci].second;
                if (cost + 0.0001f < best_cost) {
                    best_cost = cost;
                    best_item = item_id;
                    tied = false;
                } else if (std::fabs(cost - best_cost) <= 0.0001f) {
                    tied = true;
                }
            }
            if (best_item >= 0 && !tied) {
                best_item_for_detection[di->first] = best_item;
            } else if (best_item >= 0) {
                tied_detections.insert(di->first);
            }
        }

        for (std::map<int, int>::const_iterator best =
                 best_detection_for_item.begin();
             best != best_detection_for_item.end(); ++best) {
            if (tied_items.count(best->first) ||
                tied_detections.count(best->second)) continue;
            const int detection_index = best->second;
            if (!best_item_for_detection.count(detection_index) ||
                best_item_for_detection[detection_index] != best->first ||
                planned_claimed_detection_indices.count(detection_index)) {
                continue;
            }
            planned_known_item_owner[best->first] = detection_index;
            planned_claimed_detection_indices.insert(detection_index);
            owner_by_detection[detection_index] = best->first;
            made_progress = true;
        }
    }

    return owner_by_detection;
}

void SessionManager::reserve_visible_known_detections_(
        const BBox& hand_box,
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
    const std::map<int, int> owner_by_detection =
        build_mutually_unique_hand_static_owner_by_detection_(
            hand_box, detections, *claimed_detection_indices, *known_item_owner);

    for (std::map<int, int>::const_iterator owner = owner_by_detection.begin();
         owner != owner_by_detection.end(); ++owner) {
        const int detection_index = owner->first;
        const int item_id = owner->second;
        if (claimed_detection_indices->count(detection_index) ||
            known_item_owner->count(item_id)) {
            continue;
        }
        (*known_item_owner)[item_id] = detection_index;
        claimed_detection_indices->insert(detection_index);
    }
}

void SessionManager::confirm_rearrange_(OperationTrack& track,
                                        const BBox& release_box,
                                        float score, int frame_id) {
    if (track.item_id <= 0) return;
    std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
    if (item == working_inventory_.end()) return;
    item->second.box = release_box;
    item->second.base_box = release_box;
    item->second.score = score;
    item->second.updated_frame = frame_id;
    item->second.status = ItemStatus::VISIBLE;
    item->second.block_ids.clear();
    track.placed_box = release_box;
    track.has_placed_box = true;
    track.drop_confirmed = true;
    track.state = OperationTrackState::PLACED;
    track.contact_state = ContactState::NONE;
    track.resolution = ExistingItemResolution::MOVED_CONFIRMED;
    track.release_reason = ReleaseReason::NONE;
    track.needs_no_hand_settlement = false;
    track.stable_near_original_no_hand_count = 0;
    track.has_stable_near_original_box = false;
    track.stable_near_original_box = BBox();
    track.b_claim_ambiguous = false;
    track.no_hand_candidate_ambiguous = false;
    track.claim_grace_remaining = 0;
    track.no_hand_missing_count = 0;
    track.has_tentative_b_box = false;
    track.tentative_b_match_count = 0;
    track.tentative_b_started_touching_hand = false;
    confirmed_moved_ids_.insert(track.item_id);
    confirmed_position_update_ids_.insert(track.item_id);
    set_live_state_(&track, LiveObservationState::PLACED, false,
                    "moved-confirmed-after-direct-evidence");
    trace_track_("STATE", track, "confirm-rearrange");
}

void SessionManager::confirm_position_rebase_(OperationTrack& track,
                                              const BBox& release_box,
                                              float score, int frame_id,
                                              bool emit_moved,
                                              const char* reason) {
    if (track.item_id <= 0 || track.is_suspect_new || release_box.area() <= 0.0f) {
        return;
    }
    std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
    if (item == working_inventory_.end()) return;

    // `emit_moved` only controls the business event threshold.  An indirect
    // nudge below that threshold is still an INDIRECT-MOVE, not a detector
    // STABLE-REBASE; retain that distinction in the diagnostic trace.
    const bool indirect_move = track.indirect_move_candidate ||
        track.indirect_predecessor_item_id > 0 || track.indirect_cluster_id > 0;
    const int indirect_predecessor_item_id = track.indirect_predecessor_item_id;
    const int indirect_cluster_id = track.indirect_cluster_id;
    const bool formal_move = boxes_differ_as_move(track.original_box, release_box);

    item->second.box = release_box;
    item->second.base_box = release_box;
    item->second.score = score;
    item->second.updated_frame = frame_id;
    item->second.status = ItemStatus::VISIBLE;
    item->second.block_ids.clear();

    track.placed_box = release_box;
    track.has_placed_box = true;
    track.drop_confirmed = emit_moved;
    track.state = OperationTrackState::NORMAL;
    track.contact_state = ContactState::NONE;
    track.resolution = emit_moved
        ? ExistingItemResolution::MOVED_CONFIRMED
        : ExistingItemResolution::POSITION_REBASED;
    track.release_reason = ReleaseReason::NONE;
    track.needs_no_hand_settlement = false;
    track.stable_near_original_no_hand_count = 0;
    track.has_stable_near_original_box = false;
    track.stable_near_original_box = BBox();
    track.b_claim_ambiguous = false;
    track.contact_path_ambiguous = false;
    track.no_hand_candidate_ambiguous = false;
    track.claim_grace_remaining = 0;
    track.no_hand_missing_count = 0;
    track.has_tentative_b_box = false;
    track.tentative_b_match_count = 0;
    track.tentative_b_started_touching_hand = false;
    track.has_reappear_candidate_box = false;
    track.reappear_candidate_match_count = 0;
    track.reappear_candidate_started_touching_hand = false;
    track.reappearance_pending = false;
    track.indirect_move_candidate = false;
    track.indirect_assignment_ambiguous = false;
    track.indirect_no_hand_detection_index = -1;
    track.indirect_no_hand_match_count = 0;
    track.has_indirect_candidate_box = false;
    track.indirect_candidate_box = BBox();

    confirmed_position_update_ids_.insert(track.item_id);
    if (emit_moved) {
        confirmed_moved_ids_.insert(track.item_id);
    } else {
        confirmed_moved_ids_.erase(track.item_id);
    }
    set_live_state_(&track, LiveObservationState::POSSIBLE_MOVED, false,
                    reason ? reason : "position-rebased-after-direct-evidence");
    const char* trace_tag = indirect_move ? "INDIRECT-MOVE" : "STABLE-REBASE";
    trace_(trace_tag,
           "item=%d predecessor=%d cluster=%d action=%s original=(%.1f,%.1f,%.1f,%.1f) "
           "box=(%.1f,%.1f,%.1f,%.1f) "
           "formal-move=%d emit-moved=%d",
           track.item_id, indirect_predecessor_item_id, indirect_cluster_id,
           emit_moved ? "confirm-moved" : "confirm-position-only",
           track.original_box.x1, track.original_box.y1,
           track.original_box.x2, track.original_box.y2,
           release_box.x1, release_box.y1, release_box.x2, release_box.y2,
           formal_move ? 1 : 0, emit_moved ? 1 : 0);
    trace_track_(trace_tag, track,
                 reason ? reason : "position-rebased-after-direct-evidence");
}

void SessionManager::release_not_held_(OperationTrack& track, bool occluded,
                                       ReleaseReason reason,
                                       int evidence_detection_index,
                                       const BBox* evidence_box,
                                       const char* caller) {
    const OperationTrackState old_state = track.state;
    const ContactState old_contact_state = track.contact_state;
    const int old_hold = track.hold_evidence_count;
    const int old_not_hold = track.not_hold_evidence_count;
    const bool had_reappear_candidate = track.has_reappear_candidate_box;
    const int old_stable_near_original_count =
        track.stable_near_original_no_hand_count;
    const bool had_stable_near_original_box =
        track.has_stable_near_original_box;
    // 有手帧里的“仍在原位”只能说明当前这一刻没有移动。手可能在同一
    // 连续操作中随后再次拿起它，因此要保留一个可重新激活的锚点，并等
    // 无手直接帧完成真正静态结算。完整遮挡和无手原位证据才是终态。
    const bool keep_reopen_anchor = !occluded && hand_present_ &&
        (reason == ReleaseReason::ORIGINAL_DETECTION ||
         reason == ReleaseReason::CONTACT_RETURNED_ORIGINAL);
    const BBox reopen_anchor = track.has_last_seen_box
        ? track.last_seen_box : track.original_box;
    if (track.item_id > 0) {
        std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
        if (item != working_inventory_.end()) item->second.status =
            occluded ? ItemStatus::OCCLUDED : ItemStatus::VISIBLE;
        released_hand_candidate_ids_.insert(track.item_id);
    }
    track.state = OperationTrackState::NORMAL;
    track.contact_state = ContactState::NONE;
    track.release_reason = reason;
    track.resolution = reason == ReleaseReason::FULLY_OCCLUDED
        ? ExistingItemResolution::OCCLUDED_CONFIRMED
        : ExistingItemResolution::STATIC_CONFIRMED;
    track.needs_no_hand_settlement = keep_reopen_anchor;
    track.stable_near_original_no_hand_count = 0;
    track.has_stable_near_original_box = false;
    track.stable_near_original_box = BBox();
    track.shelter_or_hold = false;
    track.hold_and_move = false;
    track.hold_evidence_count = 0;
    track.not_hold_evidence_count = 0;
    track.has_reappear_candidate_box = false;
    track.reappear_candidate_match_count = 0;
    track.drop_evidence_count = 0;
    track.reappearance_pending = false;
    track.reappear_candidate_started_touching_hand = false;
    track.b_claim_ambiguous = false;
    track.contact_path_ambiguous = false;
    track.no_hand_candidate_ambiguous = false;
    track.claim_grace_remaining = 0;
    track.has_tentative_b_box = false;
    track.tentative_b_match_count = 0;
    track.tentative_b_started_touching_hand = false;
    track.has_first_hand_block_box = false;
    track.has_last_hand_block_box = false;
    track.move_values.clear();
    track.track.clear();
    track.observed_move_values.clear();
    track.observed_track.clear();
    track.hand_move_values.clear();
    track.contact_started_touching_hand = false;
    track.no_hand_missing_count = 0;
    track.indirect_move_candidate = false;
    track.indirect_assignment_ambiguous = false;
    track.indirect_predecessor_item_id = -1;
    track.indirect_cluster_id = -1;
    track.indirect_no_hand_detection_index = -1;
    track.indirect_no_hand_match_count = 0;
    track.has_indirect_candidate_box = false;
    track.indirect_candidate_box = BBox();
    if (keep_reopen_anchor) {
        track.last_seen_box = reopen_anchor;
        track.has_last_seen_box = reopen_anchor.area() > 0.0f;
        track.hand_estimate_anchor_box = reopen_anchor;
        track.has_hand_estimate_anchor_box = reopen_anchor.area() > 0.0f;
        if (track.has_hand_estimate_anchor_box) track.track.push_back(reopen_anchor);
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
    } else {
        track.has_hand_estimate_anchor_box = false;
        track.hand_track_start_index = -1;
    }
    if (evidence_box) {
        trace_("RELEASE",
               "item=%d old_state=%s old_contact=%s -> NORMAL reason=%s "
               "original_evidence=%d evidence_index=%d evidence_box=(%.1f,%.1f,%.1f,%.1f) "
               "caller=%s hold=%d not_hold=%d candidate_cleared=%d "
               "stable-near-original-before=%d stable-near-original-box=%d provisional=%d",
               track.item_id, operation_track_state_name(old_state),
               contact_state_name(old_contact_state), release_reason_name(reason),
               reason == ReleaseReason::ORIGINAL_DETECTION ||
                       reason == ReleaseReason::CONTACT_RETURNED_ORIGINAL ? 1 : 0,
               evidence_detection_index, evidence_box->x1, evidence_box->y1,
               evidence_box->x2, evidence_box->y2, caller ? caller : "NONE",
               old_hold, old_not_hold,
               had_reappear_candidate ? 1 : 0,
               old_stable_near_original_count,
               had_stable_near_original_box ? 1 : 0,
               keep_reopen_anchor ? 1 : 0);
    } else {
        trace_("RELEASE",
               "item=%d old_state=%s old_contact=%s -> NORMAL reason=%s "
               "original_evidence=0 evidence_index=-1 evidence_box=NONE hold=%d "
               "not_hold=%d caller=%s candidate_cleared=%d "
               "stable-near-original-before=%d stable-near-original-box=%d provisional=%d",
               track.item_id, operation_track_state_name(old_state),
               contact_state_name(old_contact_state), release_reason_name(reason),
               old_hold, old_not_hold, caller ? caller : "NONE",
               had_reappear_candidate ? 1 : 0,
               old_stable_near_original_count,
               had_stable_near_original_box ? 1 : 0,
               keep_reopen_anchor ? 1 : 0);
    }
    trace_track_("STATE", track, keep_reopen_anchor
                 ? "provisional-static-confirmed-awaiting-no-hand"
                 : "released-with-confirmed-resolution");
}

void SessionManager::reset_stable_near_original_no_hand_evidence_(
        OperationTrack* track, const char* reason) {
    if (!track || track->is_suspect_new ||
        (track->stable_near_original_no_hand_count == 0 &&
         !track->has_stable_near_original_box)) {
        return;
    }
    trace_("STATIC-SETTLE",
           "item=%d phase=%s detection=-1 stable-count=%d/%d "
           "action=reset-stable-near-original reason=%s",
           track->item_id, trace_hand_phase_ ? "HAND" : "NO_HAND",
           track->stable_near_original_no_hand_count,
           FLOW3_NO_HAND_D_CONFIRM_FRAMES,
           reason ? reason : "NONE");
    track->stable_near_original_no_hand_count = 0;
    track->has_stable_near_original_box = false;
    track->stable_near_original_box = BBox();
}

bool SessionManager::try_release_stable_near_original_no_hand_(
        OperationTrack* track, int detection_index, const Detection& detection,
        const std::map<int, int>& independent_static_owner_by_detection,
        const char* source) {
    if (!track) return false;

    OperationTrack& c = *track;
    const std::map<int, InventoryItem>::iterator working =
        working_inventory_.find(c.item_id);
    const std::map<int, int>::const_iterator static_owner =
        independent_static_owner_by_detection.find(detection_index);
    const bool operation_start_old_c =
        !c.is_suspect_new && c.item_id > 0 &&
        operation_start_inventory_.count(c.item_id) > 0;
    const bool provisional_static_after_hand = operation_start_old_c &&
        c.state == OperationTrackState::NORMAL &&
        c.contact_state == ContactState::NONE &&
        c.resolution == ExistingItemResolution::STATIC_CONFIRMED &&
        c.needs_no_hand_settlement && c.has_hand_estimate_anchor_box &&
        c.hand_estimate_anchor_box.area() > 0.0f;
    const bool direct_unique_owner = detection_index >= 0 &&
        static_owner != independent_static_owner_by_detection.end() &&
        static_owner->second == c.item_id;
    const bool stable_identity_match =
        strict_match_box(c.cls_id, c.original_box,
                         detection.cls_id, detection.box) &&
        partial_match_box(c.cls_id, c.original_box,
                          detection.cls_id, detection.box,
                          FLOW3_TRACK_PARTIAL_IOM);
    const float center_delta = center_distance(c.original_box, detection.box);
    const float normalized_center_delta =
        normalized_nearby_distance(c.original_box, detection.box);
    const bool formal_move = boxes_differ_as_move(c.original_box, detection.box);
    // 细节15：这不是“位移超过正式门槛就一定 MOVED”。没有直接手持或
    // 间接推动链的旧 C，若自己的最终框连续、唯一、尺度一致，就可以做
    // STABLE-REBASE；它只同步位置，不产生业务 MOVED。
    const bool stable_offset = center_delta > FLOW3_CONTACT_OBJECT_MOVE_EPS;
    const bool hand_or_contact_move_evidence =
        c.hold_and_move || has_meaningful_hand_move(c) ||
        c.contact_state != ContactState::NONE;
    const bool reappearance_claim_or_path_ambiguity =
        c.reappearance_pending || c.b_claim_ambiguous ||
        c.contact_path_ambiguous || c.no_hand_candidate_ambiguous ||
        c.claim_grace_remaining > 0;
    const bool has_prior_stable_box = c.has_stable_near_original_box;
    const bool prior_evidence_consistent =
        (c.stable_near_original_no_hand_count == 0) == !has_prior_stable_box;
    const bool previous_stable_match = !has_prior_stable_box ||
        (strict_match_box(c.cls_id, c.stable_near_original_box,
                          detection.cls_id, detection.box) &&
         partial_match_box(c.cls_id, c.stable_near_original_box,
                           detection.cls_id, detection.box,
                           FLOW3_TRACK_PARTIAL_IOM));
    const bool working_item_is_visible =
        working != working_inventory_.end() &&
        working->second.status == ItemStatus::VISIBLE &&
        !pending_out_ids_.count(c.item_id);

    const char* reason = "eligible";
    if (!provisional_static_after_hand) {
        reason = "not-operation-start-provisional-static-c";
    } else if (!direct_unique_owner) {
        reason = "no-direct-unique-static-owner";
    } else if (!stable_identity_match) {
        reason = "not-stable-identity-match";
    } else if (!stable_offset) {
        reason = "not-stable-offset";
    } else if (hand_or_contact_move_evidence) {
        reason = "hand-or-contact-move-evidence";
    } else if (reappearance_claim_or_path_ambiguity) {
        reason = "reappearance-or-claim-ambiguity";
    } else if (!working_item_is_visible) {
        reason = "missing-visible-working-inventory-item";
    } else if (!prior_evidence_consistent) {
        reason = "inconsistent-previous-static-evidence";
    } else if (!previous_stable_match) {
        reason = "not-continuous-with-previous-static-frame";
    }
    const bool eligible = provisional_static_after_hand && direct_unique_owner &&
        stable_identity_match && stable_offset &&
        !hand_or_contact_move_evidence &&
        !reappearance_claim_or_path_ambiguity && working_item_is_visible &&
        prior_evidence_consistent && previous_stable_match;

    if (!eligible) {
        trace_("STATIC-SETTLE",
               "item=%d phase=NO_HAND detection=%d source=%s "
               "center-distance-px=%.3f normalized-center-distance=%.4f iom=%.4f "
               "width-ratio-diff=%.4f height-ratio-diff=%.4f formal-move=%d "
               "has-real-move-evidence=%d unique-static-owner=%d "
               "previous-stable-present=%d previous-stable-match=%d "
               "stable-count=%d/%d action=reset-stable-near-original reason=%s",
               c.item_id, detection_index, source ? source : "NONE",
               center_delta, normalized_center_delta, iom(c.original_box, detection.box),
               ratio_difference(c.original_box.w(), detection.box.w()),
               ratio_difference(c.original_box.h(), detection.box.h()),
               formal_move ? 1 : 0, hand_or_contact_move_evidence ? 1 : 0,
               direct_unique_owner ? 1 : 0, has_prior_stable_box ? 1 : 0,
               previous_stable_match ? 1 : 0,
               c.stable_near_original_no_hand_count,
               FLOW3_NO_HAND_D_CONFIRM_FRAMES, reason);
        reset_stable_near_original_no_hand_evidence_(track, reason);
        return false;
    }

    ++c.stable_near_original_no_hand_count;
    c.stable_near_original_box = detection.box;
    c.has_stable_near_original_box = true;
    if (c.stable_near_original_no_hand_count <
        FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
        trace_("STATIC-SETTLE",
               "item=%d phase=NO_HAND detection=%d source=%s "
               "center-distance-px=%.3f normalized-center-distance=%.4f iom=%.4f "
               "width-ratio-diff=%.4f height-ratio-diff=%.4f formal-move=%d "
               "has-real-move-evidence=0 unique-static-owner=1 "
               "previous-stable-present=%d previous-stable-match=%d "
               "stable-count=%d/%d action=wait-stable-near-original reason=eligible",
               c.item_id, detection_index, source ? source : "NONE",
               center_delta, normalized_center_delta, iom(c.original_box, detection.box),
               ratio_difference(c.original_box.w(), detection.box.w()),
               ratio_difference(c.original_box.h(), detection.box.h()), formal_move ? 1 : 0,
               has_prior_stable_box ? 1 : 0, previous_stable_match ? 1 : 0,
               c.stable_near_original_no_hand_count,
               FLOW3_NO_HAND_D_CONFIRM_FRAMES);
        return false;
    }

    const int stable_count_before_release = c.stable_near_original_no_hand_count;
    const bool needs_settlement_before_release = c.needs_no_hand_settlement;
    // 不把 unresolved C-D alias 当作这里的阻塞条件：正是 C 用自己的直接
    // 证据完成位置重同步后，r15 才能安全判断 runtime-only D 是否应该回收。
    confirm_position_rebase_(c, detection.box, detection.score, trace_frame_id_, false,
                             "stable-rebase-after-continuous-direct-evidence");
    // 保留 r16 的可追溯 release 原因，供 C-D alias 清理和已有回放断言区分
    // “原位静态 release”与“连续稳定框位置重同步”。
    c.release_reason = ReleaseReason::STABLE_NEAR_ORIGINAL_NO_HAND;
    trace_("STATIC-SETTLE",
           "item=%d phase=NO_HAND detection=%d source=%s "
           "center-distance-px=%.3f normalized-center-distance=%.4f iom=%.4f "
               "width-ratio-diff=%.4f height-ratio-diff=%.4f formal-move=%d "
           "has-real-move-evidence=0 unique-static-owner=1 "
           "previous-stable-present=%d previous-stable-match=%d "
               "stable-count=%d/%d action=confirm-stable-rebase "
               "release-reason=POSITION_REBASED needs-settle-before=%d "
           "needs-settle-after=%d inventory-update=existing-item-only event=none",
           c.item_id, detection_index, source ? source : "NONE",
           center_delta, normalized_center_delta, iom(c.original_box, detection.box),
           ratio_difference(c.original_box.w(), detection.box.w()),
           ratio_difference(c.original_box.h(), detection.box.h()), formal_move ? 1 : 0,
           has_prior_stable_box ? 1 : 0, previous_stable_match ? 1 : 0,
           stable_count_before_release, FLOW3_NO_HAND_D_CONFIRM_FRAMES,
           needs_settlement_before_release ? 1 : 0,
           c.needs_no_hand_settlement ? 1 : 0);
    return true;
}

void SessionManager::advance_claim_grace_(
        const std::set<int>& new_existing_track_ids) {
    // “两张后续有效帧”按真正执行逐帧状态机的帧计数。手框微小且整帧被
    // 跳过时不会来到这里；无手收尾帧也算有效帧，避免手离开后保护期永远
    // 不结束。新建于本帧的 C 不递减，故 t0/t1/t2 均受保护，t3 才成熟。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || track.item_id <= 0 ||
            !is_active_runtime_track(track) ||
            track.claim_grace_remaining <= 0 ||
            new_existing_track_ids.count(track.item_id)) {
            continue;
        }
        --track.claim_grace_remaining;
        if (track.claim_grace_remaining == 0) {
            // 保存下来的本地 B 证据现在可作为正式候选的起点，但真正的
            // 全局认领仍必须由下一张成熟帧的唯一仲裁完成。
            seed_reappear_from_tentative_b(&track);
            printf("[3.0] item#%d 的 B 认领保护期结束，后续可参与唯一仲裁\n",
                   track.item_id);
            trace_track_("STATE", track, "claim-grace-ended");
        }
    }
}

void SessionManager::update_existing_hand_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        const MoveValue& delta, std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
    std::vector<int> promote_keys;
    // 对 HAND_* + hold_and_move=false，静止手帧只能更新局部观测，不能把
    // “还在原位”或“跟手移动”当作新的有效证据。
    const bool hand_moved = move_length(delta) >= TRACK_HAND_MOVE_EPS;
    bool has_hand_visible_suspect = false;
    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        if (it->second.is_suspect_new &&
            it->second.suspect_source == SuspectSource::HAND_VISIBLE_D &&
            (it->second.state == OperationTrackState::HAND_PARTIAL_BLOCKED ||
             it->second.state == OperationTrackState::HAND_FULL_BLOCKED)) {
            has_hand_visible_suspect = true;
            break;
        }
    }
    std::map<int, int> static_owner_by_detection;
    if (has_hand_visible_suspect) {
        // reserve_visible_known_detections_() 在本函数之后才实际写入静态 C 的
        // claim。先在副本上得到完全相同的预约计划，避免已有 HAND_VISIBLE_D
        // 按 map 顺序抢走本应唯一属于静态旧 C 的严格框。
        static_owner_by_detection =
            build_mutually_unique_hand_static_owner_by_detection_(
                hand_box, detections, *claimed_detection_indices, *known_item_owner);
    }
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.state != OperationTrackState::HAND_PARTIAL_BLOCKED &&
            track.state != OperationTrackState::HAND_FULL_BLOCKED) {
            continue;
        }

        const BBox expected = estimated_box(track);
        // 已有库存物品后续仍按 e2/e1 更新当前 HAND 状态；这里使用完整
        // 估计框，而不是 last_hand_block_box 这类局部 YOLO 框。若手已
        // 移开到 r < e2，则保留候选状态等待放下/收尾，不凭这一帧直接释放。
        if (!track.is_suspect_new) {
            if (hand_fully_covers(hand_box, expected)) {
                track.state = OperationTrackState::HAND_FULL_BLOCKED;
            } else if (hand_affects(hand_box, expected)) {
                track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
            }
        }
        int observed_index = unique_detection_for_box(
            detections, *claimed_detection_indices, track.cls_id, expected,
            true, true);
        if (observed_index < 0) {
            observed_index = best_detection_for_box(
                detections, *claimed_detection_indices, track.cls_id, expected,
                true, true);
        }
        if (observed_index < 0 && track.has_last_hand_block_box) {
            const BBox local_expected = move_box(track.last_hand_block_box, delta);
            observed_index = unique_detection_for_box(
                detections, *claimed_detection_indices, track.cls_id, local_expected,
                true, true);
            if (observed_index < 0) {
                observed_index = best_detection_for_box(
                    detections, *claimed_detection_indices, track.cls_id, local_expected,
                    true, true);
            }
            if (observed_index < 0) {
                observed_index = unique_hand_affected_detection_for_box(
                    detections, *claimed_detection_indices, track.cls_id,
                    local_expected, hand_box);
            }
        }
        if (observed_index < 0 && track.has_last_seen_box) {
            observed_index = unique_detection_for_box(
                detections, *claimed_detection_indices, track.cls_id,
                track.last_seen_box, true, true);
            if (observed_index < 0) {
                observed_index = best_detection_for_box(
                    detections, *claimed_detection_indices, track.cls_id,
                    track.last_seen_box, true, true);
            }
        }
        bool observed_matches_reappear_candidate = false;
        if (observed_index < 0 && !track.is_suspect_new &&
            track.has_reappear_candidate_box) {
            observed_index = unique_detection_for_box(
                detections, *claimed_detection_indices, track.cls_id,
                track.reappear_candidate_box, true, true);
            if (observed_index < 0) {
                observed_index = best_detection_for_box(
                    detections, *claimed_detection_indices, track.cls_id,
                    track.reappear_candidate_box, true, true);
            }
            observed_matches_reappear_candidate = observed_index >= 0;
        }
        int old_position_index = unique_detection_at_old_position(
            detections, *claimed_detection_indices, track);
        // CONTACT_* 转入 HAND_* 后，B 仍可能与最初 A.box 有较大 IoM。
        // 普通 HAND_* 的“局部重叠即旧位置”规则在这里会把已经推开的 B
        // 误当成 A 原地未动。因此该分支只接受真正仍贴近 original_box 的
        // 真实框；其他框交给 observed_track / 无手逐帧收尾。
        if (track.has_hand_estimate_anchor_box && old_position_index >= 0 &&
            !contact_detection_is_at_original(
                track, detections[old_position_index])) {
            old_position_index = -1;
        }
        const bool old_clean = old_position_is_clean(detections, track, working_inventory_);

        // HAND_* 的普通候选也必须经过同类成熟 C 的一对一仲裁。此前只在
        // scan 阶段检查这一点，导致 update 按 map 顺序先抢走 B，第二个 C
        // 根本没有机会表达“同样合理”的歧义。
        if (!track.is_suspect_new && !is_claim_protected(track) &&
            observed_index >= 0 &&
            !contact_detection_is_at_original(track, detections[observed_index])) {
            const int mature_owner = unique_c_reappear_owner_for_detection(
                detections[observed_index], track_buffer_, *known_item_owner);
            if (mature_owner == -2) {
                mark_mature_hand_b_ambiguity(detections[observed_index],
                                              &track_buffer_, *known_item_owner);
                observed_index = -1;
                observed_matches_reappear_candidate = false;
                track.reappearance_pending = true;
            } else if (mature_owner >= 0 && mature_owner != track.item_id) {
                observed_index = -1;
                observed_matches_reappear_candidate = false;
                track.reappearance_pending = true;
            } else {
                // 当前 B 没有被另一条成熟 HAND_* 轨迹解释；若此前只是
                // 暂时歧义，本帧的唯一本地匹配可以解除它。
                track.b_claim_ambiguous = false;
            }
        }

        if (track.is_suspect_new) {
            if (observed_index >= 0) {
                const Detection& d = detections[observed_index];
                const std::map<int, int>::const_iterator static_owner =
                    static_owner_by_detection.find(observed_index);
                if (track.suspect_source == SuspectSource::HAND_VISIBLE_D &&
                    static_owner != static_owner_by_detection.end() &&
                    operation_start_inventory_.count(static_owner->second)) {
                    trace_("D-GUARD",
                           "suspect=%d cls=%d action=yield-to-mutually-unique-static-old-c "
                           "detection=%d old-item=%d self-match=%d promote=0 claimed-write=0",
                           track.suspect_id, track.cls_id, observed_index,
                           static_owner->second, track.self_match_count);
                    continue;
                }
                claimed_detection_indices->insert(observed_index);
                if (track.item_id > 0) {
                    (*known_item_owner)[track.item_id] = observed_index;
                }
                ++track.self_match_count;
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                if (hand_touches_detection(hand_box, d.box)) {
                    track.last_hand_block_box = d.box;
                    track.has_last_hand_block_box = true;
                }
                if (!track.promoted_to_working_inventory &&
                    track.self_match_count >= NEW_ITEM_CONFIRM_FRAMES) {
                    promote_keys.push_back(it->first);
                }
                if (track.promoted_to_working_inventory &&
                    !hand_touches_detection(hand_box, d.box)) {
                    track.placed_box = d.box;
                    track.has_placed_box = true;
                    track.drop_confirmed = true;
                    track.state = OperationTrackState::PLACED;
                    std::map<int, InventoryItem>::iterator item =
                        working_inventory_.find(track.item_id);
                    if (item != working_inventory_.end()) {
                        update_seen(item->second, d, trace_frame_id_);
                        item->second.base_box = d.box;
                    }
                }
            }
            continue;
        }

        if (is_claim_protected(track)) {
            // 新建 HAND_* C 在保护期内仍按自己的预计框/局部框寻找 B；但
            // 除“明确仍在旧位置”的检测外，不能抢占 B 的全局归属。
            if (old_position_index >= 0) {
                const Detection& old_d = detections[old_position_index];
                claimed_detection_indices->insert(old_position_index);
                (*known_item_owner)[track.item_id] = old_position_index;
                track.b_claim_ambiguous = false;
                track.has_tentative_b_box = false;
                track.tentative_b_match_count = 0;
                track.tentative_b_started_touching_hand = false;
                track.has_reappear_candidate_box = false;
                track.reappear_candidate_match_count = 0;
                track.reappear_candidate_started_touching_hand = false;
                if (hand_moved) {
                    ++track.not_hold_evidence_count;
                    track.hold_evidence_count = 0;
                }
                track.last_hand_block_box = old_d.box;
                track.has_last_hand_block_box = true;
                if (hand_moved && track.not_hold_evidence_count >=
                    FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                    std::map<int, InventoryItem>::iterator item =
                        working_inventory_.find(track.item_id);
                    if (item != working_inventory_.end()) {
                        update_seen(item->second, old_d, trace_frame_id_);
                    }
                    release_not_held_(track, false,
                                      ReleaseReason::ORIGINAL_DETECTION,
                                      old_position_index,
                                      &old_d.box,
                                      "hand-track-protected-original");
                }
            } else if (observed_index >= 0) {
                const bool touching = hand_touches_detection(
                    hand_box, detections[observed_index].box);
                record_tentative_b(&track, detections[observed_index],
                                   touching);
                if (hand_moved && old_clean &&
                    (touching || track.tentative_b_started_touching_hand)) {
                    ++track.hold_evidence_count;
                    track.not_hold_evidence_count = 0;
                }
                track.reappearance_pending = true;
            } else {
                // 没有本地候选也只是未知，不能在保护期内累积 HOLD/OUT。
                track.reappearance_pending = true;
            }
            continue;
        }

        // C 曾经不可见后重新出现的 B，或已经有过首次 B 的 C，必须先走
        // candidate 自匹配。即使当前 B 刚好落回预计框，也不能因为一帧框
        // 跳动而直接把它认成 C / D。
        const bool should_observe_reappear_candidate = observed_index >= 0 &&
            (track.reappearance_pending || track.has_reappear_candidate_box ||
             observed_matches_reappear_candidate);

        if (track.hold_and_move) {
            if (observed_index >= 0) {
                const Detection& d = detections[observed_index];
                claimed_detection_indices->insert(observed_index);
                if (track.item_id > 0) {
                    (*known_item_owner)[track.item_id] = observed_index;
                }
                BBox previous = track.has_last_seen_box ? track.last_seen_box :
                    (track.has_last_hand_block_box ? track.last_hand_block_box : expected);
                bool candidate_ready = true;
                if (should_observe_reappear_candidate) {
                    if (track.has_reappear_candidate_box) {
                        previous = track.reappear_candidate_box;
                    }
                    candidate_ready = update_reappear_candidate(
                        &track, d, hand_touches_detection(hand_box, d.box));
                }
                if (!candidate_ready) {
                    // 首次 B 仅防止它被登记成 D；不能作为放下或整理的单帧证据。
                    continue;
                }
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                if (hand_touches_detection(hand_box, d.box)) {
                    track.last_hand_block_box = d.box;
                    track.has_last_hand_block_box = true;
                }
                const float hand_distance = move_length(delta);
                const float object_distance = center_distance(previous, d.box);
                const bool falls_behind = hand_distance >= TRACK_HAND_MOVE_EPS &&
                    object_distance <= hand_distance * FLOW3_DROP_FALL_BEHIND_RATIO &&
                    iom(previous, d.box) >= FLOW3_TRACK_PARTIAL_IOM;
                const bool detaches_from_hand = !hand_touches_detection(hand_box, d.box);
                const bool becomes_more_complete = becomes_more_like_complete_box(
                    track, previous, d.box);
                if (falls_behind && (detaches_from_hand || becomes_more_complete)) {
                    ++track.drop_evidence_count;
                } else {
                    track.drop_evidence_count = 0;
                }
                if (track.drop_evidence_count >= FLOW3_DROP_EVIDENCE_REQUIRED) {
                    confirm_rearrange_(track, d.box, d.score, trace_frame_id_);
                }
            } else {
                // B 暂时再次被手挡住时不能凭“看不见”判断放下；但连续放下
                // 证据被打断，下一次重新出现仍要从候选开始确认。
                track.reappearance_pending = true;
                track.drop_evidence_count = 0;
            }
            continue;
        }

        // False 代表“待确认”。先看旧位置，再看估计轨迹；模糊帧不改计数。
        if (old_position_index >= 0) {
            const Detection& old_d = detections[old_position_index];
            claimed_detection_indices->insert(old_position_index);
            if (track.item_id > 0) {
                (*known_item_owner)[track.item_id] = old_position_index;
            }
            track.b_claim_ambiguous = false;
            if (hand_moved) {
                ++track.not_hold_evidence_count;
                track.hold_evidence_count = 0;
            }
            track.last_hand_block_box = old_d.box;
            track.has_last_hand_block_box = true;
            if (hand_moved &&
                track.not_hold_evidence_count >= FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
                if (item != working_inventory_.end()) {
                    update_seen(item->second, old_d, trace_frame_id_);
                }
                release_not_held_(track, false,
                                  ReleaseReason::ORIGINAL_DETECTION,
                                  old_position_index,
                                  &old_d.box,
                                  "hand-track-original");
            }
            continue;
        }

        if (should_observe_reappear_candidate) {
            const Detection& d = detections[observed_index];
            claimed_detection_indices->insert(observed_index);
            if (track.item_id > 0) {
                (*known_item_owner)[track.item_id] = observed_index;
            }
            const bool candidate_ready = update_reappear_candidate(
                &track, d, hand_touches_detection(hand_box, d.box));
            if (candidate_ready) {
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                if (hand_touches_detection(hand_box, d.box)) {
                    track.last_hand_block_box = d.box;
                    track.has_last_hand_block_box = true;
                }
            }
            // 有手首帧 B 必须贴手才可用这条“候选 B 连续自匹配”链路建立
            // hold；否则保留到无手逐帧收尾按更严格的路径规则处理。
            if (hand_moved && candidate_ready &&
                track.reappear_candidate_started_touching_hand &&
                old_clean && !detection_strictly_matches_other_item(
                    d, track.item_id, working_inventory_)) {
                track.hold_evidence_count = std::max(
                    track.hold_evidence_count + 1,
                    track.reappear_candidate_match_count);
                track.not_hold_evidence_count = 0;
            }
            if (hand_moved &&
                track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
                track.hold_and_move = true;
                track.hold_evidence_count = 0;
                track.not_hold_evidence_count = 0;
            }
            continue;
        }

        if (observed_index >= 0 && old_clean &&
            !detection_strictly_matches_other_item(detections[observed_index], track.item_id,
                                                   working_inventory_)) {
            const Detection& d = detections[observed_index];
            claimed_detection_indices->insert(observed_index);
            if (track.item_id > 0) {
                (*known_item_owner)[track.item_id] = observed_index;
            }
            if (hand_moved) {
                ++track.hold_evidence_count;
                track.not_hold_evidence_count = 0;
            }
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;
            if (hand_touches_detection(hand_box, d.box)) {
                track.last_hand_block_box = d.box;
                track.has_last_hand_block_box = true;
            }
            if (hand_moved &&
                track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
                track.hold_and_move = true;
                track.hold_evidence_count = 0;
                track.not_hold_evidence_count = 0;
            }
        } else if (observed_index >= 0) {
            // 该框不能证明移动，但可作为下一帧局部连续性的参考。
            track.last_hand_block_box = detections[observed_index].box;
            track.has_last_hand_block_box = true;
        } else {
            track.reappearance_pending = true;
            // HAND_FULL + hold_and_move=False 时，没有可直接使用的 B。只有
            // 原位置的 C、已确认遮挡物、手仍覆盖原位置、或干净的旧位置
            // 离开这四种情形能改变证据；其余一律保持模糊。
            if (track.state == OperationTrackState::HAND_FULL_BLOCKED) {
                if (confirmed_blocker_covers_old_c(
                        track, detections, *known_item_owner, track_buffer_)) {
                    if (hand_moved) {
                        ++track.not_hold_evidence_count;
                        track.hold_evidence_count = 0;
                    }
                    if (hand_moved && track.not_hold_evidence_count >=
                        FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                        // 已确认的 blocker 只能说明 C 可能仍在旧位置被遮住，
                        // 不能替代“C 原位置有独立检测”的静态结论。保留 C
                        // 的轨迹和无手结算资格，避免之后把同一实体登记为 D。
                        trace_track_("STATE", track,
                                     "confirmed-blocker-keeps-unresolved");
                    }
                } else if (hand_affects(hand_box, track.original_box)) {
                    // 手仍盖在 C 原位置：既不能说明拿起，也不能说明没拿起。
                } else if (!any_detection_at_old_position(detections, track)) {
                    if (hand_moved) {
                        ++track.hold_evidence_count;
                        track.not_hold_evidence_count = 0;
                    }
                    if (hand_moved &&
                        track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
                        track.hold_and_move = true;
                        track.hold_evidence_count = 0;
                        track.not_hold_evidence_count = 0;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < promote_keys.size(); ++i) {
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(promote_keys[i]);
        if (track == track_buffer_.end() || !track->second.has_last_seen_box) continue;
        Detection d;
        d.box = track->second.last_seen_box;
        d.cls_id = track->second.cls_id;
        d.score = 0.0f;
        promote_suspect_(promote_keys[i], d, trace_frame_id_);
    }
}

void SessionManager::reopen_released_static_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        const MoveValue& delta) {
    // 有手阶段已确认“目前仍在原位”的 C 不是一次操作的最终结论。若同一
    // 只手随后发生有效位移，并且本帧没有 C 自己的唯一原位检测，就把它
    // 恢复为 HAND_*，从最后一个可靠原位框重新开始记录手部路径。
    if (move_length(delta) < TRACK_HAND_MOVE_EPS) return;

    const std::set<int> no_claimed_detections;
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || track.item_id <= 0 ||
            track.state != OperationTrackState::NORMAL ||
            track.contact_state != ContactState::NONE ||
            !track.needs_no_hand_settlement ||
            track.resolution != ExistingItemResolution::STATIC_CONFIRMED) {
            continue;
        }
        if (unique_detection_at_old_position(detections, no_claimed_detections,
                                             track) >= 0) {
            trace_track_("STATE", track,
                         "provisional-static-kept-by-current-original-detection");
            continue;
        }

        const BBox anchor = track.has_last_seen_box ? track.last_seen_box : track.original_box;
        if (anchor.area() <= 0.0f) {
            trace_track_("STATE", track,
                         "cannot-reopen-static-track-without-valid-anchor");
            continue;
        }

        released_hand_candidate_ids_.erase(track.item_id);
        track.state = hand_fully_covers(hand_box, move_box(anchor, delta))
            ? OperationTrackState::HAND_FULL_BLOCKED
            : OperationTrackState::HAND_PARTIAL_BLOCKED;
        track.contact_state = ContactState::NONE;
        track.resolution = ExistingItemResolution::NONE;
        track.release_reason = ReleaseReason::NONE;
        track.needs_no_hand_settlement = true;
        track.shelter_or_hold = true;
        track.hold_and_move = false;
        track.hold_evidence_count = 0;
        track.not_hold_evidence_count = 0;
        track.drop_confirmed = false;
        track.drop_evidence_count = 0;
        track.claim_grace_remaining = 0;
        track.has_tentative_b_box = false;
        track.tentative_b_match_count = 0;
        track.tentative_b_started_touching_hand = false;
        track.has_reappear_candidate_box = false;
        track.reappear_candidate_match_count = 0;
        track.reappear_candidate_started_touching_hand = false;
        track.reappearance_pending = true;
        track.contact_path_ambiguous = false;
        track.b_claim_ambiguous = false;
        track.no_hand_candidate_ambiguous = false;
        track.no_hand_missing_count = 0;
        track.last_seen_box = anchor;
        track.has_last_seen_box = true;
        track.first_hand_block_box = anchor;
        track.last_hand_block_box = anchor;
        track.has_first_hand_block_box = true;
        track.has_last_hand_block_box = true;
        track.hand_estimate_anchor_box = anchor;
        track.has_hand_estimate_anchor_box = true;
        track.move_values.clear();
        track.track.clear();
        track.observed_move_values.clear();
        track.observed_track.clear();
        track.hand_move_values.clear();
        track.track.push_back(anchor);
        track.move_values.push_back(delta);
        track.track.push_back(estimated_box(track));
        track.hand_track_start_index = std::max(
            0, static_cast<int>(hand_track_.size()) - 2);
        trace_("STATE",
               "item=%d reopen-after-static-release-lost-original anchor=(%.1f,%.1f,%.1f,%.1f) "
               "delta=(%.1f,%.1f) expected=(%.1f,%.1f,%.1f,%.1f)",
               track.item_id, anchor.x1, anchor.y1, anchor.x2, anchor.y2,
               delta.dx, delta.dy, estimated_box(track).x1, estimated_box(track).y1,
               estimated_box(track).x2, estimated_box(track).y2);
        trace_track_("STATE", track, "reopen-after-static-release-lost-original");
    }
}

void SessionManager::scan_or_update_suspects_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        const std::map<int, int>& known_item_owner,
        bool /*first_hand_frame*/) {
    // scan 内也会把 B 暂认给 C。复制一份本帧归属，避免同一个尚未确认的 C
    // 在同一帧连续抢走两个同类框；外部在 scan 后不再需要这个临时映射。
    std::map<int, int> effective_known_item_owner = known_item_owner;
    for (size_t di = 0; di < detections.size(); ++di) {
        if (claimed_detection_indices->count(static_cast<int>(di))) continue;
        const Detection& d = detections[di];
        const bool hand_visible_d = hand_touches_detection(hand_box, d.box);
        // D 已经被放到 C 原位置时，手可能继续移开，因此 D 不一定还贴手。
        // 只要它是唯一一个覆盖“当前看不见的 C”原位置的未认领框，也必须
        // 预登记；否则 C 会在后续无手阶段被错误当成 OUT。
        const int replacement_owner = unique_c_replacement_owner_for_detection(
            d, track_buffer_, effective_known_item_owner);
        if (!hand_visible_d && replacement_owner < 0) {
            continue;
        }

        // 先排除普通静态库存、已放下的旧 C。正在 HAND_* 的旧 C 不在这里
        // 用宽松局部匹配直接吞框：同类贴手 B 应由下面的 candidate 规则
        // 唯一归属，不能因为帧间跳动直接落进 D。
        bool belongs_to_known_item = false;
        for (std::map<int, InventoryItem>::const_iterator it = working_inventory_.begin();
             it != working_inventory_.end(); ++it) {
            // 这个旧 item 已经在本帧认领了另一个检测框。当前 D 即使与
            // 它的旧框局部相似，也不能再次借用同一身份；否则相邻同类
            // 物品会永远进不了疑似 D 链路。
            if (effective_known_item_owner.count(it->first) &&
                effective_known_item_owner.find(it->first)->second !=
                    static_cast<int>(di)) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(it->first);
            if (runtime && (is_active_existing_hand_track(*runtime) ||
                            is_active_contact_track(*runtime))) continue;
            const BBox reference = it->second.base_box.area() > 0.0f
                ? it->second.base_box : it->second.box;
            if (strict_match(it->second, d) || partial_match(it->second, d) ||
                hand_partial_match_box(it->second.cls_id, reference, d.cls_id, d.box)) {
                belongs_to_known_item = true;
                break;
            }
        }
        if (!belongs_to_known_item) {
            for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
                 it != track_buffer_.end(); ++it) {
                const OperationTrack& track = it->second;
                if (!track.is_suspect_new || track.cls_id != d.cls_id ||
                    track.state == OperationTrackState::NORMAL) {
                    continue;
                }
                if (track.item_id > 0 && effective_known_item_owner.count(track.item_id) &&
                    effective_known_item_owner.find(track.item_id)->second !=
                        static_cast<int>(di)) {
                    continue;
                }
                const bool matches_estimate =
                    track_match_box(track.cls_id, estimated_box(track), d.cls_id, d.box);
                const bool matches_last_seen = track.has_last_seen_box &&
                    track_match_box(track.cls_id, track.last_seen_box, d.cls_id, d.box);
                const bool matches_last_hand_part = track.has_last_hand_block_box &&
                    (track_match_box(track.cls_id, track.last_hand_block_box, d.cls_id, d.box) ||
                     hand_partial_match_box(track.cls_id, track.last_hand_block_box,
                                            d.cls_id, d.box));
                if (matches_estimate || matches_last_seen || matches_last_hand_part) {
                    belongs_to_known_item = true;
                    break;
                }
            }
        }
        if (belongs_to_known_item) continue;

        // 细节5的关键优先级：一个手边同类 B 若只有一个无自身检测的活动 C
        // 可以解释，就先记录为 C 的 reappear_candidate，而不是新 D。若多个
        // C 都合理，宁可保持未决，也绝不因为 map 顺序生成一条 D 链路。
        if (hand_visible_d) {
            const int c_owner = unique_c_reappear_owner_for_detection(
                d, track_buffer_, effective_known_item_owner);
            if (c_owner == -2) {
                mark_mature_hand_b_ambiguity(d, &track_buffer_,
                                              effective_known_item_owner);
                printf("[3.0] 同类贴手 B cls=%d 可属于多个 HAND_* C，保持未决\n",
                       d.cls_id);
                continue;
            }
            if (c_owner >= 0) {
                std::map<int, OperationTrack>::iterator c = track_buffer_.find(c_owner);
                if (c != track_buffer_.end()) {
                    c->second.b_claim_ambiguous = false;
                    if (c->second.has_reappear_candidate_box &&
                        track_match_box(c->second.cls_id,
                                        c->second.reappear_candidate_box,
                                        d.cls_id, d.box)) {
                        const bool candidate_ready =
                            update_reappear_candidate(&c->second, d, true);
                        trace_("MATCH",
                               "item=%d detection=%zu source=hand-reappear update=1 count=%d ready=%d",
                               c->second.item_id, di,
                               c->second.reappear_candidate_match_count,
                               candidate_ready ? 1 : 0);
                    } else {
                        start_reappear_candidate(&c->second, d, true);
                        trace_("MATCH",
                               "item=%d detection=%zu source=hand-reappear start=1 count=%d",
                               c->second.item_id, di,
                               c->second.reappear_candidate_match_count);
                    }
                    claimed_detection_indices->insert(static_cast<int>(di));
                    effective_known_item_owner[c->second.item_id] =
                        static_cast<int>(di);
                    printf("[3.0] item#%d 将同类贴手 B 暂记为重新出现候选"
                           "（等待自匹配）\n", c->second.item_id);
                    continue;
                }
            }
        }

        // 保护期内的旧 C 仍可做本地路径匹配，但不能把 B 变成排他归属。
        // 因而只要存在同类、尚未解决的保护期 C，本帧就暂不创建 D；能接上
        // 自己路径的 C 同时保存 tentative B，之后由成熟轨迹重新仲裁。
        bool deferred_by_protected_existing = false;
        for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            OperationTrack& c = it->second;
            if (!is_claim_protected(c) || c.cls_id != d.cls_id ||
                effective_known_item_owner.count(c.item_id)) {
                continue;
            }
            const bool path_matches = detection_can_belong_to_active_track(d, c);
            if (path_matches) {
                record_tentative_b(&c, d, hand_visible_d);
            }
            // hand_visible_d 是“当前手操作正在接触这个同类框”的强上下文；
            // 即使框跳动暂时脱离路径，也先留在未决池，避免 t0/t1 误建 D。
            if (path_matches || hand_visible_d || replacement_owner >= 0) {
                deferred_by_protected_existing = true;
            }
        }
        if (deferred_by_protected_existing) {
            printf("[3.0] 同类 B cls=%d 仍处于旧 C 保护期，暂不建立 D\n",
                   d.cls_id);
            continue;
        }

        // CONTACT_* 的 B 即使本帧因框跳动没有被 update 认领，也不能直接
        // 进入 D。先保留为旧物品的未决解释，等待下一帧真实观测。
        bool belongs_to_contact_track = false;
        for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            if (is_active_contact_track(it->second) &&
                is_claim_mature(it->second) &&
                detection_can_belong_to_active_track(d, it->second)) {
                belongs_to_contact_track = true;
                break;
            }
        }
        if (belongs_to_contact_track) continue;

        // 细节11：旧 C 即使本帧已经临时认领了另一个同类框，也不等于
        // 它已经结算。此时第二个未认领框必须实时建成 D 轨迹，但只要 C
        // 仍是活动的操作开始旧库存，就将 D 隔离为 C-D alias，禁止它在
        // 有手阶段取得 working_inventory_ 的排他身份。
        std::set<int> alias_old_item_ids;
        for (std::map<int, InventoryItem>::const_iterator old =
                 operation_start_inventory_.begin();
             old != operation_start_inventory_.end(); ++old) {
            if (old->second.cls_id != d.cls_id ||
                !working_inventory_.count(old->first) ||
                pending_in_ids_.count(old->first)) {
                continue;
            }
            std::map<int, OperationTrack>::const_iterator runtime =
                track_buffer_.find(old->first);
            if (runtime == track_buffer_.end() ||
                !is_unresolved_operation_start_old_track(runtime->second) ||
                !is_active_runtime_track(runtime->second)) {
                continue;
            }
            std::map<int, int>::const_iterator owner =
                effective_known_item_owner.find(old->first);
            // 仅覆盖本次实机漏洞的窄分支：C 已经临时拥有另一个检测框，
            // 当前 d 仍未认领。没有任何 owner 的 C 继续沿既有保护/未决
            // 逻辑处理，避免改变已验证的同类相邻和 CONTACT_* 路径。
            if (owner == effective_known_item_owner.end() ||
                owner->second == static_cast<int>(di)) {
                continue;
            }
            alias_old_item_ids.insert(old->first);
        }
        if (!alias_old_item_ids.empty()) {
            OperationTrack track;
            const int key = new_suspect_id_();
            track.suspect_id = key;
            track.is_suspect_new = true;
            track.pending_d_quarantined_by_old_c = true;
            track.conflicting_old_item_ids = alias_old_item_ids;
            track.suspect_source = replacement_owner >= 0
                ? SuspectSource::C_POSITION_REPLACEMENT_D
                : SuspectSource::HAND_VISIBLE_D;
            track.cls_id = d.cls_id;
            track.original_box = d.box;
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;
            if (hand_visible_d) {
                track.first_hand_block_box = d.box;
                track.last_hand_block_box = d.box;
                track.has_first_hand_block_box = true;
                track.has_last_hand_block_box = true;
            }
            track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
            track.shelter_or_hold = true;
            track.self_match_count = 1;
            track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
            track.track.push_back(d.box);
            track_buffer_[key] = track;
            for (std::set<int>::const_iterator old_id = alias_old_item_ids.begin();
                 old_id != alias_old_item_ids.end(); ++old_id) {
                std::map<int, OperationTrack>::iterator old = track_buffer_.find(*old_id);
                if (old != track_buffer_.end()) {
                    old->second.conflicting_suspect_keys.insert(key);
                    trace_("C-D-ALIAS",
                           "old-item=%d suspect=%d detection=%zu phase=HAND "
                           "relation=shared-or-unresolved action=create "
                           "reason=same-class-unsettled-old-even-if-current-owner-exists",
                           *old_id, key, di);
                }
            }
            claimed_detection_indices->insert(static_cast<int>(di));
            trace_("D-GUARD",
                   "candidate=%zu cls=%d action=quarantine-pending-d suspect=%d "
                   "conflicting-old-count=%zu reason=unresolved-same-class-old-even-if-current-owner-exists "
                   "hand-touching=%d formal-owner-authority=0",
                   di, d.cls_id, key, alias_old_item_ids.size(),
                   hand_visible_d ? 1 : 0);
            trace_track_("STATE", track_buffer_[key],
                         "create-quarantined-hand-visible-suspect");
            continue;
        }

        // 即使当前成熟 C 的宽松路径暂时没有命中，其他同类旧 C 仍可能
        // 正在等待自己的原位置/轨迹证据。此时 B 只能进未决池，不能利用
        // “没有候选”这一瞬间伪造 D；只有同类旧 C 都有独立归属或已明确
        // 结束后，才允许开始新 D 链路。
        bool unresolved_same_class_old = false;
        for (std::map<int, InventoryItem>::const_iterator old =
                 working_inventory_.begin(); old != working_inventory_.end(); ++old) {
            if (old->second.cls_id != d.cls_id ||
                pending_in_ids_.count(old->first) ||
                old->second.status == ItemStatus::OCCLUDED ||
                effective_known_item_owner.count(old->first)) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(old->first);
            if (runtime && runtime->is_suspect_new) continue;
            if (runtime && runtime->state == OperationTrackState::PLACED) continue;
            if (runtime && is_active_runtime_track(*runtime)) {
                unresolved_same_class_old = true;
                break;
            }
            // 没有运行时轨迹但本轮没有独立 owner，说明普通静态 C 也尚未
            // 解决；保守地等待下一帧，而不是把同类 B 计成数量增长。
            unresolved_same_class_old = true;
            break;
        }
        if (unresolved_same_class_old) {
            printf("[3.0] 同类旧 C 尚未得到独立归属，B cls=%d 进入未决池\n",
                   d.cls_id);
            continue;
        }

        // 如果没有在 update_existing_hand_tracks_ 中找到，是一个真正新的 D。
        OperationTrack track;
        const int key = new_suspect_id_();
        track.suspect_id = key;
        track.is_suspect_new = true;
        track.suspect_source = replacement_owner >= 0
            ? SuspectSource::C_POSITION_REPLACEMENT_D
            : SuspectSource::HAND_VISIBLE_D;
        track.cls_id = d.cls_id;
        track.original_box = d.box;
        track.last_seen_box = d.box;
        track.has_last_seen_box = true;
        // C 原位置替代 D 可能已不贴手。此时它仍有 last_seen_box 和候选
        // 轨迹，但不伪造“被手遮挡时的局部框”。
        if (hand_visible_d) {
            track.first_hand_block_box = d.box;
            track.last_hand_block_box = d.box;
            track.has_first_hand_block_box = true;
            track.has_last_hand_block_box = true;
        }
        track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
        track.shelter_or_hold = true;
        track.self_match_count = 1;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track.track.push_back(d.box);
        track_buffer_[key] = track;
        claimed_detection_indices->insert(static_cast<int>(di));
        printf("[3.0] 预登记疑似新物品 D suspect#%d cls=%d source=%s\n",
               key, d.cls_id, suspect_source_name(track.suspect_source));
        trace_track_("STATE", track_buffer_[key], "create-hand-visible-suspect");
    }
}

void SessionManager::mark_newly_hand_blocked_items_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        std::set<int>* new_existing_track_ids) {
    // 已经确认“这次没有被拿起”的物品，直到手真正离开前都不重复入 HAND_*。
    for (std::set<int>::iterator it = released_hand_candidate_ids_.begin();
         it != released_hand_candidate_ids_.end();) {
        std::map<int, InventoryItem>::const_iterator item = working_inventory_.find(*it);
        if (item == working_inventory_.end() || !hand_affects(hand_box, item->second.box)) {
            it = released_hand_candidate_ids_.erase(it);
        } else {
            ++it;
        }
    }

    for (std::map<int, InventoryItem>::iterator it = working_inventory_.begin();
         it != working_inventory_.end(); ++it) {
        InventoryItem& item = it->second;
        OperationTrack* existing = find_runtime_for_item_(item.item_id);
        if (existing && is_active_runtime_track(*existing) &&
            existing->state != OperationTrackState::PLACED) {
            continue;
        }
        if (item.status == ItemStatus::OCCLUDED ||
            released_hand_candidate_ids_.count(item.item_id)) continue;
        // reserve_visible_known_detections_ 已经在本帧为它保留了自己的
        // 严格框。它仍可在后续帧因漏检重新进入 HAND_*，但不能在这一帧
        // 把旁边的新 D 局部框认成自己的遮挡框。
        if (known_item_owner->count(item.item_id)) continue;

        const BBox reference = item.base_box.area() > 0.0f ? item.base_box : item.box;
        const bool full = hand_fully_covers(hand_box, reference);
        const bool partial = !full && hand_affects(hand_box, reference);
        // e2 <= r < e1 即使当前没有 A 的 YOLO 框，也必须进入
        // HAND_PARTIAL；是否有局部框只决定能否更新 last_hand_block_box。
        if (!partial && !full) continue;

        int observed_index = unique_hand_affected_detection_for_box(
            detections, *claimed_detection_indices, item.cls_id, reference, hand_box);
        if (observed_index < 0) {
            observed_index = best_hand_affected_detection_for_box(
                detections, *claimed_detection_indices, item.cls_id, reference, hand_box);
        }
        // 同一局部框若同时合理地属于两个同类旧库存，不能按 map 顺序硬认领。
        // 保持它未决；后续 scan 也会看到它“可能属于旧库存”，因此不会误建 D。
        if (observed_index >= 0 &&
            hand_affected_existing_candidate_count(working_inventory_,
                                                    detections[observed_index], hand_box) != 1) {
            observed_index = -1;
        }

        OperationTrack track;
        track.item_id = item.item_id;
        track.cls_id = item.cls_id;
        track.original_box = reference;
        track.needs_no_hand_settlement = true;
        track.shelter_or_hold = true;
        track.claim_grace_remaining = FLOW3_NEW_TRACK_CLAIM_GRACE_FRAMES;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track.track.push_back(track.original_box);
        track.state = full ? OperationTrackState::HAND_FULL_BLOCKED
                           : OperationTrackState::HAND_PARTIAL_BLOCKED;
        // 首帧没有可靠 C 框时，下一次同类 B 即使落回预计位置，也必须先
        // 建立重新出现候选并等待自匹配，不能一帧确认移动。
        track.reappearance_pending = observed_index < 0;
        if (observed_index >= 0) {
            const Detection& d = detections[observed_index];
            if (contact_detection_is_at_original(track, d) ||
                !is_claim_protected(track)) {
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                track.first_hand_block_box = d.box;
                track.last_hand_block_box = d.box;
                track.has_first_hand_block_box = true;
                track.has_last_hand_block_box = true;
                claimed_detection_indices->insert(observed_index);
                (*known_item_owner)[item.item_id] = observed_index;
            } else {
                // 首帧已跑离旧位置的同类框只能记作本地 B；两帧保护期内
                // 不得以它阻止成熟 C 或真正 D 的仲裁。
                record_tentative_b(&track, d, hand_touches_detection(hand_box, d.box));
                track.reappearance_pending = true;
            }
        }
        // 已经放下过又再次被手接触时，覆盖旧运行时记录即可。
        for (std::map<int, OperationTrack>::iterator rt = track_buffer_.begin();
             rt != track_buffer_.end();) {
            if (rt->second.item_id == item.item_id) {
                trace_track_("STATE", rt->second,
                             "replace-runtime-with-new-hand-blocked-track");
                rt = track_buffer_.erase(rt);
            } else {
                ++rt;
            }
        }
        // C 重新被手接触时会重建自己的 runtime；仍在隔离中的 D 保留了
        // D -> C 的关系，必须用它恢复新 C -> D 的反向索引。否则后续同一
        // 无手框只会被 C 看到，D 会被错误当作“缺失的 stale alias”。
        for (std::map<int, OperationTrack>::const_iterator suspect =
                 track_buffer_.begin(); suspect != track_buffer_.end(); ++suspect) {
            if (!suspect->second.is_suspect_new ||
                !suspect->second.pending_d_quarantined_by_old_c ||
                !suspect->second.conflicting_old_item_ids.count(item.item_id)) {
                continue;
            }
            track.conflicting_suspect_keys.insert(suspect->first);
            trace_("C-D-ALIAS",
                   "old-item=%d suspect=%d phase=HAND relation=restored "
                   "action=rebind-after-retouch",
                   item.item_id, suspect->first);
        }
        track_buffer_[item.item_id] = track;
        if (new_existing_track_ids) new_existing_track_ids->insert(item.item_id);
        trace_track_("STATE", track_buffer_[item.item_id],
                     "create-hand-blocked-existing-item");
    }
}

void SessionManager::apply_suspect_cover_evidence_(
        const BBox& hand_box, const std::vector<Detection>& /*detections*/,
        bool hand_moved) {
    for (std::map<int, OperationTrack>::const_iterator dit = track_buffer_.begin();
         dit != track_buffer_.end(); ++dit) {
        const OperationTrack& d = dit->second;
        if (!d.is_suspect_new ||
            (!d.has_last_seen_box && !d.has_last_hand_block_box)) {
            continue;
        }
        // C_POSITION_REPLACEMENT_D 可能已经离开手框，只有 last_seen_box，
        // 不能再把 last_hand_block_box 当成 D 唯一可用的遮挡位置。
        const BBox d_box = d.has_placed_box ? d.placed_box :
            (d.has_last_seen_box ? d.last_seen_box : d.last_hand_block_box);
        for (std::map<int, OperationTrack>::iterator cit = track_buffer_.begin();
             cit != track_buffer_.end(); ++cit) {
            OperationTrack& c = cit->second;
            if (c.is_suspect_new || c.state == OperationTrackState::NORMAL ||
                c.state == OperationTrackState::PLACED || c.original_box.area() <= 0.0f) {
                continue;
            }
            std::vector<BBox> covers;
            covers.push_back(hand_box);
            covers.push_back(d_box);
            const bool union_covers_c = fully_covered_by(c.original_box, covers);
            const bool d_covers_c = fully_covered_by(c.original_box,
                                                      std::vector<BBox>(1, d_box));
            if (!union_covers_c) continue;
            // D + 手的并集只能在手实际移动的有效帧中构成 C 的反向证据。
            // 静止手帧仍属于 HAND_* 的模糊帧，不能改变 hold/not_hold 或释放 C。
            if (!hand_moved) continue;
            // 尚未连续确认并放下的 D 只是“可能挡住 C”的解释，不能改变 C
            // 的 hold/not-hold 计数；否则一次误检又会把 C 提前释放。
            if (!d.promoted_to_working_inventory || !d.drop_confirmed) continue;
            c.hold_evidence_count = 0;
            ++c.not_hold_evidence_count;
            if (d_covers_c) {
                std::map<int, InventoryItem>::iterator item = working_inventory_.find(c.item_id);
                if (item != working_inventory_.end()) {
                    item->second.status = ItemStatus::OCCLUDED;
                    item->second.block_ids.insert(d.item_id);
                }
                release_not_held_(c, true, ReleaseReason::FULLY_OCCLUDED,
                                  -1, nullptr,
                                  "hand-confirmed-full-occlusion");
            }
        }
    }
}

void SessionManager::process_effective_hand_frame_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        bool first_hand_frame) {
    MoveValue delta;
    if (!first_hand_frame) {
        delta.dx = hand_box.cx() - old_hand_box_.cx();
        delta.dy = hand_box.cy() - old_hand_box_.cy();
        hand_track_.push_back(hand_box);
        append_move_to_existing_hand_tracks_(delta);
    }
    const bool hand_moved = !first_hand_frame &&
        move_length(delta) >= TRACK_HAND_MOVE_EPS;

    // 无手阶段的间接成员必须在连续直接帧里确认；手重新出现会打断这条
    // 连续性。保留 operation-start 身份和既有直接轨迹，但不能把旧的首帧
    // 证据拼接到下一次无手阶段。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || !track.indirect_move_candidate ||
            existing_track_is_terminal(track)) {
            continue;
        }
        if (track.indirect_no_hand_match_count > 0 ||
            track.has_indirect_candidate_box) {
            trace_("INDIRECT-MOVE",
                   "item=%d action=reset-no-hand-continuity reason=hand-returned",
                   track.item_id);
        }
        track.indirect_no_hand_detection_index = -1;
        track.indirect_no_hand_match_count = 0;
        track.has_indirect_candidate_box = false;
        track.indirect_candidate_box = BBox();
        track.indirect_assignment_ambiguous = false;
    }

    // 先更新已有 HAND_*；随后先为仍稳定可见的旧库存保留本帧自己的
    // 严格框，再处理新进入 HAND_* 的物品。这个先后顺序很关键：若旧
    // 苹果已有完整框，旁边贴手的新苹果不能先被“局部可能属于旧苹果”
    // 的规则抢走。
    std::set<int> claimed;
    std::map<int, int> known_item_owner;
    std::set<int> new_existing_track_ids;
    if (!first_hand_frame) {
        // 低覆盖率 CONTACT_* 先用物品真实检测框更新；这里不能使用手位移
        // 推算的 estimated_box。
        update_existing_contact_tracks_(hand_box, detections, &claimed,
                                        &known_item_owner);
        update_existing_hand_tracks_(hand_box, detections, delta, &claimed,
                                     &known_item_owner);
        // 原位暂时确认后的 C 若在本帧再次随手离开原位置，必须先恢复
        // HAND_* 身份链，再让静态预约或 D 扫描处理剩余检测框。
        reopen_released_static_tracks_(hand_box, detections, delta);
    }
    reserve_visible_known_detections_(hand_box, detections, &claimed,
                                      &known_item_owner);
    mark_new_contact_candidates_(hand_box, detections, &claimed,
                                 &known_item_owner, &new_existing_track_ids);
    mark_newly_hand_blocked_items_(hand_box, detections, &claimed,
                                   &known_item_owner, &new_existing_track_ids);
    scan_or_update_suspects_(hand_box, detections, &claimed, known_item_owner,
                             first_hand_frame);
    apply_suspect_cover_evidence_(hand_box, detections, hand_moved);
    advance_claim_grace_(new_existing_track_ids);
    // 有手阶段每张有效帧都必须维护实时观察层。这里不改变库存或事件，
    // 只把 C/D/alias 当前的可撤销结论写入运行时记录和调试日志。
    update_hand_live_states_();
}

void SessionManager::register_post_hand_reveal_suspects_(
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices) {
    // 首次完整 B 必须紧接着本次手操作出现。若允许很久之后的任意无手框
    // 触发，就会把普通漏检重现或静态误检错误写成 IN。
    if (no_hand_streak_ <= 0 ||
        no_hand_streak_ > FLOW3_POST_HAND_REVEAL_WINDOW_FRAMES) {
        return;
    }

    for (size_t di = 0; di < detections.size(); ++di) {
        const int detection_index = static_cast<int>(di);
        if (claimed_detection_indices->count(detection_index)) continue;
        const Detection& d = detections[di];

        // 先完成旧 C、已有 D 的认领。剩余 B 若能合理属于任意旧库存或
        // 活动轨迹，也必须保持未决，不能另建一个“全程被挡住的新 D”。
        const bool matches_old_inventory = detection_matches_old_working_inventory(
            d, working_inventory_, operation_start_inventory_);
        const bool conflicts_active_track =
            detection_conflicts_with_active_track(d, track_buffer_);
        const bool blocked_by_protected_track =
            protected_existing_track_blocks_post_hand_d(d, track_buffer_);
        if (matches_old_inventory || conflicts_active_track ||
            blocked_by_protected_track) {
            trace_("D-GUARD",
                   "candidate=%d cls=%d reject=known-old:%d active-track:%d protected:%d",
                   detection_index, d.cls_id, matches_old_inventory ? 1 : 0,
                   conflicts_active_track ? 1 : 0,
                   blocked_by_protected_track ? 1 : 0);
            continue;
        }
        if (!hand_track_touches_detection(hand_track_, d)) {
            trace_("D-GUARD", "candidate=%d cls=%d reject=outside-hand-track",
                   detection_index, d.cls_id);
            continue;
        }

        // 同类旧 C 若在本轮仍没有自己的独立检测/轨迹结论，手离开后
        // 首次出现的 B 也只能保持未决。否则 C 的一次漏检会被误写成
        // POST_HAND_REVEAL_D，库存数量会逐次膨胀。
        bool unresolved_same_class_old = false;
        for (std::map<int, InventoryItem>::const_iterator old =
                 working_inventory_.begin(); old != working_inventory_.end(); ++old) {
            if (old->second.cls_id != d.cls_id ||
                pending_in_ids_.count(old->first) ||
                old->second.status == ItemStatus::OCCLUDED) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(old->first);
            if (runtime && runtime->is_suspect_new) continue;
            if (existing_item_resolved_without_current_detection(runtime)) {
                trace_("D-GUARD",
                       "candidate=%d cls=%d old=%d disposition=terminal-%s",
                       detection_index, d.cls_id, old->first,
                       existing_resolution_name(runtime->resolution));
                continue;
            }
            bool independent_detection = false;
            for (size_t oi = 0; oi < detections.size(); ++oi) {
                if (detections[oi].cls_id != old->second.cls_id ||
                    detections[oi].box.area() <= 0.0f ||
                    center_distance(detections[oi].box, d.box) <= 4.0f) {
                    continue;
                }
                if (strict_match_box(old->second.cls_id, old->second.box,
                                     detections[oi].cls_id, detections[oi].box) ||
                    partial_match_box(old->second.cls_id, old->second.box,
                                      detections[oi].cls_id, detections[oi].box)) {
                    independent_detection = true;
                    break;
                }
            }
            const bool runtime_ambiguous = runtime &&
                (runtime->b_claim_ambiguous || runtime->contact_path_ambiguous ||
                 runtime->no_hand_candidate_ambiguous);
            // 即使 runtime 指针存在，只要它只是 NORMAL 且没有独立静态框，
            // 也不能把它当成已解决。STATIC_CONFIRMED 只对当前确有独立框的
            // 情形成立；离开当前帧后缺少该框仍要阻止同类 D。
            const bool old_still_unresolved = !independent_detection &&
                !existing_item_resolved_without_current_detection(runtime);
            if (runtime_ambiguous || old_still_unresolved) {
                trace_("D-GUARD",
                       "candidate=%d cls=%d blocked-by-old=%d independent=%d "
                       "runtime=%s resolution=%s ambiguous=%d",
                       detection_index, d.cls_id, old->first,
                       independent_detection ? 1 : 0,
                       runtime ? operation_track_state_name(runtime->state) : "NONE",
                       runtime ? existing_resolution_name(runtime->resolution) : "NONE",
                       runtime_ambiguous ? 1 : 0);
                unresolved_same_class_old = true;
                break;
            }
            trace_("D-GUARD",
                   "candidate=%d cls=%d old=%d disposition=independent-static runtime=%s resolution=%s",
                   detection_index, d.cls_id, old->first,
                   runtime ? operation_track_state_name(runtime->state) : "NONE",
                   runtime ? existing_resolution_name(runtime->resolution) : "NONE");
        }
        if (unresolved_same_class_old) {
            printf("[3.0] 手离开后同类旧 C 尚未解决，B cls=%d 保持未决\n",
                   d.cls_id);
            trace_("D-GUARD", "candidate=%d cls=%d result=keep-unresolved",
                   detection_index, d.cls_id);
            continue;
        }
        trace_("D-GUARD",
               "candidate=%d cls=%d result=create-d all-same-class-old-settled",
               detection_index, d.cls_id);

        OperationTrack track;
        const int key = new_suspect_id_();
        track.suspect_id = key;
        track.is_suspect_new = true;
        track.suspect_source = SuspectSource::POST_HAND_REVEAL_D;
        track.cls_id = d.cls_id;
        // 这类 D 的 B 已经是完整框；它没有也不应伪造 first_hand_block_box。
        track.original_box = d.box;
        track.last_seen_box = d.box;
        track.has_last_seen_box = true;
        track.placed_box = d.box;
        track.has_placed_box = true;
        track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
        track.shelter_or_hold = true;
        track.self_match_count = 1;
        track.no_hand_self_match_count = 1;
        track.hand_track_start_index = 0;
        track.post_hand_reveal_no_hand_streak = no_hand_streak_;
        track.track.push_back(d.box);
        track_buffer_[key] = track;
        claimed_detection_indices->insert(detection_index);
        printf("[3.0] 手离开后预登记疑似新物品 D suspect#%d cls=%d source=%s\n",
               key, d.cls_id, suspect_source_name(track.suspect_source));
        trace_track_("D-GUARD", track_buffer_[key], "create-post-hand-reveal-d");
    }
}

void SessionManager::prepare_indirect_move_candidates_(
        const std::vector<Detection>& detections) {
    // 这一步只在无手阶段准备可撤销的 B/C 候选。最终是否更新库存仍要等
    // 各自连续的直接无手框，不能由 A 的手轨迹或 detection input index 决定。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new) continue;
        track.indirect_no_hand_detection_index = -1;
        if (track.indirect_move_candidate && !existing_track_is_terminal(track)) {
            track.indirect_assignment_ambiguous = false;
        }
    }

    std::map<int, IndirectMoveSource> sources;
    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        const OperationTrack& track = it->second;
        BBox observed;
        if (!direct_track_has_observed_motion(track, &observed)) continue;
        std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(track.item_id);
        if (original == operation_start_inventory_.end()) continue;
        const BBox start = original->second.base_box.area() > 0.0f
            ? original->second.base_box : original->second.box;
        const float dx = observed.cx() - start.cx();
        const float dy = observed.cy() - start.cy();
        const float length = std::sqrt(dx * dx + dy * dy);
        if (start.area() <= 0.0f || length <= 0.0f) continue;

        IndirectMoveSource source;
        source.item_id = track.item_id;
        source.original_box = start;
        source.observed_box = observed;
        source.axis_x = dx / length;
        source.axis_y = dy / length;
        sources[source.item_id] = source;
    }
    if (sources.empty()) return;

    const std::map<int, int> static_owner_by_detection =
        build_independent_no_hand_static_owner_by_detection(
            detections, working_inventory_, operation_start_inventory_, track_buffer_);
    std::map<int, IndirectMoveAssignment> assignments;
    std::map<int, int> assignment_owner_by_detection;
    std::set<int> ambiguous_item_ids;

    for (std::map<int, IndirectMoveSource>::const_iterator source_it = sources.begin();
         source_it != sources.end(); ++source_it) {
        const IndirectMoveSource& source = source_it->second;
        std::vector<int> cluster_item_ids(1, source.item_id);
        std::set<int> possible_indirect_item_ids;

        // 以 operation-start 的物理相邻关系扩展连通分量。随后仍要通过每个
        // 成员自己的最终框、方向和连续无手帧确认，所以“相邻”本身不产生事件。
        for (size_t cursor = 0; cursor < cluster_item_ids.size(); ++cursor) {
            const int anchor_id = cluster_item_ids[cursor];
            std::map<int, InventoryItem>::const_iterator anchor =
                operation_start_inventory_.find(anchor_id);
            if (anchor == operation_start_inventory_.end()) continue;
            const BBox anchor_box = anchor->second.base_box.area() > 0.0f
                ? anchor->second.base_box : anchor->second.box;
            for (std::map<int, InventoryItem>::const_iterator candidate =
                     operation_start_inventory_.begin();
                 candidate != operation_start_inventory_.end(); ++candidate) {
                const int item_id = candidate->first;
                if (std::find(cluster_item_ids.begin(), cluster_item_ids.end(), item_id) !=
                    cluster_item_ids.end()) {
                    continue;
                }
                if (!working_inventory_.count(item_id) || pending_out_ids_.count(item_id)) {
                    continue;
                }
                const OperationTrack* runtime = find_runtime_for_item_(item_id);
                if (runtime && !runtime->is_suspect_new &&
                    existing_track_is_terminal(*runtime) && item_id != source.item_id) {
                    continue;
                }
                const BBox candidate_box = candidate->second.base_box.area() > 0.0f
                    ? candidate->second.base_box : candidate->second.box;
                if (boxes_are_adjacent_for_indirect_chain(anchor_box, candidate_box)) {
                    cluster_item_ids.push_back(item_id);
                }
            }
        }
        if (cluster_item_ids.size() <= 1) continue;
        trace_("CLUSTER-ASSIGN",
               "cluster=%d axis=(%.4f,%.4f) source-start=(%.1f,%.1f,%.1f,%.1f) "
               "source-observed=(%.1f,%.1f,%.1f,%.1f) start-members=%zu action=evaluate",
               source.item_id, source.axis_x, source.axis_y,
               source.original_box.x1, source.original_box.y1,
               source.original_box.x2, source.original_box.y2,
               source.observed_box.x1, source.observed_box.y1,
               source.observed_box.x2, source.observed_box.y2,
               cluster_item_ids.size());

        std::map<int, std::vector<int> > item_ids_by_class;
        for (size_t i = 0; i < cluster_item_ids.size(); ++i) {
            const int item_id = cluster_item_ids[i];
            std::map<int, InventoryItem>::const_iterator original =
                operation_start_inventory_.find(item_id);
            if (original == operation_start_inventory_.end() ||
                !working_inventory_.count(item_id) || pending_out_ids_.count(item_id)) {
                continue;
            }
            item_ids_by_class[original->second.cls_id].push_back(item_id);
        }

        // 在按 cls_id 分组前先记录整个簇中已有自身前向候选的旧成员。后续
        // 任一类别的漏检或歧义都会使整簇无法安全分配，不能因为失败类别
        // 恰好排在 B 前面，就让尚未遍历的 B 回落为普通静态收尾。
        for (size_t mi = 0; mi < cluster_item_ids.size(); ++mi) {
            const int item_id = cluster_item_ids[mi];
            if (sources.count(item_id)) continue;
            const InventoryItem& original =
                operation_start_inventory_.find(item_id)->second;
            const BBox start = original.base_box.area() > 0.0f
                ? original.base_box : original.box;
            for (size_t di = 0; di < detections.size(); ++di) {
                const Detection& detection = detections[di];
                if (detection.cls_id != original.cls_id ||
                    detection_conflicts_with_pending_suspect_d(detection, track_buffer_)) {
                    continue;
                }
                const std::map<int, int>::const_iterator static_owner =
                    static_owner_by_detection.find(static_cast<int>(di));
                if (static_owner != static_owner_by_detection.end() &&
                    std::find(cluster_item_ids.begin(), cluster_item_ids.end(),
                              static_owner->second) == cluster_item_ids.end()) {
                    continue;
                }
                if (!indirect_member_matches_detection(original, detection)) continue;
                const float dx = detection.box.cx() - start.cx();
                const float dy = detection.box.cy() - start.cy();
                if (center_distance(start, detection.box) > 0.0f &&
                    dx * source.axis_x + dy * source.axis_y > 0.0f) {
                    possible_indirect_item_ids.insert(item_id);
                    break;
                }
            }
        }

        std::map<int, int> final_detection_by_item;
        bool cluster_assignment_valid = true;
        for (std::map<int, std::vector<int> >::const_iterator grouped =
                 item_ids_by_class.begin(); grouped != item_ids_by_class.end(); ++grouped) {
            const int cls_id = grouped->first;
            std::vector<int> members = grouped->second;
            std::vector<int> candidates;
            for (size_t di = 0; di < detections.size(); ++di) {
                const Detection& detection = detections[di];
                if (detection.cls_id != cls_id ||
                    detection_conflicts_with_pending_suspect_d(detection, track_buffer_)) {
                    continue;
                }
                const std::map<int, int>::const_iterator static_owner =
                    static_owner_by_detection.find(static_cast<int>(di));
                if (static_owner != static_owner_by_detection.end() &&
                    std::find(cluster_item_ids.begin(), cluster_item_ids.end(),
                              static_owner->second) == cluster_item_ids.end()) {
                    continue;
                }

                bool matches_cluster_member = false;
                for (size_t mi = 0; mi < members.size(); ++mi) {
                    const int item_id = members[mi];
                    const std::map<int, IndirectMoveSource>::const_iterator direct =
                        sources.find(item_id);
                    if (direct != sources.end()) {
                        matches_cluster_member =
                            track_match_box(cls_id, direct->second.observed_box,
                                            detection.cls_id, detection.box) ||
                            strict_match_box(cls_id, direct->second.observed_box,
                                             detection.cls_id, detection.box);
                    } else {
                        const std::map<int, InventoryItem>::const_iterator original =
                            operation_start_inventory_.find(item_id);
                        matches_cluster_member = original != operation_start_inventory_.end() &&
                            indirect_member_matches_detection(original->second, detection);
                    }
                    if (matches_cluster_member) break;
                }
                if (matches_cluster_member) candidates.push_back(static_cast<int>(di));
            }

            std::sort(members.begin(), members.end(),
                      [this, &source](int left, int right) {
                          const InventoryItem& left_item =
                              operation_start_inventory_.find(left)->second;
                          const InventoryItem& right_item =
                              operation_start_inventory_.find(right)->second;
                          const BBox left_box = left_item.base_box.area() > 0.0f
                              ? left_item.base_box : left_item.box;
                          const BBox right_box = right_item.base_box.area() > 0.0f
                              ? right_item.base_box : right_item.box;
                          return indirect_projection(left_box, source.axis_x, source.axis_y) <
                              indirect_projection(right_box, source.axis_x, source.axis_y);
                      });
            std::sort(candidates.begin(), candidates.end(),
                      [&detections, &source](int left, int right) {
                          return indirect_projection(detections[left].box,
                                                     source.axis_x, source.axis_y) <
                              indirect_projection(detections[right].box,
                                                  source.axis_x, source.axis_y);
                      });

            bool has_projection_tie = false;
            for (size_t i = 1; i < members.size(); ++i) {
                const InventoryItem& previous =
                    operation_start_inventory_.find(members[i - 1])->second;
                const InventoryItem& current =
                    operation_start_inventory_.find(members[i])->second;
                const BBox previous_box = previous.base_box.area() > 0.0f
                    ? previous.base_box : previous.box;
                const BBox current_box = current.base_box.area() > 0.0f
                    ? current.base_box : current.box;
                if (std::fabs(indirect_projection(previous_box, source.axis_x, source.axis_y) -
                              indirect_projection(current_box, source.axis_x, source.axis_y)) <=
                    0.0001f) {
                    has_projection_tie = true;
                }
            }
            for (size_t i = 1; i < candidates.size(); ++i) {
                if (std::fabs(indirect_projection(detections[candidates[i - 1]].box,
                                                   source.axis_x, source.axis_y) -
                              indirect_projection(detections[candidates[i]].box,
                                                   source.axis_x, source.axis_y)) <= 0.0001f) {
                    has_projection_tie = true;
                }
            }

            if (candidates.size() != members.size() || has_projection_tie) {
                cluster_assignment_valid = false;
            } else {
                for (size_t i = 0; i < members.size(); ++i) {
                    const int item_id = members[i];
                    const Detection& detection = detections[candidates[i]];
                    bool own_identity_match = false;
                    const std::map<int, IndirectMoveSource>::const_iterator direct =
                        sources.find(item_id);
                    if (direct != sources.end()) {
                        own_identity_match = track_match_box(
                            cls_id, direct->second.observed_box,
                            detection.cls_id, detection.box);
                    } else {
                        own_identity_match = indirect_member_matches_detection(
                            operation_start_inventory_.find(item_id)->second, detection);
                    }
                    if (!own_identity_match) {
                        cluster_assignment_valid = false;
                        break;
                    }
                    final_detection_by_item[item_id] = candidates[i];
                }
            }

            if (!cluster_assignment_valid) {
                trace_("CLUSTER-ASSIGN",
                       "cluster=%d cls=%d axis=(%.4f,%.4f) start-members=%zu "
                       "final-candidates=%zu projection-tie=%d action=reject-unresolved",
                       source.item_id, cls_id, source.axis_x, source.axis_y,
                       members.size(), candidates.size(), has_projection_tie ? 1 : 0);
                ambiguous_item_ids.insert(possible_indirect_item_ids.begin(),
                                          possible_indirect_item_ids.end());
                break;
            }
        }
        if (!cluster_assignment_valid) continue;

        std::vector<int> ordered_cluster(cluster_item_ids);
        std::sort(ordered_cluster.begin(), ordered_cluster.end(),
                  [this, &source](int left, int right) {
                      const InventoryItem& left_item =
                          operation_start_inventory_.find(left)->second;
                      const InventoryItem& right_item =
                          operation_start_inventory_.find(right)->second;
                      const BBox left_box = left_item.base_box.area() > 0.0f
                          ? left_item.base_box : left_item.box;
                      const BBox right_box = right_item.base_box.area() > 0.0f
                          ? right_item.base_box : right_item.box;
                      return indirect_projection(left_box, source.axis_x, source.axis_y) <
                          indirect_projection(right_box, source.axis_x, source.axis_y);
                  });
        std::map<int, bool> member_moves_forward;
        for (size_t i = 0; i < ordered_cluster.size(); ++i) {
            const int item_id = ordered_cluster[i];
            std::map<int, int>::const_iterator final = final_detection_by_item.find(item_id);
            if (final == final_detection_by_item.end()) continue;
            const InventoryItem& original =
                operation_start_inventory_.find(item_id)->second;
            const BBox start = original.base_box.area() > 0.0f
                ? original.base_box : original.box;
            const BBox& finish = detections[final->second].box;
            const float dx = finish.cx() - start.cx();
            const float dy = finish.cy() - start.cy();
            member_moves_forward[item_id] = sources.count(item_id) ||
                (center_distance(start, finish) > 0.0f &&
                 dx * source.axis_x + dy * source.axis_y > 0.0f);
        }

        size_t source_position = ordered_cluster.size();
        for (size_t i = 0; i < ordered_cluster.size(); ++i) {
            if (ordered_cluster[i] == source.item_id) {
                source_position = i;
                break;
            }
        }
        if (source_position == ordered_cluster.size()) continue;

        for (size_t i = source_position + 1; i < ordered_cluster.size(); ++i) {
            const int item_id = ordered_cluster[i];
            if (sources.count(item_id) || !member_moves_forward[item_id]) continue;
            const int predecessor = ordered_cluster[i - 1];
            if (!member_moves_forward[predecessor]) continue;
            // 前驱本身无法安全归属时，后续成员也不能把它当作已确认推动链
            // 的一环。保留整条下游链未决，比跳过中间身份继续确认更安全。
            if (ambiguous_item_ids.count(predecessor)) {
                ambiguous_item_ids.insert(item_id);
                continue;
            }
            std::map<int, int>::const_iterator final = final_detection_by_item.find(item_id);
            if (final == final_detection_by_item.end()) continue;

            const OperationTrack* runtime = find_runtime_for_item_(item_id);
            if ((runtime && runtime->is_suspect_new) ||
                (runtime && is_active_runtime_track(*runtime) &&
                 !runtime->indirect_move_candidate) ||
                (runtime && old_track_has_unresolved_alias_(*runtime))) {
                ambiguous_item_ids.insert(item_id);
                continue;
            }

            IndirectMoveAssignment assignment;
            assignment.item_id = item_id;
            assignment.detection_index = final->second;
            assignment.predecessor_item_id = predecessor;
            assignment.cluster_id = source.item_id;
            std::map<int, IndirectMoveAssignment>::const_iterator existing =
                assignments.find(item_id);
            if (existing != assignments.end() &&
                (existing->second.detection_index != assignment.detection_index ||
                 existing->second.predecessor_item_id != assignment.predecessor_item_id)) {
                ambiguous_item_ids.insert(item_id);
                continue;
            }
            std::map<int, int>::const_iterator detection_owner =
                assignment_owner_by_detection.find(assignment.detection_index);
            if (detection_owner != assignment_owner_by_detection.end() &&
                detection_owner->second != item_id) {
                ambiguous_item_ids.insert(item_id);
                ambiguous_item_ids.insert(detection_owner->second);
                continue;
            }
            assignments[item_id] = assignment;
            assignment_owner_by_detection[assignment.detection_index] = item_id;
        }
    }

    const auto ensure_indirect_track = [this](int item_id) -> OperationTrack* {
        OperationTrack* existing = find_runtime_for_item_(item_id);
        if (existing) return existing;
        std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(item_id);
        if (original == operation_start_inventory_.end()) return nullptr;
        OperationTrack track;
        track.item_id = item_id;
        track.cls_id = original->second.cls_id;
        track.original_box = original->second.base_box.area() > 0.0f
            ? original->second.base_box : original->second.box;
        track.last_seen_box = track.original_box;
        track.has_last_seen_box = track.original_box.area() > 0.0f;
        track.needs_no_hand_settlement = true;
        track_buffer_[item_id] = track;
        return &track_buffer_[item_id];
    };

    for (std::set<int>::const_iterator item = ambiguous_item_ids.begin();
         item != ambiguous_item_ids.end(); ++item) {
        OperationTrack* track = ensure_indirect_track(*item);
        if (!track || track->is_suspect_new ||
            (is_active_runtime_track(*track) && !track->indirect_move_candidate)) {
            continue;
        }
        track->indirect_move_candidate = true;
        track->indirect_assignment_ambiguous = true;
        track->indirect_no_hand_detection_index = -1;
        track->indirect_no_hand_match_count = 0;
        track->needs_no_hand_settlement = true;
        track->resolution = ExistingItemResolution::NONE;
        trace_("UNRESOLVED",
               "item=%d reason=indirect-cluster-assignment-ambiguous action=wait",
               *item);
    }

    for (std::map<int, IndirectMoveAssignment>::const_iterator assigned =
             assignments.begin(); assigned != assignments.end(); ++assigned) {
        if (ambiguous_item_ids.count(assigned->first)) continue;
        const IndirectMoveAssignment& assignment = assigned->second;
        if (assignment.detection_index < 0 ||
            static_cast<size_t>(assignment.detection_index) >= detections.size()) {
            continue;
        }
        OperationTrack* track = ensure_indirect_track(assignment.item_id);
        if (!track || track->is_suspect_new ||
            (is_active_runtime_track(*track) && !track->indirect_move_candidate) ||
            old_track_has_unresolved_alias_(*track)) {
            continue;
        }
        const Detection& detection = detections[assignment.detection_index];
        const bool continuous = track->has_indirect_candidate_box &&
            track_match_box(track->cls_id, track->indirect_candidate_box,
                            detection.cls_id, detection.box);
        if (!continuous) {
            track->indirect_no_hand_match_count = 1;
        } else {
            ++track->indirect_no_hand_match_count;
        }
        track->indirect_move_candidate = true;
        track->indirect_assignment_ambiguous = false;
        track->indirect_predecessor_item_id = assignment.predecessor_item_id;
        track->indirect_cluster_id = assignment.cluster_id;
        track->indirect_no_hand_detection_index = assignment.detection_index;
        track->indirect_candidate_box = detection.box;
        track->has_indirect_candidate_box = true;
        track->needs_no_hand_settlement = true;
        track->resolution = ExistingItemResolution::NONE;
        set_live_state_(track, LiveObservationState::POSSIBLE_MOVED, true,
                        "indirect-chain-candidate");
        const std::map<int, IndirectMoveSource>::const_iterator source =
            sources.find(assignment.cluster_id);
        const float axis_x = source == sources.end() ? 0.0f : source->second.axis_x;
        const float axis_y = source == sources.end() ? 0.0f : source->second.axis_y;
        trace_("CLUSTER-ASSIGN",
               "cluster=%d axis=(%.4f,%.4f) item=%d predecessor=%d detection=%d "
               "count=%d/%d original=(%.1f,%.1f,%.1f,%.1f) "
               "final=(%.1f,%.1f,%.1f,%.1f)",
               assignment.cluster_id, axis_x, axis_y,
               assignment.item_id, assignment.predecessor_item_id,
               assignment.detection_index,
               track->indirect_no_hand_match_count, FLOW3_NO_HAND_D_CONFIRM_FRAMES,
               track->original_box.x1, track->original_box.y1,
               track->original_box.x2, track->original_box.y2,
               detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2);
    }

    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || !track.indirect_move_candidate ||
            existing_track_is_terminal(track) ||
            assignments.count(track.item_id) || ambiguous_item_ids.count(track.item_id)) {
            continue;
        }
        track.indirect_assignment_ambiguous = true;
        track.indirect_no_hand_detection_index = -1;
        track.indirect_no_hand_match_count = 0;
        trace_("UNRESOLVED",
               "item=%d reason=indirect-candidate-lost-continuous-assignment action=wait",
               track.item_id);
    }
}

void SessionManager::observe_no_hand_frame_(const std::vector<Detection>& detections) {
    std::set<int> claimed;
    std::vector<int> promote_keys;
    std::set<int> discard_keys;
    std::set<int> discard_settled_old_c_duplicate_keys;
    std::set<int> discard_stale_alias_keys;

    // 细节15：先完成移动簇的批量候选映射，再让既有 C/D 逐项处理。这样
    // 普通静态预约不能按 map 遍历顺序抢走已经唯一属于移动簇成员的框。
    prepare_indirect_move_candidates_(detections);
    const std::map<int, int> independent_static_owner_by_detection =
        build_independent_no_hand_static_owner_by_detection(
            detections, working_inventory_, operation_start_inventory_, track_buffer_);

    // “本帧存在无法唯一归属的路径候选”是瞬时证据；每张直接无手帧都重新
    // 计算，不能让上一帧歧义永久阻塞，也不能把本帧歧义拿去累计 OUT。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        it->second.no_hand_detection_index = -1;
        if (!it->second.is_suspect_new) {
            it->second.no_hand_candidate_ambiguous = false;
        } else if (it->second.pending_d_quarantined_by_old_c) {
            it->second.alias_no_hand_matched_this_frame = false;
        }
    }

    // 对已经由批量分配得到唯一最终框的间接成员，先保留自己的框并只累计
    // 自己的连续无手直接观测。首帧不写库存；第二张连续帧才做位置更新。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || !track.indirect_move_candidate) continue;
        if (track.indirect_assignment_ambiguous ||
            track.indirect_no_hand_detection_index < 0 ||
            static_cast<size_t>(track.indirect_no_hand_detection_index) >= detections.size()) {
            trace_("UNRESOLVED",
                   "item=%d reason=indirect-cluster-no-unique-direct-detection action=wait",
                   track.item_id);
            continue;
        }
        const int detection_index = track.indirect_no_hand_detection_index;
        if (claimed.count(detection_index)) {
            track.indirect_assignment_ambiguous = true;
            track.indirect_no_hand_match_count = 0;
            trace_("UNRESOLVED",
                   "item=%d detection=%d reason=indirect-cluster-detection-already-claimed action=wait",
                   track.item_id, detection_index);
            continue;
        }
        const Detection& detection = detections[detection_index];
        claimed.insert(detection_index);
        track.no_hand_detection_index = detection_index;
        if (track.indirect_no_hand_match_count < FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
            trace_("INDIRECT-MOVE",
                   "item=%d predecessor=%d cluster=%d detection=%d action=wait-direct-continuity "
                   "count=%d/%d",
                   track.item_id, track.indirect_predecessor_item_id,
                   track.indirect_cluster_id, detection_index,
                   track.indirect_no_hand_match_count,
                   FLOW3_NO_HAND_D_CONFIRM_FRAMES);
            continue;
        }
        const bool emit_moved = boxes_differ_as_move(track.original_box, detection.box);
        confirm_position_rebase_(track, detection.box, detection.score, trace_frame_id_,
                                 emit_moved,
                                 "indirect-move-after-continuous-direct-evidence");
    }

    // 认领顺序必须是：已有 C -> 没有手离开后新建的已有 D -> 剩余 B 的
    // POST_HAND_REVEAL_D。track_buffer_ 的负 id 是 D，若直接按 map 遍历，
    // D 会先于 C 抢框，故显式分两轮处理。
    for (int phase = 0; phase < 2; ++phase) {
        const bool process_suspects = phase == 1;
        for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            OperationTrack& track = it->second;
            const bool existing_needs_settlement =
                existing_item_needs_settlement(track);
            if (track.is_suspect_new != process_suspects ||
                (!is_active_runtime_track(track) && !existing_needs_settlement)) {
                continue;
            }

            // 间接成员已在上面的批量阶段处理。未满连续帧时必须保持未决，
            // 不能重新退回逐项 C->B/D 逻辑而破坏本帧顺序保持的映射。
            if (!track.is_suspect_new && track.indirect_move_candidate) {
                continue;
            }

            // CONTACT_* 在有手阶段保持 state=NORMAL，因此不能落入旧的
            // HAND_* 分支。手离开后只按原位置和真实 observed_track 收尾。
            if (!track.is_suspect_new &&
                track.contact_state != ContactState::NONE) {
                reset_stable_near_original_no_hand_evidence_(
                    &track, "contact-track-active");
                int contact_found = unique_contact_original_detection(
                    detections, claimed, track, working_inventory_);
                if (contact_found >= 0) {
                    const Detection& d = detections[contact_found];
                    claimed.insert(contact_found);
                    track.has_tentative_b_box = false;
                    track.tentative_b_match_count = 0;
                    track.tentative_b_started_touching_hand = false;
                    track.has_reappear_candidate_box = false;
                    track.reappear_candidate_match_count = 0;
                    track.reappear_candidate_started_touching_hand = false;
                    const bool was_moving =
                        track.contact_state == ContactState::CONTACT_MOVING;
                    append_contact_observation(&track, d, false);
                    track.hold_evidence_count = 0;
                    ++track.not_hold_evidence_count;
                    if (track.not_hold_evidence_count >=
                        FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, d, trace_frame_id_);
                        }
                        release_not_held_(track, false,
                                          ReleaseReason::CONTACT_RETURNED_ORIGINAL,
                                          contact_found,
                                          &d.box,
                                          "no-hand-contact-returned-original");
                    } else if (was_moving) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, d, trace_frame_id_);
                        }
                    }
                    continue;
                }

                const int contact_found_on_path = unique_contact_detection_for_track(
                    detections, claimed, track, working_inventory_, track_buffer_);
                if (contact_found_on_path >= 0) {
                    const Detection& d = detections[contact_found_on_path];
                    if (is_claim_protected(track)) {
                        // 无手帧也属于保护期的后续有效帧：允许 C 记录自己
                        // 的本地 B，但不把它从其他 C/D 的候选集合中拿走。
                        record_tentative_b(&track, d, false);
                        append_contact_observation(&track, d, false);
                        continue;
                    }
                    claimed.insert(contact_found_on_path);
                    const bool was_moving =
                        track.contact_state == ContactState::CONTACT_MOVING;
                    if (!was_moving) {
                        const bool candidate_ready = update_reappear_candidate(
                            &track, d, false);
                        if (candidate_ready) {
                            track.contact_state = ContactState::CONTACT_MOVING;
                            track.hold_and_move = true;
                            track.hold_evidence_count = 0;
                            track.not_hold_evidence_count = 0;
                        }
                    }
                    append_contact_observation(&track, d, false);
                    if (track.item_id > 0) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, d, trace_frame_id_);
                        }
                    }
                    continue;
                }
                if (has_contact_path_candidate(detections, claimed, track,
                                               working_inventory_)) {
                    track.contact_path_ambiguous = true;
                    trace_track_("MATCH", track, "contact-path-ambiguous");
                }
                // 没有原位置或唯一轨迹 B 时保持未决；不能仅因一帧缺失 OUT。
                trace_track_("MATCH", track, "contact-no-unique-candidate");
                continue;
            }
            BBox reference = track.has_placed_box ? track.placed_box : estimated_box(track);
            if (track.state == OperationTrackState::PLACED && track.has_placed_box) {
                reference = track.placed_box;
            }
            int found = -1;
            bool found_at_original_position = false;
            bool found_as_reappear_candidate = false;
            std::set<int> excluded_static_neighbor_detections;
            std::set<int> candidate_claimed;
            const char* found_source = "NONE";
            bool stale_reappear_candidate_cleared = false;
            bool stale_tentative_b_cleared = false;

            // 细节10：已有 reappear candidate 仍然优先，但若它被本帧唯一严格
            // 原位框明确证伪为静止邻居 B，就只能说明“不是 A”。跳过该框后要在
            // 同一无手帧继续查找 A 的预计位置/真实路径，不能直接永久未决。
            for (;;) {
                candidate_claimed = claimed;
                candidate_claimed.insert(excluded_static_neighbor_detections.begin(),
                                         excluded_static_neighbor_detections.end());
                found = -1;
                found_at_original_position = false;
                found_as_reappear_candidate = false;
                bool found_from_saved_reappear_candidate = false;
                found_source = "NONE";

                // 细节5的无手收尾顺序：已有 C 必须先回查原位置；若 C 仍在原处，
                // 后面的动态路径和 B 候选都不能把它改判成移动。
                if (!track.is_suspect_new && track.original_box.area() > 0.0f) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                      track.cls_id, track.original_box,
                                                      true, false);
                    found_at_original_position = found >= 0;
                    if (found < 0) {
                        const int nearest = best_detection_for_box(
                            detections, candidate_claimed, track.cls_id,
                            track.original_box, true, false);
                        if (nearest >= 0 && has_unique_operation_start_owner(
                                detections[nearest], track.item_id,
                                operation_start_inventory_)) {
                            found = nearest;
                            found_at_original_position = true;
                        }
                    }
                    if (found >= 0) found_source = "ORIGINAL";
                    // 同上：由 CONTACT_* 转来的物品已拥有真实 B 路径。B 与
                    // 原框局部重叠并不等于它仍在原位，必须再检查中心位移。
                    if (track.has_hand_estimate_anchor_box && found >= 0 &&
                        !contact_detection_is_at_original(track, detections[found])) {
                        found = -1;
                        found_at_original_position = false;
                        found_source = "NONE";
                    }
                }
                // 已在有手阶段看到过 B 时，真实 B 候选比“旧框 + 手位移”的估计
                // 更可信，但它同样需要连续自匹配，不能在本帧直接确认整理。
                if (found < 0 && !track.is_suspect_new &&
                    track.has_reappear_candidate_box) {
                    found = unique_detection_for_box(
                        detections, candidate_claimed, track.cls_id,
                        track.reappear_candidate_box, true, true);
                    if (found < 0) {
                        found = best_detection_for_box(
                            detections, candidate_claimed, track.cls_id,
                            track.reappear_candidate_box, true, true);
                    }
                    found_as_reappear_candidate = found >= 0;
                    found_from_saved_reappear_candidate = found >= 0;
                    if (found >= 0) found_source = "REAPPEAR_CANDIDATE";
                }
                if (found < 0) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                     track.cls_id, reference, true, true);
                    if (found >= 0) found_source = "REFERENCE";
                }
                // D 刚放下时，完整框可能和“手中局部框 + 手位移”差异很大。它已经
                // 通过前序来源建立了链路，收尾首帧应优先尝试自身最近一次真实观测。
                // 普通旧库存不走这个宽松兜底。
                if (found < 0 && track.is_suspect_new && track.has_last_seen_box) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                      track.cls_id, track.last_seen_box,
                                                      true, true);
                    if (found >= 0) found_source = "SUSPECT_LAST_SEEN";
                }
                if (found < 0 && track.is_suspect_new && track.has_last_hand_block_box) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                      track.cls_id, track.last_hand_block_box,
                                                      true, true);
                    if (found >= 0) found_source = "SUSPECT_HAND_BLOCK";
                }
                if (found < 0 && track.is_suspect_new) {
                    found = unique_d_reappearance_detection_for_track(
                        detections, candidate_claimed, it->first, track,
                        working_inventory_, track_buffer_);
                    if (found >= 0) {
                        found_source = "SUSPECT_REAPPEAR";
                        printf("[3.0] suspect#%d 通过局部→完整路径重现匹配到无手检测\n",
                               track.suspect_id);
                    }
                }
                if (found < 0 && !track.is_suspect_new) {
                    found = unique_no_hand_reappear_detection_for_track(
                        detections, candidate_claimed, it->first, track,
                        working_inventory_, operation_start_inventory_,
                        pending_in_ids_, track_buffer_);
                    found_as_reappear_candidate = found >= 0;
                    if (found >= 0) {
                        found_source = "UNIQUE_REAPPEAR";
                        printf("[3.0] item#%d 在无手帧将同类 B 暂记为重新出现候选\n",
                               track.item_id);
                    }
                }

                const StrictDetectionOwner fallback_strict_owner = found >= 0
                    ? strict_owner_for_detection(
                        detections[found], track.item_id, working_inventory_,
                        operation_start_inventory_, pending_in_ids_, track_buffer_)
                    : StrictDetectionOwner();
                if (found < 0 || track.is_suspect_new || is_claim_protected(track) ||
                    found_at_original_position ||
                    contact_detection_is_at_original(track, detections[found]) ||
                    !strict_owner_blocks_old_c(fallback_strict_owner.kind)) {
                    break;
                }

                const std::map<int, int>::const_iterator static_owner =
                    independent_static_owner_by_detection.find(found);
                if (static_owner == independent_static_owner_by_detection.end() ||
                    static_owner->second == track.item_id) {
                    // 另一旧 C 没有本帧独立严格原位证据，仍属于细节8的真实
                    // 同类歧义；交给下面原有的 ambiguous 分支处理。
                    break;
                }

                const bool candidate_matches_static_neighbor =
                    track.has_reappear_candidate_box &&
                    track_match_box(track.cls_id, track.reappear_candidate_box,
                                    detections[found].cls_id, detections[found].box);
                const bool tentative_matches_static_neighbor =
                    track.has_tentative_b_box &&
                    track_match_box(track.cls_id, track.tentative_b_box,
                                    detections[found].cls_id, detections[found].box);
                if (candidate_matches_static_neighbor) {
                    track.has_reappear_candidate_box = false;
                    track.reappear_candidate_match_count = 0;
                    track.reappear_candidate_started_touching_hand = false;
                    track.drop_evidence_count = 0;
                    stale_reappear_candidate_cleared = true;
                }
                if (tentative_matches_static_neighbor) {
                    track.has_tentative_b_box = false;
                    track.tentative_b_match_count = 0;
                    track.tentative_b_started_touching_hand = false;
                    stale_tentative_b_cleared = true;
                }
                // 即使本次候选来自 reference/path 而非保存的 candidate，排除后
                // 的下一个真实框也必须重新走连续确认，不能单帧确认 MOVED。
                track.reappearance_pending = true;
                trace_(
                    "MATCH",
                    "item=%d detection=%d source=%s other-owner=%d "
                    "other-owner-evidence=direct-original-mutually-unique "
                    "action=exclude-independent-static-neighbor-and-fallback "
                    "candidate-cleared=%d tentative-cleared=%d saved-candidate-source=%d "
                    "box=(%.1f,%.1f,%.1f,%.1f)",
                    track.item_id, found, found_source, static_owner->second,
                    candidate_matches_static_neighbor ? 1 : 0,
                    tentative_matches_static_neighbor ? 1 : 0,
                    found_from_saved_reappear_candidate ? 1 : 0,
                    detections[found].box.x1, detections[found].box.y1,
                    detections[found].box.x2, detections[found].box.y2);
                trace_track_("MATCH", track,
                             "exclude-static-neighbor-candidate-continue-fallback");
                excluded_static_neighbor_detections.insert(found);
            }

            if (!track.is_suspect_new && is_claim_protected(track)) {
                reset_stable_near_original_no_hand_evidence_(
                    &track, "active-claim-grace");
                if (found < 0) {
                    // 保护期内的缺失只是未决，不产生 HOLD/OUT 证据；同时
                    // register_post_hand_reveal_suspects_ 会继续屏蔽同类 D。
                    continue;
                }
                const Detection& provisional = detections[found];
                const bool at_original = found_at_original_position ||
                    contact_detection_is_at_original(track, provisional);
                if (at_original) {
                    claimed.insert(found);
                    track.b_claim_ambiguous = false;
                    track.has_tentative_b_box = false;
                    track.tentative_b_match_count = 0;
                    track.tentative_b_started_touching_hand = false;
                    track.has_reappear_candidate_box = false;
                    track.reappear_candidate_match_count = 0;
                    track.reappear_candidate_started_touching_hand = false;
                    ++track.not_hold_evidence_count;
                    track.hold_evidence_count = 0;
                    track.last_hand_block_box = provisional.box;
                    track.has_last_hand_block_box = true;
                    if (track.not_hold_evidence_count >=
                        FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, provisional, trace_frame_id_);
                        }
                        release_not_held_(track, false,
                                          ReleaseReason::ORIGINAL_DETECTION,
                                          found,
                                          &provisional.box,
                                          "no-hand-protected-original");
                    }
                } else {
                    record_tentative_b(&track, provisional, false);
                    track.reappearance_pending = true;
                }
                continue;
            }
            if (!track.is_suspect_new && found >= 0 &&
                !found_at_original_position &&
                !contact_detection_is_at_original(track, detections[found])) {
                const StrictDetectionOwner strict_owner = strict_owner_for_detection(
                    detections[found], track.item_id, working_inventory_,
                    operation_start_inventory_, pending_in_ids_, track_buffer_);
                const bool is_current_c_alias =
                    strict_owner_is_quarantined_alias_of_old_c(
                        strict_owner, track.item_id, track_buffer_);
                if ((strict_owner.kind == StrictOwnerKind::PENDING_D ||
                     strict_owner.kind == StrictOwnerKind::QUARANTINED_PENDING_D) &&
                    !is_current_c_alias) {
                    // Pending D 没有 old C 的排他权限，也不能反过来被 C 单帧
                    // 抢走。保留双方独立证据链，日志明确它是 runtime D 而非
                    // "other old item"。
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d defer=pending-runtime-owner "
                           "owner-kind=%s owner-id=%d owner-runtime=%d "
                           "owner-authority=provisional-runtime-only",
                           track.item_id, found,
                           strict_owner_kind_name(strict_owner.kind),
                           strict_owner.item_id, strict_owner.runtime_key);
                    trace_track_("MATCH", track,
                                 "no-hand-candidate-belongs-to-pending-runtime-d");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "pending-runtime-owner");
                    continue;
                }
                if (strict_owner_blocks_old_c(strict_owner.kind)) {
                    // A 的预计轨迹可以碰巧经过 B 的原位，但一个完整框若已经
                    // 严格属于另一件旧库存 B，就不能被 A 先抢走。它既不是 A
                    // 的 MOVED 证据，也不是 A 缺失的 OUT 证据，应保持未决。
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d reject=strict-owner "
                           "owner-kind=%s owner-id=%d owner-runtime=%d "
                           "excluded-static-neighbors=%zu stale-candidate-reset=%d "
                           "tentative-reset=%d fallback-source=%s result=real-ambiguity",
                           track.item_id, found, strict_owner_kind_name(strict_owner.kind),
                           strict_owner.item_id, strict_owner.runtime_key,
                           excluded_static_neighbor_detections.size(),
                           stale_reappear_candidate_cleared ? 1 : 0,
                           stale_tentative_b_cleared ? 1 : 0, found_source);
                    trace_track_("MATCH", track,
                                 "no-hand-candidate-strictly-owned-by-old-or-confirmed-item");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "reappearance-or-claim-ambiguity");
                    continue;
                }
            }
            if (!track.is_suspect_new && found >= 0 &&
                !found_at_original_position &&
                !contact_detection_is_at_original(track, detections[found])) {
                const std::map<int, int> no_known_owner;
                const int mature_owner = unique_c_reappear_owner_for_detection(
                    detections[found], track_buffer_, no_known_owner);
                if (mature_owner == -2) {
                    mark_mature_hand_b_ambiguity(detections[found], &track_buffer_,
                                                  no_known_owner);
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d excluded-static-neighbors=%zu "
                           "stale-candidate-reset=%d tentative-reset=%d fallback-source=%s "
                           "result=real-shared-ambiguity",
                           track.item_id, found,
                           excluded_static_neighbor_detections.size(),
                           stale_reappear_candidate_cleared ? 1 : 0,
                           stale_tentative_b_cleared ? 1 : 0, found_source);
                    trace_track_("MATCH", track, "no-hand-shared-candidate");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "reappearance-or-claim-ambiguity");
                    continue;
                }
                if (mature_owner >= 0 && mature_owner != track.item_id) {
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d excluded-static-neighbors=%zu "
                           "stale-candidate-reset=%d tentative-reset=%d fallback-source=%s "
                           "result=real-other-c-owner",
                           track.item_id, found,
                           excluded_static_neighbor_detections.size(),
                           stale_reappear_candidate_cleared ? 1 : 0,
                           stale_tentative_b_cleared ? 1 : 0, found_source);
                    trace_track_("MATCH", track, "no-hand-candidate-owned-by-other-c");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "reappearance-or-claim-ambiguity");
                    continue;
                }
                track.b_claim_ambiguous = false;
            }
            if (found < 0) {
                if (!track.is_suspect_new) {
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "no-direct-unique-owner");
                    track.no_hand_candidate_ambiguous =
                        has_ambiguous_no_hand_reappear_candidate(
                            detections, candidate_claimed, it->first, track,
                            working_inventory_, operation_start_inventory_,
                            pending_in_ids_, track_buffer_,
                            independent_static_owner_by_detection);
                    trace_track_("MATCH", track,
                                 track.no_hand_candidate_ambiguous
                                     ? "no-hand-path-ambiguous"
                                     : "no-hand-no-path-candidate");
                    if (!excluded_static_neighbor_detections.empty()) {
                        trace_("MATCH",
                               "item=%d excluded-static-neighbors=%zu "
                               "stale-candidate-reset=%d tentative-reset=%d "
                               "fallback-source=NONE fallback-detection=-1 "
                               "ambiguous-after-exclusion=%d result=%s",
                               track.item_id,
                               excluded_static_neighbor_detections.size(),
                               stale_reappear_candidate_cleared ? 1 : 0,
                               stale_tentative_b_cleared ? 1 : 0,
                               track.no_hand_candidate_ambiguous ? 1 : 0,
                               track.no_hand_candidate_ambiguous
                                   ? "real-path-ambiguity"
                                   : "excluded-static-neighbor-then-no-a-candidate");
                    }
                }
                // 对 C-D alias：当前框若已被 C 直接认领，D 只是同一框的
                // 重复运行时解释，必须等待第二张直接无手帧再合并；不能按
                // 普通 D 的“本帧未自匹配”规则立刻丢掉，否则 C 无法获得
                // 连续的 alias 仲裁证据。
                if (track.is_suspect_new &&
                    track.pending_d_quarantined_by_old_c) {
                    if (track.alias_no_hand_matched_this_frame) {
                        trace_("C-D-ALIAS",
                               "old-count=%zu suspect=%d phase=NO_HAND relation=shared-detection "
                               "direct-count=%d action=%s",
                               track.conflicting_old_item_ids.size(), track.suspect_id,
                               track.alias_no_hand_match_count,
                               track.alias_no_hand_match_count >=
                                       FLOW3_NO_HAND_D_CONFIRM_FRAMES
                                   ? "discard-duplicate"
                                   : "wait-second-no-hand");
                        if (track.alias_no_hand_match_count >=
                            FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                            discard_keys.insert(it->first);
                        }
                    } else {
                        track.alias_no_hand_match_count = 0;
                        const bool all_old_settled =
                            all_conflicting_old_c_independently_settled(
                                track, track_buffer_);
                        const bool runtime_only =
                            !track.promoted_to_working_inventory && track.item_id <= 0;
                        if (all_old_settled && runtime_only) {
                            ++track.alias_no_hand_missing_count;
                            const bool discard_stale_alias =
                                track.alias_no_hand_missing_count >=
                                FLOW3_NO_HAND_D_CONFIRM_FRAMES;
                            trace_(
                                "C-D-ALIAS",
                                "old-count=%zu suspect=%d phase=NO_HAND relation=no-direct-pair "
                                "all-conflicting-old-settled=1 runtime-only=1 "
                                "missing-count=%d/%d action=%s",
                                track.conflicting_old_item_ids.size(), track.suspect_id,
                                track.alias_no_hand_missing_count,
                                FLOW3_NO_HAND_D_CONFIRM_FRAMES,
                                discard_stale_alias
                                    ? "discard-stale-alias-after-settled-old-c-no-hand-missing"
                                    : "wait-stale-alias-evidence");
                            trace_track_(
                                "C-D-ALIAS", track,
                                discard_stale_alias
                                    ? "stale-alias-no-direct-evidence-confirmed"
                                    : "stale-alias-no-direct-evidence-wait");
                            if (discard_stale_alias) {
                                discard_keys.insert(it->first);
                                discard_stale_alias_keys.insert(it->first);
                            }
                        } else {
                            const int previous_missing_count =
                                track.alias_no_hand_missing_count;
                            track.alias_no_hand_missing_count = 0;
                            trace_(
                                "C-D-ALIAS",
                                "old-count=%zu suspect=%d phase=NO_HAND relation=no-direct-pair "
                                "all-conflicting-old-settled=%d runtime-only=%d "
                                "missing-count=%d/%d action=%s",
                                track.conflicting_old_item_ids.size(), track.suspect_id,
                                all_old_settled ? 1 : 0, runtime_only ? 1 : 0,
                                previous_missing_count,
                                FLOW3_NO_HAND_D_CONFIRM_FRAMES,
                                previous_missing_count > 0
                                    ? "reset-stale-alias-evidence"
                                    : "keep-unresolved");
                            trace_track_("C-D-ALIAS", track,
                                         all_old_settled
                                             ? "quarantined-alias-not-runtime-only"
                                             : "conflicting-old-c-not-yet-settled");
                        }
                    }
                    continue;
                }
                // 普通 D 必须在后续直接无手帧中连续自匹配。未达到门槛就
                // 消失，不保留为悬空 IN，更不能让它在后续随机帧重新凑次数。
                if (track.is_suspect_new &&
                    track.no_hand_self_match_count < FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                    discard_keys.insert(it->first);
                }
                continue;
            }
            const Detection& d = detections[found];
            bool shared_quarantined_alias = false;
            if (!track.is_suspect_new) {
                track.no_hand_detection_index = found;
                for (std::set<int>::const_iterator key =
                         track.conflicting_suspect_keys.begin();
                     key != track.conflicting_suspect_keys.end(); ++key) {
                    std::map<int, OperationTrack>::iterator suspect =
                        track_buffer_.find(*key);
                    if (suspect == track_buffer_.end() ||
                        !suspect->second.pending_d_quarantined_by_old_c ||
                        !suspect->second.conflicting_old_item_ids.count(track.item_id) ||
                        !quarantined_suspect_matches_detection(suspect->second, d)) {
                        continue;
                    }
                    OperationTrack& alias_d = suspect->second;
                    alias_d.alias_no_hand_matched_this_frame = true;
                    ++alias_d.alias_no_hand_match_count;
                    alias_d.alias_no_hand_missing_count = 0;
                    track.alias_no_hand_match_count =
                        alias_d.alias_no_hand_match_count;
                    track.reappearance_pending = true;
                    shared_quarantined_alias = true;
                    trace_("C-D-ALIAS",
                           "old-item=%d suspect=%d detection=%d phase=NO_HAND "
                           "relation=shared-detection direct-count=%d "
                           "action=%s",
                           track.item_id, alias_d.suspect_id, found,
                           alias_d.alias_no_hand_match_count,
                           alias_d.alias_no_hand_match_count >=
                                   FLOW3_NO_HAND_D_CONFIRM_FRAMES
                               ? "assign-to-old-c-discard-duplicate"
                               : "wait-second-no-hand");
                    if (alias_d.alias_no_hand_match_count >=
                        FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                        discard_keys.insert(*key);
                    }
                }
            }
            claimed.insert(found);
            if (!track.is_suspect_new) {
                track.no_hand_candidate_ambiguous = false;
                trace_("MATCH",
                       "item=%d detection=%d source=%s excluded-static-neighbors=%zu "
                       "stale-candidate-reset=%d tentative-reset=%d "
                       "ambiguous-after-exclusion=0 result=fallback-selected "
                       "box=(%.1f,%.1f,%.1f,%.1f)",
                       track.item_id, found,
                       found_source,
                       excluded_static_neighbor_detections.size(),
                       stale_reappear_candidate_cleared ? 1 : 0,
                       stale_tentative_b_cleared ? 1 : 0,
                       d.box.x1, d.box.y1, d.box.x2, d.box.y2);
            }

            if (track.is_suspect_new) {
                if (track.pending_d_quarantined_by_old_c) {
                    // 任意直接无手证据（独立、共享或尚待仲裁）都会中断“D 消失”
                    // 的连续计数。不能把被直接证据打断的缺失片段拼起来。
                    track.alias_no_hand_missing_count = 0;
                }
                if (track.pending_d_quarantined_by_old_c &&
                    quarantined_suspect_detection_is_duplicate_of_settled_old_c(
                        track, track_buffer_, found, d)) {
                    // C 已先独立结算；D 当前又只命中同一个 C 框时，D 不再
                    // 具有独立新增物品的任何证据。立即回收该重复 runtime
                    // identity，避免下一帧把它累积为 IN。
                    discard_keys.insert(it->first);
                    discard_settled_old_c_duplicate_keys.insert(it->first);
                    trace_("C-D-ALIAS",
                           "suspect=%d detection=%d phase=NO_HAND "
                           "relation=duplicate-settled-old-c "
                           "action=discard-duplicate-settled-old-c",
                           track.suspect_id, found);
                    continue;
                }
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                ++track.no_hand_self_match_count;
                track.no_hand_detection_index = found;
                if (track.pending_d_quarantined_by_old_c) {
                    if (!quarantined_suspect_has_distinct_old_c_detections(
                            track, track_buffer_, found, d)) {
                        track.alias_no_hand_match_count = 0;
                        trace_("C-D-ALIAS",
                               "suspect=%d detection=%d phase=NO_HAND relation=not-independent "
                               "action=keep-unresolved",
                               track.suspect_id, found);
                        continue;
                    }
                    track.alias_no_hand_matched_this_frame = true;
                    ++track.alias_no_hand_match_count;
                    trace_("C-D-ALIAS",
                           "suspect=%d detection=%d phase=NO_HAND relation=distinct-detections "
                           "old-count=%zu direct-count=%d action=%s",
                           track.suspect_id, found, track.conflicting_old_item_ids.size(),
                           track.alias_no_hand_match_count,
                           track.alias_no_hand_match_count >=
                                   FLOW3_NO_HAND_D_CONFIRM_FRAMES
                               ? "confirm-independent-d"
                               : "wait-second-no-hand");
                    if (track.alias_no_hand_match_count <
                        FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                        continue;
                    }
                    const int released_key = it->first;
                    unlink_quarantined_suspect_(released_key,
                                                "confirm-independent-d");
                    set_live_state_(&track, LiveObservationState::PROVISIONAL_D,
                                    true, "independent-no-hand-d-confirmed");
                }
            }
            if (track.is_suspect_new && !track.promoted_to_working_inventory) {
                ++track.self_match_count;
                if (track.self_match_count >= NEW_ITEM_CONFIRM_FRAMES) {
                    promote_keys.push_back(it->first);
                }
            }
            if (track.is_suspect_new && track.promoted_to_working_inventory) {
                track.placed_box = d.box;
                track.has_placed_box = true;
                track.drop_confirmed = true;
                std::map<int, InventoryItem>::iterator item =
                    working_inventory_.find(track.item_id);
                if (item != working_inventory_.end()) {
                    update_seen(item->second, d, trace_frame_id_);
                    item->second.base_box = d.box;
                }
                continue;
            }
            if (!track.is_suspect_new) {
                const bool still_at_original = found_at_original_position ||
                    (!track.has_hand_estimate_anchor_box &&
                     partial_match_box(track.cls_id, track.original_box,
                                       d.cls_id, d.box,
                                       FLOW3_TRACK_PARTIAL_IOM));
                if (still_at_original) {
                    std::map<int, InventoryItem>::iterator item =
                        working_inventory_.find(track.item_id);
                    if (item != working_inventory_.end()) {
                        update_seen(item->second, d, trace_frame_id_);
                    }
                    release_not_held_(track, false,
                                      ReleaseReason::ORIGINAL_DETECTION,
                                      found,
                                      &d.box,
                                      "no-hand-original");
                } else {
                    // r16：普通原位 release 未成立时，只有这个旧 C 自己
                    // 连续两张唯一、尺度一致的无手直接框才可位置重同步。
                    // 这里绝不改动 D 的 alias 仲裁；C 结算后，后续 phase=1
                    // 仍完全走 r15。
                    const bool may_be_stable_rebase_old_c =
                        track.state == OperationTrackState::NORMAL &&
                        track.contact_state == ContactState::NONE &&
                        track.resolution == ExistingItemResolution::STATIC_CONFIRMED &&
                        track.needs_no_hand_settlement &&
                        track.has_hand_estimate_anchor_box;
                    if (may_be_stable_rebase_old_c &&
                        try_release_stable_near_original_no_hand_(
                            &track, found, d,
                            independent_static_owner_by_detection,
                            found_source)) {
                        continue;
                    }
                    const bool should_observe_candidate =
                        found_as_reappear_candidate || track.reappearance_pending ||
                        track.has_reappear_candidate_box;
                    bool candidate_ready = true;
                    if (should_observe_candidate) {
                        const bool had_candidate = track.has_reappear_candidate_box;
                        candidate_ready = update_reappear_candidate(&track, d, false);
                        trace_("MATCH",
                               "item=%d detection=%d source=no-hand-reappear action=%s count=%d ready=%d",
                               track.item_id, found, had_candidate ? "update" : "start",
                               track.reappear_candidate_match_count,
                               candidate_ready ? 1 : 0);
                    }
                    if (!candidate_ready) continue;
                    track.last_seen_box = d.box;
                    track.has_last_seen_box = true;
                    if ((track.hold_and_move || has_meaningful_hand_move(track)) &&
                        boxes_differ_as_move(track.original_box, d.box) &&
                        (!shared_quarantined_alias ||
                         track.alias_no_hand_match_count >=
                             FLOW3_NO_HAND_D_CONFIRM_FRAMES)) {
                        confirm_rearrange_(track, d.box, d.score, trace_frame_id_);
                    }
                }
            }
        }

        // 第一轮 C 处理结束后，正常静态 C 也要先保留自己的严格框，再让 D
        // 用宽松的局部/路径规则查找。这样严格属于旧库存的 B 不会被 D 抢走。
        if (phase == 0) {
            reserve_unique_no_hand_static_inventory_detections(
                detections, working_inventory_, operation_start_inventory_, track_buffer_,
                &claimed);
        }
    }

    for (std::set<int>::const_iterator discard = discard_keys.begin();
         discard != discard_keys.end(); ++discard) {
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(*discard);
        if (track == track_buffer_.end()) continue;
        const bool discard_alias_duplicate =
            track->second.pending_d_quarantined_by_old_c;
        const bool duplicate_of_settled_old_c =
            discard_settled_old_c_duplicate_keys.count(*discard) > 0;
        const bool stale_alias_after_settled_old_c =
            discard_stale_alias_keys.count(*discard) > 0;
        if (discard_alias_duplicate) {
            unlink_quarantined_suspect_(
                *discard,
                stale_alias_after_settled_old_c
                    ? "discard-stale-alias-after-settled-old-c-no-hand-missing"
                    : (duplicate_of_settled_old_c
                        ? "discard-duplicate-settled-old-c"
                        : "discard-duplicate"));
        }
        if (stale_alias_after_settled_old_c) {
            printf("[3.0] suspect#%d 的关联旧 C 已独立结算，且连续无手帧无直接证据，"
                   "按过期 alias 丢弃\n", track->second.suspect_id);
            trace_(
                "C-D-ALIAS",
                "suspect=%d phase=NO_HAND "
                "action=discard-stale-alias-after-settled-old-c-no-hand-missing "
                "working-inventory-write=0 pending-in-write=0 event=none",
                track->second.suspect_id);
            trace_track_("C-D-ALIAS", track->second,
                         "discard-stale-alias-after-settled-old-c-no-hand-missing");
        } else if (duplicate_of_settled_old_c) {
            printf("[3.0] suspect#%d 只命中已独立结算旧 C 的框，"
                   "按重复身份丢弃\n", track->second.suspect_id);
            trace_track_("C-D-ALIAS", track->second,
                         "discard-quarantined-duplicate-of-settled-old-c");
        } else if (discard_alias_duplicate) {
            printf("[3.0] suspect#%d 与旧 C 的共享无手框已连续确认，"
                   "按重复身份丢弃\n", track->second.suspect_id);
            trace_track_("C-D-ALIAS", track->second,
                         "discard-quarantined-duplicate-after-shared-no-hand-confirmation");
        } else {
            printf("[3.0] suspect#%d 的手离开后完整框未连续确认，丢弃候选\n",
                   track->second.suspect_id);
            trace_track_("STATE", track->second,
                         "discard-suspect-missing-continuous-no-hand-confirmation");
        }
        if (track->second.promoted_to_working_inventory && track->second.item_id > 0) {
            const int item_id = track->second.item_id;
            working_inventory_.erase(item_id);
            pending_in_ids_.erase(item_id);
            confirmed_moved_ids_.erase(item_id);
            confirmed_position_update_ids_.erase(item_id);
            pending_out_ids_.erase(item_id);
        }
        track_buffer_.erase(track);
    }

    for (size_t i = 0; i < promote_keys.size(); ++i) {
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(promote_keys[i]);
        if (track == track_buffer_.end() || !track->second.has_last_seen_box) continue;
        Detection d;
        d.box = track->second.last_seen_box;
        d.cls_id = track->second.cls_id;
        d.score = 0.0f;
        promote_suspect_(promote_keys[i], d, trace_frame_id_);
        track->second.placed_box = d.box;
        track->second.has_placed_box = true;
        track->second.drop_confirmed = true;
    }

    // 所有旧 C、已有 D 都已优先处理完成；此时剩余 B 才能成为“全程被手
    // 遮挡、手离开后首次显现”的 D。
    register_post_hand_reveal_suspects_(detections, &claimed);
    const std::set<int> no_new_existing_tracks;
    advance_claim_grace_(no_new_existing_tracks);
}

void SessionManager::mark_pending_out_(int item_id) {
    pending_out_ids_.insert(item_id);
    OperationTrack* track = find_runtime_for_item_(item_id);
    if (track && !track->is_suspect_new) {
        track->resolution = ExistingItemResolution::OUT_CONFIRMED;
        track->release_reason = ReleaseReason::NONE;
        track->needs_no_hand_settlement = false;
        track->stable_near_original_no_hand_count = 0;
        track->has_stable_near_original_box = false;
        track->stable_near_original_box = BBox();
        trace_track_("NO-HAND", *track, "confirm-out-after-direct-missing-frames");
    }
}

void SessionManager::clear_visible_count_settlement_(bool restore_uncommitted_outs) {
    for (std::map<int, int>::const_iterator it = visible_count_missing_counts_.begin();
         it != visible_count_missing_counts_.end(); ++it) {
        OperationTrack* track = find_runtime_for_item_(it->first);
        if (track && !track->is_suspect_new) {
            track->no_hand_missing_count = 0;
        }
    }
    if (restore_uncommitted_outs) {
        for (std::set<int>::const_iterator it = visible_count_confirmed_out_ids_.begin();
             it != visible_count_confirmed_out_ids_.end(); ++it) {
            pending_out_ids_.erase(*it);
            OperationTrack* track = find_runtime_for_item_(*it);
            if (track && !track->is_suspect_new &&
                track->resolution == ExistingItemResolution::OUT_CONFIRMED) {
                track->resolution = ExistingItemResolution::NONE;
                track->needs_no_hand_settlement = true;
                track->release_reason = ReleaseReason::NONE;
                trace_track_("VISIBLE-COUNT", *track,
                             "retract-uncommitted-visible-count-out");
            }
        }
    }
    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    visible_count_out_candidate_ids_.clear();
    visible_count_missing_counts_.clear();
    visible_count_confirmed_out_ids_.clear();
    visible_count_prior_survivors_by_cls_.clear();
    visible_count_prior_survivor_boxes_by_cls_.clear();
}

void SessionManager::prepare_visible_count_settlement_(
        const std::vector<Detection>& detections) {
    std::map<int, std::vector<int> > detection_indices_by_cls;
    for (size_t di = 0; di < detections.size(); ++di) {
        if (detections[di].cls_id >= 0 && detections[di].box.area() > 0.0f) {
            detection_indices_by_cls[detections[di].cls_id].push_back(
                static_cast<int>(di));
        }
    }

    std::map<int, std::vector<int> > old_visible_ids_by_cls;
    std::set<int> classes_with_runtime;
    for (std::map<int, InventoryItem>::const_iterator original =
             operation_start_inventory_.begin();
         original != operation_start_inventory_.end(); ++original) {
        std::map<int, InventoryItem>::const_iterator current =
            working_inventory_.find(original->first);
        if (current == working_inventory_.end() ||
            original->second.status == ItemStatus::OCCLUDED ||
            current->second.status == ItemStatus::OCCLUDED) {
            continue;
        }
        const bool confirmed_by_visible_count =
            visible_count_confirmed_out_ids_.count(original->first) > 0;
        if (pending_out_ids_.count(original->first) && !confirmed_by_visible_count) {
            continue;
        }
        const OperationTrack* runtime = find_runtime_for_item_(original->first);
        if (runtime && runtime->is_suspect_new) continue;
        if (runtime && !confirmed_by_visible_count &&
            (runtime->resolution == ExistingItemResolution::OUT_CONFIRMED ||
             runtime->resolution == ExistingItemResolution::OCCLUDED_CONFIRMED)) {
            continue;
        }
        old_visible_ids_by_cls[original->second.cls_id].push_back(original->first);
        if (runtime && (is_active_runtime_track(*runtime) ||
                        existing_item_needs_settlement(*runtime) ||
                        confirmed_by_visible_count)) {
            classes_with_runtime.insert(original->second.cls_id);
        }
    }

    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    std::set<int> next_out_candidates;
    std::set<int> active_deficit_classes;
    std::set<int> continuity_reset_item_ids;

    for (std::map<int, std::vector<int> >::const_iterator group =
             old_visible_ids_by_cls.begin();
         group != old_visible_ids_by_cls.end(); ++group) {
        const int cls_id = group->first;
        const std::vector<int>& old_ids = group->second;
        std::map<int, std::vector<int> >::const_iterator detections_it =
            detection_indices_by_cls.find(cls_id);
        const std::vector<int> empty_detection_indices;
        const std::vector<int>& detection_indices =
            detections_it == detection_indices_by_cls.end()
                ? empty_detection_indices : detections_it->second;

        bool has_protected_old_track = false;
        for (size_t oi = 0; oi < old_ids.size(); ++oi) {
            const OperationTrack* runtime = find_runtime_for_item_(old_ids[oi]);
            if (runtime && is_claim_protected(*runtime)) {
                has_protected_old_track = true;
                break;
            }
        }
        if (old_ids.size() < 2 || !classes_with_runtime.count(cls_id) ||
            has_protected_old_track || detection_indices.size() >= old_ids.size()) {
            continue;
        }

        active_deficit_classes.insert(cls_id);
        std::set<int> remaining_items(old_ids.begin(), old_ids.end());
        std::set<int> remaining_detections;
        std::set<int> confirmed_d_detection_indices;
        for (size_t di = 0; di < detection_indices.size(); ++di) {
            const int detection_index = detection_indices[di];
            const Detection& detection = detections[detection_index];
            const bool owned_by_confirmed_d = detection_matches_confirmed_suspect_d(
                detection, track_buffer_);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d input box=(%.1f,%.1f,%.1f,%.1f) score=%.3f confirmed-d=%d",
                   cls_id, detection_index, detection.box.x1, detection.box.y1,
                   detection.box.x2, detection.box.y2, detection.score,
                   owned_by_confirmed_d ? 1 : 0);
            if (!owned_by_confirmed_d) {
                remaining_detections.insert(detection_index);
            } else {
                confirmed_d_detection_indices.insert(detection_index);
            }
        }
        std::set<int> survivors;
        std::map<int, const char*> assignment_reasons;

        // 先用双方都唯一的严格原位框锁住静止邻居。这样 B 的独立框不会
        // 被 A 的宽松路径再次解释为歧义。
        bool made_progress = true;
        while (made_progress) {
            made_progress = false;
            std::map<int, std::vector<int> > item_candidates;
            std::map<int, std::vector<int> > detection_candidates;
            for (std::set<int>::const_iterator item_id = remaining_items.begin();
                 item_id != remaining_items.end(); ++item_id) {
                std::map<int, InventoryItem>::const_iterator original =
                    operation_start_inventory_.find(*item_id);
                if (original == operation_start_inventory_.end()) continue;
                const BBox reference = original->second.base_box.area() > 0.0f
                    ? original->second.base_box : original->second.box;
                for (std::set<int>::const_iterator detection_index =
                         remaining_detections.begin();
                     detection_index != remaining_detections.end(); ++detection_index) {
                    if (!strict_match_box(original->second.cls_id, reference,
                                          detections[*detection_index].cls_id,
                                          detections[*detection_index].box)) {
                        continue;
                    }
                    item_candidates[*item_id].push_back(*detection_index);
                    detection_candidates[*detection_index].push_back(*item_id);
                }
            }
            for (std::map<int, std::vector<int> >::const_iterator item =
                     item_candidates.begin();
                 item != item_candidates.end(); ++item) {
                if (item->second.size() != 1) continue;
                const int detection_index = item->second.front();
                if (detection_candidates[detection_index].size() != 1 ||
                    !remaining_items.count(item->first) ||
                    !remaining_detections.count(detection_index)) {
                    continue;
                }
                visible_count_detection_owner_[detection_index] = item->first;
                visible_count_survivor_ids_.insert(item->first);
                survivors.insert(item->first);
                assignment_reasons[detection_index] = "unique-direct-static";
                remaining_items.erase(item->first);
                remaining_detections.erase(detection_index);
                trace_("VISIBLE-COUNT",
                       "cls=%d detection=%d reserved-for-item=%d reason=unique-direct-static",
                       cls_id, detection_index, item->first);
                made_progress = true;
            }
        }

        // 只有画面真的不可区分时才使用上一张无手帧的存活 id 作为稳定决胜；
        // 它不是 map 遍历顺序，也不会单独证明 MOVED。
        const std::map<int, std::set<int> >::const_iterator previous_survivors_it =
            visible_count_prior_survivors_by_cls_.find(cls_id);
        const std::set<int> empty_previous_survivors;
        const std::set<int>& previous_survivors =
            previous_survivors_it == visible_count_prior_survivors_by_cls_.end()
                ? empty_previous_survivors : previous_survivors_it->second;
        const std::map<int, std::map<int, BBox> >::const_iterator previous_boxes_it =
            visible_count_prior_survivor_boxes_by_cls_.find(cls_id);
        const std::map<int, BBox> empty_previous_boxes;
        const std::map<int, BBox>& previous_survivor_boxes =
            previous_boxes_it == visible_count_prior_survivor_boxes_by_cls_.end()
                ? empty_previous_boxes : previous_boxes_it->second;
        for (std::set<int>::const_iterator prior = previous_survivors.begin();
             prior != previous_survivors.end(); ++prior) {
            if (!remaining_items.count(*prior) || remaining_detections.empty()) continue;
            const std::map<int, BBox>::const_iterator previous_box =
                previous_survivor_boxes.find(*prior);
            if (previous_box == previous_survivor_boxes.end()) continue;
            const OperationTrack* runtime = find_runtime_for_item_(*prior);
            std::map<int, InventoryItem>::const_iterator original =
                operation_start_inventory_.find(*prior);
            if (original == operation_start_inventory_.end()) continue;
            int best_detection = -1;
            float best_cost = std::numeric_limits<float>::infinity();
            for (std::set<int>::const_iterator detection_index =
                     remaining_detections.begin();
                 detection_index != remaining_detections.end(); ++detection_index) {
                if (!visible_count_survivor_box_is_continuous(
                        cls_id, previous_box->second,
                        detections[*detection_index].box)) {
                    trace_("VISIBLE-COUNT",
                           "cls=%d detection=%d prior-survivor=%d action=reject-prior reason=box-discontinuous",
                           cls_id, *detection_index, *prior);
                    continue;
                }
                const float cost = visible_count_owner_cost(
                    original->second, runtime, detections[*detection_index]);
                if (cost + 0.0001f < best_cost ||
                    (std::fabs(cost - best_cost) <= 0.0001f &&
                     (best_detection < 0 || *detection_index < best_detection))) {
                    best_cost = cost;
                    best_detection = *detection_index;
                }
            }
            if (best_detection < 0) continue;
            visible_count_detection_owner_[best_detection] = *prior;
            visible_count_survivor_ids_.insert(*prior);
            survivors.insert(*prior);
            assignment_reasons[best_detection] = "previous-no-hand-survivor";
            remaining_items.erase(*prior);
            remaining_detections.erase(best_detection);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d reserved-for-item=%d reason=previous-no-hand-survivor",
                   cls_id, best_detection, *prior);
        }

        // 仍不可区分时，选择全局最小成本 pair；完全相同才按 item_id /
        // detection index 决胜，保证两张连续无手帧不会因遍历顺序换身份。
        while (!remaining_items.empty() && !remaining_detections.empty()) {
            int best_item = -1;
            int best_detection = -1;
            float best_cost = std::numeric_limits<float>::infinity();
            for (std::set<int>::const_iterator item_id = remaining_items.begin();
                 item_id != remaining_items.end(); ++item_id) {
                const OperationTrack* runtime = find_runtime_for_item_(*item_id);
                std::map<int, InventoryItem>::const_iterator original =
                    operation_start_inventory_.find(*item_id);
                if (original == operation_start_inventory_.end()) continue;
                for (std::set<int>::const_iterator detection_index =
                         remaining_detections.begin();
                     detection_index != remaining_detections.end(); ++detection_index) {
                    const float cost = visible_count_owner_cost(
                        original->second, runtime, detections[*detection_index]);
                    const bool better_cost = cost + 0.0001f < best_cost;
                    const bool equal_cost = std::fabs(cost - best_cost) <= 0.0001f;
                    const bool better_tie = equal_cost &&
                        (best_item < 0 || *item_id < best_item ||
                         (*item_id == best_item && *detection_index < best_detection));
                    if (better_cost || better_tie) {
                        best_cost = cost;
                        best_item = *item_id;
                        best_detection = *detection_index;
                    }
                }
            }
            if (best_item < 0 || best_detection < 0) break;
            visible_count_detection_owner_[best_detection] = best_item;
            visible_count_survivor_ids_.insert(best_item);
            survivors.insert(best_item);
            assignment_reasons[best_detection] = "min-cost";
            remaining_items.erase(best_item);
            remaining_detections.erase(best_detection);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d reserved-for-item=%d reason=min-cost",
                   cls_id, best_detection, best_item);
        }

        std::map<int, BBox> current_survivor_boxes;
        for (std::map<int, int>::const_iterator assignment =
                 visible_count_detection_owner_.begin();
             assignment != visible_count_detection_owner_.end(); ++assignment) {
            if (!survivors.count(assignment->second) ||
                detections[assignment->first].cls_id != cls_id) {
                continue;
            }
            current_survivor_boxes[assignment->second] = detections[assignment->first].box;
        }
        bool survivor_box_discontinuous = false;
        for (std::set<int>::const_iterator prior = previous_survivors.begin();
             prior != previous_survivors.end(); ++prior) {
            if (!survivors.count(*prior)) continue;
            const std::map<int, BBox>::const_iterator previous_box =
                previous_survivor_boxes.find(*prior);
            const std::map<int, BBox>::const_iterator current_box =
                current_survivor_boxes.find(*prior);
            if (previous_box == previous_survivor_boxes.end() ||
                current_box == current_survivor_boxes.end() ||
                !visible_count_survivor_box_is_continuous(
                    cls_id, previous_box->second, current_box->second)) {
                survivor_box_discontinuous = true;
                trace_("VISIBLE-COUNT",
                       "cls=%d survivor=%d action=clear-deficit reason=survivor-box-discontinuous "
                       "previous=(%.1f,%.1f,%.1f,%.1f) current=(%.1f,%.1f,%.1f,%.1f)",
                       cls_id, *prior,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.x1,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.y1,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.x2,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.y2,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.x1,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.y1,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.x2,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.y2);
            }
        }
        if (survivor_box_discontinuous) {
            continuity_reset_item_ids.insert(old_ids.begin(), old_ids.end());
        }
        visible_count_prior_survivors_by_cls_[cls_id] = survivors;
        visible_count_prior_survivor_boxes_by_cls_[cls_id] = current_survivor_boxes;
        for (size_t oi = 0; oi < old_ids.size(); ++oi) {
            if (!survivors.count(old_ids[oi])) {
                next_out_candidates.insert(old_ids[oi]);
            }
        }
        for (size_t di = 0; di < detection_indices.size(); ++di) {
            const int detection_index = detection_indices[di];
            const Detection& detection = detections[detection_index];
            if (confirmed_d_detection_indices.count(detection_index)) {
                trace_("VISIBLE-COUNT",
                       "cls=%d detection=%d decision=confirmed-d box=(%.1f,%.1f,%.1f,%.1f) score=%.3f",
                       cls_id, detection_index, detection.box.x1, detection.box.y1,
                       detection.box.x2, detection.box.y2, detection.score);
                continue;
            }
            const std::map<int, int>::const_iterator owner =
                visible_count_detection_owner_.find(detection_index);
            if (owner == visible_count_detection_owner_.end()) {
                trace_("VISIBLE-COUNT",
                       "cls=%d detection=%d decision=unassigned box=(%.1f,%.1f,%.1f,%.1f) score=%.3f",
                       cls_id, detection_index, detection.box.x1, detection.box.y1,
                       detection.box.x2, detection.box.y2, detection.score);
                continue;
            }
            const std::map<int, const char*>::const_iterator reason =
                assignment_reasons.find(detection_index);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d decision=reserve item=%d reason=%s box=(%.1f,%.1f,%.1f,%.1f) score=%.3f",
                   cls_id, detection_index, owner->second,
                   reason == assignment_reasons.end() ? "unknown" : reason->second,
                   detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2,
                   detection.score);
        }
        for (size_t oi = 0; oi < old_ids.size(); ++oi) {
            if (survivors.count(old_ids[oi])) continue;
            for (std::map<int, int>::const_iterator assignment =
                     visible_count_detection_owner_.begin();
                 assignment != visible_count_detection_owner_.end(); ++assignment) {
                if (detections[assignment->first].cls_id != cls_id) continue;
                trace_("VISIBLE-COUNT",
                       "detection=%d reserved_for_item=%d excluded_from_item=%d "
                       "reason=independent-same-class-survivor",
                       assignment->first, assignment->second, old_ids[oi]);
            }
        }
        trace_("VISIBLE-COUNT",
               "cls=%d old-visible=%zu raw-detections=%zu retained-old=%zu deficit=%zu",
               cls_id, old_ids.size(), detection_indices.size(), survivors.size(),
               old_ids.size() - survivors.size());
    }

    for (std::map<int, std::set<int> >::iterator prior =
             visible_count_prior_survivors_by_cls_.begin();
         prior != visible_count_prior_survivors_by_cls_.end();) {
        if (!active_deficit_classes.count(prior->first)) {
            visible_count_prior_survivors_by_cls_.erase(prior++);
        } else {
            ++prior;
        }
    }
    for (std::map<int, std::map<int, BBox> >::iterator prior =
             visible_count_prior_survivor_boxes_by_cls_.begin();
         prior != visible_count_prior_survivor_boxes_by_cls_.end();) {
        if (!active_deficit_classes.count(prior->first)) {
            visible_count_prior_survivor_boxes_by_cls_.erase(prior++);
        } else {
            ++prior;
        }
    }

    // 数量恢复、类别不再短缺或手再次出现之前，都不能让上一轮缺失证据在
    // 随机帧继续累积；已标记但尚未提交的 OUT 也要一并撤销。
    for (std::map<int, int>::iterator missing = visible_count_missing_counts_.begin();
         missing != visible_count_missing_counts_.end();) {
        const bool reset_for_discontinuity =
            continuity_reset_item_ids.count(missing->first) > 0;
        if (next_out_candidates.count(missing->first) && !reset_for_discontinuity) {
            ++missing;
            continue;
        }
        const int item_id = missing->first;
        if (visible_count_confirmed_out_ids_.count(item_id)) {
            pending_out_ids_.erase(item_id);
            visible_count_confirmed_out_ids_.erase(item_id);
            OperationTrack* track = find_runtime_for_item_(item_id);
            if (track && !track->is_suspect_new &&
                track->resolution == ExistingItemResolution::OUT_CONFIRMED) {
                track->resolution = ExistingItemResolution::NONE;
                track->needs_no_hand_settlement = true;
                track->release_reason = ReleaseReason::NONE;
            }
            trace_("VISIBLE-COUNT",
                   "item=%d action=retract-confirmed-out reason=%s",
                   item_id, reset_for_discontinuity
                       ? "survivor-box-discontinuous" : "visible-count-recovered");
        }
        OperationTrack* track = find_runtime_for_item_(item_id);
        if (track && !track->is_suspect_new) track->no_hand_missing_count = 0;
        missing = visible_count_missing_counts_.erase(missing);
    }

    visible_count_out_candidate_ids_.swap(next_out_candidates);
    for (std::set<int>::const_iterator candidate =
             visible_count_out_candidate_ids_.begin();
         candidate != visible_count_out_candidate_ids_.end(); ++candidate) {
        const int old_count = visible_count_missing_counts_[*candidate];
        const int new_count = old_count + 1;
        visible_count_missing_counts_[*candidate] = new_count;
        OperationTrack* track = find_runtime_for_item_(*candidate);
        if (track && !track->is_suspect_new) {
            track->no_hand_missing_count = new_count;
        }
        trace_("VISIBLE-COUNT",
               "item=%d action=count-deficit old=%d new=%d threshold=%d",
               *candidate, old_count, new_count, FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        if (new_count >= FLOW3_NO_HAND_OUT_MISSING_FRAMES &&
            !pending_out_ids_.count(*candidate)) {
            mark_pending_out_(*candidate);
            visible_count_confirmed_out_ids_.insert(*candidate);
            if (track && !track->is_suspect_new) {
                trace_track_("VISIBLE-COUNT", *track,
                             "confirm-out-visible-count-deficit");
            } else {
                trace_("VISIBLE-COUNT",
                       "item=%d action=confirm-out reason=visible-count-deficit",
                       *candidate);
            }
        }
    }
}

void SessionManager::refresh_confirmed_blockers_(const std::set<int>& /*observed_working_ids*/) {
    // 最终无手直接帧中统一重算。手还在时不按任意框交集写 block_ids，
    // 避免旧 2.0 版本那种 blocker 不断膨胀的问题。
}

bool SessionManager::has_unresolved_no_hand_state_(
        const std::set<int>& observed_item_ids,
        const std::set<int>& fully_occluded_item_ids) {
    bool unresolved = false;

    // 数量不足 OUT 候选中可能有“本次手操作没有单独建轨”的旧 C。它们也
    // 必须让整轮操作等待到第二张直接无手帧，不能因为没有 OperationTrack
    // 就在首帧把旧库存原样提交回去。
    for (std::set<int>::const_iterator item =
             visible_count_out_candidate_ids_.begin();
         item != visible_count_out_candidate_ids_.end(); ++item) {
        if (observed_item_ids.count(*item) || fully_occluded_item_ids.count(*item)) {
            continue;
        }
        if (pending_out_ids_.count(*item)) {
            trace_("VISIBLE-COUNT",
                   "item=%d settlement=out-confirmed-visible-count-deficit",
                   *item);
        } else {
            unresolved = true;
            trace_("VISIBLE-COUNT",
                   "item=%d settlement=defer-visible-count-deficit count=%d",
                   *item, visible_count_missing_counts_[*item]);
        }
    }

    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;

        if (track.is_suspect_new) {
            if (!is_active_runtime_track(track)) continue;
            // D 只有在至少两张直接无手帧中连续匹配到自己后才可提交 IN。
            // 这是单对象时序证据，不是多检测框投票。
            if (!track.promoted_to_working_inventory || track.item_id <= 0 ||
                !observed_item_ids.count(track.item_id) ||
                track.no_hand_self_match_count < FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                unresolved = true;
                trace_track_("NO-HAND", track, "suspect-not-yet-directly-confirmed");
            }
            continue;
        }

        if (!existing_item_needs_settlement(track)) continue;

        const bool is_visible_count_out_candidate =
            visible_count_out_candidate_ids_.count(track.item_id) &&
            !observed_item_ids.count(track.item_id) &&
            !fully_occluded_item_ids.count(track.item_id);
        const bool is_visible_count_survivor =
            visible_count_survivor_ids_.count(track.item_id) &&
            observed_item_ids.count(track.item_id);
        if (is_visible_count_out_candidate) {
            // 该 C 已经在本帧同类一对一分配中没有任何可见框。邻居 B 的框
            // 已被独占保留，不能再作为 A 的歧义候选阻止连续 OUT 计数。
            trace_track_("VISIBLE-COUNT", track,
                         pending_out_ids_.count(track.item_id)
                             ? "out-confirmed-visible-count-deficit"
                             : "await-visible-count-deficit-confirmation");
            continue;
        }
        if (is_visible_count_survivor) {
            // 强制保留只代表一个可见实例，不使它绕过本轮“连续无手”要求；
            // 本函数被调用前已由 prepare_visible_count_settlement_ 保证两帧
            // 都会重新建立同一保留关系。
            track.no_hand_missing_count = 0;
            trace_track_("VISIBLE-COUNT", track,
                         "directly-observed-visible-count-survivor");
            continue;
        }

        // 状态被意外变成 NORMAL 绝不是“旧 C 已结案”。此时没有可靠活动
        // 轨迹可安全累计 OUT，必须保持未决并阻止本轮提交。
        if (!is_active_runtime_track(track)) {
            unresolved = true;
            trace_track_("NO-HAND", track, "normal-track-still-needs-settlement");
            continue;
        }

        if (is_claim_protected(track) || track.b_claim_ambiguous ||
            track.contact_path_ambiguous || track.no_hand_candidate_ambiguous) {
            unresolved = true;
            trace_track_("NO-HAND", track,
                         track.no_hand_candidate_ambiguous
                             ? "ambiguous-no-hand-path-candidate"
                             : "claim-or-contact-ambiguity");
            continue;
        }
        if (old_track_has_unresolved_alias_(track)) {
            unresolved = true;
            trace_track_("C-D-ALIAS", track,
                         "old-c-awaits-quarantined-pending-d-no-hand-arbitration");
            continue;
        }
        if (observed_item_ids.count(track.item_id)) {
            // 手离开后首次重新出现的同类 B 仍要在下一张直接无手帧中和
            // 自己连续匹配。不能让本帧的一对一绑定绕过这条身份确认链路。
            if (track.has_reappear_candidate_box &&
                !reappear_candidate_is_confirmed(track)) {
                unresolved = true;
                trace_track_("NO-HAND", track, "reappear-candidate-needs-second-frame");
                continue;
            }
            track.no_hand_missing_count = 0;
            trace_track_("NO-HAND", track, "directly-observed");
            continue;
        }
        if (fully_occluded_item_ids.count(track.item_id)) {
            track.no_hand_missing_count = 0;
            track.resolution = ExistingItemResolution::OCCLUDED_CONFIRMED;
            track.release_reason = ReleaseReason::FULLY_OCCLUDED;
            track.needs_no_hand_settlement = false;
            track.stable_near_original_no_hand_count = 0;
            track.has_stable_near_original_box = false;
            track.stable_near_original_box = BBox();
            trace_track_("NO-HAND", track, "fully-occluded-by-confirmed-front-item");
            continue;
        }
        if (pending_out_ids_.count(track.item_id)) continue;

        // CONTACT_CANDIDATE 没有足够的真实移动证据，继续保持未决；只有
        // CONTACT_MOVING 才可沿实际观察路径累计无手缺失并最终 OUT。
        const bool contact_out_evidence =
            track.contact_state == ContactState::CONTACT_MOVING &&
            track.hold_and_move && !track.observed_move_values.empty();
        const bool hand_out_evidence =
            track.contact_state == ContactState::NONE &&
            track.state != OperationTrackState::NORMAL &&
            (track.hold_and_move || has_meaningful_hand_move(track));
        if (!contact_out_evidence && !hand_out_evidence) {
            unresolved = true;
            trace_track_("NO-HAND", track, "missing-without-sufficient-out-evidence");
            continue;
        }

        const int old_missing_count = track.no_hand_missing_count;
        ++track.no_hand_missing_count;
        trace_("NO-HAND", "item=%d direct-missing-count=%d->%d threshold=%d",
               track.item_id, old_missing_count, track.no_hand_missing_count,
               FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        trace_track_("NO-HAND", track, "direct-missing-frame");
        if (track.no_hand_missing_count >= FLOW3_NO_HAND_OUT_MISSING_FRAMES) {
            mark_pending_out_(track.item_id);
        } else {
            unresolved = true;
        }
    }

    // 最终保险：即使此前某个旧 C 的运行时记录已经意外失活，只要同类 D
    // 要正式 IN，而该旧 C 在本帧没有独立观测/遮挡/已确认结束，仍拒绝提交。
    for (std::set<int>::const_iterator pending = pending_in_ids_.begin();
         pending != pending_in_ids_.end(); ++pending) {
        std::map<int, InventoryItem>::const_iterator d = working_inventory_.find(*pending);
        if (d == working_inventory_.end()) continue;
        for (std::map<int, InventoryItem>::const_iterator old =
                 operation_start_inventory_.begin();
             old != operation_start_inventory_.end(); ++old) {
            if (old->second.cls_id != d->second.cls_id ||
                pending_out_ids_.count(old->first) ||
                observed_item_ids.count(old->first) ||
                fully_occluded_item_ids.count(old->first)) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(old->first);
            if (existing_item_resolved_without_current_detection(runtime)) continue;
            unresolved = true;
            trace_("SETTLE",
                   "block-commit pending-d=%d cls=%d unresolved-old=%d runtime=%s resolution=%s",
                   *pending, d->second.cls_id, old->first,
                   runtime ? operation_track_state_name(runtime->state) : "NONE",
                   runtime ? existing_resolution_name(runtime->resolution) : "NONE");
            break;
        }
    }
    return unresolved;
}

SettlementResult SessionManager::settle_no_hand_frame_(
        const std::vector<Detection>& detections, int frame_id) {
    SettlementResult result;
    if (!working_inventory_active_) return result;

    // 细节9：先得到本帧同类“一框一物品”的唯一保留关系。它只在可见数量
    // 已不足时填补单摄像头无法观察个体身份的空白，普通 C->B/D 仲裁不变。
    prepare_visible_count_settlement_(detections);

    std::map<int, InventoryItem> final_items = working_inventory_;
    const std::vector<Detection>& observed = detections;
    std::vector<int> observation_owner(observed.size(), -1);
    std::map<int, int> item_to_observation;
    std::map<int, BBox> references;
    std::set<int> track_priority_ids;
    std::set<int> ambiguous_ids;
    std::set<int> all_ids;
    std::map<int, BBox> original_references;
    std::set<int> original_position_ids;

    for (std::map<int, InventoryItem>::const_iterator it = final_items.begin();
         it != final_items.end(); ++it) {
        all_ids.insert(it->first);
        references[it->first] = it->second.base_box.area() > 0.0f
            ? it->second.base_box : it->second.box;
        const OperationTrack* runtime = find_runtime_for_item_(it->first);
        const bool runtime_in_claim_grace = runtime && is_claim_protected(*runtime);
        const bool runtime_has_ambiguous_b = runtime &&
            (runtime->b_claim_ambiguous || runtime->contact_path_ambiguous);
        if (runtime_has_ambiguous_b) ambiguous_ids.insert(it->first);
        if (!runtime_in_claim_grace && runtime && runtime->has_placed_box) {
            references[it->first] = runtime->placed_box;
        } else if (!runtime_in_claim_grace && runtime && runtime->is_suspect_new &&
                   runtime->has_last_seen_box) {
            references[it->first] = runtime->last_seen_box;
        }
        else if (!runtime_in_claim_grace && runtime && !runtime->observed_track.empty()) {
            references[it->first] = runtime->observed_track.back();
        }
        else if (!runtime_in_claim_grace && runtime && !runtime->is_suspect_new &&
                 reappear_candidate_is_confirmed(*runtime)) {
            references[it->first] = runtime->reappear_candidate_box;
        }
        else if (!runtime_in_claim_grace && runtime && !runtime->track.empty()) {
            references[it->first] = runtime->track.back();
        }
        if (!runtime_has_ambiguous_b &&
            (pending_in_ids_.count(it->first) || confirmed_moved_ids_.count(it->first) ||
             confirmed_position_update_ids_.count(it->first) ||
            (!runtime_in_claim_grace && runtime && (runtime->hold_and_move ||
                         runtime->contact_state != ContactState::NONE ||
                         (has_meaningful_hand_move(*runtime) &&
                          runtime->state != OperationTrackState::NORMAL))))) {
            track_priority_ids.insert(it->first);
        }
    }

    // 同类 B 仍有多个成熟来源时，暂时把这些 C 从所有“动态/宽松”绑定
    // 阶段移出；后面的原位置/缺失处理会保留旧库存，不按 item_id 强选一个。
    for (std::set<int>::const_iterator it = ambiguous_ids.begin();
         it != ambiguous_ids.end(); ++it) {
        all_ids.erase(*it);
        track_priority_ids.erase(*it);
    }

    // 已由可见数量结算保留给某个旧 C 的框必须先占用，不能让同一个框随后
    // 又被另一个同类 C 的宽松轨迹绑定。即使旧 C 的运行时路径有歧义，这个
    // 强制绑定也只表示“保留一个可见实例”，不单独生成 MOVED。
    for (std::map<int, int>::const_iterator assignment =
             visible_count_detection_owner_.begin();
         assignment != visible_count_detection_owner_.end(); ++assignment) {
        const int detection_index = assignment->first;
        const int item_id = assignment->second;
        if (detection_index < 0 ||
            static_cast<size_t>(detection_index) >= observed.size() ||
            !final_items.count(item_id) || pending_out_ids_.count(item_id) ||
            observation_owner[detection_index] >= 0 ||
            item_to_observation.count(item_id)) {
            continue;
        }
        if (final_items[item_id].cls_id != observed[detection_index].cls_id) continue;
        item_to_observation[item_id] = detection_index;
        observation_owner[detection_index] = item_id;
        trace_("VISIBLE-COUNT",
               "detection=%d reserved-for-item=%d action=force-no-hand-observation",
               detection_index, item_id);
    }

    // 动态轨迹没有唯一终点时，不能因为当前框暂时漏检就直接把旧物品
    // 当成 OUT。把本次操作开始前的原位置作为第二条独立证据源；它只
    // 用于旧库存，不允许刚提升的 D 抢回自己的终点框。
    for (std::map<int, InventoryItem>::const_iterator it =
             operation_start_inventory_.begin();
         it != operation_start_inventory_.end(); ++it) {
        if (!final_items.count(it->first) || pending_in_ids_.count(it->first)) continue;
        original_position_ids.insert(it->first);
        original_references[it->first] = it->second.base_box.area() > 0.0f
            ? it->second.base_box : it->second.box;
    }
    for (std::set<int>::const_iterator it = ambiguous_ids.begin();
         it != ambiguous_ids.end(); ++it) {
        original_position_ids.erase(*it);
    }

    // 绑定顺序：已确认移动/新 D 的终点 -> 普通严格匹配 -> 局部匹配。
    bind_mutually_unique(final_items, track_priority_ids, observed,
                         &references, &item_to_observation, &observation_owner, true, false);
    bind_mutually_unique(final_items, all_ids, observed,
                         &references, &item_to_observation, &observation_owner, false, false);
    // 若物品在手尚未离开时已经掉队并停在候选路径中段，终点框不会命中；
    // 在原位置回查前再做一次整条 path 的唯一最近绑定，避免这类物品被 OUT。
    bind_mutually_unique_track_paths(final_items, track_priority_ids, observed,
                                     track_buffer_, &item_to_observation, &observation_owner);
    // 再回查旧位置。这样 hold_and_move 尚未凑满、或只被手短暂擦过的
    // 物品，只要在原位重新出现，就不会因为轨迹参考框漂移而被误判出库。
    bind_mutually_unique(final_items, original_position_ids, observed,
                         &original_references, &item_to_observation, &observation_owner,
                         false, false);
    bind_mutually_unique(final_items, original_position_ids, observed,
                         &original_references, &item_to_observation, &observation_owner,
                         false, true);
    bind_mutually_unique(final_items, all_ids, observed,
                         &references, &item_to_observation, &observation_owner, false, true);

    // 文档允许 HAND_* 下尚未来得及凑满两次证据的物品，在手离开后的直接
    // 无手帧中由自己的候选轨迹补确认整理。这里仍要求它命中了自己的轨迹参考框，
    // 不把普通未匹配框当作移动终点。
    for (std::map<int, int>::const_iterator it = item_to_observation.begin();
         it != item_to_observation.end(); ++it) {
        const OperationTrack* runtime = find_runtime_for_item_(it->first);
        std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(it->first);
        if (!runtime || original == operation_start_inventory_.end() ||
            !is_active_runtime_track(*runtime) || is_claim_protected(*runtime)) {
            continue;
        }
        // CONTACT_CANDIDATE 必须先凑够两次有效 B 才能确认整理；仅有一条
        // observed_move_values 不能绕过 CONTACT_MOVING 门槛。普通 HAND_*
        // 仍保留原有的 move_values/hold_and_move 收尾规则。
        const bool has_contact_confirmation =
            runtime->contact_state == ContactState::CONTACT_MOVING &&
            runtime->hold_and_move;
        const bool has_hand_confirmation = runtime->contact_state == ContactState::NONE &&
            (runtime->hold_and_move || has_meaningful_hand_move(*runtime));
        if (!has_contact_confirmation && !has_hand_confirmation &&
            runtime->observed_move_values.empty()) {
            continue;
        }
        if (runtime->contact_state != ContactState::NONE &&
            !has_contact_confirmation) {
            continue;
        }
        const bool matches_endpoint = track_match_box(
            original->second.cls_id, references[it->first],
            observed[it->second].cls_id, observed[it->second].box);
        const bool matches_any_path = track_path_match_cost(
            original->second, *runtime, observed[it->second]) <
            std::numeric_limits<float>::infinity();
        if ((matches_endpoint || matches_any_path) &&
            boxes_differ_as_move(original->second.base_box, observed[it->second].box)) {
            confirmed_moved_ids_.insert(it->first);
        }
    }

    std::set<int> observed_ids;
    for (std::map<int, int>::const_iterator it = item_to_observation.begin();
         it != item_to_observation.end(); ++it) {
        observed_ids.insert(it->first);
        InventoryItem& item = final_items[it->first];
        update_seen(item, observed[it->second], frame_id);
        if (pending_in_ids_.count(it->first) || confirmed_moved_ids_.count(it->first) ||
            confirmed_position_update_ids_.count(it->first)) {
            item.base_box = observed[it->second].box;
        }
        item.status = ItemStatus::VISIBLE;
        item.block_ids.clear();
    }

    // 仅“本轮确认移动/确认入库，且在当前无手帧直接出现”的物品可成为前景遮挡物。
    std::set<int> confirmed_front_ids;
    for (std::set<int>::const_iterator it = observed_ids.begin(); it != observed_ids.end(); ++it) {
        if (pending_in_ids_.count(*it) || confirmed_moved_ids_.count(*it)) {
            confirmed_front_ids.insert(*it);
        }
    }

    std::set<int> fully_occluded_ids;
    for (std::map<int, InventoryItem>::iterator it = final_items.begin();
         it != final_items.end(); ++it) {
        if (observed_ids.count(it->first)) continue;
        std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(it->first);
        if (original == operation_start_inventory_.end()) continue;

        std::vector<int> blockers;
        std::vector<BBox> cover_boxes;
        for (std::set<int>::const_iterator front_id = confirmed_front_ids.begin();
             front_id != confirmed_front_ids.end(); ++front_id) {
            if (*front_id == it->first) continue;
            std::map<int, InventoryItem>::const_iterator front = final_items.find(*front_id);
            if (front == final_items.end()) continue;
            const BBox front_box = front->second.base_box.area() > 0.0f
                ? front->second.base_box : front->second.box;
            if (intersection_area(original->second.base_box, front_box) <=
                BLOCK_OVERLAP_AREA_EPS) {
                continue;
            }
            // 先收集所有已确认前景框，再由矩形差集计算它们的覆盖并集。
            // 单个新 D 的部分覆盖不能把缺失 C 直接定为 OCCLUDED。
            blockers.push_back(*front_id);
            cover_boxes.push_back(front_box);
        }
        if (fully_covered_by(original->second.base_box, cover_boxes)) {
            it->second = original->second;
            it->second.status = ItemStatus::OCCLUDED;
            it->second.block_ids.clear();
            it->second.block_ids.insert(blockers.begin(), blockers.end());
            fully_occluded_ids.insert(it->first);
            continue;
        }

        // 不完整遮挡和本帧缺失都不是最终状态。旧 C 是否可 OUT 由下面的
        // 连续直接无手缺失证据统一处理；在此之前保留原库存身份。
        it->second = original->second;
    }

    const bool has_unresolved_state = has_unresolved_no_hand_state_(
        observed_ids, fully_occluded_ids);

    for (std::set<int>::const_iterator out = pending_out_ids_.begin();
         out != pending_out_ids_.end(); ++out) {
        final_items.erase(*out);
    }

    if (has_unresolved_state) {
        trace_("SETTLE",
               "defer-commit inventory=%zu tracks=%zu pending-in=%zu pending-out=%zu",
               final_items.size(), track_buffer_.size(), pending_in_ids_.size(),
               pending_out_ids_.size());
        return result;
    }

    // blocker 只保留存在且真正覆盖的已确认前景物品。
    for (std::map<int, InventoryItem>::iterator it = final_items.begin();
         it != final_items.end(); ++it) {
        for (std::set<int>::iterator blocker = it->second.block_ids.begin();
             blocker != it->second.block_ids.end();) {
            std::map<int, InventoryItem>::const_iterator front = final_items.find(*blocker);
            if (front == final_items.end() || *blocker == it->first ||
                !confirmed_front_ids.count(*blocker) ||
                intersection_area(it->second.base_box, front->second.base_box) <=
                    BLOCK_OVERLAP_AREA_EPS) {
                blocker = it->second.block_ids.erase(blocker);
            } else {
                ++blocker;
            }
        }
        if (it->second.status == ItemStatus::OCCLUDED && it->second.block_ids.empty() &&
            observed_ids.count(it->first)) {
            it->second.status = ItemStatus::VISIBLE;
        }
    }

    // 最后才生成正式事件；整个过程中没有任何“未绑定无手框自动 IN”。
    std::vector<InventoryEvent> events;
    for (std::map<int, InventoryItem>::const_iterator old = operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        std::map<int, InventoryItem>::const_iterator now = final_items.find(old->first);
        if (now == final_items.end()) {
            if (pending_out_ids_.count(old->first)) {
                events.push_back(make_event(EventKind::OUT, old->second));
            }
            continue;
        }
        if (confirmed_moved_ids_.count(old->first) &&
            boxes_differ_as_move(old->second.base_box, now->second.base_box)) {
            events.push_back(make_event(EventKind::MOVED, now->second,
                                        old->second.base_box, now->second.base_box));
        }
        if (old->second.status != now->second.status) {
            if (now->second.status == ItemStatus::OCCLUDED) {
                events.push_back(make_event(EventKind::OCCLUDED, now->second));
            } else if (old->second.status == ItemStatus::OCCLUDED &&
                       now->second.status == ItemStatus::VISIBLE) {
                events.push_back(make_event(EventKind::REVEALED, now->second));
            }
        }
    }
    for (std::set<int>::const_iterator in = pending_in_ids_.begin();
         in != pending_in_ids_.end(); ++in) {
        std::map<int, InventoryItem>::const_iterator item = final_items.find(*in);
        if (item != final_items.end() && observed_ids.count(*in)) {
            events.push_back(make_event(EventKind::IN, item->second));
        }
    }

    for (size_t si = 0; si < observation_owner.size(); ++si) {
        if (observation_owner[si] < 0) {
            printf("[3.0] 无手直接帧出现未绑定检测 cls=%d；没有 D 证据链，不自动 IN\n",
                   observed[si].cls_id);
            trace_("SETTLE", "unbound-detection=%zu cls=%d result=no-auto-in",
                   si, observed[si].cls_id);
        }
    }

    trace_("SETTLE",
           "commit inventory-before=%zu inventory-after=%zu events=%zu pending-in=%zu pending-out=%zu",
           inventory_.size(), final_items.size(), events.size(), pending_in_ids_.size(),
           pending_out_ids_.size());
    for (std::map<int, OperationTrack>::const_iterator track = track_buffer_.begin();
         track != track_buffer_.end(); ++track) {
        trace_track_("SETTLE", track->second, "commit-final-track");
    }
    for (size_t event_index = 0; event_index < events.size(); ++event_index) {
        const InventoryEvent& event = events[event_index];
        trace_("SETTLE",
               "event=%s item=%d cls=%d box=(%.1f,%.1f,%.1f,%.1f) "
               "before=(%.1f,%.1f,%.1f,%.1f) after=(%.1f,%.1f,%.1f,%.1f)",
               event_kind_name(event.kind), event.item_id, event.cls_id,
               event.box.x1, event.box.y1, event.box.x2, event.box.y2,
               event.before_box.x1, event.before_box.y1,
               event.before_box.x2, event.before_box.y2,
               event.after_box.x1, event.after_box.y1,
               event.after_box.x2, event.after_box.y2);
    }

    inventory_.replace_all(final_items, working_next_item_id_);
    rebuild_persistent_item_index_();
    result.committed = true;
    result.happened = !events.empty();
    result.events.swap(events);
    clear_runtime_after_commit_();
    return result;
}

void SessionManager::clear_runtime_after_commit_() {
    trace_("STATE", "clear-operation-runtime-after-commit");
    reset_operation_runtime_();
}

void SessionManager::print_inventory() const {
    const size_t visible_count = inventory_.count_by_status(ItemStatus::VISIBLE);
    const size_t occluded_count = inventory_.count_by_status(ItemStatus::OCCLUDED);
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────┐\n");
    printf("  │  在库清单 │ 可见: %-3zu │ 遮挡: %-3zu │ 共: %-3zu            │\n",
           visible_count, occluded_count, visible_count + occluded_count);
    printf("  ├────┬──────────────┬────────┬────────┬──────────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 遮挡数 │ 位置（当前可见框中心） │\n");
    printf("  ├────┼──────────────┼────────┼────────┼──────────────────────────┤\n");
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        const InventoryItem& item = it->second;
        printf("  │ %-2d │ %-12s │ %-6s │ %-6zu │ (%4.0f,%4.0f)             │\n",
               item.item_id, cls_id_to_chinese(item.cls_id),
               item.status == ItemStatus::VISIBLE ? "可见" : "遮挡",
               item.block_ids.size(), item.box.cx(), item.box.cy());
    }
    printf("  └────┴──────────────┴────────┴────────┴──────────────────────────┘\n\n");
}

FrameProcessResult SessionManager::process_frame(
        const std::vector<Detection>& food_detections,
        const std::vector<BBox>& hand_boxes, int frame_id, long long time_ms) {
    FrameProcessResult output;
    trace_frame_id_ = frame_id;
    trace_hand_phase_ = !hand_boxes.empty();
    current_time_ms_ = time_ms;
    hand_present_ = !hand_boxes.empty();
    trace_("FRAME", "foods=%zu hands=%zu operation_active=%d tracks=%zu no_hand_streak=%d",
           food_detections.size(), hand_boxes.size(),
           working_inventory_active_ ? 1 : 0, track_buffer_.size(), no_hand_streak_);
    for (size_t i = 0; i < food_detections.size(); ++i) {
        const Detection& detection = food_detections[i];
        trace_("FOOD-DETECTION",
               "input-index=%zu cls=%d score=%.3f box=(%.1f,%.1f,%.1f,%.1f) "
               "business-input=1 osd-business-visible=1",
               i, detection.cls_id, detection.score,
               detection.box.x1, detection.box.y1,
               detection.box.x2, detection.box.y2);
    }

    if (!session_active_) {
        trace_("FRAME", "ignore-inactive-session foods=%zu hands=%zu",
               food_detections.size(), hand_boxes.size());
        return output;
    }

    if (hand_present_) {
        for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            OperationTrack& track = it->second;
            // r16 的稳定位置重同步证据只允许来自连续无手帧。手重新出现时，
            // 无论手是否接触该 C，都不能把前后两段无手证据拼在一起。
            if (!track.is_suspect_new) {
                reset_stable_near_original_no_hand_evidence_(
                    &track, "hand-returned");
                continue;
            }
            if (!track.pending_d_quarantined_by_old_c ||
                track.alias_no_hand_missing_count == 0) {
                continue;
            }
            trace_("C-D-ALIAS",
                   "old-count=%zu suspect=%d phase=HAND relation=hand-returned "
                   "missing-count=%d/%d action=reset-stale-alias-evidence",
                   track.conflicting_old_item_ids.size(), track.suspect_id,
                   track.alias_no_hand_missing_count,
                   FLOW3_NO_HAND_D_CONFIRM_FRAMES);
            track.alias_no_hand_missing_count = 0;
        }
        // 细节9的数量不足只属于一段连续无手收尾。手重新进入时，上一段
        // 尚未提交的“一框一物品”候选必须撤销，不能跨操作继续凑 OUT 帧数。
        if (!visible_count_missing_counts_.empty() ||
            !visible_count_confirmed_out_ids_.empty()) {
            clear_visible_count_settlement_(true);
        }
        finalize_initial_check_before_hand_();
        no_hand_streak_ = 0;
        if (!has_local_inventory_) return output;
        const BBox hand = choose_primary_hand(hand_boxes);
        if (!working_inventory_active_) {
            begin_working_operation_(hand, food_detections);
            return output;
        }
        // 手框不动时只有在没有任何未决 HAND_* / CONTACT_* 轨迹的情况下
        // 才能整帧跳过。手指可能在手腕不动时推动物品，所以活动轨迹必须
        // 继续逐帧检查真实物品检测框。
        bool has_active_runtime = false;
        for (std::map<int, OperationTrack>::const_iterator it =
                 track_buffer_.begin(); it != track_buffer_.end(); ++it) {
            if (is_active_runtime_track(it->second)) {
                has_active_runtime = true;
                break;
            }
        }
        if (!has_active_runtime && has_old_hand_box_ &&
            hand_boxes_effectively_same(old_hand_box_, hand)) {
            trace_("FRAME", "skip-stationary-hand-no-active-track");
            return output;
        }
        process_effective_hand_frame_(hand, food_detections, false);
        old_hand_box_ = hand;
        has_old_hand_box_ = true;
        return output;
    }

    ++no_hand_streak_;
    output.no_hand_frame_processed = true;
    if (working_inventory_active_) {
        // 手刚离开的每一帧都直接供 D/C 候选完成认领、连续自匹配和缺失证据。
        // 不缓存、不平均框，也不做多数投票。
        observe_no_hand_frame_(food_detections);
        has_old_hand_box_ = false;
        output.settlement = settle_no_hand_frame_(food_detections, frame_id);
        return output;
    }

    if (initial_check_state_ == InitialCheckState::WAITING ||
        initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_FRAME) {
        if (initial_check_state_ == InitialCheckState::WAITING) {
            validate_initial_no_hand_frame_(food_detections);
            initial_check_state_ = InitialCheckState::DONE;
        } else {
            initialize_from_bootstrap_no_hand_frame_(food_detections, frame_id);
        }
    }
    return output;
}

}  // namespace fridge
