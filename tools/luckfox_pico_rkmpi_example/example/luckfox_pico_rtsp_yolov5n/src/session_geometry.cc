// ============================================================================
//  session_geometry.cc
//  3.0 session internal geometry, matching, and box helpers
// ============================================================================
#include "session_internal.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace fridge {
namespace session_internal {

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
                       float iom_threshold) {
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
           track.resolution != ExistingItemResolution::OUT_CONFIRMED &&
           track.resolution != ExistingItemResolution::OCCLUDED_CONFIRMED;
}

bool existing_item_resolved_without_current_detection(const OperationTrack* track) {
    if (!track || track->is_suspect_new) return false;
    return track->resolution == ExistingItemResolution::MOVED_CONFIRMED ||
           track->resolution == ExistingItemResolution::OUT_CONFIRMED ||
           track->resolution == ExistingItemResolution::OCCLUDED_CONFIRMED;
}

bool existing_track_is_terminal(const OperationTrack& track) {
    return !track.is_suspect_new &&
           (track.resolution == ExistingItemResolution::MOVED_CONFIRMED ||
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
                          const BBox& before,
                          const BBox& after) {
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
        // 不能在每一步差集时套用最终的总面积阈值：多个单独很小的
        // 未覆盖残片加起来仍可能超过阈值。这里只丢弃零面积残片，
        // COVER_REMAINING_AREA_EPS 仅在 fully_covered_by() 的总面积判断中使用。
        if (pieces[i].area() > 0.0f) output->push_back(pieces[i]);
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

}  // namespace session_internal
}  // namespace fridge
