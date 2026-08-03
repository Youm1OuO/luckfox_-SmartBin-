// ============================================================================
//  session.cc
//  3.0：工作库存 + HAND_* 候选 + 疑似新物品 D + 无手逐帧条件提交
// ============================================================================
#include "session.h"
#include "fridge_config.h"

#include <algorithm>
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

bool reappear_candidate_is_confirmed(const OperationTrack& track) {
    return track.has_reappear_candidate_box &&
           track.reappear_candidate_match_count >=
               FLOW3_REAPPEAR_CANDIDATE_CONFIRM_FRAMES;
}

const char* identity_resolution_name(IdentityResolution resolution) {
    switch (resolution) {
        case IdentityResolution::UNRESOLVED: return "UNRESOLVED";
        case IdentityResolution::AT_ORIGIN: return "AT_ORIGIN";
        case IdentityResolution::AT_NEW_POSITION: return "AT_NEW_POSITION";
        case IdentityResolution::OCCLUDED_CONFIRMED: return "OCCLUDED_CONFIRMED";
        case IdentityResolution::OUT_CONFIRMED: return "OUT_CONFIRMED";
    }
    return "UNKNOWN";
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

void clear_reappear_candidate(OperationTrack* track, bool keep_pending) {
    if (!track) return;
    track->has_reappear_candidate_box = false;
    track->reappear_candidate_match_count = 0;
    track->reappear_candidate_last_direct_frame = -1;
    track->drop_evidence_count = 0;
    track->reappearance_pending = keep_pending;
    track->reappear_candidate_started_touching_hand = false;
}

void start_reappear_candidate(OperationTrack* track, const Detection& detection,
                              bool started_touching_hand, int direct_frame) {
    if (!track) return;
    track->reappear_candidate_box = detection.box;
    track->has_reappear_candidate_box = true;
    track->reappear_candidate_match_count = 1;
    track->reappear_candidate_last_direct_frame = direct_frame;
    track->drop_evidence_count = 0;
    track->reappearance_pending = false;
    track->reappear_candidate_started_touching_hand = started_touching_hand;
}

// 返回本次 B 是否让候选达到“连续自匹配”门槛。若 B 已不再像原来的
// candidate，则把它当作新的首次候选重新开始；不会把两个 B 混成一个 C。
bool update_reappear_candidate(OperationTrack* track, const Detection& detection,
                               bool started_touching_hand, int direct_frame) {
    if (!track) return false;
    // 同一张直接帧的局部路径与 FrameOwnership 只是同一个 B 的两种观察入口，
    // 只能计为一次自匹配。若中间漏掉一张直接帧，则重新从当前 B 开始。
    if (track->reappear_candidate_last_direct_frame == direct_frame) {
        return reappear_candidate_is_confirmed(*track);
    }
    if (!track->has_reappear_candidate_box ||
        track->reappear_candidate_last_direct_frame != direct_frame - 1 ||
        !track_match_box(track->cls_id, track->reappear_candidate_box,
                         detection.cls_id, detection.box)) {
        start_reappear_candidate(track, detection, started_touching_hand, direct_frame);
        return false;
    }
    ++track->reappear_candidate_match_count;
    track->reappear_candidate_last_direct_frame = direct_frame;
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

// CONTACT_* 只记录物品自己的真实位移。它后来转入 HAND_* 时，不能因为
// hand 的累计位移从新锚点开始而遗失此前已经观察到的有效推/拉证据。
bool has_meaningful_contact_move(const OperationTrack& track) {
    for (size_t i = 0; i < track.observed_move_values.size(); ++i) {
        if (move_length(track.observed_move_values[i]) >=
            FLOW3_CONTACT_OBJECT_MOVE_EPS) {
            return true;
        }
    }
    return false;
}

// `reappear_candidate` 只说明同一 B 在连续直接帧稳定存在；旧 C 最终移动
// 还必须已有一次可信的手部、真实 CONTACT 物体位移或放下证据。
bool has_valid_old_c_move_evidence(const OperationTrack& track) {
    return track.hold_and_move || has_meaningful_hand_move(track) ||
           has_meaningful_contact_move(track) ||
           (track.state == OperationTrackState::PLACED && track.drop_confirmed);
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
                other_track.contact_state == ContactState::NONE) {
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
        if (!is_active_existing_hand_track(c) ||
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

void mark_hand_b_ambiguity(
        const Detection& detection, std::map<int, OperationTrack>* tracks,
        const std::map<int, int>& known_item_owner) {
    if (!tracks) return;
    for (std::map<int, OperationTrack>::iterator it = tracks->begin();
         it != tracks->end(); ++it) {
        OperationTrack& c = it->second;
        if (!is_active_existing_hand_track(c) ||
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
        const std::map<int, OperationTrack>& tracks) {
    int result = -1;
    const std::map<int, int> no_known_owner;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (claimed.count(static_cast<int>(i)) ||
            detection_strictly_matches_other_item(detections[i], track.item_id, working) ||
            !reappear_candidate_path_matches(track, detections[i])) {
            continue;
        }
        bool belongs_to_existing_d = false;
        for (std::map<int, OperationTrack>::const_iterator other = tracks.begin();
             other != tracks.end(); ++other) {
            if (!other->second.is_suspect_new ||
                other->second.state == OperationTrackState::NORMAL ||
                other->second.cls_id != detections[i].cls_id) {
                continue;
            }
            const OperationTrack& d = other->second;
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
    if (!is_active_runtime_track(track) || track.cls_id != detection.cls_id) {
        return false;
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

}  // namespace

SessionManager::SessionManager() {}

void SessionManager::rebuild_persistent_item_index_() {
    item_by_id_.clear();
    std::map<int, InventoryItem>& items = inventory_.mutable_items();
    for (std::map<int, InventoryItem>::iterator it = items.begin(); it != items.end(); ++it) {
        item_by_id_[it->first] = &it->second;
    }
}

void SessionManager::reset_operation_runtime_() {
    working_inventory_.clear();
    no_hand_direct_inventory_.clear();
    operation_start_inventory_.clear();
    working_next_item_id_ = inventory_.next_item_id();
    working_inventory_active_ = false;
    track_buffer_.clear();
    pending_in_ids_.clear();
    pending_out_ids_.clear();
    confirmed_moved_ids_.clear();
    released_hand_candidate_ids_.clear();
    frame_ownership_ = FrameOwnership();
    old_item_resolution_.clear();
    unresolved_same_class_claims_.clear();
    baseline_collision_groups_.clear();
    current_d_runtime_to_detection_.clear();
    current_detection_to_d_runtime_.clear();
    direct_frame_sequence_ = 0;
    hand_track_.clear();
    has_old_hand_box_ = false;
    next_suspect_id_ = -1;
    no_hand_streak_ = 0;
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

void SessionManager::reset_frame_ownership_(size_t detection_count) {
    frame_ownership_.old_item_to_detection.clear();
    frame_ownership_.detection_to_old_item.assign(detection_count, -1);
    frame_ownership_.item_candidates.clear();
    frame_ownership_.detection_candidates.clear();
}

void SessionManager::begin_direct_frame_(const std::vector<Detection>& detections) {
    ++direct_frame_sequence_;
    reset_frame_ownership_(detections.size());
    current_d_runtime_to_detection_.clear();
    current_detection_to_d_runtime_.assign(detections.size(), 0);
    refresh_unresolved_same_class_claims_(detections);
}

void SessionManager::detect_baseline_collision_groups_() {
    baseline_collision_groups_.clear();
    std::vector<int> item_ids;
    for (std::map<int, InventoryItem>::const_iterator it =
             operation_start_inventory_.begin();
         it != operation_start_inventory_.end(); ++it) {
        item_ids.push_back(it->first);
    }
    std::vector<int> parent(item_ids.size());
    for (size_t i = 0; i < parent.size(); ++i) parent[i] = static_cast<int>(i);

    for (size_t i = 0; i < item_ids.size(); ++i) {
        const InventoryItem& left = operation_start_inventory_[item_ids[i]];
        const BBox left_box = left.base_box.area() > 0.0f ? left.base_box : left.box;
        for (size_t j = i + 1; j < item_ids.size(); ++j) {
            const InventoryItem& right = operation_start_inventory_[item_ids[j]];
            const BBox right_box = right.base_box.area() > 0.0f ? right.base_box : right.box;
            if (left.cls_id != right.cls_id ||
                !strict_match_box(left.cls_id, left_box, right.cls_id, right_box)) {
                continue;
            }
            int root_i = static_cast<int>(i);
            while (parent[root_i] != root_i) root_i = parent[root_i];
            int root_j = static_cast<int>(j);
            while (parent[root_j] != root_j) root_j = parent[root_j];
            if (root_i != root_j) parent[root_j] = root_i;
        }
    }

    std::map<int, std::set<int> > groups;
    for (size_t i = 0; i < item_ids.size(); ++i) {
        int root = static_cast<int>(i);
        while (parent[root] != root) root = parent[root];
        groups[root].insert(item_ids[i]);
    }
    for (std::map<int, std::set<int> >::const_iterator it = groups.begin();
         it != groups.end(); ++it) {
        if (it->second.size() > 1) {
            baseline_collision_groups_.push_back(it->second);
            printf("[INIT-IDENTITY] collision_group=");
            for (std::set<int>::const_iterator id = it->second.begin();
                 id != it->second.end(); ++id) {
                printf("%s%d", id == it->second.begin() ? "" : ",", *id);
            }
            printf("\n");
        }
    }
}

bool SessionManager::old_item_is_resolved_(int item_id) const {
    std::map<int, IdentityResolution>::const_iterator it =
        old_item_resolution_.find(item_id);
    return it != old_item_resolution_.end() &&
           it->second != IdentityResolution::UNRESOLVED;
}

bool SessionManager::baseline_collision_group_resolved_(
        const std::set<int>& group) const {
    std::set<int> used_detections;
    for (std::set<int>::const_iterator id = group.begin(); id != group.end(); ++id) {
        std::map<int, IdentityResolution>::const_iterator resolution =
            old_item_resolution_.find(*id);
        if (resolution == old_item_resolution_.end() ||
            resolution->second == IdentityResolution::UNRESOLVED) {
            return false;
        }
        if (resolution->second != IdentityResolution::AT_ORIGIN &&
            resolution->second != IdentityResolution::AT_NEW_POSITION) {
            continue;
        }
        std::map<int, int>::const_iterator owner =
            frame_ownership_.old_item_to_detection.find(*id);
        if (owner == frame_ownership_.old_item_to_detection.end() ||
            used_detections.count(owner->second)) {
            return false;
        }
        used_detections.insert(owner->second);
    }
    return true;
}

void SessionManager::assign_frame_owner_(int item_id, int detection_index,
                                         IdentityResolution resolution) {
    if (detection_index < 0 ||
        static_cast<size_t>(detection_index) >=
            frame_ownership_.detection_to_old_item.size()) {
        return;
    }
    std::map<int, int>::const_iterator old_owner =
        frame_ownership_.old_item_to_detection.find(item_id);
    if ((old_owner != frame_ownership_.old_item_to_detection.end() &&
         old_owner->second != detection_index) ||
        (frame_ownership_.detection_to_old_item[detection_index] >= 0 &&
         frame_ownership_.detection_to_old_item[detection_index] != item_id)) {
        return;
    }
    frame_ownership_.old_item_to_detection[item_id] = detection_index;
    frame_ownership_.detection_to_old_item[detection_index] = item_id;
    old_item_resolution_[item_id] = resolution;
}

std::set<int> SessionManager::possible_old_owners_for_detection_(
        int detection_index, const std::vector<Detection>& detections) const {
    std::set<int> possible;
    if (detection_index < 0 ||
        static_cast<size_t>(detection_index) >= detections.size()) {
        return possible;
    }
    std::map<int, std::vector<int> >::const_iterator candidates =
        frame_ownership_.detection_candidates.find(detection_index);
    if (candidates != frame_ownership_.detection_candidates.end()) {
        for (size_t i = 0; i < candidates->second.size(); ++i) {
            if (!old_item_is_resolved_(candidates->second[i])) {
                possible.insert(candidates->second[i]);
            }
        }
    }

    const Detection& detection = detections[detection_index];
    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        const OperationTrack& track = it->second;
        if (track.is_suspect_new || track.item_id <= 0 ||
            old_item_is_resolved_(track.item_id) ||
            !is_active_runtime_track(track) ||
            track.cls_id != detection.cls_id) {
            continue;
        }
        if (detection_can_belong_to_active_track(detection, track)) {
            possible.insert(track.item_id);
        }
    }

    // 基线中已经重叠到无法区分的同类 C，不能让其中一件暂时没有 owner
    // 时，把另一件附近或同类的新框直接解释成 D。组内必须先形成独立结论。
    for (size_t gi = 0; gi < baseline_collision_groups_.size(); ++gi) {
        const std::set<int>& group = baseline_collision_groups_[gi];
        if (group.empty() || baseline_collision_group_resolved_(group)) continue;
        std::map<int, InventoryItem>::const_iterator first =
            operation_start_inventory_.find(*group.begin());
        if (first == operation_start_inventory_.end() ||
            first->second.cls_id != detection.cls_id) {
            continue;
        }
        possible.insert(group.begin(), group.end());
    }
    return possible;
}

void SessionManager::build_frame_ownership_(
        const std::vector<Detection>& detections) {
    reset_frame_ownership_(detections.size());

    // 本帧的可见结论必须来自当前直接检测；此前帧的 AT_ORIGIN 不能在
    // 漏检后继续伪装为可见。已经有完整遮挡记录的旧 C 可以暂时保留该
    // 结论，直到当前帧真的重新得到独立 B。
    for (std::map<int, InventoryItem>::const_iterator old =
             operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        if (pending_out_ids_.count(old->first)) {
            old_item_resolution_[old->first] = IdentityResolution::OUT_CONFIRMED;
        } else if (old->second.status == ItemStatus::OCCLUDED) {
            old_item_resolution_[old->first] =
                IdentityResolution::OCCLUDED_CONFIRMED;
        } else {
            old_item_resolution_[old->first] = IdentityResolution::UNRESOLVED;
        }
    }

    std::map<int, std::set<int> > strict_origin_by_item;
    std::map<int, std::set<int> > strict_origin_by_detection;
    std::map<int, std::set<int> > path_by_item;
    std::map<int, std::set<int> > path_by_detection;

    const auto append_candidate = [this](int item_id, int detection_index) {
        std::vector<int>& item_candidates =
            frame_ownership_.item_candidates[item_id];
        if (std::find(item_candidates.begin(), item_candidates.end(),
                      detection_index) == item_candidates.end()) {
            item_candidates.push_back(detection_index);
        }
        std::vector<int>& detection_candidates =
            frame_ownership_.detection_candidates[detection_index];
        if (std::find(detection_candidates.begin(), detection_candidates.end(), item_id) ==
            detection_candidates.end()) {
            detection_candidates.push_back(item_id);
        }
    };
    const auto detection_is_current_d_owner = [this](int detection_index) {
        return detection_index >= 0 &&
            static_cast<size_t>(detection_index) < current_detection_to_d_runtime_.size() &&
            current_detection_to_d_runtime_[detection_index] != 0;
    };

    // 先同时收集全部候选边。这里故意不按“静态库存先抢框”的顺序写 owner：
    // 一个 B 若同时落在静态 C 原位和活动 C 路径上，必须保留歧义。
    for (std::map<int, InventoryItem>::const_iterator old =
             operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        if (!working_inventory_.count(old->first) || pending_out_ids_.count(old->first)) {
            continue;
        }
        const BBox original_box = old->second.base_box.area() > 0.0f
            ? old->second.base_box : old->second.box;
        const OperationTrack* runtime = find_runtime_for_item_(old->first);
        const bool has_active_path = runtime && !runtime->is_suspect_new &&
            is_active_runtime_track(*runtime);
        for (size_t di = 0; di < detections.size(); ++di) {
            const Detection& detection = detections[di];
            bool strict_origin = strict_match_box(old->second.cls_id, original_box,
                                                  detection.cls_id, detection.box);
            // CONTACT_* 已经得到真实物品路径、或它刚以该路径转入 HAND_* 时，
            // 宽松严格窗口不能把“已推开但仍有局部重叠”的 B 又叫回原位置。
            if (strict_origin && runtime &&
                (runtime->contact_state != ContactState::NONE ||
                 runtime->has_hand_estimate_anchor_box) &&
                !contact_detection_is_at_original(*runtime, detection)) {
                strict_origin = false;
            }
            const bool weak_origin = strict_origin || partial_match_box(
                old->second.cls_id, original_box, detection.cls_id, detection.box);
            const bool path_match = has_active_path &&
                detection_can_belong_to_active_track(detection, *runtime);
            if (!weak_origin && !path_match) continue;
            append_candidate(old->first, static_cast<int>(di));
            if (strict_origin) {
                strict_origin_by_item[old->first].insert(static_cast<int>(di));
                strict_origin_by_detection[static_cast<int>(di)].insert(old->first);
            }
            if (path_match) {
                path_by_item[old->first].insert(static_cast<int>(di));
                path_by_detection[static_cast<int>(di)].insert(old->first);
            }
        }
    }

    // 单个 C 若有多个严格 B，只有其中一个相对次优框具有足够明显的既有
    // 严格尺度领先时才可先选中它。这样“旧苹果完整框 + 相邻新苹果”可
    // 保住完整框；而两条几乎重合的旧苹果到同一个中央 B 仍不会被 2 像素
    // 的偶然差异强行分开。
    std::map<int, int> decisive_origin_by_item;
    std::map<int, std::set<int> > decisive_origin_by_detection;
    for (std::map<int, std::set<int> >::const_iterator item =
             strict_origin_by_item.begin(); item != strict_origin_by_item.end(); ++item) {
        std::map<int, InventoryItem>::const_iterator old =
            operation_start_inventory_.find(item->first);
        if (old == operation_start_inventory_.end()) continue;
        const BBox original_box = old->second.base_box.area() > 0.0f
            ? old->second.base_box : old->second.box;
        float best_cost = std::numeric_limits<float>::infinity();
        float second_cost = std::numeric_limits<float>::infinity();
        int best_detection = -1;
        for (std::set<int>::const_iterator detection = item->second.begin();
             detection != item->second.end(); ++detection) {
            const float cost = strict_match_cost(old->second.cls_id, original_box,
                                                 detections[*detection]);
            if (cost < best_cost) {
                second_cost = best_cost;
                best_cost = cost;
                best_detection = *detection;
            } else if (cost < second_cost) {
                second_cost = cost;
            }
        }
        const bool clearly_leading = best_detection >= 0 &&
            (second_cost == std::numeric_limits<float>::infinity() ||
             second_cost - best_cost >= INVENTORY_STRICT_CENTER_NORM * 0.5f);
        if (!clearly_leading) continue;
        decisive_origin_by_item[item->first] = best_detection;
        decisive_origin_by_detection[best_detection].insert(item->first);
    }

    // 只有 C 和 B 两侧都不存在另一个同等强解释时，原位置严格匹配才能锁定。
    // 自己同一条 C 的宽松路径不会制造冲突；其他 C 的路径会。
    for (std::map<int, int>::const_iterator item = decisive_origin_by_item.begin();
         item != decisive_origin_by_item.end(); ++item) {
        const int detection_index = item->second;
        // 已有 D 若在本帧已独占这个 B，C/D 不能同时登记成 owner。此时
        // 保留 C 的候选边和未决状态，等待下一张直接帧解除歧义。
        bool conflict = detection_is_current_d_owner(detection_index);
        for (std::set<int>::const_iterator other =
                 decisive_origin_by_detection[detection_index].begin();
             other != decisive_origin_by_detection[detection_index].end(); ++other) {
            if (*other != item->first) {
                conflict = true;
                break;
            }
        }
        if (!conflict) {
            for (std::set<int>::const_iterator other =
                     path_by_detection[detection_index].begin();
                 other != path_by_detection[detection_index].end(); ++other) {
                if (*other != item->first) {
                    conflict = true;
                    break;
                }
            }
        }
        if (!conflict) {
            assign_frame_owner_(item->first, detection_index,
                                IdentityResolution::AT_ORIGIN);
        }
    }

    // 再让仍未解决的活动 CONTACT/HAND/PLACED C 尝试认领剩余 B。B 不能
    // 严格属于其他旧 C，且不能同时被另一活动旧 C 的路径解释。
    for (std::map<int, std::set<int> >::const_iterator item = path_by_item.begin();
         item != path_by_item.end(); ++item) {
        if (frame_ownership_.old_item_to_detection.count(item->first)) continue;
        std::vector<int> unique_path_detections;
        for (std::set<int>::const_iterator detection = item->second.begin();
             detection != item->second.end(); ++detection) {
            if (frame_ownership_.detection_to_old_item[*detection] >= 0 ||
                detection_is_current_d_owner(*detection)) {
                continue;
            }
            bool conflict = false;
            for (std::set<int>::const_iterator other =
                     decisive_origin_by_detection[*detection].begin();
                 other != decisive_origin_by_detection[*detection].end(); ++other) {
                if (*other != item->first) {
                    conflict = true;
                    break;
                }
            }
            if (!conflict) {
                for (std::set<int>::const_iterator other =
                         path_by_detection[*detection].begin();
                     other != path_by_detection[*detection].end(); ++other) {
                    if (*other != item->first) {
                        conflict = true;
                        break;
                    }
                }
            }
            if (!conflict) unique_path_detections.push_back(*detection);
        }
        if (unique_path_detections.size() != 1) continue;

        const OperationTrack* runtime = find_runtime_for_item_(item->first);
        bool confirmed_new_position = false;
        if (!hand_present_ && runtime) {
            confirmed_new_position = confirmed_moved_ids_.count(item->first) ||
                (runtime->state == OperationTrackState::PLACED &&
                 runtime->drop_confirmed) ||
                (reappear_candidate_is_confirmed(*runtime) &&
                 has_valid_old_c_move_evidence(*runtime));
        }
        assign_frame_owner_(item->first, unique_path_detections.front(),
                            confirmed_new_position
                                ? IdentityResolution::AT_NEW_POSITION
                                : IdentityResolution::UNRESOLVED);
    }

    discard_claims_resolved_by_old_owner_(detections);
    capture_unresolved_same_class_claims_(detections);
}

void SessionManager::synchronize_frame_ownership_candidates_(
        const std::vector<Detection>& detections, const BBox* hand_box) {
    std::set<int> owned_item_ids;
    for (std::map<int, int>::const_iterator owner =
             frame_ownership_.old_item_to_detection.begin();
         owner != frame_ownership_.old_item_to_detection.end(); ++owner) {
        const int item_id = owner->first;
        const int detection_index = owner->second;
        owned_item_ids.insert(item_id);
        if (detection_index < 0 ||
            static_cast<size_t>(detection_index) >= detections.size()) {
            continue;
        }

        OperationTrack* track = find_runtime_for_item_(item_id);
        if (!track || track->is_suspect_new ||
            !(is_active_existing_hand_track(*track) ||
              is_active_contact_track(*track))) {
            continue;
        }
        std::map<int, IdentityResolution>::const_iterator resolution =
            old_item_resolution_.find(item_id);
        const IdentityResolution current_resolution =
            resolution == old_item_resolution_.end()
                ? IdentityResolution::UNRESOLVED : resolution->second;
        const Detection& detection = detections[detection_index];

        // 原位直接框不是“重新出现”路径；它会撤销之前未完成的移动链。
        if (current_resolution != IdentityResolution::UNRESOLVED ||
            contact_detection_is_at_original(*track, detection)) {
            if (track->has_reappear_candidate_box || track->reappearance_pending) {
                clear_reappear_candidate(track, false);
            }
            continue;
        }

        // FrameOwnership 已经证明本帧 C->B 唯一。无论局部匹配是否先找到 B，
        // 都要在同一张直接帧开始/推进 C 自己的连续自匹配；这不会把 C 直接
        // 设为 AT_NEW_POSITION，最终确认仍只在后续无手直接帧发生。
        const bool touching_hand = hand_box &&
            hand_touches_detection(*hand_box, detection.box);
        update_reappear_candidate(track, detection, touching_hand,
                                  direct_frame_sequence_);
    }

    // 连续确认必须是连续直接帧。B 消失、被另一 C 竞争，或不再获得当前帧
    // owner 时，撤销旧候选；不能隔帧借用之前的 count 生成 MOVED/OUT/IN。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || track.item_id <= 0 ||
            !(is_active_existing_hand_track(track) || is_active_contact_track(track)) ||
            owned_item_ids.count(track.item_id)) {
            continue;
        }
        std::map<int, IdentityResolution>::const_iterator resolution =
            old_item_resolution_.find(track.item_id);
        if (resolution != old_item_resolution_.end() &&
            resolution->second != IdentityResolution::UNRESOLVED) {
            continue;
        }
        if (track.has_reappear_candidate_box) {
            clear_reappear_candidate(&track, true);
        } else {
            track.reappearance_pending = true;
        }
    }
}

void SessionManager::log_frame_ownership_() const {
    for (size_t di = 0; di < frame_ownership_.detection_to_old_item.size(); ++di) {
        const int owner = frame_ownership_.detection_to_old_item[di];
        std::map<int, std::vector<int> >::const_iterator candidates =
            frame_ownership_.detection_candidates.find(static_cast<int>(di));
        printf("[OWNERSHIP] frame=%d B=%zu candidates=", direct_frame_sequence_, di);
        if (candidates == frame_ownership_.detection_candidates.end() ||
            candidates->second.empty()) {
            printf("NONE");
        } else {
            for (size_t i = 0; i < candidates->second.size(); ++i) {
                printf("%sC#%d", i == 0 ? "" : ",", candidates->second[i]);
            }
        }
        if (owner >= 0) {
            std::map<int, IdentityResolution>::const_iterator resolution =
                old_item_resolution_.find(owner);
            const IdentityResolution current_resolution =
                resolution == old_item_resolution_.end()
                    ? IdentityResolution::UNRESOLVED : resolution->second;
            printf(" owner=C#%d resolution=%s\n", owner,
                   identity_resolution_name(current_resolution));
        } else {
            const bool ambiguous = candidates != frame_ownership_.detection_candidates.end() &&
                !candidates->second.empty();
            printf(" owner=%s resolution=UNRESOLVED\n",
                   ambiguous ? "AMBIGUOUS" : "NONE");
        }
    }
}

void SessionManager::upsert_unresolved_same_class_claim_(
        const Detection& detection, const std::set<int>& possible_old_item_ids,
        bool has_hand_source, bool first_seen_in_post_hand_window) {
    if (detection.box.area() <= 0.0f || possible_old_item_ids.empty()) return;

    UnresolvedSameClassClaim* claim = nullptr;
    for (size_t i = 0; i < unresolved_same_class_claims_.size(); ++i) {
        UnresolvedSameClassClaim& candidate = unresolved_same_class_claims_[i];
        if (candidate.cls_id == detection.cls_id && candidate.has_last_box &&
            track_match_box(candidate.cls_id, candidate.last_box,
                            detection.cls_id, detection.box)) {
            claim = &candidate;
            break;
        }
    }
    if (!claim) {
        UnresolvedSameClassClaim created;
        created.cls_id = detection.cls_id;
        created.first_box = detection.box;
        created.last_box = detection.box;
        created.has_last_box = true;
        created.direct_self_match_count = 1;
        created.last_seen_direct_frame = direct_frame_sequence_;
        unresolved_same_class_claims_.push_back(created);
        claim = &unresolved_same_class_claims_.back();
    } else if (claim->last_seen_direct_frame != direct_frame_sequence_) {
        claim->direct_self_match_count =
            claim->last_seen_direct_frame == direct_frame_sequence_ - 1
                ? claim->direct_self_match_count + 1 : 1;
        claim->last_box = detection.box;
        claim->has_last_box = true;
        claim->last_seen_direct_frame = direct_frame_sequence_;
    }
    claim->has_hand_source = claim->has_hand_source || has_hand_source;
    claim->first_seen_in_post_hand_window =
        claim->first_seen_in_post_hand_window || first_seen_in_post_hand_window;
    claim->possible_old_item_ids.insert(possible_old_item_ids.begin(),
                                        possible_old_item_ids.end());
}

void SessionManager::refresh_unresolved_same_class_claims_(
        const std::vector<Detection>& detections) {
    for (std::vector<UnresolvedSameClassClaim>::iterator claim =
             unresolved_same_class_claims_.begin();
         claim != unresolved_same_class_claims_.end();) {
        int matched_detection = -1;
        bool ambiguous = false;
        for (size_t di = 0; di < detections.size(); ++di) {
            if (detections[di].cls_id != claim->cls_id || !claim->has_last_box ||
                !track_match_box(claim->cls_id, claim->last_box,
                                 detections[di].cls_id, detections[di].box)) {
                continue;
            }
            if (matched_detection >= 0) {
                ambiguous = true;
                break;
            }
            matched_detection = static_cast<int>(di);
        }
        if (matched_detection < 0) {
            claim = unresolved_same_class_claims_.erase(claim);
            continue;
        }
        if (!ambiguous && claim->last_seen_direct_frame != direct_frame_sequence_) {
            claim->direct_self_match_count =
                claim->last_seen_direct_frame == direct_frame_sequence_ - 1
                    ? claim->direct_self_match_count + 1 : 1;
            claim->last_box = detections[matched_detection].box;
            claim->has_last_box = true;
            claim->last_seen_direct_frame = direct_frame_sequence_;
        }
        ++claim;
    }
}

void SessionManager::discard_claims_resolved_by_old_owner_(
        const std::vector<Detection>& detections) {
    for (std::vector<UnresolvedSameClassClaim>::iterator claim =
             unresolved_same_class_claims_.begin();
         claim != unresolved_same_class_claims_.end();) {
        bool resolved_by_old_owner = false;
        for (std::map<int, int>::const_iterator owner =
                 frame_ownership_.old_item_to_detection.begin();
             owner != frame_ownership_.old_item_to_detection.end(); ++owner) {
            if (!claim->possible_old_item_ids.count(owner->first) ||
                !old_item_is_resolved_(owner->first) || owner->second < 0 ||
                static_cast<size_t>(owner->second) >= detections.size() ||
                !claim->has_last_box ||
                !track_match_box(claim->cls_id, claim->last_box,
                                 detections[owner->second].cls_id,
                                 detections[owner->second].box)) {
                continue;
            }
            resolved_by_old_owner = true;
            break;
        }
        if (resolved_by_old_owner) {
            claim = unresolved_same_class_claims_.erase(claim);
        } else {
            ++claim;
        }
    }
}

void SessionManager::capture_unresolved_same_class_claims_(
        const std::vector<Detection>& detections) {
    for (size_t di = 0; di < detections.size(); ++di) {
        if (frame_ownership_.detection_to_old_item[di] >= 0 ||
            (di < current_detection_to_d_runtime_.size() &&
             current_detection_to_d_runtime_[di] != 0)) {
            continue;
        }
        const std::set<int> possible = possible_old_owners_for_detection_(
            static_cast<int>(di), detections);
        if (possible.empty()) continue;
        const bool hand_source = hand_present_ &&
            hand_track_touches_detection(hand_track_, detections[di]);
        const bool post_hand_source = !hand_present_ && no_hand_streak_ > 0 &&
            no_hand_streak_ <= FLOW3_POST_HAND_REVEAL_WINDOW_FRAMES &&
            hand_track_touches_detection(hand_track_, detections[di]);
        upsert_unresolved_same_class_claim_(detections[di], possible, hand_source,
                                             post_hand_source);
    }
}

bool SessionManager::can_start_new_d_for_detection_(
        int detection_index, const std::vector<Detection>& detections,
        bool has_hand_source, bool first_seen_in_post_hand_window) {
    const auto log_gate = [this, detection_index](bool allow, const char* blocked_by) {
        printf("[D-GATE] frame=%d B=%d allow=%d blocked_by=%s\n",
               direct_frame_sequence_, detection_index, allow ? 1 : 0, blocked_by);
    };
    if (detection_index < 0 ||
        static_cast<size_t>(detection_index) >= detections.size() ||
        static_cast<size_t>(detection_index) >=
            frame_ownership_.detection_to_old_item.size()) {
        log_gate(false, "invalid");
        return false;
    }
    const int old_owner = frame_ownership_.detection_to_old_item[detection_index];
    if (old_owner >= 0) {
        char blocked_by[32];
        snprintf(blocked_by, sizeof(blocked_by), "C#%d", old_owner);
        log_gate(false, blocked_by);
        return false;
    }
    if (static_cast<size_t>(detection_index) < current_detection_to_d_runtime_.size() &&
        current_detection_to_d_runtime_[detection_index] != 0) {
        log_gate(false, "existing-D");
        return false;
    }

    const std::set<int> possible = possible_old_owners_for_detection_(
        detection_index, detections);
    if (!possible.empty()) {
        upsert_unresolved_same_class_claim_(detections[detection_index], possible,
                                             has_hand_source,
                                             first_seen_in_post_hand_window);
        printf("[D-GATE] frame=%d B=%d allow=0 blocked_by=", direct_frame_sequence_,
               detection_index);
        for (std::set<int>::const_iterator item_id = possible.begin();
             item_id != possible.end(); ++item_id) {
            printf("%sC#%d", item_id == possible.begin() ? "" : ",", *item_id);
        }
        printf("\n");
        printf("[3.0] B cls=%d 仍可能属于旧 C，保持 claim，不建立 D\n",
               detections[detection_index].cls_id);
        return false;
    }
    if (!has_hand_source && !first_seen_in_post_hand_window) {
        log_gate(false, "no-hand-source");
        return false;
    }

    // 两个近到仍可被同一条直接自匹配链解释的同类未归属 B，不能靠检测
    // 遍历顺序各自建立 D。没有旧 C 可归属时暂时不提升，等待下一帧分开。
    for (size_t di = 0; di < detections.size(); ++di) {
        if (static_cast<int>(di) == detection_index ||
            frame_ownership_.detection_to_old_item[di] >= 0 ||
            (di < current_detection_to_d_runtime_.size() &&
             current_detection_to_d_runtime_[di] != 0) ||
            detections[di].cls_id != detections[detection_index].cls_id) {
            continue;
        }
        if (track_match_box(detections[detection_index].cls_id,
                            detections[detection_index].box,
                            detections[di].cls_id, detections[di].box)) {
            log_gate(false, "collision");
            printf("[3.0] 当前同类未归属 B 仍互相歧义，暂不建立 D\n");
            return false;
        }
    }

    // claim 守卫已经解除时，当前帧才是 D 的第 1 张确认帧。删除同一 B 的
    // 旧声明，防止它在 D 已开始后继续无意义地阻挡提交。
    for (std::vector<UnresolvedSameClassClaim>::iterator claim =
             unresolved_same_class_claims_.begin();
         claim != unresolved_same_class_claims_.end();) {
        if (claim->cls_id == detections[detection_index].cls_id && claim->has_last_box &&
            track_match_box(claim->cls_id, claim->last_box,
                            detections[detection_index].cls_id,
                            detections[detection_index].box)) {
            claim = unresolved_same_class_claims_.erase(claim);
        } else {
            ++claim;
        }
    }
    log_gate(true, "NONE");
    return true;
}

void SessionManager::mark_current_d_owner_(int runtime_key, int detection_index) {
    if (detection_index < 0 ||
        static_cast<size_t>(detection_index) >= current_detection_to_d_runtime_.size() ||
        frame_ownership_.detection_to_old_item[detection_index] >= 0) {
        return;
    }
    const int existing = current_detection_to_d_runtime_[detection_index];
    if (existing != 0 && existing != runtime_key) return;
    std::map<int, int>::const_iterator previous =
        current_d_runtime_to_detection_.find(runtime_key);
    if (previous != current_d_runtime_to_detection_.end() &&
        previous->second != detection_index) {
        return;
    }
    current_detection_to_d_runtime_[detection_index] = runtime_key;
    current_d_runtime_to_detection_[runtime_key] = detection_index;
}

void SessionManager::begin_working_operation_(const BBox& hand_box,
                                               const std::vector<Detection>& detections) {
    working_inventory_ = inventory_.items();
    no_hand_direct_inventory_.clear();
    operation_start_inventory_ = inventory_.items();
    working_next_item_id_ = inventory_.next_item_id();
    working_inventory_active_ = true;
    track_buffer_.clear();
    pending_in_ids_.clear();
    pending_out_ids_.clear();
    confirmed_moved_ids_.clear();
    released_hand_candidate_ids_.clear();
    frame_ownership_ = FrameOwnership();
    old_item_resolution_.clear();
    unresolved_same_class_claims_.clear();
    baseline_collision_groups_.clear();
    current_d_runtime_to_detection_.clear();
    current_detection_to_d_runtime_.clear();
    direct_frame_sequence_ = 0;
    for (std::map<int, InventoryItem>::const_iterator old =
             operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        old_item_resolution_[old->first] = old->second.status == ItemStatus::OCCLUDED
            ? IdentityResolution::OCCLUDED_CONFIRMED
            : IdentityResolution::UNRESOLVED;
    }
    detect_baseline_collision_groups_();
    hand_track_.clear();
    hand_track_.push_back(hand_box);
    old_hand_box_ = hand_box;
    has_old_hand_box_ = true;
    no_hand_streak_ = 0;
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
    std::vector<int> release_keys;
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (!is_active_contact_track(track)) continue;

        // 若 CONTACT_* 已经用真实 B 看见过物品的新位置，覆盖率必须相对
        // 当前真实位置计算；仍拿最初 A.box 计算会使“先推、后握住”永远
        // 停留在 CONTACT_*。
        const BBox reference = track.has_last_seen_box ? track.last_seen_box
            : track.original_box;
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
        claimed_detection_indices->insert(observed_index);
        if (track.item_id > 0) (*known_item_owner)[track.item_id] = observed_index;
        append_contact_observation(&track, detection, touching_hand);

        if (!at_original) {
            // 当前帧唯一的 C/B 关系会在稍后的 FrameOwnership 中立即占用 B；
            // 此处只累计 C 自己的连续自匹配，不能把首次 B 直接当成 MOVED。
            update_reappear_candidate(&track, detection, touching_hand,
                                      direct_frame_sequence_);
        }

        if (track.contact_state == ContactState::CONTACT_MOVING) {
            std::map<int, InventoryItem>::iterator item =
                working_inventory_.find(track.item_id);
            if (item != working_inventory_.end()) update_seen(item->second, detection, 0);
            continue;
        }

        if (at_original) {
            clear_reappear_candidate(&track, false);
            ++track.not_hold_evidence_count;
            track.hold_evidence_count = 0;
        } else if (object_move >= FLOW3_CONTACT_OBJECT_MOVE_EPS &&
                   (had_touch_before || touching_hand)) {
            // 只要第一次有效 B 已经与手相贴，后续 B 可在手离开后继续沿
            // observed_track 确认，不把手框方向当作物品方向。
            ++track.hold_evidence_count;
            track.not_hold_evidence_count = 0;
        }

        if (track.not_hold_evidence_count >= FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
            release_keys.push_back(it->first);
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
        std::map<int, OperationTrack>::iterator it = track_buffer_.find(release_keys[i]);
        if (it != track_buffer_.end() &&
            it->second.contact_state == ContactState::CONTACT_CANDIDATE) {
            release_not_held_(it->second, false);
        }
    }
}

void SessionManager::mark_new_contact_candidates_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
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
        track.shelter_or_hold = true;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track_buffer_[item.item_id] = track;
        OperationTrack& created = track_buffer_[item.item_id];

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
            if (!claimed_detection_indices->count(observed_index)) {
                claimed_detection_indices->insert(observed_index);
            }
            (*known_item_owner)[item.item_id] = observed_index;
            append_contact_observation(&created, detections[observed_index], touching);
            if (!at_original) {
                start_reappear_candidate(&created, detections[observed_index], touching,
                                         direct_frame_sequence_);
            }
        }
        printf("[3.0] item#%d 进入 CONTACT_CANDIDATE（低覆盖率且手相贴）\n",
               item.item_id);
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
    const int item_id = working_next_item_id_++;
    working_inventory_[item_id] = make_inventory_item(item_id, detection,
                                                       frame_id, current_time_ms_);
    // 初次检测可能只是局部框；它只作为临时位置，后续无手直接帧会刷新。
    track.item_id = item_id;
    track.promoted_to_working_inventory = true;
    pending_in_ids_.insert(item_id);
    printf("[3.0] suspect#%d 已提升为工作库存 item#%d（尚未正式 IN）\n",
           track.suspect_id, item_id);
}

void SessionManager::reserve_visible_known_detections_(
        const BBox& hand_box,
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
    // 只处理没有正在进行 HAND_* / PLACED 轨迹的普通旧物品。正在移动的
    // 物品必须由自己的轨迹逻辑认领，不能被这里按原位置抢走。
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
                known_item_owner->count(item->first)) {
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
                if (claimed_detection_indices->count(detection_index)) continue;
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
                claimed_detection_indices->count(detection_index)) {
                continue;
            }
            (*known_item_owner)[best->first] = detection_index;
            claimed_detection_indices->insert(detection_index);
            made_progress = true;
        }
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
    track.b_claim_ambiguous = false;
    track.no_hand_missing_count = 0;
    confirmed_moved_ids_.insert(track.item_id);
}

void SessionManager::release_not_held_(OperationTrack& track, bool occluded) {
    if (track.item_id > 0) {
        std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
        if (item != working_inventory_.end()) item->second.status =
            occluded ? ItemStatus::OCCLUDED : ItemStatus::VISIBLE;
        released_hand_candidate_ids_.insert(track.item_id);
    }
    track.state = OperationTrackState::NORMAL;
    track.contact_state = ContactState::NONE;
    track.shelter_or_hold = false;
    track.hold_and_move = false;
    track.hold_evidence_count = 0;
    track.not_hold_evidence_count = 0;
    clear_reappear_candidate(&track, false);
    track.b_claim_ambiguous = false;
    track.has_first_hand_block_box = false;
    track.has_last_hand_block_box = false;
    track.has_hand_estimate_anchor_box = false;
    track.move_values.clear();
    track.track.clear();
    track.observed_move_values.clear();
    track.observed_track.clear();
    track.hand_move_values.clear();
    track.contact_started_touching_hand = false;
    track.hand_track_start_index = -1;
    track.no_hand_missing_count = 0;
}

void SessionManager::update_existing_hand_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        const MoveValue& delta, std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
    std::vector<int> promote_keys;
    // 对 HAND_* + hold_and_move=false，静止手帧只能更新局部观测，不能把
    // “还在原位”或“跟手移动”当作新的有效证据。
    const bool hand_moved = move_length(delta) >= TRACK_HAND_MOVE_EPS;
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

        // HAND_* 的普通候选必须在当前帧和其他同类 C 一起仲裁。轨迹刚创建
        // 不会延迟这次仲裁；真正有多个 C 能解释 B 时才保持歧义。
        if (!track.is_suspect_new &&
            observed_index >= 0 &&
            !contact_detection_is_at_original(track, detections[observed_index])) {
            const int candidate_owner = unique_c_reappear_owner_for_detection(
                detections[observed_index], track_buffer_, *known_item_owner);
            if (candidate_owner == -2) {
                mark_hand_b_ambiguity(detections[observed_index],
                                      &track_buffer_, *known_item_owner);
                observed_index = -1;
                observed_matches_reappear_candidate = false;
                track.reappearance_pending = true;
            } else if (candidate_owner >= 0 && candidate_owner != track.item_id) {
                observed_index = -1;
                observed_matches_reappear_candidate = false;
                track.reappearance_pending = true;
            } else {
                // 当前 B 没有被另一条 HAND_* 轨迹解释；若此前只是
                // 暂时歧义，本帧的唯一本地匹配可以解除它。
                track.b_claim_ambiguous = false;
            }
        }

        if (track.is_suspect_new) {
            if (observed_index >= 0) {
                const Detection& d = detections[observed_index];
                claimed_detection_indices->insert(observed_index);
                mark_current_d_owner_(it->first, observed_index);
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
                        update_seen(item->second, d, item->second.updated_frame + 1);
                        item->second.base_box = d.box;
                    }
                }
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
                        &track, d, hand_touches_detection(hand_box, d.box),
                        direct_frame_sequence_);
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
                    confirm_rearrange_(track, d.box, d.score, 0);
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
                if (item != working_inventory_.end()) update_seen(item->second, old_d, 0);
                release_not_held_(track, false);
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
                &track, d, hand_touches_detection(hand_box, d.box),
                direct_frame_sequence_);
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
                        release_not_held_(track, false);
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
        promote_suspect_(promote_keys[i], d, 0);
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
                mark_hand_b_ambiguity(d, &track_buffer_,
                                      effective_known_item_owner);
                upsert_unresolved_same_class_claim_(
                    d, possible_old_owners_for_detection_(
                           static_cast<int>(di), detections),
                    true, false);
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
                        update_reappear_candidate(&c->second, d, true,
                                                  direct_frame_sequence_);
                    } else {
                        start_reappear_candidate(&c->second, d, true,
                                                 direct_frame_sequence_);
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

        // CONTACT_* 的 B 即使本帧因框跳动没有被 update 认领，也不能直接
        // 进入 D。先保留为旧物品的未决解释，等待下一帧真实观测。
        bool belongs_to_contact_track = false;
        for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            if (is_active_contact_track(it->second) &&
                detection_can_belong_to_active_track(d, it->second)) {
                belongs_to_contact_track = true;
                break;
            }
        }
        if (belongs_to_contact_track) {
            upsert_unresolved_same_class_claim_(
                d, possible_old_owners_for_detection_(static_cast<int>(di), detections),
                hand_visible_d, false);
            continue;
        }

        // 即使当前 C 的宽松路径暂时没有命中，其他同类旧 C 仍可能
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
            upsert_unresolved_same_class_claim_(
                d, possible_old_owners_for_detection_(static_cast<int>(di), detections),
                hand_visible_d, false);
            printf("[3.0] 同类旧 C 尚未得到独立归属，B cls=%d 进入未决池\n",
                   d.cls_id);
            continue;
        }

        if (!can_start_new_d_for_detection_(static_cast<int>(di), detections,
                                             hand_visible_d || replacement_owner >= 0,
                                             false)) {
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
        mark_current_d_owner_(key, static_cast<int>(di));
        printf("[3.0] 预登记疑似新物品 D suspect#%d cls=%d source=%s\n",
               key, d.cls_id, suspect_source_name(track.suspect_source));
    }
}

void SessionManager::mark_newly_hand_blocked_items_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
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
        track.shelter_or_hold = true;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track.track.push_back(track.original_box);
        track.state = full ? OperationTrackState::HAND_FULL_BLOCKED
                           : OperationTrackState::HAND_PARTIAL_BLOCKED;
        // 首帧没有可靠 C 框时，下一次同类 B 即使落回预计位置，也必须先
        // 建立重新出现候选并等待自匹配，不能一帧确认移动。
        track.reappearance_pending = observed_index < 0;
        if (observed_index >= 0) {
            const Detection& d = detections[observed_index];
            const bool at_original = contact_detection_is_at_original(track, d);
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;
            track.first_hand_block_box = d.box;
            track.last_hand_block_box = d.box;
            track.has_first_hand_block_box = true;
            track.has_last_hand_block_box = true;
            claimed_detection_indices->insert(observed_index);
            (*known_item_owner)[item.item_id] = observed_index;
            if (!at_original) {
                start_reappear_candidate(&track, d,
                                         hand_touches_detection(hand_box, d.box),
                                         direct_frame_sequence_);
                track.reappearance_pending = true;
            }
        }
        // 已经放下过又再次被手接触时，覆盖旧运行时记录即可。
        for (std::map<int, OperationTrack>::iterator rt = track_buffer_.begin();
             rt != track_buffer_.end();) {
            if (rt->second.item_id == item.item_id) rt = track_buffer_.erase(rt);
            else ++rt;
        }
        track_buffer_[item.item_id] = track;
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
                release_not_held_(c, true);
            }
        }
    }
}

void SessionManager::process_effective_hand_frame_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        bool first_hand_frame) {
    begin_direct_frame_(detections);
    MoveValue delta;
    if (!first_hand_frame) {
        delta.dx = hand_box.cx() - old_hand_box_.cx();
        delta.dy = hand_box.cy() - old_hand_box_.cy();
        hand_track_.push_back(hand_box);
        append_move_to_existing_hand_tracks_(delta);
    }
    const bool hand_moved = !first_hand_frame &&
        move_length(delta) >= TRACK_HAND_MOVE_EPS;

    // 先更新已有 HAND_*；随后先为仍稳定可见的旧库存保留本帧自己的
    // 严格框，再处理新进入 HAND_* 的物品。这个先后顺序很关键：若旧
    // 苹果已有完整框，旁边贴手的新苹果不能先被“局部可能属于旧苹果”
    // 的规则抢走。
    std::set<int> claimed;
    std::map<int, int> known_item_owner;
    if (!first_hand_frame) {
        // 低覆盖率 CONTACT_* 先用物品真实检测框更新；这里不能使用手位移
        // 推算的 estimated_box。
        update_existing_contact_tracks_(hand_box, detections, &claimed,
                                        &known_item_owner);
        update_existing_hand_tracks_(hand_box, detections, delta, &claimed,
                                     &known_item_owner);
    }
    reserve_visible_known_detections_(hand_box, detections, &claimed,
                                      &known_item_owner);
    mark_new_contact_candidates_(hand_box, detections, &claimed,
                                 &known_item_owner);
    mark_newly_hand_blocked_items_(hand_box, detections, &claimed,
                                   &known_item_owner);

    // 上面的 HAND/CONTACT 分支只负责更新各自路径。D 扫描前重新统一当前
    // 帧的 C/B 归属，并把所有唯一旧 C owner（包括尚待连续确认移动的 C）
    // 写回 claimed，避免局部分支把同一 B 再登记成 D。
    build_frame_ownership_(detections);
    synchronize_frame_ownership_candidates_(detections, &hand_box);
    for (std::map<int, int>::const_iterator owner =
             frame_ownership_.old_item_to_detection.begin();
         owner != frame_ownership_.old_item_to_detection.end(); ++owner) {
        claimed.insert(owner->second);
        known_item_owner[owner->first] = owner->second;
    }
    scan_or_update_suspects_(hand_box, detections, &claimed, known_item_owner,
                             first_hand_frame);
    apply_suspect_cover_evidence_(hand_box, detections, hand_moved);
    // scan 可能刚把 B 暂存为 C 的 reappear_candidate；本帧最终 owner 与
    // claim 必须反映这个更新，而不是停留在 scan 前的局部判断。
    build_frame_ownership_(detections);
    log_frame_ownership_();
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
        const bool touches_hand_path = hand_track_touches_detection(hand_track_, d);
        if (!touches_hand_path) continue;

        // 同类旧 C 若在本轮仍没有自己的独立检测/轨迹结论，手离开后
        // 首次出现的 B 也只能保持未决。否则 C 的一次漏检会被误写成
        // POST_HAND_REVEAL_D，库存数量会逐次膨胀。
        if (!can_start_new_d_for_detection_(detection_index, detections,
                                             false, true)) {
            continue;
        }

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
        mark_current_d_owner_(key, detection_index);
        printf("[3.0] 手离开后预登记疑似新物品 D suspect#%d cls=%d source=%s\n",
               key, d.cls_id, suspect_source_name(track.suspect_source));
    }
}

void SessionManager::observe_no_hand_frame_(const std::vector<Detection>& detections,
                                            int frame_id) {
    begin_direct_frame_(detections);
    std::set<int> claimed;
    std::vector<int> promote_keys;
    std::set<int> discard_keys;

    // 认领顺序必须是：已有 C -> 没有手离开后新建的已有 D -> 剩余 B 的
    // POST_HAND_REVEAL_D。track_buffer_ 的负 id 是 D，若直接按 map 遍历，
    // D 会先于 C 抢框，故显式分两轮处理。
    for (int phase = 0; phase < 2; ++phase) {
        const bool process_suspects = phase == 1;
        for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            OperationTrack& track = it->second;
            if (track.is_suspect_new != process_suspects ||
                !is_active_runtime_track(track)) continue;

            // CONTACT_* 在有手阶段保持 state=NORMAL，因此不能落入旧的
            // HAND_* 分支。手离开后只按原位置和真实 observed_track 收尾。
            if (!track.is_suspect_new &&
                track.contact_state != ContactState::NONE) {
                int contact_found = unique_contact_original_detection(
                    detections, claimed, track, working_inventory_);
                if (contact_found >= 0) {
                    const Detection& d = detections[contact_found];
                    claimed.insert(contact_found);
                    clear_reappear_candidate(&track, false);
                    const bool was_moving =
                        track.contact_state == ContactState::CONTACT_MOVING;
                    append_contact_observation(&track, d, false);
                    track.hold_evidence_count = 0;
                    ++track.not_hold_evidence_count;
                    if (track.not_hold_evidence_count >=
                        FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) update_seen(item->second, d, 0);
                        release_not_held_(track, false);
                    } else if (was_moving) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) update_seen(item->second, d, 0);
                    }
                    continue;
                }

                const int contact_found_on_path = unique_contact_detection_for_track(
                    detections, claimed, track, working_inventory_, track_buffer_);
                if (contact_found_on_path >= 0) {
                    const Detection& d = detections[contact_found_on_path];
                    claimed.insert(contact_found_on_path);
                    const bool was_moving =
                        track.contact_state == ContactState::CONTACT_MOVING;
                    if (!was_moving) {
                        const bool candidate_ready = update_reappear_candidate(
                            &track, d, false, direct_frame_sequence_);
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
                        if (item != working_inventory_.end()) update_seen(item->second, d, 0);
                    }
                    continue;
                }
                if (has_contact_path_candidate(detections, claimed, track,
                                               working_inventory_)) {
                    track.contact_path_ambiguous = true;
                }
                // 没有原位置或唯一轨迹 B 时保持未决；不能仅因一帧缺失 OUT。
                continue;
            }
            BBox reference = track.has_placed_box ? track.placed_box : estimated_box(track);
            if (track.state == OperationTrackState::PLACED && track.has_placed_box) {
                reference = track.placed_box;
            }
            int found = -1;
            bool found_at_original_position = false;
            bool found_as_reappear_candidate = false;

            // 细节5的无手收尾顺序：已有 C 必须先回查原位置；若 C 仍在原处，
            // 后面的动态路径和 B 候选都不能把它改判成移动。
            if (!track.is_suspect_new && track.original_box.area() > 0.0f) {
                found = unique_detection_for_box(detections, claimed, track.cls_id,
                                                  track.original_box, true, false);
                found_at_original_position = found >= 0;
                if (found < 0) {
                    const int nearest = best_detection_for_box(
                        detections, claimed, track.cls_id, track.original_box, true, false);
                    if (nearest >= 0 && has_unique_operation_start_owner(
                            detections[nearest], track.item_id,
                            operation_start_inventory_)) {
                        found = nearest;
                        found_at_original_position = true;
                    }
                }
                // 同上：由 CONTACT_* 转来的物品已拥有真实 B 路径。B 与
                // 原框局部重叠并不等于它仍在原位，必须再检查中心位移。
                if (track.has_hand_estimate_anchor_box && found >= 0 &&
                    !contact_detection_is_at_original(track, detections[found])) {
                    found = -1;
                    found_at_original_position = false;
                }
            }
            // 已在有手阶段看到过 B 时，真实 B 候选比“旧框 + 手位移”的估计
            // 更可信，但它同样需要连续自匹配，不能在本帧直接确认整理。
            if (found < 0 && !track.is_suspect_new &&
                track.has_reappear_candidate_box) {
                found = unique_detection_for_box(detections, claimed, track.cls_id,
                                                  track.reappear_candidate_box, true, true);
                if (found < 0) {
                    found = best_detection_for_box(detections, claimed, track.cls_id,
                                                   track.reappear_candidate_box, true, true);
                }
                found_as_reappear_candidate = found >= 0;
            }
            if (found < 0) {
                found = unique_detection_for_box(detections, claimed, track.cls_id,
                                                 reference, true, true);
            }
            // D 刚放下时，完整框可能和“手中局部框 + 手位移”差异很大。它已经
            // 通过前序来源建立了链路，收尾首帧应优先尝试自身最近一次真实观测。
            // 普通旧库存不走这个宽松兜底。
            if (found < 0 && track.is_suspect_new && track.has_last_seen_box) {
                found = unique_detection_for_box(detections, claimed, track.cls_id,
                                                  track.last_seen_box, true, true);
            }
            if (found < 0 && track.is_suspect_new && track.has_last_hand_block_box) {
                found = unique_detection_for_box(detections, claimed, track.cls_id,
                                                  track.last_hand_block_box, true, true);
            }
            if (found < 0 && track.is_suspect_new) {
                found = unique_d_reappearance_detection_for_track(
                    detections, claimed, it->first, track, working_inventory_, track_buffer_);
                if (found >= 0) {
                    printf("[3.0] suspect#%d 通过局部→完整路径重现匹配到无手检测\n",
                           track.suspect_id);
                }
            }
            if (found < 0 && !track.is_suspect_new) {
                found = unique_no_hand_reappear_detection_for_track(
                    detections, claimed, it->first, track, working_inventory_, track_buffer_);
                found_as_reappear_candidate = found >= 0;
                if (found >= 0) {
                    printf("[3.0] item#%d 在无手帧将同类 B 暂记为重新出现候选\n",
                           track.item_id);
                }
            }

            if (!track.is_suspect_new && found >= 0 &&
                !found_at_original_position &&
                !contact_detection_is_at_original(track, detections[found])) {
                const std::map<int, int> no_known_owner;
                const int candidate_owner = unique_c_reappear_owner_for_detection(
                    detections[found], track_buffer_, no_known_owner);
                if (candidate_owner == -2) {
                    mark_hand_b_ambiguity(detections[found], &track_buffer_,
                                          no_known_owner);
                    track.reappearance_pending = true;
                    continue;
                }
                if (candidate_owner >= 0 && candidate_owner != track.item_id) {
                    track.reappearance_pending = true;
                    continue;
                }
                track.b_claim_ambiguous = false;
            }
            if (found < 0) {
                // D 必须在后续直接无手帧中连续自匹配。未达到门槛就消失，
                // 不保留为悬空 IN，更不能让它在后续随机帧重新凑次数。
                if (track.is_suspect_new &&
                    track.no_hand_self_match_count < FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                    discard_keys.insert(it->first);
                }
                continue;
            }
            const Detection& d = detections[found];
            claimed.insert(found);

            if (track.is_suspect_new) {
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                mark_current_d_owner_(it->first, found);
                ++track.no_hand_self_match_count;
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
                    update_seen(item->second, d, 0);
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
                    if (item != working_inventory_.end()) update_seen(item->second, d, 0);
                    release_not_held_(track, false);
                } else {
                    const bool should_observe_candidate =
                        found_as_reappear_candidate || track.reappearance_pending ||
                        track.has_reappear_candidate_box;
                    bool candidate_ready = true;
                    if (should_observe_candidate) {
                        candidate_ready = update_reappear_candidate(
                            &track, d, false, direct_frame_sequence_);
                    }
                    if (!candidate_ready) continue;
                    track.last_seen_box = d.box;
                    track.has_last_seen_box = true;
                    if (has_valid_old_c_move_evidence(track) &&
                        boxes_differ_as_move(track.original_box, d.box)) {
                        confirm_rearrange_(track, d.box, d.score, 0);
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
            // C 的本地路径、原位回查已经都推进到当前直接帧；现在统一形成
            // owner，再进入已有 D 的处理。这样 D 不会用另一套匹配重抢 C 的 B。
            build_frame_ownership_(detections);
            synchronize_frame_ownership_candidates_(detections, nullptr);
            for (std::map<int, int>::const_iterator owner =
                     frame_ownership_.old_item_to_detection.begin();
                 owner != frame_ownership_.old_item_to_detection.end(); ++owner) {
                // C 的唯一路径 B 即使尚未完成移动确认，也已经是当前帧的
                // owner，已有 D 不能再用同一个 B 累计自己的直接确认。
                claimed.insert(owner->second);
            }
        }
    }

    for (std::set<int>::const_iterator discard = discard_keys.begin();
         discard != discard_keys.end(); ++discard) {
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(*discard);
        if (track == track_buffer_.end()) continue;
        printf("[3.0] suspect#%d 的手离开后完整框未连续确认，丢弃候选\n",
               track->second.suspect_id);
        if (track->second.promoted_to_working_inventory && track->second.item_id > 0) {
            const int item_id = track->second.item_id;
            working_inventory_.erase(item_id);
            pending_in_ids_.erase(item_id);
            confirmed_moved_ids_.erase(item_id);
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
        promote_suspect_(promote_keys[i], d, 0);
        track->second.placed_box = d.box;
        track->second.has_placed_box = true;
        track->second.drop_confirmed = true;
    }

    // 所有旧 C、已有 D 都已优先处理完成；此时剩余 B 才能成为“全程被手
    // 遮挡、手离开后首次显现”的 D。这里先重建一次共享 owner，D 准入
    // 和稍后的结算读取完全相同的解释。
    build_frame_ownership_(detections);
    register_post_hand_reveal_suspects_(detections, &claimed);
    build_frame_ownership_(detections);
    log_frame_ownership_();
    apply_no_hand_direct_frame_(detections, frame_id);
    advance_no_hand_old_item_resolutions_();

    for (size_t di = 0; di < detections.size(); ++di) {
        if (frame_ownership_.detection_to_old_item[di] < 0 &&
            (di >= current_detection_to_d_runtime_.size() ||
             current_detection_to_d_runtime_[di] == 0)) {
            printf("[3.0] 无手直接帧出现未归属 B cls=%d；没有 D 证据链，不自动 IN\n",
                   detections[di].cls_id);
        }
    }
}

void SessionManager::apply_no_hand_direct_frame_(
        const std::vector<Detection>& detections, int frame_id) {
    no_hand_direct_inventory_ = working_inventory_;
    std::map<int, InventoryItem>& direct_items = no_hand_direct_inventory_;
    std::set<int> observed_item_ids;
    std::set<int> confirmed_front_ids;

    // C 和已有 D 的 B 都已在 observe_no_hand_frame_ 中完成唯一归属；这里只把
    // 当前直接帧已决定的结果写入结算投影，供遮挡、OUT 证据和提交共同读取。
    for (std::map<int, int>::const_iterator owner =
             frame_ownership_.old_item_to_detection.begin();
         owner != frame_ownership_.old_item_to_detection.end(); ++owner) {
        if (owner->second < 0 || static_cast<size_t>(owner->second) >= detections.size()) {
            continue;
        }
        std::map<int, InventoryItem>::iterator item =
            direct_items.find(owner->first);
        std::map<int, IdentityResolution>::const_iterator resolution =
            old_item_resolution_.find(owner->first);
        if (item == direct_items.end() || resolution == old_item_resolution_.end() ||
            (resolution->second != IdentityResolution::AT_ORIGIN &&
             resolution->second != IdentityResolution::AT_NEW_POSITION)) {
            continue;
        }
        update_seen(item->second, detections[owner->second], frame_id);
        item->second.status = ItemStatus::VISIBLE;
        item->second.block_ids.clear();
        observed_item_ids.insert(owner->first);
        if (resolution->second == IdentityResolution::AT_NEW_POSITION) {
            item->second.base_box = detections[owner->second].box;
            confirmed_front_ids.insert(owner->first);
        }
    }

    for (std::map<int, int>::const_iterator owner =
             current_d_runtime_to_detection_.begin();
         owner != current_d_runtime_to_detection_.end(); ++owner) {
        if (owner->second < 0 || static_cast<size_t>(owner->second) >= detections.size()) {
            continue;
        }
        std::map<int, OperationTrack>::const_iterator runtime =
            track_buffer_.find(owner->first);
        if (runtime == track_buffer_.end() || !runtime->second.is_suspect_new ||
            !runtime->second.promoted_to_working_inventory ||
            runtime->second.item_id <= 0) {
            continue;
        }
        std::map<int, InventoryItem>::iterator item =
            direct_items.find(runtime->second.item_id);
        if (item == direct_items.end()) continue;
        update_seen(item->second, detections[owner->second], frame_id);
        item->second.base_box = detections[owner->second].box;
        item->second.status = ItemStatus::VISIBLE;
        item->second.block_ids.clear();
        observed_item_ids.insert(runtime->second.item_id);
        if (runtime->second.no_hand_self_match_count >=
            FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
            confirmed_front_ids.insert(runtime->second.item_id);
        }
    }

    // 完整遮挡也只使用当前直接帧中已经确认的移动 C 或 D 前景框。未归属 B
    // 不能成为 blocker，更不能间接让旧 C 被判 OUT。
    for (std::map<int, InventoryItem>::const_iterator old =
             operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        if (observed_item_ids.count(old->first) || pending_out_ids_.count(old->first)) {
            continue;
        }
        std::map<int, InventoryItem>::iterator item = direct_items.find(old->first);
        if (item == direct_items.end()) continue;
        std::map<int, IdentityResolution>::const_iterator existing_resolution =
            old_item_resolution_.find(old->first);
        if (existing_resolution != old_item_resolution_.end() &&
            existing_resolution->second == IdentityResolution::OCCLUDED_CONFIRMED) {
            item->second = old->second;
            continue;
        }

        std::vector<int> blockers;
        std::vector<BBox> cover_boxes;
        const BBox original_box = old->second.base_box.area() > 0.0f
            ? old->second.base_box : old->second.box;
        for (std::set<int>::const_iterator front_id = confirmed_front_ids.begin();
             front_id != confirmed_front_ids.end(); ++front_id) {
            if (*front_id == old->first) continue;
            std::map<int, InventoryItem>::const_iterator front =
                direct_items.find(*front_id);
            if (front == direct_items.end()) continue;
            const BBox front_box = front->second.base_box.area() > 0.0f
                ? front->second.base_box : front->second.box;
            if (intersection_area(original_box, front_box) <= BLOCK_OVERLAP_AREA_EPS) {
                continue;
            }
            blockers.push_back(*front_id);
            cover_boxes.push_back(front_box);
        }
        if (fully_covered_by(original_box, cover_boxes)) {
            item->second = old->second;
            item->second.status = ItemStatus::OCCLUDED;
            item->second.block_ids.clear();
            item->second.block_ids.insert(blockers.begin(), blockers.end());
            old_item_resolution_[old->first] = IdentityResolution::OCCLUDED_CONFIRMED;
        } else {
            // 当前帧没有 B 并不是 OUT；缺失证据随后仅在主状态机中推进。
            item->second = old->second;
        }
    }

    // block_ids 只描述本张直接帧已确认且仍覆盖的前景物品。
    for (std::map<int, InventoryItem>::iterator item = direct_items.begin();
         item != direct_items.end(); ++item) {
        for (std::set<int>::iterator blocker = item->second.block_ids.begin();
             blocker != item->second.block_ids.end();) {
            std::map<int, InventoryItem>::const_iterator front =
                direct_items.find(*blocker);
            if (front == direct_items.end() || *blocker == item->first ||
                !confirmed_front_ids.count(*blocker) ||
                intersection_area(item->second.base_box, front->second.base_box) <=
                    BLOCK_OVERLAP_AREA_EPS) {
                blocker = item->second.block_ids.erase(blocker);
            } else {
                ++blocker;
            }
        }
    }
}

void SessionManager::advance_no_hand_old_item_resolutions_() {
    for (std::map<int, InventoryItem>::const_iterator old =
             operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        OperationTrack* track = find_runtime_for_item_(old->first);
        IdentityResolution resolution = IdentityResolution::UNRESOLVED;
        std::map<int, IdentityResolution>::const_iterator current =
            old_item_resolution_.find(old->first);
        if (current != old_item_resolution_.end()) resolution = current->second;
        std::map<int, std::vector<int> >::const_iterator candidates =
            frame_ownership_.item_candidates.find(old->first);
        if (resolution != IdentityResolution::UNRESOLVED ||
            frame_ownership_.old_item_to_detection.count(old->first) ||
            (candidates != frame_ownership_.item_candidates.end() &&
             !candidates->second.empty())) {
            if (track) track->no_hand_missing_count = 0;
            continue;
        }

        bool claim_or_collision_blocks_out = false;
        for (size_t ci = 0; ci < unresolved_same_class_claims_.size(); ++ci) {
            if (unresolved_same_class_claims_[ci].possible_old_item_ids.count(old->first)) {
                claim_or_collision_blocks_out = true;
                break;
            }
        }
        if (!claim_or_collision_blocks_out) {
            for (size_t gi = 0; gi < baseline_collision_groups_.size(); ++gi) {
                const std::set<int>& group = baseline_collision_groups_[gi];
                if (group.count(old->first) &&
                    !baseline_collision_group_resolved_(group)) {
                    claim_or_collision_blocks_out = true;
                    break;
                }
            }
        }
        if (claim_or_collision_blocks_out || !track || !is_active_runtime_track(*track) ||
            track->b_claim_ambiguous ||
            track->contact_path_ambiguous) {
            if (track) track->no_hand_missing_count = 0;
            continue;
        }

        const bool contact_out_evidence =
            track->contact_state == ContactState::CONTACT_MOVING &&
            track->hold_and_move && !track->observed_move_values.empty();
        const bool hand_out_evidence =
            track->contact_state == ContactState::NONE &&
            track->state != OperationTrackState::NORMAL &&
            (track->hold_and_move || has_meaningful_hand_move(*track) ||
             has_meaningful_contact_move(*track));
        if (!contact_out_evidence && !hand_out_evidence) {
            track->no_hand_missing_count = 0;
            continue;
        }

        ++track->no_hand_missing_count;
        if (track->no_hand_missing_count >= FLOW3_NO_HAND_OUT_MISSING_FRAMES) {
            mark_pending_out_(old->first);
            old_item_resolution_[old->first] = IdentityResolution::OUT_CONFIRMED;
        }
    }
}

void SessionManager::mark_pending_out_(int item_id) {
    pending_out_ids_.insert(item_id);
}

bool SessionManager::has_unresolved_no_hand_state_() {
    bool ownership_unresolved = false;

    // 这里只读取 observe_no_hand_frame_ 已经形成的结论；缺失计数和 OUT
    // 只能在逐帧主流程中推进，结算阶段不能再产生新的身份推断。
    for (std::map<int, InventoryItem>::const_iterator old =
             operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        IdentityResolution resolution = IdentityResolution::UNRESOLVED;
        std::map<int, IdentityResolution>::const_iterator current =
            old_item_resolution_.find(old->first);
        if (current != old_item_resolution_.end()) resolution = current->second;
        if (resolution == IdentityResolution::UNRESOLVED) ownership_unresolved = true;
    }

    // claim 是“可能旧 C，也可能 D”的持久声明。未成为旧 C 或已通过 D
    // 门的实际对象前，它必须阻止结算，不能把两帧存在误解释为 IN。
    if (!unresolved_same_class_claims_.empty()) ownership_unresolved = true;
    for (size_t gi = 0; gi < baseline_collision_groups_.size(); ++gi) {
        if (!baseline_collision_group_resolved_(baseline_collision_groups_[gi])) {
            ownership_unresolved = true;
        }
    }

    for (std::map<int, OperationTrack>::const_iterator track = track_buffer_.begin();
         track != track_buffer_.end(); ++track) {
        const OperationTrack& d = track->second;
        if (!d.is_suspect_new) continue;
        std::map<int, int>::const_iterator owner =
            current_d_runtime_to_detection_.find(track->first);
        if (!d.promoted_to_working_inventory || d.item_id <= 0 ||
            owner == current_d_runtime_to_detection_.end() ||
            d.no_hand_self_match_count < FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
            ownership_unresolved = true;
        }
    }
    return ownership_unresolved;
}

SettlementResult SessionManager::settle_no_hand_frame_() {
    SettlementResult result;
    if (!working_inventory_active_) return result;

    // observe_no_hand_frame_ 已完成本帧 owner、遮挡和 OUT 证据更新。这里仅
    // 检查提交门并应用工作副本，不能接触 detections 或修改身份状态。
    if (has_unresolved_no_hand_state_()) return result;

    std::map<int, InventoryItem> ownership_final_items = no_hand_direct_inventory_;
    for (std::set<int>::const_iterator out = pending_out_ids_.begin();
         out != pending_out_ids_.end(); ++out) {
        std::map<int, IdentityResolution>::const_iterator resolution =
            old_item_resolution_.find(*out);
        if (resolution != old_item_resolution_.end() &&
            resolution->second == IdentityResolution::OUT_CONFIRMED) {
            ownership_final_items.erase(*out);
        }
    }
    std::vector<InventoryEvent> ownership_events;
    for (std::map<int, InventoryItem>::const_iterator old =
             operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        std::map<int, InventoryItem>::const_iterator now =
            ownership_final_items.find(old->first);
        std::map<int, IdentityResolution>::const_iterator resolution =
            old_item_resolution_.find(old->first);
        if (now == ownership_final_items.end()) {
            if (resolution != old_item_resolution_.end() &&
                resolution->second == IdentityResolution::OUT_CONFIRMED) {
                ownership_events.push_back(make_event(EventKind::OUT, old->second));
            }
            continue;
        }
        if (resolution != old_item_resolution_.end() &&
            resolution->second == IdentityResolution::AT_NEW_POSITION &&
            boxes_differ_as_move(old->second.base_box, now->second.base_box)) {
            ownership_events.push_back(make_event(EventKind::MOVED, now->second,
                                                   old->second.base_box,
                                                   now->second.base_box));
        }
        if (old->second.status != now->second.status) {
            if (now->second.status == ItemStatus::OCCLUDED) {
                ownership_events.push_back(make_event(EventKind::OCCLUDED, now->second));
            } else if (old->second.status == ItemStatus::OCCLUDED &&
                       now->second.status == ItemStatus::VISIBLE) {
                ownership_events.push_back(make_event(EventKind::REVEALED, now->second));
            }
        }
    }
    for (std::set<int>::const_iterator in = pending_in_ids_.begin();
         in != pending_in_ids_.end(); ++in) {
        std::map<int, InventoryItem>::const_iterator item =
            ownership_final_items.find(*in);
        if (item != ownership_final_items.end()) {
            ownership_events.push_back(make_event(EventKind::IN, item->second));
        }
    }

    inventory_.replace_all(ownership_final_items, working_next_item_id_);
    rebuild_persistent_item_index_();
    result.committed = true;
    result.happened = !ownership_events.empty();
    result.events.swap(ownership_events);
    clear_runtime_after_commit_();
    return result;
}

void SessionManager::clear_runtime_after_commit_() {
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
    current_time_ms_ = time_ms;
    hand_present_ = !hand_boxes.empty();
    if (!session_active_) return output;

    if (hand_present_) {
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
        observe_no_hand_frame_(food_detections, frame_id);
        has_old_hand_box_ = false;
        output.settlement = settle_no_hand_frame_();
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
