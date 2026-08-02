// ============================================================================
//  session.cc
//  3.0：工作库存 + HAND_* 候选 + 疑似新物品 D + 无手稳定快照提交
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

bool strict_match(const InventoryItem& item, const VotingItem& observed) {
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

bool partial_match(const InventoryItem& item, const VotingItem& observed) {
    return partial_match_box(item.cls_id, item.box, observed.cls_id, observed.box);
}

// 仅用于“手正在遮挡”的逐帧认领。被手挡住时，YOLO 经常只给出原物品的一部分，
// 用普通 partial_match 的高 IoM 阈值会把已有 C 错当成新的 D。这里仍要求：
// 同类、局部框有足够部分落在旧框内、局部框不会比旧框异常大、中心也不能太远。
// 它不是无手稳定快照的身份匹配，绝不能在别处滥用。
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
    return move_box(track.original_box, total_move(track));
}

float move_length(const MoveValue& delta) {
    return std::sqrt(delta.dx * delta.dx + delta.dy * delta.dy);
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

InventoryItem make_inventory_item(int item_id, const VotingItem& detection,
                                  int frame_id, long long time_ms) {
    InventoryItem item;
    item.item_id = item_id;
    item.cls_id = detection.cls_id;
    item.box = detection.box;
    item.base_box = detection.box;
    item.score = detection.best_score;
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

void update_seen(InventoryItem& item, const VotingItem& detection, int frame_id) {
    item.box = detection.box;
    item.score = detection.best_score;
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
    if (track.state == OperationTrackState::NORMAL ||
        track.cls_id != detection.cls_id) {
        return false;
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
    for (size_t i = 0; i < track.track.size(); ++i) {
        if (track_match_box(track.cls_id, track.track[i],
                            detection.cls_id, detection.box)) {
            return true;
        }
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

bool detection_conflicts_with_active_track(
        const Detection& detection,
        const std::map<int, OperationTrack>& tracks) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (detection_can_belong_to_active_track(detection, it->second)) return true;
    }
    return false;
}

bool has_active_runtime_for_item(const std::map<int, OperationTrack>& tracks,
                                 int item_id) {
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (it->second.item_id == item_id &&
            it->second.state != OperationTrackState::NORMAL) {
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

// 给稳定快照建立严格的一对一绑定：只有 item 和 snapshot 两侧都唯一才绑定。
void bind_mutually_unique(const std::map<int, InventoryItem>& items,
                          const std::set<int>& candidate_item_ids,
                          const std::vector<VotingItem>& observed,
                          std::map<int, BBox>* references,
                          std::map<int, int>* item_to_snapshot,
                          std::vector<int>* snapshot_owner,
                          bool track_mode, bool partial_mode) {
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        std::map<int, std::vector<int> > item_candidates;
        std::vector<std::vector<int> > snapshot_candidates(observed.size());
        for (std::set<int>::const_iterator id = candidate_item_ids.begin();
             id != candidate_item_ids.end(); ++id) {
            if (item_to_snapshot->count(*id)) continue;
            std::map<int, InventoryItem>::const_iterator item_it = items.find(*id);
            if (item_it == items.end()) continue;
            const BBox reference = references->count(*id) ? (*references)[*id] :
                item_it->second.box;
            for (size_t si = 0; si < observed.size(); ++si) {
                if ((*snapshot_owner)[si] >= 0 || observed[si].cls_id != item_it->second.cls_id) {
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
                snapshot_candidates[si].push_back(*id);
            }
        }
        for (std::map<int, std::vector<int> >::const_iterator it = item_candidates.begin();
             it != item_candidates.end(); ++it) {
            if (it->second.size() != 1) continue;
            const int si = it->second.front();
            if (snapshot_candidates[si].size() != 1 || (*snapshot_owner)[si] >= 0) continue;
            (*item_to_snapshot)[it->first] = si;
            (*snapshot_owner)[si] = it->first;
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

// 返回快照框到该物品“完整候选轨迹”的最小匹配代价。它只在终点严格匹配
// 和原位置回查都没有解决时使用，避免把手持物体在路径中途放下的情况误判 OUT。
float track_path_match_cost(const InventoryItem& item, const OperationTrack& track,
                            const VotingItem& observed) {
    if (item.cls_id != observed.cls_id || track.cls_id != observed.cls_id) {
        return std::numeric_limits<float>::infinity();
    }
    float best_cost = std::numeric_limits<float>::infinity();
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
    return best_cost;
}

// 路径可能有多个点、同类物品也可能有多条路径，所以这里同样只提交双方
// 都是“唯一最近”的 pair。任何平分都保持未决，交给后续的旧位置回查或 OUT。
void bind_mutually_unique_track_paths(
        const std::map<int, InventoryItem>& items,
        const std::set<int>& candidate_item_ids,
        const std::vector<VotingItem>& observed,
        const std::map<int, OperationTrack>& tracks,
        std::map<int, int>* item_to_snapshot,
        std::vector<int>* snapshot_owner) {
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        std::map<int, int> best_snapshot_for_item;
        std::set<int> tied_items;
        std::map<int, std::vector<std::pair<int, float> > > candidates_for_snapshot;

        for (std::set<int>::const_iterator id = candidate_item_ids.begin();
             id != candidate_item_ids.end(); ++id) {
            if (item_to_snapshot->count(*id)) continue;
            std::map<int, InventoryItem>::const_iterator item = items.find(*id);
            const OperationTrack* track = find_track_for_item(tracks, *id);
            if (item == items.end() || !track || track->track.empty()) continue;

            float best_cost = std::numeric_limits<float>::infinity();
            int best_snapshot = -1;
            bool tied = false;
            for (size_t si = 0; si < observed.size(); ++si) {
                if ((*snapshot_owner)[si] >= 0) continue;
                const float cost = track_path_match_cost(item->second, *track, observed[si]);
                if (!(cost < std::numeric_limits<float>::infinity())) continue;
                candidates_for_snapshot[static_cast<int>(si)].push_back(
                    std::make_pair(*id, cost));
                if (cost + 0.0001f < best_cost) {
                    best_cost = cost;
                    best_snapshot = static_cast<int>(si);
                    tied = false;
                } else if (std::fabs(cost - best_cost) <= 0.0001f) {
                    tied = true;
                }
            }
            if (best_snapshot >= 0 && !tied) {
                best_snapshot_for_item[*id] = best_snapshot;
            } else if (best_snapshot >= 0) {
                tied_items.insert(*id);
            }
        }

        std::map<int, int> best_item_for_snapshot;
        std::set<int> tied_snapshots;
        for (std::map<int, std::vector<std::pair<int, float> > >::const_iterator si =
                 candidates_for_snapshot.begin();
             si != candidates_for_snapshot.end(); ++si) {
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
                best_item_for_snapshot[si->first] = best_item;
            } else if (best_item >= 0) {
                tied_snapshots.insert(si->first);
            }
        }

        for (std::map<int, int>::const_iterator item = best_snapshot_for_item.begin();
             item != best_snapshot_for_item.end(); ++item) {
            const int snapshot_index = item->second;
            if (tied_items.count(item->first) || tied_snapshots.count(snapshot_index) ||
                (*snapshot_owner)[snapshot_index] >= 0 ||
                !best_item_for_snapshot.count(snapshot_index) ||
                best_item_for_snapshot[snapshot_index] != item->first) {
                continue;
            }
            (*item_to_snapshot)[item->first] = snapshot_index;
            (*snapshot_owner)[snapshot_index] = item->first;
            made_progress = true;
        }
    }
}

}  // namespace

SessionManager::SessionManager() : no_hand_buffer_(SNAPSHOT_N, SNAPSHOT_S) {}

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
    released_hand_candidate_ids_.clear();
    hand_track_.clear();
    has_old_hand_box_ = false;
    next_suspect_id_ = -1;
    no_hand_buffer_.reset();
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
    initial_check_state_ = ALLOW_SNAPSHOT_BOOTSTRAP_WHEN_BACKEND_UNAVAILABLE
        ? InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT : InitialCheckState::NONE;
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
    no_hand_buffer_.reset();
    has_old_hand_box_ = false;
}

void SessionManager::resume_after_false_closing() {
    // CLOSING 打断了连续证据链，最安全的处理是丢弃工作副本，保留正式库存。
    if (working_inventory_active_) reset_operation_runtime_();
}

void SessionManager::validate_initial_snapshot_(const Snapshot& snapshot) const {
    int matched = 0;
    for (size_t si = 0; si < snapshot.items.size(); ++si) {
        int count = 0;
        for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
             it != inventory_.items().end(); ++it) {
            if (strict_match(it->second, snapshot.items[si]) ||
                partial_match(it->second, snapshot.items[si])) {
                ++count;
            }
        }
        if (count == 1) ++matched;
    }
    printf("[INIT-CHECK] 首张无手快照校验：%d/%zu 个框可唯一对应；库存未修改\n",
           matched, snapshot.items.size());
}

void SessionManager::initialize_from_bootstrap_snapshot_(const Snapshot& snapshot) {
    std::map<int, InventoryItem> loaded;
    int next_id = inventory_.next_item_id();
    for (size_t i = 0; i < snapshot.items.size(); ++i) {
        loaded[next_id] = make_inventory_item(next_id, snapshot.items[i],
                                               snapshot.frame_id, current_time_ms_);
        ++next_id;
    }
    inventory_.replace_all(loaded, next_id);
    rebuild_persistent_item_index_();
    has_local_inventory_ = true;
    initial_check_state_ = InitialCheckState::DONE;
    printf("[INIT] 本地稳定快照建库：%zu 个物品\n", loaded.size());
}

void SessionManager::finalize_initial_check_before_hand_() {
    if (initial_check_state_ != InitialCheckState::WAITING &&
        initial_check_state_ != InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT) return;
    if (no_hand_buffer_.empty()) {
        if (initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT) {
            Snapshot empty;
            empty.valid = true;
            initialize_from_bootstrap_snapshot_(empty);
        } else {
            initial_check_state_ = InitialCheckState::SKIPPED;
        }
        return;
    }
    Snapshot snapshot = no_hand_buffer_.take_snapshot();
    if (initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT) {
        initialize_from_bootstrap_snapshot_(snapshot);
    } else {
        validate_initial_snapshot_(snapshot);
        initial_check_state_ = InitialCheckState::DONE;
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
    released_hand_candidate_ids_.clear();
    hand_track_.clear();
    hand_track_.push_back(hand_box);
    old_hand_box_ = hand_box;
    has_old_hand_box_ = true;
    no_hand_buffer_.reset();
    no_hand_streak_ = 0;
    process_effective_hand_frame_(hand_box, detections, true);
}

void SessionManager::append_move_to_existing_hand_tracks_(const MoveValue& delta) {
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.state != OperationTrackState::HAND_PARTIAL_BLOCKED &&
            track.state != OperationTrackState::HAND_FULL_BLOCKED) {
            continue;
        }
        track.move_values.push_back(delta);
        track.track.push_back(estimated_box(track));
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
    // 初次检测可能只是局部框；它只作为临时位置，最终 stable snapshot 会刷新。
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
            if (runtime && runtime->state != OperationTrackState::NORMAL) continue;

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
    track.shelter_or_hold = false;
    track.hold_and_move = false;
    track.hold_evidence_count = 0;
    track.not_hold_evidence_count = 0;
    track.has_first_hand_block_box = false;
    track.has_last_hand_block_box = false;
    track.move_values.clear();
    track.track.clear();
    track.hand_track_start_index = -1;
}

void SessionManager::update_existing_hand_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        const MoveValue& delta, std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner) {
    std::vector<int> promote_keys;
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
        const int old_position_index = unique_detection_at_old_position(
            detections, *claimed_detection_indices, track);
        const bool old_clean = old_position_is_clean(detections, track, working_inventory_);

        if (track.is_suspect_new) {
            if (observed_index >= 0) {
                const Detection& d = detections[observed_index];
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
                        update_seen(item->second, d, item->second.updated_frame + 1);
                        item->second.base_box = d.box;
                    }
                }
            }
            continue;
        }

        if (track.hold_and_move) {
            if (observed_index >= 0) {
                const Detection& d = detections[observed_index];
                claimed_detection_indices->insert(observed_index);
                if (track.item_id > 0) {
                    (*known_item_owner)[track.item_id] = observed_index;
                }
                const BBox previous = track.has_last_hand_block_box
                    ? track.last_hand_block_box : expected;
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                track.last_hand_block_box = d.box;
                track.has_last_hand_block_box = true;
                const float hand_distance = move_length(delta);
                const float object_distance = center_distance(previous, d.box);
                const bool falls_behind = hand_distance >= TRACK_HAND_MOVE_EPS &&
                    object_distance <= hand_distance * FLOW3_DROP_FALL_BEHIND_RATIO &&
                    iom(previous, d.box) >= FLOW3_TRACK_PARTIAL_IOM;
                if (!hand_touches_detection(hand_box, d.box) || falls_behind) {
                    confirm_rearrange_(track, d.box, d.score, 0);
                }
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
            ++track.not_hold_evidence_count;
            track.hold_evidence_count = 0;
            track.last_hand_block_box = old_d.box;
            track.has_last_hand_block_box = true;
            if (track.not_hold_evidence_count >= FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
                if (item != working_inventory_.end()) update_seen(item->second, old_d, 0);
                release_not_held_(track, false);
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
            ++track.hold_evidence_count;
            track.not_hold_evidence_count = 0;
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;
            track.last_hand_block_box = d.box;
            track.has_last_hand_block_box = true;
            if (track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
                track.hold_and_move = true;
                track.hold_evidence_count = 0;
                track.not_hold_evidence_count = 0;
            }
        } else if (observed_index >= 0) {
            // 该框不能证明移动，但可作为下一帧局部连续性的参考。
            track.last_hand_block_box = detections[observed_index].box;
            track.has_last_hand_block_box = true;
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
    for (size_t di = 0; di < detections.size(); ++di) {
        if (claimed_detection_indices->count(static_cast<int>(di))) continue;
        const Detection& d = detections[di];
        const bool hand_visible_d = hand_touches_detection(hand_box, d.box);
        // D 已经被放到 C 原位置时，手可能继续移开，因此 D 不一定还贴手。
        // 只要它是唯一一个覆盖“当前看不见的 C”原位置的未认领框，也必须
        // 预登记；否则 C 会在后续无手阶段被错误当成 OUT。
        const int replacement_owner = unique_c_replacement_owner_for_detection(
            d, track_buffer_, known_item_owner);
        if (!hand_visible_d && replacement_owner < 0) {
            continue;
        }

        // 先排除已经可以认领给工作库存、或已存在 HAND_* 轨迹的物品。
        bool belongs_to_known_item = false;
        for (std::map<int, InventoryItem>::const_iterator it = working_inventory_.begin();
             it != working_inventory_.end(); ++it) {
            // 这个旧 item 已经在本帧认领了另一个检测框。当前 D 即使与
            // 它的旧框局部相似，也不能再次借用同一身份；否则相邻同类
            // 物品会永远进不了疑似 D 链路。
            if (known_item_owner.count(it->first) &&
                known_item_owner.find(it->first)->second != static_cast<int>(di)) {
                continue;
            }
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
                if (track.cls_id != d.cls_id || track.state == OperationTrackState::NORMAL) continue;
                if (track.item_id > 0 && known_item_owner.count(track.item_id) &&
                    known_item_owner.find(track.item_id)->second != static_cast<int>(di)) {
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
        if (existing && existing->state != OperationTrackState::NORMAL &&
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
        if (observed_index >= 0) {
            const Detection& d = detections[observed_index];
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;
            track.first_hand_block_box = d.box;
            track.last_hand_block_box = d.box;
            track.has_first_hand_block_box = true;
            track.has_last_hand_block_box = true;
            claimed_detection_indices->insert(observed_index);
            (*known_item_owner)[item.item_id] = observed_index;
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
        const BBox& hand_box, const std::vector<Detection>& /*detections*/) {
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
            c.hold_evidence_count = 0;
            ++c.not_hold_evidence_count;
            if (d.drop_confirmed && d.promoted_to_working_inventory && d_covers_c) {
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
    MoveValue delta;
    if (!first_hand_frame) {
        delta.dx = hand_box.cx() - old_hand_box_.cx();
        delta.dy = hand_box.cy() - old_hand_box_.cy();
        hand_track_.push_back(hand_box);
        append_move_to_existing_hand_tracks_(delta);
    }

    // 先更新已有 HAND_*；随后先为仍稳定可见的旧库存保留本帧自己的
    // 严格框，再处理新进入 HAND_* 的物品。这个先后顺序很关键：若旧
    // 苹果已有完整框，旁边贴手的新苹果不能先被“局部可能属于旧苹果”
    // 的规则抢走。
    std::set<int> claimed;
    std::map<int, int> known_item_owner;
    if (!first_hand_frame) {
        update_existing_hand_tracks_(hand_box, detections, delta, &claimed,
                                     &known_item_owner);
    }
    reserve_visible_known_detections_(hand_box, detections, &claimed,
                                      &known_item_owner);
    mark_newly_hand_blocked_items_(hand_box, detections, &claimed,
                                   &known_item_owner);
    scan_or_update_suspects_(hand_box, detections, &claimed, known_item_owner,
                             first_hand_frame);
    apply_suspect_cover_evidence_(hand_box, detections);
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
        if (detection_matches_old_working_inventory(
                d, working_inventory_, operation_start_inventory_) ||
            detection_conflicts_with_active_track(d, track_buffer_)) {
            continue;
        }
        if (!hand_track_touches_detection(hand_track_, d)) continue;

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
        track.hand_track_start_index = 0;
        track.post_hand_reveal_no_hand_streak = no_hand_streak_;
        track.track.push_back(d.box);
        track_buffer_[key] = track;
        claimed_detection_indices->insert(detection_index);
        printf("[3.0] 手离开后预登记疑似新物品 D suspect#%d cls=%d source=%s\n",
               key, d.cls_id, suspect_source_name(track.suspect_source));
    }
}

void SessionManager::observe_no_hand_frame_(const std::vector<Detection>& detections) {
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
                track.state == OperationTrackState::NORMAL) {
                continue;
            }
            BBox reference = track.has_placed_box ? track.placed_box : estimated_box(track);
            if (track.state == OperationTrackState::PLACED && track.has_placed_box) {
                reference = track.placed_box;
            }
            int found = unique_detection_for_box(detections, claimed, track.cls_id,
                                                 reference, true, true);
            if (found < 0 && track.original_box.area() > 0.0f) {
                found = unique_detection_for_box(detections, claimed, track.cls_id,
                                                  track.original_box, true, false);
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
            if (found < 0) {
                // POST_HAND_REVEAL_D 的第一张完整 B 就是它唯一的初始证据。
                // 下一张有效无手帧无法自匹配，说明该 B 不稳定，直接丢弃，
                // 而不是让它在后续随机帧中重新凑够次数。
                if (track.is_suspect_new && !track.promoted_to_working_inventory &&
                    track.suspect_source == SuspectSource::POST_HAND_REVEAL_D &&
                    no_hand_streak_ > track.post_hand_reveal_no_hand_streak) {
                    discard_keys.insert(it->first);
                }
                continue;
            }
            const Detection& d = detections[found];
            claimed.insert(found);
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;

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
                if (partial_match_box(track.cls_id, track.original_box, d.cls_id, d.box,
                                      FLOW3_TRACK_PARTIAL_IOM)) {
                    std::map<int, InventoryItem>::iterator item =
                        working_inventory_.find(track.item_id);
                    if (item != working_inventory_.end()) update_seen(item->second, d, 0);
                    release_not_held_(track, false);
                } else if (track.hold_and_move || !track.move_values.empty()) {
                    confirm_rearrange_(track, d.box, d.score, 0);
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
        printf("[3.0] suspect#%d 的手离开后完整框未连续确认，丢弃候选\n",
               track->second.suspect_id);
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
    // 遮挡、手离开后首次显现”的 D。
    register_post_hand_reveal_suspects_(detections, &claimed);
}

void SessionManager::mark_pending_out_(int item_id) {
    pending_out_ids_.insert(item_id);
}

void SessionManager::refresh_confirmed_blockers_(const std::set<int>& /*observed_working_ids*/) {
    // 最终 stable snapshot 中统一重算。手还在时不按任意框交集写 block_ids，
    // 避免旧 2.0 版本那种 blocker 不断膨胀的问题。
}

SettlementResult SessionManager::settle_stable_snapshot_(const Snapshot& snapshot) {
    SettlementResult result;
    if (!working_inventory_active_ || !snapshot.valid) return result;

    std::map<int, InventoryItem> final_items = working_inventory_;
    std::vector<VotingItem> observed = snapshot.items;
    std::vector<int> snapshot_owner(observed.size(), -1);
    std::map<int, int> item_to_snapshot;
    std::map<int, BBox> references;
    std::set<int> track_priority_ids;
    std::set<int> all_ids;
    std::map<int, BBox> original_references;
    std::set<int> original_position_ids;

    for (std::map<int, InventoryItem>::const_iterator it = final_items.begin();
         it != final_items.end(); ++it) {
        all_ids.insert(it->first);
        references[it->first] = it->second.base_box.area() > 0.0f
            ? it->second.base_box : it->second.box;
        const OperationTrack* runtime = find_runtime_for_item_(it->first);
        if (runtime && runtime->has_placed_box) references[it->first] = runtime->placed_box;
        else if (runtime && runtime->is_suspect_new && runtime->has_last_seen_box) {
            references[it->first] = runtime->last_seen_box;
        }
        else if (runtime && !runtime->track.empty()) references[it->first] = runtime->track.back();
        if (pending_in_ids_.count(it->first) || confirmed_moved_ids_.count(it->first) ||
            (runtime && (runtime->hold_and_move ||
                         (!runtime->move_values.empty() &&
                          runtime->state != OperationTrackState::NORMAL)))) {
            track_priority_ids.insert(it->first);
        }
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

    // 绑定顺序：已确认移动/新 D 的终点 -> 普通严格匹配 -> 局部匹配。
    bind_mutually_unique(final_items, track_priority_ids, observed,
                         &references, &item_to_snapshot, &snapshot_owner, true, false);
    bind_mutually_unique(final_items, all_ids, observed,
                         &references, &item_to_snapshot, &snapshot_owner, false, false);
    // 若物品在手尚未离开时已经掉队并停在候选路径中段，终点框不会命中；
    // 在原位置回查前再做一次整条 path 的唯一最近绑定，避免这类物品被 OUT。
    bind_mutually_unique_track_paths(final_items, track_priority_ids, observed,
                                     track_buffer_, &item_to_snapshot, &snapshot_owner);
    // 再回查旧位置。这样 hold_and_move 尚未凑满、或只被手短暂擦过的
    // 物品，只要在原位重新出现，就不会因为轨迹参考框漂移而被误判出库。
    bind_mutually_unique(final_items, original_position_ids, observed,
                         &original_references, &item_to_snapshot, &snapshot_owner,
                         false, false);
    bind_mutually_unique(final_items, original_position_ids, observed,
                         &original_references, &item_to_snapshot, &snapshot_owner,
                         false, true);
    bind_mutually_unique(final_items, all_ids, observed,
                         &references, &item_to_snapshot, &snapshot_owner, false, true);

    // 文档允许 HAND_* 下尚未来得及凑满两次证据的物品，在手离开后的稳定
    // 快照中由自己的候选轨迹补确认整理。这里仍要求它命中了自己的轨迹参考框，
    // 不把普通未匹配框当作移动终点。
    for (std::map<int, int>::const_iterator it = item_to_snapshot.begin();
         it != item_to_snapshot.end(); ++it) {
        const OperationTrack* runtime = find_runtime_for_item_(it->first);
        std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(it->first);
        if (!runtime || original == operation_start_inventory_.end() ||
            runtime->state == OperationTrackState::NORMAL ||
            runtime->move_values.empty()) {
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
    for (std::map<int, int>::const_iterator it = item_to_snapshot.begin();
         it != item_to_snapshot.end(); ++it) {
        observed_ids.insert(it->first);
        InventoryItem& item = final_items[it->first];
        update_seen(item, observed[it->second], snapshot.frame_id);
        if (pending_in_ids_.count(it->first) || confirmed_moved_ids_.count(it->first)) {
            item.base_box = observed[it->second].box;
        }
        item.status = ItemStatus::VISIBLE;
        item.block_ids.clear();
    }

    // 未稳定出现的疑似 D 不允许入库；它只属于本次临时操作。
    for (std::set<int>::const_iterator it = pending_in_ids_.begin();
         it != pending_in_ids_.end(); ++it) {
        if (!observed_ids.count(*it)) {
            printf("[3.0] 已提升的 D item#%d 未能唯一绑定到稳定快照；取消本轮 IN\n",
                   *it);
            final_items.erase(*it);
        }
    }

    // 仅“本轮确认移动/确认入库，且在稳定快照中出现”的物品可成为前景遮挡物。
    std::set<int> confirmed_front_ids;
    for (std::set<int>::const_iterator it = observed_ids.begin(); it != observed_ids.end(); ++it) {
        if (pending_in_ids_.count(*it) || confirmed_moved_ids_.count(*it)) {
            confirmed_front_ids.insert(*it);
        }
    }

    for (std::map<int, InventoryItem>::iterator it = final_items.begin();
         it != final_items.end(); ++it) {
        if (observed_ids.count(it->first)) continue;
        std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(it->first);
        if (original == operation_start_inventory_.end()) continue;

        std::vector<int> blockers;
        for (std::set<int>::const_iterator front_id = confirmed_front_ids.begin();
             front_id != confirmed_front_ids.end(); ++front_id) {
            if (*front_id == it->first) continue;
            std::map<int, InventoryItem>::const_iterator front = final_items.find(*front_id);
            if (front == final_items.end()) continue;
            const bool full = fully_covered_by(original->second.base_box,
                                                std::vector<BBox>(1, front->second.base_box));
            const bool partial_new = pending_in_ids_.count(*front_id) &&
                cover_ratio(original->second.base_box, front->second.base_box) >=
                    FLOW3_D_PARTIAL_COVER_RATIO;
            if (full || partial_new) blockers.push_back(*front_id);
        }
        if (!blockers.empty()) {
            it->second = original->second;
            it->second.status = ItemStatus::OCCLUDED;
            it->second.block_ids.clear();
            it->second.block_ids.insert(blockers.begin(), blockers.end());
            continue;
        }

        const OperationTrack* runtime = find_runtime_for_item_(it->first);
        // 到这里说明：D 的最终遮挡解释已经处理过，且旧位置与动态参考
        // 位置都没有把 C 认领回来。move_values 非空表示手确实在该候选
        // 上完成过有效移动；即使两次正向证据尚未凑满，也要按文档在收尾
        // 阶段把“整条候选路径都找不到 C”作为出库证据。没有任何有效移动
        // 的普通漏检仍然保留，不会凭空 OUT。
        const bool strong_out_evidence = pending_out_ids_.count(it->first) ||
            (runtime && runtime->state != OperationTrackState::NORMAL &&
             (runtime->hold_and_move || !runtime->move_values.empty()));
        if (strong_out_evidence) {
            mark_pending_out_(it->first);
        } else {
            // 无法证明 OUT：严格保留本次操作开始前的正式物品，绝不把缺失
            // 误变成 OUT，也不让一个未绑定框取代它。
            it->second = original->second;
        }
    }

    for (std::set<int>::const_iterator out = pending_out_ids_.begin();
         out != pending_out_ids_.end(); ++out) {
        final_items.erase(*out);
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

    // 最后才生成正式事件；整个过程中没有任何“未绑定快照框自动 IN”。
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

    for (size_t si = 0; si < snapshot_owner.size(); ++si) {
        if (snapshot_owner[si] < 0) {
            printf("[3.0] 稳定快照出现未绑定检测 cls=%d；没有 D 证据链，不自动 IN\n",
                   observed[si].cls_id);
        }
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
        no_hand_buffer_.reset();
        no_hand_streak_ = 0;
        if (!has_local_inventory_) return output;
        const BBox hand = choose_primary_hand(hand_boxes);
        if (!working_inventory_active_) {
            begin_working_operation_(hand, food_detections);
            return output;
        }
        // 文档定义：微小移动帧整体不存在，不能更新 old_hand、D、轨迹或计数。
        if (has_old_hand_box_ &&
            center_distance(old_hand_box_, hand) < HAND_MICRO_MOVE_SKIP_EPS) {
            return output;
        }
        process_effective_hand_frame_(hand, food_detections, false);
        old_hand_box_ = hand;
        has_old_hand_box_ = true;
        return output;
    }

    ++no_hand_streak_;
    if (working_inventory_active_) {
        // 手刚离开的每一帧都先供 D/候选轨迹做最终确认；随后再进入稳定快照投票。
        observe_no_hand_frame_(food_detections);
        has_old_hand_box_ = false;
        no_hand_buffer_.push(food_detections, frame_id);
        if (!no_hand_buffer_.full()) return output;
        Snapshot snapshot = no_hand_buffer_.take_snapshot();
        output.stable_snapshot_generated = true;
        output.settlement = settle_stable_snapshot_(snapshot);
        return output;
    }

    if (initial_check_state_ == InitialCheckState::WAITING ||
        initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT) {
        no_hand_buffer_.push(food_detections, frame_id);
        if (!no_hand_buffer_.full()) return output;
        Snapshot snapshot = no_hand_buffer_.take_snapshot();
        output.stable_snapshot_generated = true;
        if (initial_check_state_ == InitialCheckState::WAITING) {
            validate_initial_snapshot_(snapshot);
            initial_check_state_ = InitialCheckState::DONE;
        } else {
            initialize_from_bootstrap_snapshot_(snapshot);
        }
    }
    return output;
}

}  // namespace fridge
