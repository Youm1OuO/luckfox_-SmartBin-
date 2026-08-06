// ============================================================================
//  session_ownership.cc
//  3.0 session ownership planning and matching arbitration
// ============================================================================
#include "session.h"
#include "session_internal.h"
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
namespace session_internal {

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

SameClassCandidateContext build_same_class_candidate_context(
        const Detection& detection, int detection_index,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks,
        const std::map<int, int>& known_item_owner) {
    SameClassCandidateContext context;

    // 直接 owner 只是本帧的高优先级事实。它不能让同类、仍未结案的旧 C
    // 从另一个候选框的竞争关系中消失，因此下面会独立收集 viable old C。
    for (std::map<int, int>::const_iterator owner = known_item_owner.begin();
         owner != known_item_owner.end(); ++owner) {
        if (owner->second == detection_index && operation_start.count(owner->first)) {
            context.direct_old_item_ids.insert(owner->first);
        }
    }

    for (std::map<int, InventoryItem>::const_iterator old = operation_start.begin();
         old != operation_start.end(); ++old) {
        if (old->second.cls_id != detection.cls_id ||
            !working.count(old->first) || pending_in_ids.count(old->first)) {
            continue;
        }
        std::map<int, OperationTrack>::const_iterator runtime = tracks.find(old->first);
        if (runtime == tracks.end() ||
            !is_unresolved_operation_start_old_track(runtime->second) ||
            !is_active_runtime_track(runtime->second)) {
            continue;
        }
        if (detection_can_belong_to_active_track(detection, runtime->second)) {
            context.viable_unresolved_old_item_ids.insert(old->first);
        }
    }

    for (std::map<int, OperationTrack>::const_iterator track = tracks.begin();
         track != tracks.end(); ++track) {
        if (!track->second.is_suspect_new ||
            !is_active_runtime_track(track->second) ||
            track->second.cls_id != detection.cls_id) {
            continue;
        }
        if (detection_can_belong_to_active_track(detection, track->second)) {
            context.matching_suspect_runtime_keys.insert(track->first);
        }
    }
    return context;
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

// 最终库存位置变化必须以操作开始时的完整可靠框为尺度。不能复用
// normalized_nearby_distance()：它会按两个框中较小的对角线归一化，而被手
// 遮挡后的局部框恰好可能更小。这里始终只使用 before 的完整框，并把尺度
// clamp 到可调范围，避免极小/极大检测框把结果放大或缩小得失真。
float final_motion_reference_diagonal(const BBox& before) {
    if (before.area() <= 0.0f) return 0.0f;
    return std::max(
        FLOW3_MOTION_REFERENCE_DIAGONAL_MIN_PX,
        std::min(FLOW3_MOTION_REFERENCE_DIAGONAL_MAX_PX, diagonal(before)));
}

float normalized_final_motion_distance(const BBox& before, const BBox& after) {
    const float reference_diagonal = final_motion_reference_diagonal(before);
    if (reference_diagonal <= 0.0f || after.area() <= 0.0f) {
        return std::numeric_limits<float>::infinity();
    }
    return center_distance(before, after) / reference_diagonal;
}

bool boxes_differ_as_move(const BBox& before, const BBox& after) {
    return normalized_final_motion_distance(before, after) >=
        FLOW3_FORMAL_MOVE_CENTER_SHIFT_NORM;
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

int hand_direct_old_owner_strength_rank(HandDirectOldOwnerStrength strength) {
    switch (strength) {
        case HandDirectOldOwnerStrength::STRICT:
            return 0;
        case HandDirectOldOwnerStrength::LOCAL_CONTINUOUS:
            return 1;
        case HandDirectOldOwnerStrength::LOCAL_WEAK:
            return 2;
    }
    return 3;
}

int hand_direct_old_owner_local_continuity_rank(
        HandDirectOldOwnerCandidate::LocalContinuity continuity) {
    switch (continuity) {
        case HandDirectOldOwnerCandidate::LocalContinuity::STRICT_LAST_HAND_BOX:
            return 0;
        case HandDirectOldOwnerCandidate::LocalContinuity::TRACK_LAST_HAND_BOX:
            return 1;
        case HandDirectOldOwnerCandidate::LocalContinuity::PARTIAL_LAST_HAND_BOX:
            return 2;
        case HandDirectOldOwnerCandidate::LocalContinuity::NONE:
            return 3;
    }
    return 4;
}

bool hand_direct_old_owner_candidate_better(
        const HandDirectOldOwnerCandidate& left,
        const HandDirectOldOwnerCandidate& right) {
    const int left_rank = hand_direct_old_owner_strength_rank(left.strength);
    const int right_rank = hand_direct_old_owner_strength_rank(right.strength);
    if (left_rank != right_rank) return left_rank < right_rank;
    if (left.strength == HandDirectOldOwnerStrength::STRICT) {
        return left.cost + 0.0001f < right.cost;
    }
    if (left.strength == HandDirectOldOwnerStrength::LOCAL_CONTINUOUS) {
        return hand_direct_old_owner_local_continuity_rank(left.local_continuity) <
            hand_direct_old_owner_local_continuity_rank(right.local_continuity);
    }
    // 两条 LOCAL_WEAK 只能保护原位置，不能靠小框中心距离强行分配 owner。
    return false;
}

bool hand_direct_old_owner_candidate_tied(
        const HandDirectOldOwnerCandidate& left,
        const HandDirectOldOwnerCandidate& right) {
    if (hand_direct_old_owner_strength_rank(left.strength) !=
        hand_direct_old_owner_strength_rank(right.strength)) {
        return false;
    }
    if (left.strength == HandDirectOldOwnerStrength::STRICT) {
        return std::fabs(left.cost - right.cost) <= 0.0001f;
    }
    if (left.strength == HandDirectOldOwnerStrength::LOCAL_CONTINUOUS) {
        return left.local_continuity == right.local_continuity;
    }
    return true;
}

int direct_old_owner_detection_for_item(const HandDirectOldOwnerPlan& plan,
                                        int item_id) {
    std::map<int, int>::const_iterator owner = plan.detection_by_item.find(item_id);
    return owner == plan.detection_by_item.end() ? -1 : owner->second;
}

HandDirectOldOwnerStrength direct_old_owner_strength_for_detection(
        const HandDirectOldOwnerPlan& plan, int detection_index) {
    std::map<int, HandDirectOldOwnerStrength>::const_iterator strength =
        plan.strength_by_detection.find(detection_index);
    return strength == plan.strength_by_detection.end()
        ? HandDirectOldOwnerStrength::LOCAL_WEAK : strength->second;
}

// 当前 C 自己的 LOCAL_CONTINUOUS 框仍可用于它自己的真实路径更新；严格
// 原位框在调用方会先走原位分支，弱局部框只保护所有权、不作为移动正向证据。
std::set<int> claimed_with_other_direct_old_owners(
        const std::set<int>& claimed, const HandDirectOldOwnerPlan& plan,
        int current_item_id) {
    std::set<int> filtered(claimed);
    filtered.insert(plan.ambiguous_detection_indices.begin(),
                    plan.ambiguous_detection_indices.end());
    for (std::map<int, int>::const_iterator owner = plan.owner_by_detection.begin();
         owner != plan.owner_by_detection.end(); ++owner) {
        const HandDirectOldOwnerStrength strength =
            direct_old_owner_strength_for_detection(plan, owner->first);
        if (owner->second != current_item_id ||
            strength == HandDirectOldOwnerStrength::LOCAL_WEAK) {
            filtered.insert(owner->first);
        }
    }
    return filtered;
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

}  // namespace session_internal

using namespace session_internal;

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
            // 细节15：静态预约只允许真正静态的旧 C 参加竞争。若先把被手
            // 影响的 C 写进 candidates_for_detection，即使它随后不能获得
            // 自己的预约，也会以更低成本挡住另一件静态旧 C。
            if (hand_affects(hand_box, reference)) continue;
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
            if (best_detection >= 0 && !tied) {
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

HandDirectOldOwnerPlan
SessionManager::build_mutually_unique_hand_direct_old_owner_by_detection_(
        const std::vector<Detection>& detections,
        const std::set<int>& claimed_seed,
        const std::map<int, int>& known_item_owner_seed) const {
    HandDirectOldOwnerPlan plan;
    std::map<int, std::vector<HandDirectOldOwnerCandidate> > candidates_by_item;
    std::set<int> remaining_items;
    std::set<int> remaining_detections;

    for (size_t di = 0; di < detections.size(); ++di) {
        if (!claimed_seed.count(static_cast<int>(di))) {
            remaining_detections.insert(static_cast<int>(di));
        }
    }

    // 这一步只从 operation-start 旧库存出发。普通 C 只产生严格原位候选；
    // CONTACT/HAND C 才可在严格失败后使用自己的连续局部框，最后才允许
    // "小框被旧完整框包含"的弱局部候选。PLACED/MOVED C 仅允许严格原位
    // 证据，以免它自己的已放下终点被错误当成回滚证据。
    for (std::map<int, InventoryItem>::const_iterator original =
             operation_start_inventory_.begin();
         original != operation_start_inventory_.end(); ++original) {
        const int item_id = original->first;
        if (known_item_owner_seed.count(item_id) ||
            !working_inventory_.count(item_id) || pending_in_ids_.count(item_id)) {
            continue;
        }
        const OperationTrack* runtime = find_runtime_for_item_(item_id);
        if (runtime && (runtime->is_suspect_new ||
                        runtime->resolution == ExistingItemResolution::OUT_CONFIRMED ||
                        runtime->resolution == ExistingItemResolution::OCCLUDED_CONFIRMED)) {
            continue;
        }

        const BBox original_box = runtime && runtime->original_box.area() > 0.0f
            ? runtime->original_box
            : (original->second.base_box.area() > 0.0f
                   ? original->second.base_box : original->second.box);
        if (original_box.area() <= 0.0f) continue;

        const bool moved_or_placed = runtime &&
            (runtime->state == OperationTrackState::PLACED ||
             runtime->resolution == ExistingItemResolution::MOVED_CONFIRMED);
        // CONTACT_* 在保护期内仍沿既有本地 tentative 路径运行，不能提前
        // 把 B 写成全局排他 owner；保护期结束后才把其 last_hand_block_box
        // 纳入本计划的局部连续仲裁。这样既保留推/拉连续路径，也避免成熟
        // 的同类 CONTACT 轨迹按 map 顺序借走同一个 B。
        const bool allows_local = runtime && !moved_or_placed &&
            (is_active_existing_hand_track(*runtime) ||
             (is_active_contact_track(*runtime) && !is_claim_protected(*runtime)));

        for (size_t di = 0; di < detections.size(); ++di) {
            const int detection_index = static_cast<int>(di);
            if (!remaining_detections.count(detection_index) ||
                detections[di].cls_id != original->second.cls_id) {
                continue;
            }

            HandDirectOldOwnerCandidate candidate;
            candidate.item_id = item_id;
            candidate.detection_index = detection_index;
            candidate.cost = strict_match_cost(original->second.cls_id, original_box,
                                               detections[di]);
            // 对活动 CONTACT，严格几何范围可能覆盖“相对原框已推开超过
            // 12px”的真实 B；这类框必须继续走既有 CONTACT tentative/hold
            // 路径，不能在直接所有权计划里提前变成严格原位证据。HAND
            // 恢复路径则保留严格候选，供无手灰区结算使用。
            const bool contact_origin_guard = runtime &&
                is_active_contact_track(*runtime) &&
                !contact_detection_is_at_original(*runtime, detections[di]);
            if (candidate.cost < std::numeric_limits<float>::infinity() &&
                !contact_origin_guard) {
                candidate.strength = HandDirectOldOwnerStrength::STRICT;
                candidates_by_item[item_id].push_back(candidate);
                continue;
            }

            if (!allows_local) continue;
            if (runtime->has_last_hand_block_box) {
                const bool strict_last_hand = strict_match_box(
                    runtime->cls_id, runtime->last_hand_block_box,
                    detections[di].cls_id, detections[di].box);
                const bool track_last_hand = !strict_last_hand && track_match_box(
                    runtime->cls_id, runtime->last_hand_block_box,
                    detections[di].cls_id, detections[di].box);
                const bool partial_last_hand = !strict_last_hand && !track_last_hand &&
                    hand_partial_match_box(runtime->cls_id, runtime->last_hand_block_box,
                                           detections[di].cls_id, detections[di].box);
                if (strict_last_hand || track_last_hand || partial_last_hand) {
                    candidate.strength = HandDirectOldOwnerStrength::LOCAL_CONTINUOUS;
                    candidate.local_continuity = strict_last_hand
                        ? HandDirectOldOwnerCandidate::LocalContinuity::STRICT_LAST_HAND_BOX
                        : (track_last_hand
                            ? HandDirectOldOwnerCandidate::LocalContinuity::TRACK_LAST_HAND_BOX
                            : HandDirectOldOwnerCandidate::LocalContinuity::PARTIAL_LAST_HAND_BOX);
                    candidates_by_item[item_id].push_back(candidate);
                    continue;
                }
            }
            if (hand_partial_match_box(original->second.cls_id, original_box,
                                       detections[di].cls_id, detections[di].box)) {
                candidate.strength = HandDirectOldOwnerStrength::LOCAL_WEAK;
                candidates_by_item[item_id].push_back(candidate);
            }
        }
        if (!candidates_by_item[item_id].empty()) remaining_items.insert(item_id);
    }

    // 迭代的双方唯一预约。严格 > 连续局部 > 弱局部；严格候选同级才按
    // 成本裁决，局部候选必须有更强的自身连续性，否则保留歧义。无进展时
    // 绝不回退到 map/item_id 顺序或局部框中心距离。
    bool made_progress = true;
    while (made_progress) {
        made_progress = false;
        std::map<int, HandDirectOldOwnerCandidate> best_for_item;
        std::set<int> tied_items;
        for (std::set<int>::const_iterator item = remaining_items.begin();
             item != remaining_items.end(); ++item) {
            const std::vector<HandDirectOldOwnerCandidate>& candidates =
                candidates_by_item[*item];
            HandDirectOldOwnerCandidate best;
            bool has_best = false;
            bool tied = false;
            for (size_t ci = 0; ci < candidates.size(); ++ci) {
                if (!remaining_detections.count(candidates[ci].detection_index)) continue;
                if (!has_best || hand_direct_old_owner_candidate_better(candidates[ci], best)) {
                    best = candidates[ci];
                    has_best = true;
                    tied = false;
                } else if (hand_direct_old_owner_candidate_tied(candidates[ci], best)) {
                    tied = true;
                }
            }
            if (has_best && !tied) best_for_item[*item] = best;
            if (has_best && tied) tied_items.insert(*item);
        }

        std::map<int, HandDirectOldOwnerCandidate> best_for_detection;
        std::set<int> tied_detections;
        for (std::set<int>::const_iterator item = remaining_items.begin();
             item != remaining_items.end(); ++item) {
            const std::vector<HandDirectOldOwnerCandidate>& candidates =
                candidates_by_item[*item];
            for (size_t ci = 0; ci < candidates.size(); ++ci) {
                const HandDirectOldOwnerCandidate& candidate = candidates[ci];
                if (!remaining_detections.count(candidate.detection_index)) continue;
                std::map<int, HandDirectOldOwnerCandidate>::iterator best =
                    best_for_detection.find(candidate.detection_index);
                if (best == best_for_detection.end() ||
                    hand_direct_old_owner_candidate_better(candidate, best->second)) {
                    best_for_detection[candidate.detection_index] = candidate;
                    tied_detections.erase(candidate.detection_index);
                } else if (hand_direct_old_owner_candidate_tied(candidate, best->second)) {
                    tied_detections.insert(candidate.detection_index);
                }
            }
        }

        std::vector<HandDirectOldOwnerCandidate> accepted;
        for (std::map<int, HandDirectOldOwnerCandidate>::const_iterator best =
                 best_for_item.begin(); best != best_for_item.end(); ++best) {
            if (tied_items.count(best->first) ||
                tied_detections.count(best->second.detection_index)) {
                continue;
            }
            std::map<int, HandDirectOldOwnerCandidate>::const_iterator detection_best =
                best_for_detection.find(best->second.detection_index);
            if (detection_best == best_for_detection.end() ||
                detection_best->second.item_id != best->first) {
                continue;
            }
            accepted.push_back(best->second);
        }

        for (size_t ai = 0; ai < accepted.size(); ++ai) {
            const HandDirectOldOwnerCandidate& candidate = accepted[ai];
            if (!remaining_items.count(candidate.item_id) ||
                !remaining_detections.count(candidate.detection_index)) {
                continue;
            }
            plan.owner_by_detection[candidate.detection_index] = candidate.item_id;
            plan.detection_by_item[candidate.item_id] = candidate.detection_index;
            plan.strength_by_detection[candidate.detection_index] = candidate.strength;
            remaining_items.erase(candidate.item_id);
            remaining_detections.erase(candidate.detection_index);
            made_progress = true;
        }
    }

    for (std::set<int>::const_iterator item = remaining_items.begin();
         item != remaining_items.end(); ++item) {
        const std::vector<HandDirectOldOwnerCandidate>& candidates =
            candidates_by_item[*item];
        for (size_t ci = 0; ci < candidates.size(); ++ci) {
            if (remaining_detections.count(candidates[ci].detection_index)) {
                plan.ambiguous_detection_indices.insert(candidates[ci].detection_index);
            }
        }
    }
    return plan;
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

}  // namespace fridge
