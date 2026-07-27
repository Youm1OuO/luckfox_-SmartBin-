// ============================================================================
//  session.cc
//  单库存快照业务实现。
//
//  这里故意不依赖旧 ByteTrackLite：当前可执行文件只运行本文件的
//  OperationTrack，避免 track_id 与 item_id 混在一起。
// ============================================================================
#include "session.h"
#include "fridge_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <set>

namespace fridge {
namespace {

float ratio_difference(float a, float b) {
    float larger = std::max(std::fabs(a), std::fabs(b));
    if (larger <= 0.001f) return 0.0f;
    return std::fabs(a - b) / larger;
}

// 快照身份匹配有两种结果：普通完整框匹配，或“局部小框被完整框包含”。
// 后者只是一种兜底，绝不能反过来把一个大框当作小框的同一物品。
enum class IdentityMatchKind { NONE, NORMAL, PARTIAL };

enum class SnapshotRole { NONE, DIRECT, MOVED, REVEALED, RETURNED, NEW_ITEM };

struct IdentityMatch {
    IdentityMatchKind kind = IdentityMatchKind::NONE;
    float score = std::numeric_limits<float>::max();
};

struct SnapshotBinding {
    int item_id = -1;
    SnapshotRole role = SnapshotRole::NONE;
    IdentityMatchKind match_kind = IdentityMatchKind::NONE;
};

// 整个新业务层统一使用这个匹配方法：中心距离、宽、长。
// 不同场景只传不同阈值；FULL_HAND_OCCLUDED 会把宽高阈值放宽到 1。
bool box_match(const BBox& a, const BBox& b,
               float center_norm_limit,
               float width_ratio_limit,
               float height_ratio_limit) {
    if (a.area() <= 0.0f || b.area() <= 0.0f) return false;
    if (normalized_nearby_distance(a, b) > center_norm_limit) return false;
    if (ratio_difference(a.w(), b.w()) > width_ratio_limit) return false;
    if (ratio_difference(a.h(), b.h()) > height_ratio_limit) return false;
    return true;
}

float box_distance_score(const BBox& a, const BBox& b) {
    return normalized_nearby_distance(a, b) +
           0.25f * ratio_difference(a.w(), b.w()) +
           0.25f * ratio_difference(a.h(), b.h());
}

bool is_contained_partial_box(const BBox& complete_reference,
                              const BBox& observed_box) {
    if (complete_reference.area() <= 0.0f || observed_box.area() <= 0.0f) {
        return false;
    }

    // 必须是“当前框更小且被完整参考框包含”，不能把方向反过来。
    if (observed_box.area() >=
        complete_reference.area() * PARTIAL_MATCH_MAX_AREA_RATIO) {
        return false;
    }
    if (overlap_ratio_of_smaller(complete_reference, observed_box) <
        PARTIAL_MATCH_CONTAINMENT) {
        return false;
    }

    // 完全在参考框内时，局部框可能位于角落；只要求中心在框内。
    // 允许少量 YOLO 抖动时，中心也可以落在框边缘附近。
    return point_in_box(observed_box.cx(), observed_box.cy(), complete_reference) ||
           normalized_nearby_distance(complete_reference, observed_box) <=
               BOX_MATCH_CENTER_NORM;
}

IdentityMatch match_snapshot_to_reference(int cls_id, const BBox& reference,
                                          const VotingItem& observed) {
    IdentityMatch result;
    if (cls_id != observed.cls_id || reference.area() <= 0.0f) return result;

    if (box_match(observed.box, reference,
                  BOX_MATCH_CENTER_NORM,
                  BOX_MATCH_WIDTH_RATIO,
                  BOX_MATCH_HEIGHT_RATIO)) {
        result.kind = IdentityMatchKind::NORMAL;
        result.score = box_distance_score(observed.box, reference);
        return result;
    }

    if (is_contained_partial_box(reference, observed.box)) {
        result.kind = IdentityMatchKind::PARTIAL;
        // NORMAL 的分数始终小于约 1；PARTIAL 从 1 开始，保证优先普通匹配。
        result.score = 1.0f +
                       (1.0f - overlap_ratio_of_smaller(reference, observed.box)) +
                       0.20f * std::min(normalized_nearby_distance(reference,
                                                                    observed.box),
                                        1.0f);
    }
    return result;
}

IdentityMatch match_snapshot_to_item(const InventoryItem& item,
                                     const VotingItem& observed) {
    IdentityMatch result;
    if (item.cls_id != observed.cls_id) return result;

    IdentityMatch last_match = match_snapshot_to_reference(item.cls_id,
                                                            item.last_seen_box,
                                                            observed);
    IdentityMatch anchor_match;
    if (item.anchor_valid) {
        anchor_match = match_snapshot_to_reference(item.cls_id, item.anchor_box,
                                                   observed);
    }

    // 先使用普通匹配；last_seen 与 anchor 同时可用时，取几何分数更好的一个。
    if (last_match.kind == IdentityMatchKind::NORMAL ||
        anchor_match.kind == IdentityMatchKind::NORMAL) {
        if (last_match.kind == IdentityMatchKind::NORMAL &&
            (anchor_match.kind != IdentityMatchKind::NORMAL ||
             last_match.score <= anchor_match.score)) {
            return last_match;
        }
        return anchor_match;
    }

    // PARTIAL 只允许相对于可靠 anchor_box；last_seen 允许本来就是局部框。
    if (item.anchor_valid && anchor_match.kind == IdentityMatchKind::PARTIAL) {
        return anchor_match;
    }
    return result;
}

BBox item_reference_box(const InventoryItem& item) {
    return item.anchor_valid ? item.anchor_box : item.last_seen_box;
}

// 新物品没有旧 anchor 可供比较时，只能在它明确被同一快照中更大框包含的
// 情况下，保守推断当前 B 可能只是局部框。其余情况按普通新物品处理。
bool snapshot_item_is_likely_partial(const Snapshot& snapshot, int snapshot_index) {
    const BBox& box = snapshot.items[snapshot_index].box;
    for (size_t other_index = 0; other_index < snapshot.items.size(); ++other_index) {
        if ((int)other_index == snapshot_index) continue;
        if (is_contained_partial_box(snapshot.items[other_index].box, box)) {
            return true;
        }
    }
    return false;
}

struct PlannedItemMatch {
    int item_id = -1;
    IdentityMatch match;
};

// 从“比较开始时就是 wanted_status”的旧库存中，找唯一身份匹配项。
// 这里故意使用 initial_status，而不是 planned 的当前状态：本轮刚改成 OUT
// 或 OCCLUDED 的物品，不允许又被后面的快照框重新认领。
PlannedItemMatch find_unique_initial_item_match(
        const std::map<int, InventoryItem>& planned,
        const std::map<int, ItemStatus>& initial_status,
        const std::set<int>& identity_bound_items,
        const VotingItem& observed,
        ItemStatus wanted_status,
        long long current_time_ms,
        bool require_unexpired_out = false) {
    PlannedItemMatch best;
    float second_score = std::numeric_limits<float>::max();
    int matched_count = 0;

    for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
         it != planned.end(); ++it) {
        std::map<int, ItemStatus>::const_iterator status_it =
            initial_status.find(it->first);
        if (status_it == initial_status.end() || status_it->second != wanted_status) {
            continue;
        }
        if (identity_bound_items.count(it->first)) continue;

        const InventoryItem& item = it->second;
        if (require_unexpired_out && item.out_time_ms > 0 &&
            current_time_ms - item.out_time_ms > OUT_ITEM_EXPIRE_MS) {
            continue;
        }

        IdentityMatch match = match_snapshot_to_item(item, observed);
        if (match.kind == IdentityMatchKind::NONE) continue;
        ++matched_count;

        if (match.score < best.match.score) {
            second_score = best.match.score;
            best.item_id = item.item_id;
            best.match = match;
        } else if (match.score < second_score) {
            second_score = match.score;
        }
    }

    if (best.item_id < 0) return PlannedItemMatch();

    // 局部框若同时被多个同类 anchor 包含，不能靠“谁稍近一点”强行绑定。
    if (best.match.kind == IdentityMatchKind::PARTIAL && matched_count > 1) {
        return PlannedItemMatch();
    }
    if (second_score < std::numeric_limits<float>::max() &&
        second_score - best.match.score < IDENTITY_MATCH_AMBIGUITY_MARGIN) {
        return PlannedItemMatch();
    }
    return best;
}

bool has_meaningful_motion(const BBox& previous, const BBox& current) {
    return normalized_nearby_distance(previous, current) > TRACK_CREATE_MOTION_NORM;
}

bool matches_reference_box(const Detection& det, int cls_id, const BBox& reference) {
    return det.cls_id == cls_id &&
           box_match(det.box, reference,
                     TRACK_FRAME_CENTER_NORM,
                     TRACK_FRAME_WIDTH_RATIO,
                     TRACK_FRAME_HEIGHT_RATIO);
}

int find_best_unused_detection(const std::vector<Detection>& detections,
                               const std::vector<bool>& used,
                               int cls_id, const BBox& reference,
                               float center_norm_limit,
                               float width_ratio_limit,
                               float height_ratio_limit) {
    int best_index = -1;
    float best_score = std::numeric_limits<float>::max();
    for (size_t i = 0; i < detections.size(); ++i) {
        if (used[i] || detections[i].cls_id != cls_id) continue;
        if (!box_match(detections[i].box, reference,
                       center_norm_limit, width_ratio_limit, height_ratio_limit)) {
            continue;
        }
        float score = box_distance_score(detections[i].box, reference);
        if (score < best_score) {
            best_score = score;
            best_index = (int)i;
        }
    }
    return best_index;
}

bool source_is_visible(const std::vector<Detection>& detections,
                       int cls_id, const BBox& source_box) {
    for (size_t i = 0; i < detections.size(); ++i) {
        if (matches_reference_box(detections[i], cls_id, source_box)) return true;
    }
    return false;
}

BBox move_box(const BBox& box, float dx, float dy) {
    return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy);
}

BBox move_box_center_to(const BBox& box, float cx, float cy) {
    return move_box(box, cx - box.cx(), cy - box.cy());
}

bool hand_overlaps_box(const BBox& hand, const BBox& box) {
    return overlap_ratio_of_smaller(hand, box) >= TRACK_HAND_OVERLAP;
}

bool hand_fully_covers_box(const BBox& hand, const BBox& box) {
    return overlap_ratio_of_smaller(hand, box) >= TRACK_FULL_OCCLUSION_OVERLAP;
}

bool hand_near_box(const BBox& hand, const BBox& box) {
    return hand_overlaps_box(hand, box) ||
           normalized_nearby_distance(hand, box) <= TRACK_HAND_NEAR_NORM;
}

bool any_hand_near(const std::vector<BBox>& hands, const BBox& box) {
    for (size_t i = 0; i < hands.size(); ++i) {
        if (hand_near_box(hands[i], box)) return true;
    }
    return false;
}

bool any_hand_overlaps(const std::vector<BBox>& hands, const BBox& box) {
    for (size_t i = 0; i < hands.size(); ++i) {
        if (hand_overlaps_box(hands[i], box)) return true;
    }
    return false;
}

int nearest_hand_index(const std::vector<BBox>& hands, const BBox& box) {
    int best = -1;
    float best_score = std::numeric_limits<float>::max();
    for (size_t i = 0; i < hands.size(); ++i) {
        float score = normalized_nearby_distance(hands[i], box);
        if (score < best_score) {
            best_score = score;
            best = (int)i;
        }
    }
    return best;
}

bool nearby(const BBox& a, const BBox& b) {
    return normalized_nearby_distance(a, b) <= NEARBY_DISTANCE_THRESH;
}

bool similar_anchor_size(const BBox& box, const InventoryItem& item) {
    if (!item.anchor_valid) return true;
    return ratio_difference(box.w(), item.anchor_box.w()) <= TRACK_PLACED_SIZE_RATIO &&
           ratio_difference(box.h(), item.anchor_box.h()) <= TRACK_PLACED_SIZE_RATIO;
}

void set_visible(InventoryItem& item, const BBox& box, float score,
                 int frame_id, bool update_anchor) {
    item.last_seen_box = box;
    item.score = score;
    item.updated_frame = frame_id;
    item.status = ItemStatus::VISIBLE;
    item.out_time_ms = 0;
    item.no_occluder_count = 0;
    if (update_anchor) {
        item.anchor_box = box;
        item.anchor_valid = true;
    }
}

void set_occluded(InventoryItem& item) {
    item.status = ItemStatus::OCCLUDED;
    item.out_time_ms = 0;
    item.no_occluder_count = 0;
}

void set_out(InventoryItem& item, long long time_ms) {
    item.status = ItemStatus::OUT;
    item.out_time_ms = time_ms;
}

InventoryItem make_inventory_item(int item_id, const VotingItem& observed,
                                  int frame_id, long long time_ms) {
    InventoryItem item;
    item.item_id = item_id;
    item.cls_id = observed.cls_id;
    item.anchor_box = observed.box;
    item.anchor_valid = true;
    item.last_seen_box = observed.box;
    item.score = observed.best_score;
    item.status = ItemStatus::VISIBLE;
    item.no_occluder_count = 0;
    item.duplicate_count = 0;
    item.created_frame = frame_id;
    item.updated_frame = frame_id;
    item.created_time_ms = time_ms;
    item.out_time_ms = 0;
    return item;
}

struct TrackDetectionPair {
    int track_index;
    int detection_index;
    float score;
};

// 单向握手：每个快照物品必须绑定到唯一一个 planned InventoryItem，
// 且该物品最后必须是 VISIBLE。这里只检查前面已经确定的绑定，不重新猜。
bool snapshot_inventory_handshake_valid(
        const std::vector<int>& snapshot_item_to_inventory_id,
        const std::map<int, InventoryItem>& planned) {
    std::set<int> bound_ids;
    for (size_t si = 0; si < snapshot_item_to_inventory_id.size(); ++si) {
        int item_id = snapshot_item_to_inventory_id[si];
        if (item_id < 0 || !bound_ids.insert(item_id).second) return false;
        std::map<int, InventoryItem>::const_iterator it = planned.find(item_id);
        if (it == planned.end() || it->second.status != ItemStatus::VISIBLE) return false;
    }
    return true;
}

}  // namespace

SessionManager::SessionManager()
    : no_hand_buffer_(SNAPSHOT_N, SNAPSHOT_S),
      next_operation_track_id_(1),
      operation_pending_(false),
      hand_present_(false),
      no_hand_streak_(0),
      current_time_ms_(0),
      session_start_time_ms_(0),
      init_state_(InitState::WAIT_FIRST_STABLE_SNAPSHOT),
      backend_status_(BackendStatus::UNKNOWN) {}

void SessionManager::start_new_session(long long time_ms) {
    inventory_ = InventoryDB();
    no_hand_buffer_.reset();
    tracks_.clear();
    candidates_.clear();
    next_operation_track_id_ = 1;
    operation_pending_ = false;
    hand_present_ = false;
    no_hand_streak_ = 0;
    current_time_ms_ = time_ms;
    session_start_time_ms_ = time_ms;
    init_state_ = InitState::WAIT_BACKEND;
    backend_status_ = BackendStatus::UNKNOWN;
    printf("[SESSION] 新开门会话：等待后台库存或第一份连续无手稳定快照\n");
}

void SessionManager::init_from_backend(const std::vector<InventoryItem>& items,
                                       bool authoritative_empty) {
    if (init_state_ == InitState::READY) return;

    if (items.empty() && !authoritative_empty) {
        mark_backend_unavailable();
        return;
    }

    std::map<int, InventoryItem> loaded;
    int next_id = 1;
    for (size_t i = 0; i < items.size(); ++i) {
        InventoryItem item = items[i];
        if (item.item_id <= 0 || loaded.count(item.item_id) != 0) {
            item.item_id = next_id;
        }
        if (!item.anchor_valid) {
            item.anchor_box = item.last_seen_box;
            item.anchor_valid = item.anchor_box.area() > 0.0f;
        }
        item.no_occluder_count = 0;
        item.duplicate_count = 0;
        loaded[item.item_id] = item;
        next_id = std::max(next_id, item.item_id + 1);
    }
    inventory_.replace_all(loaded, next_id);
    backend_status_ = BackendStatus::TRUSTED;
    init_state_ = InitState::WAIT_FIRST_STABLE_SNAPSHOT;
    printf("[SESSION] 已载入可信后台库存：%zu 件；等待第一份稳定快照对齐\n",
           loaded.size());
}

void SessionManager::mark_backend_unavailable() {
    if (init_state_ == InitState::READY) return;
    backend_status_ = BackendStatus::NO_TRUSTED_BACKEND;
    init_state_ = InitState::WAIT_FIRST_STABLE_SNAPSHOT;
    printf("[SESSION] 无可信后台库存：第一份稳定快照将初始化本地库存\n");
}

void SessionManager::finish_session(long long time_ms) {
    current_time_ms_ = time_ms;
    no_hand_buffer_.reset();
    tracks_.clear();
    candidates_.clear();
    operation_pending_ = false;
    hand_present_ = false;
    no_hand_streak_ = 0;
    inventory_.cleanup_expired(time_ms);
    printf("[SESSION] 会话结束：可见=%zu 遮挡=%zu 出库历史=%zu\n",
           inventory_.count_by_status(ItemStatus::VISIBLE),
           inventory_.count_by_status(ItemStatus::OCCLUDED),
           inventory_.count_by_status(ItemStatus::OUT));
}

bool SessionManager::item_matches_snapshot(const InventoryItem& item,
                                           const VotingItem& observed) const {
    return match_snapshot_to_item(item, observed).kind != IdentityMatchKind::NONE;
}

bool SessionManager::item_is_bound_to_operation(int item_id) const {
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (tracks_[i].bound_item_id == item_id) return true;
    }
    for (size_t i = 0; i < candidates_.size(); ++i) {
        if (candidates_[i].bound_item_id == item_id) return true;
    }
    return false;
}

int SessionManager::find_unique_inventory_binding(
        const Detection& detection, const std::vector<BBox>& hand_boxes) const {
    // Candidate 的起点必须是“手确实盖到物品框”，不能只是手在附近。
    if (!any_hand_overlaps(hand_boxes, detection.box)) return -1;

    float best_score = std::numeric_limits<float>::max();
    float second_score = std::numeric_limits<float>::max();
    int best_id = -1;
    bool best_is_partial = false;
    int matched_count = 0;
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status != ItemStatus::VISIBLE || item.cls_id != detection.cls_id) continue;
        if (item_is_bound_to_operation(item.item_id)) continue;

        bool match_last = box_match(detection.box, item.last_seen_box,
                                    TRACK_FRAME_CENTER_NORM,
                                    TRACK_FRAME_WIDTH_RATIO,
                                    TRACK_FRAME_HEIGHT_RATIO);
        bool match_anchor = item.anchor_valid &&
                            box_match(detection.box, item.anchor_box,
                                      TRACK_FRAME_CENTER_NORM,
                                      TRACK_FRAME_WIDTH_RATIO,
                                      TRACK_FRAME_HEIGHT_RATIO);
        bool match_partial = item.anchor_valid &&
                             is_contained_partial_box(item.anchor_box, detection.box);
        if (!match_last && !match_anchor && !match_partial) continue;
        ++matched_count;

        float score = 1.0f;
        if (match_last || match_anchor) {
            score = box_distance_score(detection.box, item.last_seen_box);
            if (item.anchor_valid) {
                score = std::min(score, box_distance_score(detection.box, item.anchor_box));
            }
        } else {
            // TRACK_FRAME 的普通阈值比快照宽；局部兜底仍要排在所有普通匹配之后。
            score = 2.0f +
                    (1.0f - overlap_ratio_of_smaller(item.anchor_box, detection.box));
        }
        if (score < best_score) {
            second_score = best_score;
            best_score = score;
            best_id = item.item_id;
            best_is_partial = !match_last && !match_anchor;
        } else if (score < second_score) {
            second_score = score;
        }
    }

    // 两个同类库存物品同样接近时，宁可不建立 Candidate。
    if (best_id < 0) return -1;
    if (best_is_partial && matched_count > 1) return -1;
    if (second_score < std::numeric_limits<float>::max() &&
        second_score - best_score < IDENTITY_MATCH_AMBIGUITY_MARGIN) {
        return -1;
    }
    return best_id;
}

void SessionManager::update_tracks_while_hand_present(
        const std::vector<Detection>& food_detections,
        const std::vector<BBox>& hand_boxes, int /*frame_id*/) {
    // ---------------------------------------------------------------------
    // 1. 已经确认发生移动的正式 Track：沿用已有的逐帧更新逻辑。
    // ---------------------------------------------------------------------
    std::vector<TrackDetectionPair> track_pairs;
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        const OperationTrack& track = tracks_[ti];
        for (size_t di = 0; di < food_detections.size(); ++di) {
            const Detection& det = food_detections[di];
            if (det.cls_id != track.cls_id) continue;

            bool match = false;
            float score = 0.0f;
            if (track.state == OperationTrackState::FULL_HAND_OCCLUDED) {
                match = box_match(det.box, track.proxy_box,
                                  TRACK_REAPPEAR_CENTER_NORM, 1.0f, 1.0f);
                score = box_distance_score(det.box, track.proxy_box);
            } else if (track.has_last_yolo_box) {
                match = box_match(det.box, track.last_yolo_box,
                                  TRACK_FRAME_CENTER_NORM,
                                  TRACK_FRAME_WIDTH_RATIO,
                                  TRACK_FRAME_HEIGHT_RATIO);
                score = box_distance_score(det.box, track.last_yolo_box);
            }
            if (match) track_pairs.push_back({(int)ti, (int)di, score});
        }
    }
    std::sort(track_pairs.begin(), track_pairs.end(),
              [](const TrackDetectionPair& a, const TrackDetectionPair& b) {
                  return a.score < b.score;
              });

    std::vector<bool> track_used(tracks_.size(), false);
    std::vector<bool> det_used(food_detections.size(), false);
    for (size_t ci = 0; ci < track_pairs.size(); ++ci) {
        const TrackDetectionPair& pair = track_pairs[ci];
        if (track_used[pair.track_index] || det_used[pair.detection_index]) continue;
        track_used[pair.track_index] = true;
        det_used[pair.detection_index] = true;

        OperationTrack& track = tracks_[pair.track_index];
        const Detection& det = food_detections[pair.detection_index];
        const InventoryItem* item = inventory_.find_by_item(track.bound_item_id);
        int hand_index = nearest_hand_index(hand_boxes, track.proxy_box);

        if (track.state == OperationTrackState::FULL_HAND_OCCLUDED) {
            // 物品从完全遮挡中出现：先恢复真实 YOLO 框，下一帧再估计位移。
            track.last_yolo_box = det.box;
            track.has_last_yolo_box = true;
            if (hand_index >= 0) {
                track.last_hand_box = hand_boxes[hand_index];
                track.has_last_hand_box = true;
            }
            track.state = OperationTrackState::TRACKING_VISIBLE;
            track.path.push_back(track.proxy_box);
            continue;
        }

        BBox old_yolo = track.last_yolo_box;
        float dx = det.box.cx() - old_yolo.cx();
        float dy = det.box.cy() - old_yolo.cy();
        float raw_motion = normalized_nearby_distance(det.box, old_yolo);
        bool hands_still_cover_item = any_hand_overlaps(hand_boxes, det.box);

        if (track.state == OperationTrackState::PLACED) {
            if (raw_motion > TRACK_STILL_CENTER_NORM && hands_still_cover_item) {
                track.state = OperationTrackState::TRACKING_VISIBLE;
                track.has_release_box = false;
                track.proxy_box = move_box(track.proxy_box, dx, dy);
            }
        } else {
            bool can_confirm_placed = raw_motion <= TRACK_STILL_CENTER_NORM &&
                                      !hands_still_cover_item &&
                                      (!item || similar_anchor_size(det.box, *item));
            if (can_confirm_placed) {
                track.proxy_box = move_box_center_to(track.proxy_box,
                                                      det.box.cx(), det.box.cy());
                track.release_box = track.proxy_box;
                track.has_release_box = true;
                track.state = OperationTrackState::PLACED;
            } else {
                track.proxy_box = move_box(track.proxy_box, dx, dy);
            }
        }

        track.last_yolo_box = det.box;
        track.has_last_yolo_box = true;
        if (hand_index >= 0) {
            track.last_hand_box = hand_boxes[hand_index];
            track.has_last_hand_box = true;
        }
        track.path.push_back(track.proxy_box);
    }

    // 已确认 Track 本帧没有物品框：只有手仍与它相关，才让代理框跟手移动。
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        if (track_used[ti]) continue;
        OperationTrack& track = tracks_[ti];
        BBox reference = track.has_last_yolo_box ? track.last_yolo_box : track.proxy_box;
        int hand_index = nearest_hand_index(hand_boxes, reference);
        if (hand_index < 0) continue;

        bool hand_related = track.state == OperationTrackState::FULL_HAND_OCCLUDED
                                ? hand_near_box(hand_boxes[hand_index], track.proxy_box)
                                : hand_overlaps_box(hand_boxes[hand_index], reference);
        if (!hand_related) continue;

        const BBox& hand = hand_boxes[hand_index];
        if (track.state == OperationTrackState::PLACED) {
            track.has_release_box = false;
        } else if (track.has_last_hand_box) {
            track.proxy_box = move_box(track.proxy_box,
                                       hand.cx() - track.last_hand_box.cx(),
                                       hand.cy() - track.last_hand_box.cy());
        }
        track.last_hand_box = hand;
        track.has_last_hand_box = true;
        track.has_last_yolo_box = false;
        track.state = OperationTrackState::FULL_HAND_OCCLUDED;
        track.path.push_back(track.proxy_box);
    }

    // ---------------------------------------------------------------------
    // 2. Candidate：手碰到物品不等于物品移动。只有真实位移才升级 Track。
    // ---------------------------------------------------------------------
    std::vector<OperationCandidate> next_candidates;
    next_candidates.reserve(candidates_.size());
    for (size_t ci = 0; ci < candidates_.size(); ++ci) {
        OperationCandidate candidate = candidates_[ci];

        if (candidate.state == OperationCandidateState::VISIBLE_CANDIDATE) {
            int det_index = -1;
            if (candidate.has_last_yolo_box) {
                det_index = find_best_unused_detection(
                    food_detections, det_used, candidate.cls_id, candidate.last_yolo_box,
                    TRACK_FRAME_CENTER_NORM,
                    TRACK_FRAME_WIDTH_RATIO,
                    TRACK_FRAME_HEIGHT_RATIO);
            }

            if (det_index >= 0) {
                const Detection& det = food_detections[det_index];
                det_used[det_index] = true;
                bool moved = has_meaningful_motion(candidate.last_yolo_box, det.box);
                bool hand_still_related = any_hand_near(hand_boxes, det.box);
                if (moved && hand_still_related) {
                    int hand_index = nearest_hand_index(hand_boxes, det.box);
                    OperationTrack track;
                    track.track_id = next_operation_track_id_++;
                    track.bound_item_id = candidate.bound_item_id;
                    track.cls_id = candidate.cls_id;
                    track.start_box = candidate.source_box;
                    track.proxy_box = move_box(candidate.source_box,
                                               det.box.cx() - candidate.last_yolo_box.cx(),
                                               det.box.cy() - candidate.last_yolo_box.cy());
                    track.last_yolo_box = det.box;
                    track.has_last_yolo_box = true;
                    if (hand_index >= 0) {
                        track.last_hand_box = hand_boxes[hand_index];
                        track.has_last_hand_box = true;
                    }
                    track.state = OperationTrackState::TRACKING_VISIBLE;
                    track.path.push_back(track.start_box);
                    track.path.push_back(track.proxy_box);
                    tracks_.push_back(track);
                    printf("[OP_TRACK] 确认移动，创建 Track#%d -> item#%d\n",
                           track.track_id, track.bound_item_id);
                    continue;
                }

                candidate.last_yolo_box = det.box;
                candidate.has_last_yolo_box = true;
                int hand_index = nearest_hand_index(hand_boxes, det.box);
                if (hand_index >= 0) {
                    candidate.last_hand_box = hand_boxes[hand_index];
                    candidate.has_last_hand_box = true;
                }
                next_candidates.push_back(candidate);
                continue;
            }

            // 原先还能看到的物品突然不见，只有手完整盖住该物品时才转完全遮挡 Candidate。
            BBox reference = candidate.has_last_yolo_box ? candidate.last_yolo_box
                                                          : candidate.source_box;
            int hand_index = nearest_hand_index(hand_boxes, reference);
            if (hand_index >= 0 && hand_fully_covers_box(hand_boxes[hand_index], reference)) {
                candidate.state = OperationCandidateState::FULL_HAND_OCCLUDED_CANDIDATE;
                candidate.start_hand_box = hand_boxes[hand_index];
                candidate.last_hand_box = hand_boxes[hand_index];
                candidate.has_last_hand_box = true;
                candidate.has_last_yolo_box = false;
                next_candidates.push_back(candidate);
            }
            // 否则是普通漏检或手已离开物品：丢弃 Candidate，不建 Track。
            continue;
        }

        // FULL_HAND_OCCLUDED_CANDIDATE：原位置重新出现说明物品没被移动。
        if (source_is_visible(food_detections, candidate.cls_id, candidate.source_box)) {
            continue;
        }

        BBox hand_reference = candidate.has_last_hand_box ? candidate.last_hand_box
                                                           : candidate.source_box;
        int hand_index = nearest_hand_index(hand_boxes, hand_reference);
        if (hand_index < 0) continue;
        const BBox& hand = hand_boxes[hand_index];
        bool hand_moved = has_meaningful_motion(candidate.start_hand_box, hand);

        // 若物品在新位置重新出现，且手也已移动，直接建立可见 Track。
        int moved_det_index = -1;
        float moved_det_score = std::numeric_limits<float>::max();
        for (size_t di = 0; di < food_detections.size(); ++di) {
            if (det_used[di] || food_detections[di].cls_id != candidate.cls_id) continue;
            if (!any_hand_near(hand_boxes, food_detections[di].box)) continue;
            if (!has_meaningful_motion(candidate.source_box, food_detections[di].box)) continue;
            float score = normalized_nearby_distance(hand, food_detections[di].box);
            if (score < moved_det_score) {
                moved_det_score = score;
                moved_det_index = (int)di;
            }
        }

        if (hand_moved && moved_det_index >= 0) {
            const Detection& det = food_detections[moved_det_index];
            det_used[moved_det_index] = true;
            OperationTrack track;
            track.track_id = next_operation_track_id_++;
            track.bound_item_id = candidate.bound_item_id;
            track.cls_id = candidate.cls_id;
            track.start_box = candidate.source_box;
            track.proxy_box = move_box_center_to(candidate.source_box,
                                                  det.box.cx(), det.box.cy());
            track.last_yolo_box = det.box;
            track.has_last_yolo_box = true;
            track.last_hand_box = hand;
            track.has_last_hand_box = true;
            track.state = OperationTrackState::TRACKING_VISIBLE;
            track.path.push_back(track.start_box);
            track.path.push_back(track.proxy_box);
            tracks_.push_back(track);
            printf("[OP_TRACK] 确认完全遮挡后移动，创建 Track#%d -> item#%d\n",
                   track.track_id, track.bound_item_id);
            continue;
        }

        if (hand_moved) {
            // 原位置仍不可见，手也确实移动：正式 Track 开始使用手代理。
            OperationTrack track;
            track.track_id = next_operation_track_id_++;
            track.bound_item_id = candidate.bound_item_id;
            track.cls_id = candidate.cls_id;
            track.start_box = candidate.source_box;
            track.proxy_box = move_box(candidate.source_box,
                                       hand.cx() - candidate.start_hand_box.cx(),
                                       hand.cy() - candidate.start_hand_box.cy());
            track.last_hand_box = hand;
            track.has_last_hand_box = true;
            track.has_last_yolo_box = false;
            track.state = OperationTrackState::FULL_HAND_OCCLUDED;
            track.path.push_back(track.start_box);
            track.path.push_back(track.proxy_box);
            tracks_.push_back(track);
            printf("[OP_TRACK] 确认完全遮挡后移动，创建 Track#%d -> item#%d\n",
                   track.track_id, track.bound_item_id);
            continue;
        }

        candidate.last_hand_box = hand;
        candidate.has_last_hand_box = true;
        next_candidates.push_back(candidate);
    }
    candidates_.swap(next_candidates);

    // ---------------------------------------------------------------------
    // 3. 本帧尚未绑定的、被手实际覆盖的可见物品，只建立 Candidate。
    // ---------------------------------------------------------------------
    for (size_t di = 0; di < food_detections.size(); ++di) {
        if (det_used[di]) continue;
        const Detection& det = food_detections[di];
        int item_id = find_unique_inventory_binding(det, hand_boxes);
        if (item_id < 0) continue;
        const InventoryItem* item = inventory_.find_by_item(item_id);
        if (!item) continue;
        int hand_index = nearest_hand_index(hand_boxes, det.box);
        if (hand_index < 0) continue;

        OperationCandidate candidate;
        candidate.bound_item_id = item_id;
        candidate.cls_id = item->cls_id;
        candidate.source_box = item->anchor_valid ? item->anchor_box : item->last_seen_box;
        candidate.last_yolo_box = det.box;
        candidate.has_last_yolo_box = true;
        candidate.start_hand_box = hand_boxes[hand_index];
        candidate.last_hand_box = hand_boxes[hand_index];
        candidate.has_last_hand_box = true;
        candidate.state = OperationCandidateState::VISIBLE_CANDIDATE;
        candidates_.push_back(candidate);
        det_used[di] = true;
    }

    // ---------------------------------------------------------------------
    // 4. 手一开始就完整挡住物品：只建立完全遮挡 Candidate，不直接建 Track。
    // ---------------------------------------------------------------------
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status != ItemStatus::VISIBLE || item_is_bound_to_operation(item.item_id)) {
            continue;
        }

        BBox source = item.anchor_valid ? item.anchor_box : item.last_seen_box;
        int hand_index = -1;
        for (size_t hi = 0; hi < hand_boxes.size(); ++hi) {
            if (hand_fully_covers_box(hand_boxes[hi], source)) {
                hand_index = (int)hi;
                break;
            }
        }
        if (hand_index < 0) continue;
        if (source_is_visible(food_detections, item.cls_id, source)) continue;

        OperationCandidate candidate;
        candidate.bound_item_id = item.item_id;
        candidate.cls_id = item.cls_id;
        candidate.source_box = source;
        candidate.start_hand_box = hand_boxes[hand_index];
        candidate.last_hand_box = hand_boxes[hand_index];
        candidate.has_last_hand_box = true;
        candidate.has_last_yolo_box = false;
        candidate.state = OperationCandidateState::FULL_HAND_OCCLUDED_CANDIDATE;
        candidates_.push_back(candidate);
    }
}

void SessionManager::initialize_from_snapshot(const Snapshot& snapshot) {
    std::map<int, InventoryItem> planned = inventory_.items();
    int next_id = inventory_.next_item_id();
    std::set<int> used_items;

    for (size_t si = 0; si < snapshot.items.size(); ++si) {
        const VotingItem& observed = snapshot.items[si];
        int best_id = -1;
        float best_score = std::numeric_limits<float>::max();
        for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
             it != planned.end(); ++it) {
            const InventoryItem& item = it->second;
            if (used_items.count(item.item_id) || item.status == ItemStatus::OUT) continue;
            if (!item_matches_snapshot(item, observed)) continue;
            float score = box_distance_score(item.last_seen_box, observed.box);
            if (score < best_score) {
                best_score = score;
                best_id = item.item_id;
            }
        }
        if (best_id >= 0) {
            set_visible(planned[best_id], observed.box, observed.best_score,
                        snapshot.frame_id, false);
            used_items.insert(best_id);
        } else {
            int id = next_id++;
            planned[id] = make_inventory_item(id, observed, snapshot.frame_id, current_time_ms_);
            if (snapshot_item_is_likely_partial(snapshot, (int)si)) {
                planned[id].anchor_box = BBox();
                planned[id].anchor_valid = false;
            }
            // 新建项本来就来自这份首快照，不能在下面被误标为遮挡。
            used_items.insert(id);
        }
    }

    // 有后台库存但第一份画面暂时没看到的物品，先视为遮挡；不在开门时直接报 OUT。
    for (std::map<int, InventoryItem>::iterator it = planned.begin();
         it != planned.end(); ++it) {
        if (it->second.status == ItemStatus::VISIBLE && !used_items.count(it->first)) {
            set_occluded(it->second);
        }
    }
    inventory_.replace_all(planned, next_id);
    init_state_ = InitState::READY;
    tracks_.clear();
    candidates_.clear();
    operation_pending_ = false;
    printf("[SESSION] 第一份稳定快照已完成初始化：库存=%zu 件\n", planned.size());
    print_inventory();
}

void SessionManager::refresh_visible_items_without_operation(const Snapshot& snapshot) {
    // 无手、且自上次结算以来没有手操作：快照只能刷新已明确匹配物品的
    // last_seen_box，绝不新增、取出、整理或修改可见/遮挡状态。
    std::set<int> used_items;
    for (size_t si = 0; si < snapshot.items.size(); ++si) {
        const VotingItem& observed = snapshot.items[si];
        int best_id = -1;
        float best_score = std::numeric_limits<float>::max();
        for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
             it != inventory_.items().end(); ++it) {
            const InventoryItem& item = it->second;
            if (item.status != ItemStatus::VISIBLE || used_items.count(item.item_id)) continue;
            if (!item_matches_snapshot(item, observed)) continue;

            float score = box_distance_score(item.last_seen_box, observed.box);
            if (item.anchor_valid) {
                score = std::min(score, box_distance_score(item.anchor_box, observed.box));
            }
            if (score < best_score) {
                best_score = score;
                best_id = item.item_id;
            }
        }
        if (best_id >= 0) {
            inventory_.update_seen_item(best_id, observed.box,
                                        observed.best_score, snapshot.frame_id);
            used_items.insert(best_id);
        }
    }
}

SettlementResult SessionManager::settle_snapshot(const Snapshot& snapshot) {
    SettlementResult result;
    if (!operation_pending_) return result;

    std::map<int, InventoryItem> planned = inventory_.items();
    int next_id = inventory_.next_item_id();
    std::map<int, ItemStatus> initial_status;
    for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
         it != planned.end(); ++it) {
        initial_status[it->first] = it->second.status;
    }

    // binding/role 只属于这一次结算。每个快照物品只允许有一个身份。
    std::vector<SnapshotBinding> bindings(snapshot.items.size());
    std::set<int> identity_bound_items;
    std::set<int> resolved_initial_visible_items;

    const auto initial_status_is = [&initial_status](int item_id, ItemStatus status) {
        std::map<int, ItemStatus>::const_iterator it = initial_status.find(item_id);
        return it != initial_status.end() && it->second == status;
    };
    const auto bind_snapshot = [&bindings, &identity_bound_items](
            int snapshot_index, int item_id, SnapshotRole role,
            IdentityMatchKind match_kind) {
        bindings[snapshot_index].item_id = item_id;
        bindings[snapshot_index].role = role;
        bindings[snapshot_index].match_kind = match_kind;
        identity_bound_items.insert(item_id);
    };
    // -----------------------------------------------------------------
    // 1. 原位置仍可见：只刷新 last_seen_box，绝不因一个局部框缩小 anchor。
    // -----------------------------------------------------------------
    for (size_t si = 0; si < snapshot.items.size(); ++si) {
        const VotingItem& observed = snapshot.items[si];
        PlannedItemMatch match = find_unique_initial_item_match(
            planned, initial_status, identity_bound_items, observed,
            ItemStatus::VISIBLE, current_time_ms_);
        if (match.item_id < 0) continue;

        InventoryItem& item = planned[match.item_id];
        set_visible(item, observed.box, observed.best_score, snapshot.frame_id, false);
        bind_snapshot((int)si, item.item_id, SnapshotRole::DIRECT, match.match.kind);
        resolved_initial_visible_items.insert(item.item_id);
    }

    // -----------------------------------------------------------------
    // 2. Track 证明整理。先确认所有 MOVED，再统一处理旧位置露出和新位置遮挡，
    //    避免一个 Track 先把另一个正在移动的物品误标为遮挡。
    // -----------------------------------------------------------------
    struct ConfirmedMove {
        int item_id;
        BBox old_box;
        BBox full_new_box;
        int snapshot_index;
    };
    std::vector<ConfirmedMove> confirmed_moves;

    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        const OperationTrack& track = tracks_[ti];
        if (!initial_status_is(track.bound_item_id, ItemStatus::VISIBLE) ||
            identity_bound_items.count(track.bound_item_id) ||
            resolved_initial_visible_items.count(track.bound_item_id)) {
            continue;
        }
        std::map<int, InventoryItem>::iterator item_it = planned.find(track.bound_item_id);
        if (item_it == planned.end()) continue;

        struct TrackSnapshotMatch {
            int snapshot_index = -1;
            IdentityMatch match;
            BBox complete_reference;
        } chosen;
        int candidate_count = 0;

        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            if (bindings[si].item_id >= 0 || snapshot.items[si].cls_id != track.cls_id) {
                continue;
            }

            IdentityMatch best_match;
            BBox best_reference;
            if (track.has_release_box) {
                // 有 release_box 时只能验证它；不能失败后退回整条 path 猜。
                best_match = match_snapshot_to_reference(track.cls_id,
                                                         track.release_box,
                                                         snapshot.items[si]);
                best_reference = track.release_box;
            } else {
                for (size_t pi = 0; pi < track.path.size(); ++pi) {
                    IdentityMatch candidate = match_snapshot_to_reference(
                        track.cls_id, track.path[pi], snapshot.items[si]);
                    if (candidate.kind != IdentityMatchKind::NONE &&
                        candidate.score < best_match.score) {
                        best_match = candidate;
                        best_reference = track.path[pi];
                    }
                }
            }

            if (best_match.kind == IdentityMatchKind::NONE) continue;
            ++candidate_count;
            chosen.snapshot_index = (int)si;
            chosen.match = best_match;
            chosen.complete_reference = best_reference;
        }

        // Track 对应的目标框不唯一时，不证明整理。
        if (candidate_count != 1 || chosen.snapshot_index < 0) continue;

        InventoryItem& item = item_it->second;
        const VotingItem& observed = snapshot.items[chosen.snapshot_index];
        BBox old_box = item_reference_box(item);
        if (chosen.match.kind == IdentityMatchKind::NORMAL) {
            set_visible(item, observed.box, observed.best_score, snapshot.frame_id, true);
        } else {
            // 当前 B 只是局部框：anchor 必须保存 Track 给出的完整位置。
            set_visible(item, observed.box, observed.best_score, snapshot.frame_id, false);
            item.anchor_box = chosen.complete_reference;
            item.anchor_valid = true;
        }
        bind_snapshot(chosen.snapshot_index, item.item_id, SnapshotRole::MOVED,
                      chosen.match.kind);
        resolved_initial_visible_items.insert(item.item_id);
        confirmed_moves.push_back({item.item_id, old_box,
                                   chosen.complete_reference,
                                   chosen.snapshot_index});
        result.happened = true;
        result.events.push_back({EventKind::MOVED, item.item_id, item.cls_id,
                                 observed.box, old_box, observed.box,
                                 observed.best_score});
    }

    // 2a. A 离开旧位置后，附近原本遮挡的 C 可能露出。
    for (size_t mi = 0; mi < confirmed_moves.size(); ++mi) {
        const BBox& old_box = confirmed_moves[mi].old_box;
        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            if (bindings[si].item_id >= 0 || !nearby(old_box, snapshot.items[si].box)) {
                continue;
            }
            PlannedItemMatch match = find_unique_initial_item_match(
                planned, initial_status, identity_bound_items, snapshot.items[si],
                ItemStatus::OCCLUDED, current_time_ms_);
            if (match.item_id < 0) continue;

            InventoryItem& revealed = planned[match.item_id];
            BBox old_revealed_box = item_reference_box(revealed);
            set_visible(revealed, snapshot.items[si].box,
                        snapshot.items[si].best_score, snapshot.frame_id, false);
            bind_snapshot((int)si, revealed.item_id, SnapshotRole::REVEALED,
                          match.match.kind);
            result.happened = true;
            result.events.push_back({EventKind::REVEALED, revealed.item_id,
                                     revealed.cls_id, snapshot.items[si].box,
                                     old_revealed_box, snapshot.items[si].box,
                                     snapshot.items[si].best_score});
        }
    }

    // 2b. A 移到新位置后，原本可见却没有任何当前身份绑定的 D 可能被挡住。
    for (size_t mi = 0; mi < confirmed_moves.size(); ++mi) {
        const BBox& new_box = confirmed_moves[mi].full_new_box;
        for (std::map<int, InventoryItem>::iterator it = planned.begin();
             it != planned.end(); ++it) {
            InventoryItem& candidate = it->second;
            if (candidate.item_id == confirmed_moves[mi].item_id ||
                !initial_status_is(candidate.item_id, ItemStatus::VISIBLE) ||
                resolved_initial_visible_items.count(candidate.item_id) ||
                identity_bound_items.count(candidate.item_id)) {
                continue;
            }
            BBox candidate_box = item_reference_box(candidate);
            if (!nearby(candidate_box, new_box)) continue;

            set_occluded(candidate);
            resolved_initial_visible_items.insert(candidate.item_id);
            result.happened = true;
            result.events.push_back({EventKind::OCCLUDED, candidate.item_id,
                                     candidate.cls_id, candidate_box,
                                     candidate_box, BBox(), candidate.score});
        }
    }

    // -----------------------------------------------------------------
    // 3. 剩余旧可见 A 消失：先判断附近 C 是“旧遮挡物露出”还是未知遮挡物，
    //    然后才决定 A 是 OUT 还是 OCCLUDED。
    // -----------------------------------------------------------------
    for (std::map<int, InventoryItem>::iterator it = planned.begin();
         it != planned.end(); ++it) {
        InventoryItem& item = it->second;
        if (!initial_status_is(item.item_id, ItemStatus::VISIBLE) ||
            resolved_initial_visible_items.count(item.item_id)) {
            continue;
        }

        BBox old_box = item_reference_box(item);
        bool has_unknown_nearby = false;
        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            if (!nearby(old_box, snapshot.items[si].box)) continue;

            if (bindings[si].item_id >= 0) {
                // DIRECT 表示该物体本轮仍在原位：它在操作前已经存在，不能突然
                // 变成另一个消失物品的“新遮挡物”。REVEALED 同理，它只是原先
                // 被遮挡的旧物品重新露出。只有 MOVED / RETURNED / NEW_ITEM 才
                // 可能是本轮新来到旧位置附近的遮挡物。
                if (bindings[si].role == SnapshotRole::MOVED ||
                    bindings[si].role == SnapshotRole::RETURNED ||
                    bindings[si].role == SnapshotRole::NEW_ITEM) {
                    has_unknown_nearby = true;
                }
                continue;
            }

            PlannedItemMatch reveal = find_unique_initial_item_match(
                planned, initial_status, identity_bound_items, snapshot.items[si],
                ItemStatus::OCCLUDED, current_time_ms_);
            if (reveal.item_id < 0) {
                has_unknown_nearby = true;
                continue;
            }

            InventoryItem& revealed = planned[reveal.item_id];
            BBox old_revealed_box = item_reference_box(revealed);
            set_visible(revealed, snapshot.items[si].box,
                        snapshot.items[si].best_score, snapshot.frame_id, false);
            bind_snapshot((int)si, revealed.item_id, SnapshotRole::REVEALED,
                          reveal.match.kind);
            result.happened = true;
            result.events.push_back({EventKind::REVEALED, revealed.item_id,
                                     revealed.cls_id, snapshot.items[si].box,
                                     old_revealed_box, snapshot.items[si].box,
                                     snapshot.items[si].best_score});
        }

        if (has_unknown_nearby) {
            set_occluded(item);
            result.events.push_back({EventKind::OCCLUDED, item.item_id, item.cls_id,
                                     old_box, old_box, BBox(), item.score});
        } else {
            // 附近没有框，或附近只有“原本遮挡的物品露出”：A 都不是被新物体挡住。
            set_out(item, current_time_ms_);
            result.events.push_back({EventKind::OUT, item.item_id, item.cls_id,
                                     old_box, BBox(), BBox(), item.score});
        }
        result.happened = true;
        resolved_initial_visible_items.insert(item.item_id);
    }

    // -----------------------------------------------------------------
    // 4. 剩余 B：只允许匹配“比较开始时就遮挡/出库”的旧项；本轮刚 OUT 或
    //    OCCLUDED 的 A 绝不能在这里被重新认回，避免把 OUT + IN 合并成整理。
    // -----------------------------------------------------------------
    for (size_t si = 0; si < snapshot.items.size(); ++si) {
        if (bindings[si].item_id >= 0) continue;
        const VotingItem& observed = snapshot.items[si];

        PlannedItemMatch match = find_unique_initial_item_match(
            planned, initial_status, identity_bound_items, observed,
            ItemStatus::OCCLUDED, current_time_ms_);
        if (match.item_id >= 0) {
            InventoryItem& item = planned[match.item_id];
            BBox old_box = item_reference_box(item);
            set_visible(item, observed.box, observed.best_score, snapshot.frame_id, false);
            bind_snapshot((int)si, item.item_id, SnapshotRole::REVEALED,
                          match.match.kind);
            result.happened = true;
            result.events.push_back({EventKind::REVEALED, item.item_id, item.cls_id,
                                     observed.box, old_box, observed.box,
                                     observed.best_score});
            continue;
        }

        match = find_unique_initial_item_match(
            planned, initial_status, identity_bound_items, observed,
            ItemStatus::OUT, current_time_ms_, true);
        if (match.item_id >= 0) {
            InventoryItem& item = planned[match.item_id];
            // 普通完整框放回时可以更新 anchor；局部框仍只更新 last_seen。
            set_visible(item, observed.box, observed.best_score, snapshot.frame_id,
                        match.match.kind == IdentityMatchKind::NORMAL);
            bind_snapshot((int)si, item.item_id, SnapshotRole::RETURNED,
                          match.match.kind);
            result.happened = true;
            result.events.push_back({EventKind::IN, item.item_id, item.cls_id,
                                     observed.box, BBox(), BBox(), observed.best_score});
            continue;
        }

        int item_id = next_id++;
        planned[item_id] = make_inventory_item(item_id, observed, snapshot.frame_id,
                                                current_time_ms_);
        if (snapshot_item_is_likely_partial(snapshot, (int)si)) {
            // 新物品只有局部框时不能凭空伪造“完整 anchor”。
            planned[item_id].anchor_box = BBox();
            planned[item_id].anchor_valid = false;
        }
        bind_snapshot((int)si, item_id, SnapshotRole::NEW_ITEM,
                      IdentityMatchKind::NONE);
        result.happened = true;
        result.events.push_back({EventKind::IN, item_id, observed.cls_id,
                                 observed.box, BBox(), BBox(), observed.best_score});
    }

    // 5. 两个遮挡计数器只在“本次稳定快照结算”时更新一次，绝不按每帧加。
    for (std::map<int, InventoryItem>::iterator it = planned.begin(); it != planned.end(); ++it) {
        InventoryItem& item = it->second;
        if (item.status == ItemStatus::OUT) continue;

        int same_item_count = 0;
        bool has_near_other_object = false;
        BBox reference = item.anchor_valid ? item.anchor_box : item.last_seen_box;
        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            if (item_matches_snapshot(item, snapshot.items[si])) same_item_count++;
            if (nearby(reference, snapshot.items[si].box) &&
                !item_matches_snapshot(item, snapshot.items[si])) {
                has_near_other_object = true;
            }
        }
        item.duplicate_count = same_item_count > 1 ? item.duplicate_count + 1 : 0;

        if (item.status == ItemStatus::OCCLUDED) {
            if (operation_pending_ && same_item_count == 0 && !has_near_other_object) {
                item.no_occluder_count++;
            } else {
                item.no_occluder_count = 0;
            }
            if (item.no_occluder_count >= OCCLUDED_TO_OUT_SNAPSHOTS) {
                BBox out_box = reference;
                set_out(item, current_time_ms_);
                result.happened = true;
                result.events.push_back({EventKind::OUT, item.item_id, item.cls_id,
                                         out_box, BBox(), BBox(), item.score});
            }
        }
    }

    std::vector<int> snapshot_item_to_inventory_id(snapshot.items.size(), -1);
    for (size_t si = 0; si < bindings.size(); ++si) {
        snapshot_item_to_inventory_id[si] = bindings[si].item_id;
    }

    bool every_initial_visible_item_resolved = true;
    for (std::map<int, ItemStatus>::const_iterator it = initial_status.begin();
         it != initial_status.end(); ++it) {
        if (it->second == ItemStatus::VISIBLE &&
            !resolved_initial_visible_items.count(it->first)) {
            every_initial_visible_item_resolved = false;
            break;
        }
    }

    // 单向握手失败时，不猜 item_id、不提交半套修改；正式库存保持原样。
    if (!every_initial_visible_item_resolved ||
        !snapshot_inventory_handshake_valid(snapshot_item_to_inventory_id, planned)) {
        printf("[SESSION] 快照-库存握手失败：跳过本次库存修改\n");
        tracks_.clear();
        candidates_.clear();
        operation_pending_ = false;
        return SettlementResult();
    }

    // planned 全部完成且握手通过后，才一次性替换正式库存。
    inventory_.replace_all(planned, next_id);
    inventory_.cleanup_expired(current_time_ms_);
    tracks_.clear();
    candidates_.clear();
    operation_pending_ = false;
    return result;
}

void SessionManager::print_inventory() const {
    const size_t visible_count = inventory_.count_by_status(ItemStatus::VISIBLE);
    const size_t occluded_count = inventory_.count_by_status(ItemStatus::OCCLUDED);
    const size_t in_count = visible_count + occluded_count;

    printf("\n");
    printf("  ┌──────────────────────────────────────────────────┐\n");
    printf("  │  在库清单 │ 可见: %-3zu │ 遮挡: %-3zu │ 共: %-3zu    │\n",
           visible_count, occluded_count, in_count);
    printf("  ├────┬──────────────┬────────┬───────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 位置 (中心)           │\n");
    printf("  ├────┼──────────────┼────────┼───────────────────────┤\n");

    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status == ItemStatus::OUT) continue;
        const char* status = item.status == ItemStatus::VISIBLE ? "可见" : "遮挡";
        printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)          │\n",
               item.item_id, cls_id_to_chinese(item.cls_id), status,
               item.last_seen_box.cx(), item.last_seen_box.cy());
    }

    printf("  └────┴──────────────┴────────┴───────────────────────┘\n\n");
}

FrameProcessResult SessionManager::process_frame(
        const std::vector<Detection>& food_detections,
        const std::vector<BBox>& hand_boxes, int frame_id, long long time_ms) {
    FrameProcessResult output;
    current_time_ms_ = time_ms;
    hand_present_ = !hand_boxes.empty();

    if (hand_present_) {
        // operation_pending 的唯一开启条件：本轮真实检测到手。
        operation_pending_ = true;
        no_hand_streak_ = 0;
        no_hand_buffer_.reset();
        update_tracks_while_hand_present(food_detections, hand_boxes, frame_id);
        return output;
    }

    ++no_hand_streak_;
    // 手已离开：未升级的 Candidate 没有整理证据，直接丢弃。
    candidates_.clear();

    // 连续无手 N 帧由 SnapshotBuffer 投票合成为一份稳定快照。
    no_hand_buffer_.push(food_detections, frame_id);
    if (!no_hand_buffer_.full()) return output;

    Snapshot snapshot = no_hand_buffer_.take_snapshot();
    output.stable_snapshot_generated = true;
    if (!snapshot.valid) return output;

    if (init_state_ == InitState::WAIT_BACKEND) {
        // 主循环正常会先调用 mark_backend_unavailable/init_from_backend；这里不猜测。
        return output;
    }
    if (init_state_ == InitState::WAIT_FIRST_STABLE_SNAPSHOT) {
        if (snapshot.items.empty() &&
            current_time_ms_ - session_start_time_ms_ < FIRST_SNAPSHOT_EMPTY_GRACE_MS) {
            printf("[SESSION] 开门初期空快照，继续等待曝光稳定\n");
            return output;
        }
        initialize_from_snapshot(snapshot);
        return output;
    }

    if (operation_pending_) {
        output.settlement = settle_snapshot(snapshot);
    } else {
        // 无手操作许可时，快照只能刷新正常匹配物品的观测框。
        refresh_visible_items_without_operation(snapshot);
    }
    return output;
}

}  // namespace fridge
