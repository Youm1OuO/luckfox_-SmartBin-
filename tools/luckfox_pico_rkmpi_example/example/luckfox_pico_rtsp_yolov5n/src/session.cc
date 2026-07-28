// ============================================================================
//  session.cc
//  单框库存 + 稳定快照结算。
//
//  快照结算只使用：NORMAL / SHRINK / GROW、遮挡覆盖和 blocker_id。
//  OperationTrack 只在没有框关系冲突时，将 OUT + IN 升级为 MOVED。
// ============================================================================
#include "session.h"
#include "fridge_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

namespace fridge {
namespace {

enum class SnapshotRole { NONE, DIRECT, MOVED, REVEALED, NEW_ITEM };

struct SnapshotBinding {
    int item_id = -1;
    SnapshotRole role = SnapshotRole::NONE;
};

enum class BoxRelation { NONE, SHRINK, GROW };

struct BoxCandidate {
    int item_id = -1;
    int snapshot_index = -1;
    BoxRelation relation = BoxRelation::NONE;
    SnapshotRole role = SnapshotRole::NONE;
    bool need_blocker_left = false;
};

float ratio_difference(float a, float b) {
    const float larger = std::max(std::fabs(a), std::fabs(b));
    return larger > 0.001f ? std::fabs(a - b) / larger : 0.0f;
}

// Track 内部逐帧匹配使用。它不参与稳定快照的身份裁决。
bool track_box_match(const BBox& a, const BBox& b,
                     float center_norm_limit,
                     float width_ratio_limit,
                     float height_ratio_limit) {
    return a.area() > 0.0f && b.area() > 0.0f &&
           normalized_nearby_distance(a, b) <= center_norm_limit &&
           ratio_difference(a.w(), b.w()) <= width_ratio_limit &&
           ratio_difference(a.h(), b.h()) <= height_ratio_limit;
}

float track_box_score(const BBox& a, const BBox& b) {
    return normalized_nearby_distance(a, b) +
           0.25f * ratio_difference(a.w(), b.w()) +
           0.25f * ratio_difference(a.h(), b.h());
}

bool normal_relation(int cls_id, const BBox& last_box,
                     const VotingItem& observed) {
    if (cls_id != observed.cls_id || last_box.area() <= 0.0f ||
        observed.box.area() <= 0.0f) {
        return false;
    }
    const float ratio = box_area_ratio(last_box, observed.box);
    return iom(last_box, observed.box) >= SNAPSHOT_NORMAL_IOM &&
           normalized_center_shift(last_box, observed.box) <= SNAPSHOT_CENTER_NORMAL &&
           ratio >= SNAPSHOT_NORMAL_AREA_MIN &&
           ratio <= SNAPSHOT_NORMAL_AREA_MAX &&
           box_shape_delta(last_box, observed.box) <= SNAPSHOT_SHAPE_NORMAL;
}

bool shrink_relation(const InventoryItem& item, const VotingItem& observed) {
    if (item.cls_id != observed.cls_id) return false;
    return iom(item.last_box, observed.box) >= SNAPSHOT_CONTAIN_IOM &&
           box_area_ratio(item.last_box, observed.box) <= SNAPSHOT_SHRINK_AREA_MAX &&
           normalized_center_shift(item.last_box, observed.box) <= SNAPSHOT_CENTER_CONTAIN &&
           box_shape_delta(item.last_box, observed.box) <= SNAPSHOT_SHAPE_CONTAIN;
}

bool grow_relation(const InventoryItem& item, const VotingItem& observed) {
    if (item.cls_id != observed.cls_id) return false;
    const float ratio = box_area_ratio(item.last_box, observed.box);
    return iom(item.last_box, observed.box) >= SNAPSHOT_CONTAIN_IOM &&
           ratio >= SNAPSHOT_GROW_AREA_MIN &&
           ratio <= SNAPSHOT_GROW_AREA_MAX &&
           normalized_center_shift(item.last_box, observed.box) <= SNAPSHOT_CENTER_CONTAIN &&
           box_shape_delta(item.last_box, observed.box) <= SNAPSHOT_SHAPE_CONTAIN;
}

bool normal_reference_match(int cls_id, const BBox& reference,
                            const VotingItem& observed) {
    return normal_relation(cls_id, reference, observed);
}

bool has_meaningful_motion(const BBox& previous, const BBox& current) {
    return normalized_nearby_distance(previous, current) > TRACK_CREATE_MOTION_NORM;
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

int nearest_hand_index(const std::vector<BBox>& hands, const BBox& box) {
    int best_index = -1;
    float best_score = 999.0f;
    for (size_t i = 0; i < hands.size(); ++i) {
        const float score = normalized_nearby_distance(hands[i], box);
        if (score < best_score) {
            best_score = score;
            best_index = static_cast<int>(i);
        }
    }
    return best_index;
}

bool any_hand_overlaps(const std::vector<BBox>& hands, const BBox& box) {
    for (size_t i = 0; i < hands.size(); ++i) {
        if (hand_overlaps_box(hands[i], box)) return true;
    }
    return false;
}

bool any_hand_near(const std::vector<BBox>& hands, const BBox& box) {
    for (size_t i = 0; i < hands.size(); ++i) {
        if (hand_near_box(hands[i], box)) return true;
    }
    return false;
}

BBox move_box(const BBox& box, float dx, float dy) {
    return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy);
}

BBox move_box_center_to(const BBox& box, float cx, float cy) {
    return move_box(box, cx - box.cx(), cy - box.cy());
}

bool similar_last_box_size(const BBox& box, const InventoryItem& item) {
    return ratio_difference(box.w(), item.last_box.w()) <= TRACK_PLACED_SIZE_RATIO &&
           ratio_difference(box.h(), item.last_box.h()) <= TRACK_PLACED_SIZE_RATIO;
}

void set_visible(InventoryItem& item, const BBox& box, float score, int frame_id,
                 bool clear_blocker = false) {
    item.last_box = box;
    item.score = score;
    item.updated_frame = frame_id;
    item.status = ItemStatus::VISIBLE;
    if (clear_blocker) item.blocker_id = -1;
    item.new_item_pending = false;
}

void set_occluded(InventoryItem& item, int blocker_id) {
    item.status = ItemStatus::OCCLUDED;
    item.blocker_id = blocker_id;
    item.new_item_pending = false;
}

InventoryItem make_inventory_item(int item_id, const VotingItem& observed,
                                  int frame_id, long long time_ms) {
    InventoryItem item;
    item.item_id = item_id;
    item.cls_id = observed.cls_id;
    item.last_box = observed.box;
    item.score = observed.best_score;
    item.status = ItemStatus::VISIBLE;
    item.blocker_id = -1;
    item.new_item_pending = true;
    item.created_frame = frame_id;
    item.updated_frame = frame_id;
    item.created_time_ms = time_ms;
    return item;
}

bool contains_index(const std::vector<int>& values, int value) {
    return std::find(values.begin(), values.end(), value) != values.end();
}

void add_unique(std::vector<int>& values, int value) {
    if (!contains_index(values, value)) values.push_back(value);
}

}  // namespace

SessionManager::SessionManager()
    : no_hand_buffer_(SNAPSHOT_N, SNAPSHOT_S),
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
    operation_pending_ = false;
    hand_present_ = false;
    no_hand_streak_ = 0;
    current_time_ms_ = time_ms;
    session_start_time_ms_ = time_ms;
    init_state_ = InitState::WAIT_BACKEND;
    backend_status_ = BackendStatus::UNKNOWN;
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
        if (item.item_id <= 0 || loaded.count(item.item_id)) item.item_id = next_id;
        if (item.last_box.area() <= 0.0f) item.status = ItemStatus::VISIBLE;
        loaded[item.item_id] = item;
        next_id = std::max(next_id, item.item_id + 1);
    }
    for (std::map<int, InventoryItem>::iterator it = loaded.begin();
         it != loaded.end(); ++it) {
        if (it->second.status == ItemStatus::OCCLUDED &&
            (it->second.blocker_id < 0 || !loaded.count(it->second.blocker_id))) {
            // 后台没有 blocker 历史时不伪造 OCCLUDED；首快照只做 NORMAL 刷新。
            it->second.status = ItemStatus::VISIBLE;
            it->second.blocker_id = -1;
        }
    }
    inventory_.replace_all(loaded, next_id);
    backend_status_ = BackendStatus::TRUSTED;
    init_state_ = InitState::WAIT_FIRST_STABLE_SNAPSHOT;
}

void SessionManager::mark_backend_unavailable() {
    if (init_state_ == InitState::READY) return;
    backend_status_ = BackendStatus::NO_TRUSTED_BACKEND;
    init_state_ = InitState::WAIT_FIRST_STABLE_SNAPSHOT;
}

void SessionManager::finish_session(long long time_ms) {
    current_time_ms_ = time_ms;
    no_hand_buffer_.reset();
    tracks_.clear();
    candidates_.clear();
    operation_pending_ = false;
    hand_present_ = false;
    no_hand_streak_ = 0;
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
    if (!any_hand_overlaps(hand_boxes, detection.box)) return -1;

    int matched_item = -1;
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status != ItemStatus::VISIBLE || item.cls_id != detection.cls_id ||
            item_is_bound_to_operation(item.item_id)) {
            continue;
        }
        if (!track_box_match(item.last_box, detection.box,
                             TRACK_FRAME_CENTER_NORM,
                             TRACK_FRAME_WIDTH_RATIO,
                             TRACK_FRAME_HEIGHT_RATIO)) {
            continue;
        }
        if (matched_item >= 0) return -1;
        matched_item = item.item_id;
    }
    return matched_item;
}

void SessionManager::update_tracks_while_hand_present(
        const std::vector<Detection>& detections,
        const std::vector<BBox>& hand_boxes, int /*frame_id*/) {
    std::vector<bool> detection_used(detections.size(), false);

    // 已确认 Track：只维护路径；它最终仍需快照唯一验证才能成为 MOVED。
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        OperationTrack& track = tracks_[ti];
        const BBox reference = track.has_last_yolo_box ? track.last_yolo_box : track.proxy_box;
        int best_detection = -1;
        float best_score = 999.0f;
        for (size_t di = 0; di < detections.size(); ++di) {
            if (detection_used[di] || detections[di].cls_id != track.cls_id ||
                !track_box_match(reference, detections[di].box,
                                 TRACK_REAPPEAR_CENTER_NORM, 1.0f, 1.0f)) {
                continue;
            }
            const float score = track_box_score(reference, detections[di].box);
            if (score < best_score) {
                best_score = score;
                best_detection = static_cast<int>(di);
            }
        }

        if (best_detection >= 0) {
            const Detection& det = detections[best_detection];
            detection_used[best_detection] = true;
            track.proxy_box = move_box(track.proxy_box,
                                       det.box.cx() - reference.cx(),
                                       det.box.cy() - reference.cy());
            track.last_yolo_box = det.box;
            track.has_last_yolo_box = true;
            const int hand_index = nearest_hand_index(hand_boxes, det.box);
            if (hand_index >= 0) {
                track.last_hand_box = hand_boxes[hand_index];
                track.has_last_hand_box = true;
            }
            const InventoryItem* tracked_item = inventory_.find_by_item(track.bound_item_id);
            if (!any_hand_overlaps(hand_boxes, det.box) &&
                (!tracked_item || similar_last_box_size(det.box, *tracked_item))) {
                track.release_box = track.proxy_box;
                track.has_release_box = true;
            }
            track.path.push_back(track.proxy_box);
            continue;
        }

        const int hand_index = nearest_hand_index(hand_boxes, reference);
        if (hand_index >= 0 && hand_near_box(hand_boxes[hand_index], reference)) {
            if (track.has_last_hand_box) {
                track.proxy_box = move_box(track.proxy_box,
                                           hand_boxes[hand_index].cx() - track.last_hand_box.cx(),
                                           hand_boxes[hand_index].cy() - track.last_hand_box.cy());
            }
            track.last_hand_box = hand_boxes[hand_index];
            track.has_last_hand_box = true;
            track.has_last_yolo_box = false;
            track.has_release_box = false;
            track.path.push_back(track.proxy_box);
        }
    }

    // Candidate 只在真实位移后升级为 Track。
    std::vector<OperationCandidate> next_candidates;
    for (size_t ci = 0; ci < candidates_.size(); ++ci) {
        OperationCandidate candidate = candidates_[ci];
        const BBox reference = candidate.has_last_yolo_box ? candidate.last_yolo_box
                                                            : candidate.source_box;
        int best_detection = -1;
        for (size_t di = 0; di < detections.size(); ++di) {
            if (detection_used[di] || detections[di].cls_id != candidate.cls_id) continue;
            if (track_box_match(reference, detections[di].box,
                                TRACK_REAPPEAR_CENTER_NORM, 1.0f, 1.0f)) {
                best_detection = static_cast<int>(di);
                break;
            }
        }

        if (best_detection >= 0) {
            const Detection& det = detections[best_detection];
            detection_used[best_detection] = true;
            if (candidate.has_last_yolo_box &&
                has_meaningful_motion(candidate.last_yolo_box, det.box) &&
                any_hand_near(hand_boxes, det.box)) {
                OperationTrack track;
                track.bound_item_id = candidate.bound_item_id;
                track.cls_id = candidate.cls_id;
                track.start_box = candidate.source_box;
                track.proxy_box = move_box_center_to(candidate.source_box,
                                                      det.box.cx(), det.box.cy());
                track.last_yolo_box = det.box;
                track.has_last_yolo_box = true;
                track.path.push_back(track.start_box);
                track.path.push_back(track.proxy_box);
                tracks_.push_back(track);
                continue;
            }
            candidate.last_yolo_box = det.box;
            candidate.has_last_yolo_box = true;
            next_candidates.push_back(candidate);
            continue;
        }

        const int hand_index = nearest_hand_index(hand_boxes, reference);
        if (hand_index < 0 || !hand_fully_covers_box(hand_boxes[hand_index], reference)) {
            continue;
        }
        if (candidate.has_last_hand_box &&
            has_meaningful_motion(candidate.last_hand_box, hand_boxes[hand_index])) {
            OperationTrack track;
            track.bound_item_id = candidate.bound_item_id;
            track.cls_id = candidate.cls_id;
            track.start_box = candidate.source_box;
            track.proxy_box = move_box(candidate.source_box,
                                       hand_boxes[hand_index].cx() - candidate.last_hand_box.cx(),
                                       hand_boxes[hand_index].cy() - candidate.last_hand_box.cy());
            track.last_hand_box = hand_boxes[hand_index];
            track.has_last_hand_box = true;
            track.path.push_back(track.start_box);
            track.path.push_back(track.proxy_box);
            tracks_.push_back(track);
            continue;
        }
        candidate.last_hand_box = hand_boxes[hand_index];
        candidate.has_last_hand_box = true;
        candidate.has_last_yolo_box = false;
        next_candidates.push_back(candidate);
    }
    candidates_.swap(next_candidates);

    // 新 Candidate：手必须实际盖住一个唯一的可见库存物品。
    for (size_t di = 0; di < detections.size(); ++di) {
        if (detection_used[di]) continue;
        const int item_id = find_unique_inventory_binding(detections[di], hand_boxes);
        if (item_id < 0) continue;
        OperationCandidate candidate;
        candidate.bound_item_id = item_id;
        candidate.cls_id = detections[di].cls_id;
        candidate.source_box = inventory_.find_by_item(item_id)->last_box;
        candidate.last_yolo_box = detections[di].box;
        candidate.has_last_yolo_box = true;
        const int hand_index = nearest_hand_index(hand_boxes, detections[di].box);
        if (hand_index >= 0) {
            candidate.last_hand_box = hand_boxes[hand_index];
            candidate.has_last_hand_box = true;
        }
        candidates_.push_back(candidate);
        detection_used[di] = true;
    }
}

void SessionManager::initialize_from_snapshot(const Snapshot& snapshot) {
    std::map<int, InventoryItem> planned = inventory_.items();
    int next_id = inventory_.next_item_id();

    if (backend_status_ == BackendStatus::NO_TRUSTED_BACKEND) {
        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            const int id = next_id++;
            planned[id] = make_inventory_item(id, snapshot.items[si],
                                              snapshot.frame_id, current_time_ms_);
        }
    } else {
        // 后台可信时，首快照只做 NORMAL 刷新；不在无操作阶段擅自出入库。
        std::set<int> used_items;
        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            int matched = -1;
            for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
                 it != planned.end(); ++it) {
                if (used_items.count(it->first) || it->second.status != ItemStatus::VISIBLE ||
                    !normal_relation(it->second.cls_id, it->second.last_box, snapshot.items[si])) {
                    continue;
                }
                if (matched >= 0) {
                    matched = -1;
                    break;
                }
                matched = it->first;
            }
            if (matched >= 0) {
                set_visible(planned[matched], snapshot.items[si].box,
                            snapshot.items[si].best_score, snapshot.frame_id);
                used_items.insert(matched);
            }
        }
    }

    inventory_.replace_all(planned, next_id);
    init_state_ = InitState::READY;
    tracks_.clear();
    candidates_.clear();
    operation_pending_ = false;
}

void SessionManager::refresh_visible_items_without_operation(const Snapshot& snapshot) {
    std::map<int, std::vector<int> > item_options;
    std::vector<std::vector<int> > snapshot_options(snapshot.items.size());
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        if (it->second.status != ItemStatus::VISIBLE) continue;
        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            if (!normal_relation(it->second.cls_id, it->second.last_box, snapshot.items[si])) {
                continue;
            }
            item_options[it->first].push_back(static_cast<int>(si));
            snapshot_options[si].push_back(it->first);
        }
    }
    for (std::map<int, std::vector<int> >::const_iterator it = item_options.begin();
         it != item_options.end(); ++it) {
        if (it->second.size() != 1) continue;
        const int si = it->second.front();
        if (snapshot_options[si].size() != 1) continue;
        inventory_.update_seen_item(it->first, snapshot.items[si].box,
                                    snapshot.items[si].best_score, snapshot.frame_id);
        InventoryItem* item = inventory_.find_by_item(it->first);
        if (item) item->new_item_pending = false;
    }
}

SettlementResult SessionManager::settle_snapshot(const Snapshot& snapshot) {
    SettlementResult empty;
    if (!operation_pending_) return empty;

    std::map<int, InventoryItem> planned = inventory_.items();
    int next_id = inventory_.next_item_id();
    std::map<int, ItemStatus> initial_status;
    std::map<int, bool> planned_out;
    std::map<int, bool> unresolved_item;
    std::map<int, bool> out_used_by_grow;
    for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
         it != planned.end(); ++it) {
        initial_status[it->first] = it->second.status;
        planned_out[it->first] = false;
        unresolved_item[it->first] = false;
        out_used_by_grow[it->first] = false;
    }

    const size_t snapshot_count = snapshot.items.size();
    std::vector<SnapshotBinding> bindings(snapshot_count);
    std::vector<bool> unresolved_snapshot(snapshot_count, false);
    std::vector<bool> reserve(snapshot_count, false);
    std::vector<bool> maybe_in(snapshot_count, false);
    std::vector<int> planned_new_id(snapshot_count, -1);
    std::vector<InventoryEvent> events;
    std::set<int> bound_items;
    std::set<int> retry_snapshots;
    std::set<int> retry_items;

    const auto is_initial = [&initial_status](int item_id, ItemStatus status) {
        std::map<int, ItemStatus>::const_iterator it = initial_status.find(item_id);
        return it != initial_status.end() && it->second == status;
    };
    const auto item_bound = [&bound_items](int item_id) {
        return bound_items.count(item_id) != 0;
    };
    const auto snapshot_bound = [&bindings](int snapshot_index) {
        return bindings[snapshot_index].item_id >= 0;
    };
    const auto mark_unresolved = [&unresolved_item, &unresolved_snapshot](int item_id,
                                                                            int snapshot_index) {
        if (item_id >= 0) unresolved_item[item_id] = true;
        if (snapshot_index >= 0) unresolved_snapshot[snapshot_index] = true;
    };
    const auto bind = [&bindings, &bound_items](int snapshot_index, int item_id,
                                                 SnapshotRole role) {
        bindings[snapshot_index].item_id = item_id;
        bindings[snapshot_index].role = role;
        bound_items.insert(item_id);
    };
    const auto plan_out = [&planned, &planned_out, &events](int item_id) {
        if (planned_out[item_id]) return;
        const InventoryItem& item = planned.find(item_id)->second;
        planned_out[item_id] = true;
        events.push_back({EventKind::OUT, item_id, item.cls_id, item.last_box,
                          BBox(), BBox(), item.score});
    };
    const auto cancel_planned_out = [&planned_out, &events](int item_id) {
        if (!planned_out[item_id]) return;
        planned_out[item_id] = false;
        events.erase(std::remove_if(events.begin(), events.end(),
                                    [item_id](const InventoryEvent& event) {
                                        return event.kind == EventKind::OUT &&
                                               event.item_id == item_id;
                                    }),
                     events.end());
    };

    // 1. NORMAL：只匹配本轮开始时可见的库存物品，且必须双向唯一。
    std::map<int, std::vector<int> > normal_options;
    std::vector<std::vector<int> > normal_by_snapshot(snapshot_count);
    for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
         it != planned.end(); ++it) {
        if (!is_initial(it->first, ItemStatus::VISIBLE)) continue;
        for (size_t si = 0; si < snapshot_count; ++si) {
            if (!normal_relation(it->second.cls_id, it->second.last_box, snapshot.items[si])) {
                continue;
            }
            normal_options[it->first].push_back(static_cast<int>(si));
            normal_by_snapshot[si].push_back(it->first);
        }
    }
    for (std::map<int, std::vector<int> >::const_iterator it = normal_options.begin();
         it != normal_options.end(); ++it) {
        for (size_t oi = 0; oi < it->second.size(); ++oi) {
            const int si = it->second[oi];
            if (it->second.size() == 1 && normal_by_snapshot[si].size() == 1) {
                InventoryItem& item = planned[it->first];
                set_visible(item, snapshot.items[si].box,
                            snapshot.items[si].best_score, snapshot.frame_id);
                bind(si, it->first, SnapshotRole::DIRECT);
            } else {
                mark_unresolved(it->first, si);
            }
        }
    }

    // 2. 只收集 Track 的整理候选。Track 后面才能确认，不能抢框关系。
    std::map<int, std::vector<int> > track_options;
    std::vector<std::vector<int> > track_by_snapshot(snapshot_count);
    for (size_t ti = 0; ti < tracks_.size(); ++ti) {
        const OperationTrack& track = tracks_[ti];
        if (!is_initial(track.bound_item_id, ItemStatus::VISIBLE) ||
            item_bound(track.bound_item_id)) {
            continue;
        }
        for (size_t si = 0; si < snapshot_count; ++si) {
            if (snapshot_bound(static_cast<int>(si)) || snapshot.items[si].cls_id != track.cls_id) {
                continue;
            }
            bool matched = false;
            if (track.has_release_box) {
                matched = normal_reference_match(track.cls_id, track.release_box, snapshot.items[si]);
            } else {
                for (size_t pi = 0; pi < track.path.size(); ++pi) {
                    if (normal_reference_match(track.cls_id, track.path[pi], snapshot.items[si])) {
                        matched = true;
                        break;
                    }
                }
            }
            if (!matched) continue;
            add_unique(track_options[track.bound_item_id], static_cast<int>(si));
            add_unique(track_by_snapshot[si], track.bound_item_id);
        }
    }
    std::map<int, int> track_candidate;
    for (std::map<int, std::vector<int> >::const_iterator it = track_options.begin();
         it != track_options.end(); ++it) {
        if (it->second.size() == 1 && track_by_snapshot[it->second.front()].size() == 1) {
            track_candidate[it->first] = it->second.front();
        }
    }

    // 3 + 4. 收集 GROW / SHRINK，合并后统一执行双向唯一判断。
    std::vector<BoxCandidate> candidates;
    std::map<int, std::vector<int> > box_by_item;
    std::vector<std::vector<int> > box_by_snapshot(snapshot_count);
    const auto add_box_candidate = [&candidates, &box_by_item, &box_by_snapshot](
            const BoxCandidate& candidate) {
        const int index = static_cast<int>(candidates.size());
        candidates.push_back(candidate);
        box_by_item[candidate.item_id].push_back(index);
        box_by_snapshot[candidate.snapshot_index].push_back(index);
    };
    for (size_t si = 0; si < snapshot_count; ++si) {
        if (snapshot_bound(static_cast<int>(si))) continue;
        for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
             it != planned.end(); ++it) {
            const InventoryItem& item = it->second;
            if (item_bound(item.item_id)) continue;
            // OCCLUDED 的 A 若先以小框重现，应按 SHRINK 处理，不能同时成为
            // NORMAL/GROW 候选而制造“同一对关系自冲突”。
            const bool is_occluded = is_initial(item.item_id, ItemStatus::OCCLUDED);
            const bool is_shrink = shrink_relation(item, snapshot.items[si]);
            if (is_occluded && item.blocker_id >= 0 && !is_shrink &&
                (normal_relation(item.cls_id, item.last_box, snapshot.items[si]) ||
                 grow_relation(item, snapshot.items[si]))) {
                BoxCandidate candidate;
                candidate.item_id = item.item_id;
                candidate.snapshot_index = static_cast<int>(si);
                candidate.relation = BoxRelation::GROW;
                candidate.role = SnapshotRole::REVEALED;
                candidate.need_blocker_left = true;
                add_box_candidate(candidate);
            }
            if (is_initial(item.item_id, ItemStatus::VISIBLE) &&
                grow_relation(item, snapshot.items[si]) &&
                (item.blocker_id >= 0 || item.new_item_pending)) {
                BoxCandidate candidate;
                candidate.item_id = item.item_id;
                candidate.snapshot_index = static_cast<int>(si);
                candidate.relation = BoxRelation::GROW;
                candidate.role = SnapshotRole::DIRECT;
                candidate.need_blocker_left = item.blocker_id >= 0;
                add_box_candidate(candidate);
            }
            if (is_shrink) {
                BoxCandidate candidate;
                candidate.item_id = item.item_id;
                candidate.snapshot_index = static_cast<int>(si);
                candidate.relation = BoxRelation::SHRINK;
                candidate.role = is_initial(item.item_id, ItemStatus::OCCLUDED)
                                     ? SnapshotRole::REVEALED : SnapshotRole::DIRECT;
                add_box_candidate(candidate);
            }
        }
    }

    std::map<int, int> grow_candidate;
    std::map<int, int> shrink_candidate;
    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const BoxCandidate& candidate = candidates[ci];
        if (box_by_item[candidate.item_id].size() == 1 &&
            box_by_snapshot[candidate.snapshot_index].size() == 1) {
            if (candidate.relation == BoxRelation::GROW) {
                grow_candidate[candidate.item_id] = static_cast<int>(ci);
            } else {
                shrink_candidate[candidate.item_id] = static_cast<int>(ci);
            }
            reserve[candidate.snapshot_index] = true;
        } else {
            mark_unresolved(candidate.item_id, candidate.snapshot_index);
        }
    }

    // 框候选不存在时，Track 才能成为整理；之后才标记疑似新入库。
    for (std::map<int, int>::const_iterator it = track_candidate.begin();
         it != track_candidate.end(); ++it) {
        const int item_id = it->first;
        const int si = it->second;
        if (item_bound(item_id) || snapshot_bound(si) || unresolved_item[item_id] ||
            unresolved_snapshot[si] || !box_by_item[item_id].empty() ||
            !box_by_snapshot[si].empty()) {
            continue;
        }
        InventoryItem& item = planned[item_id];
        const BBox before = item.last_box;
        set_visible(item, snapshot.items[si].box, snapshot.items[si].best_score,
                    snapshot.frame_id, true);
        bind(si, item_id, SnapshotRole::MOVED);
        events.push_back({EventKind::MOVED, item_id, item.cls_id, snapshot.items[si].box,
                          before, snapshot.items[si].box, snapshot.items[si].best_score});
    }
    for (size_t si = 0; si < snapshot_count; ++si) {
        if (!snapshot_bound(static_cast<int>(si)) && box_by_snapshot[si].empty() &&
            !unresolved_snapshot[si]) {
            maybe_in[si] = true;
            planned_new_id[si] = next_id++;
        }
    }

    const auto blocker_inventory_id = [&bindings, &maybe_in, &planned_new_id](int si) {
        return bindings[si].item_id >= 0 ? bindings[si].item_id
                                          : (maybe_in[si] ? planned_new_id[si] : -1);
    };
    const auto can_be_blocker = [&planned, &initial_status, &bindings, &maybe_in](
            int item_id, int si) {
        if (bindings[si].item_id >= 0 &&
            (bindings[si].role == SnapshotRole::MOVED ||
             bindings[si].role == SnapshotRole::REVEALED)) {
            return true;
        }
        if (maybe_in[si]) return true;
        const InventoryItem& item = planned.find(item_id)->second;
        if (initial_status.find(item_id)->second == ItemStatus::OCCLUDED &&
            bindings[si].role == SnapshotRole::DIRECT &&
            bindings[si].item_id == item.blocker_id) {
            return true;
        }
        return item.new_item_pending && bindings[si].role == SnapshotRole::DIRECT &&
               bindings[si].item_id >= 0;
    };

    // 5. 唯一 SHRINK 候选必须由 C 覆盖缺失区域确认。
    for (std::map<int, int>::const_iterator it = shrink_candidate.begin();
         it != shrink_candidate.end(); ++it) {
        const BoxCandidate& candidate = candidates[it->second];
        InventoryItem& item = planned[candidate.item_id];
        int blocker_snapshot = -1;
        for (size_t ci = 0; ci < snapshot_count; ++ci) {
            if (static_cast<int>(ci) == candidate.snapshot_index ||
                !can_be_blocker(candidate.item_id, static_cast<int>(ci))) {
                continue;
            }
            if (missing_region_cover_ratio(item.last_box,
                                           snapshot.items[candidate.snapshot_index].box,
                                           snapshot.items[ci].box) >= SNAPSHOT_BLOCK_COVER) {
                blocker_snapshot = static_cast<int>(ci);
                break;
            }
        }
        if (blocker_snapshot < 0) {
            mark_unresolved(candidate.item_id, candidate.snapshot_index);
            continue;
        }
        const BBox before = item.last_box;
        item.last_box = snapshot.items[candidate.snapshot_index].box;
        item.score = snapshot.items[candidate.snapshot_index].best_score;
        item.updated_frame = snapshot.frame_id;
        item.status = ItemStatus::VISIBLE;
        item.blocker_id = blocker_inventory_id(blocker_snapshot);
        item.new_item_pending = false;
        bind(candidate.snapshot_index, candidate.item_id, candidate.role);
        reserve[candidate.snapshot_index] = false;
        if (candidate.role == SnapshotRole::REVEALED) {
            events.push_back({EventKind::REVEALED, item.item_id, item.cls_id,
                              item.last_box, before, item.last_box, item.score});
        }
    }

    // 当前可见 A 没有框候选：先验证完全遮挡，否则计划出库。
    const auto settle_missing_visible = [&](int item_id) {
        if (!is_initial(item_id, ItemStatus::VISIBLE) || item_bound(item_id) ||
            planned_out[item_id] || unresolved_item[item_id] || shrink_candidate.count(item_id) ||
            grow_candidate.count(item_id)) {
            return;
        }
        InventoryItem& item = planned[item_id];
        for (size_t si = 0; si < snapshot_count; ++si) {
            if (!can_be_blocker(item_id, static_cast<int>(si))) continue;
            if (cover_ratio(item.last_box, snapshot.items[si].box) >= SNAPSHOT_FULL_COVER) {
                const int blocker_id = blocker_inventory_id(static_cast<int>(si));
                set_occluded(item, blocker_id);
                events.push_back({EventKind::OCCLUDED, item_id, item.cls_id,
                                  item.last_box, item.last_box, BBox(), item.score});
                return;
            }
        }
        plan_out(item_id);
    };
    for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
         it != planned.end(); ++it) {
        settle_missing_visible(it->first);
    }

    const auto blocker_left = [&planned, &planned_out, &bindings, &snapshot](
            int item_id, int snapshot_index) {
        const InventoryItem& item = planned.find(item_id)->second;
        if (item.blocker_id < 0) return false;
        if (planned_out[item.blocker_id]) return true;
        for (size_t ei = 0; ei < bindings.size(); ++ei) {
            if (bindings[ei].item_id == item.blocker_id &&
                bindings[ei].role == SnapshotRole::MOVED) {
                return cover_ratio(snapshot.items[snapshot_index].box,
                                   snapshot.items[ei].box) <= SNAPSHOT_LEAVE_COVER_MAX;
            }
        }
        return false;
    };

    // 6. 最后确认唯一 GROW 候选。
    for (std::map<int, int>::const_iterator it = grow_candidate.begin();
         it != grow_candidate.end(); ++it) {
        const BoxCandidate& candidate = candidates[it->second];
        if (item_bound(candidate.item_id)) {
            reserve[candidate.snapshot_index] = false;
            retry_items.insert(candidate.item_id);
            retry_snapshots.insert(candidate.snapshot_index);
            continue;
        }
        if (candidate.need_blocker_left &&
            !blocker_left(candidate.item_id, candidate.snapshot_index)) {
            reserve[candidate.snapshot_index] = false;
            retry_items.insert(candidate.item_id);
            retry_snapshots.insert(candidate.snapshot_index);
            continue;
        }
        InventoryItem& item = planned[candidate.item_id];
        const BBox before = item.last_box;
        if (candidate.need_blocker_left && planned_out[item.blocker_id]) {
            out_used_by_grow[item.blocker_id] = true;
        }
        set_visible(item, snapshot.items[candidate.snapshot_index].box,
                    snapshot.items[candidate.snapshot_index].best_score, snapshot.frame_id,
                    true);
        cancel_planned_out(candidate.item_id);
        bind(candidate.snapshot_index, candidate.item_id, candidate.role);
        reserve[candidate.snapshot_index] = false;
        if (candidate.role == SnapshotRole::REVEALED) {
            events.push_back({EventKind::REVEALED, item.item_id, item.cls_id, item.last_box,
                              before, item.last_box, item.score});
        }
    }

    // 7. 被 GROW 延后的 Track，只有没有任何冲突时才能确认 MOVED。
    for (std::map<int, int>::const_iterator it = track_candidate.begin();
         it != track_candidate.end(); ++it) {
        const int item_id = it->first;
        const int si = it->second;
        if (item_bound(item_id) || snapshot_bound(si) || unresolved_item[item_id] ||
            unresolved_snapshot[si] || reserve[si] || out_used_by_grow[item_id]) {
            continue;
        }
        InventoryItem& item = planned[item_id];
        const BBox before = item.last_box;
        set_visible(item, snapshot.items[si].box, snapshot.items[si].best_score,
                    snapshot.frame_id, true);
        cancel_planned_out(item_id);
        bind(si, item_id, SnapshotRole::MOVED);
        events.push_back({EventKind::MOVED, item_id, item.cls_id, item.last_box,
                          before, item.last_box, item.score});
    }

    // 8. GROW 失败后统一回到普通规则：B 可成为新物品，A 再判断遮挡/出库。
    for (std::set<int>::const_iterator it = retry_snapshots.begin();
         it != retry_snapshots.end(); ++it) {
        const int si = *it;
        if (snapshot_bound(si) || unresolved_snapshot[si]) continue;
        if (!maybe_in[si]) {
            maybe_in[si] = true;
            planned_new_id[si] = next_id++;
        }
    }
    for (std::set<int>::const_iterator it = retry_items.begin();
         it != retry_items.end(); ++it) {
        settle_missing_visible(*it);
    }

    // 9. 最后把真正多出来的 B 新建为活跃库存；不回查历史出库物品。
    for (size_t si = 0; si < snapshot_count; ++si) {
        if (snapshot_bound(static_cast<int>(si))) continue;
        if (!maybe_in[si] || unresolved_snapshot[si]) {
            unresolved_snapshot[si] = true;
            continue;
        }
        const int item_id = planned_new_id[si];
        planned[item_id] = make_inventory_item(item_id, snapshot.items[si],
                                                snapshot.frame_id, current_time_ms_);
        bind(static_cast<int>(si), item_id, SnapshotRole::NEW_ITEM);
        events.push_back({EventKind::IN, item_id, snapshot.items[si].cls_id,
                          snapshot.items[si].box, BBox(), BBox(),
                          snapshot.items[si].best_score});
    }

    // 不唯一、未绑定、或仍指向本轮出库 blocker 时，整轮不提交。
    for (std::map<int, bool>::const_iterator it = unresolved_item.begin();
         it != unresolved_item.end(); ++it) {
        if (it->second) return SettlementResult();
    }
    for (size_t si = 0; si < snapshot_count; ++si) {
        if (unresolved_snapshot[si] || !snapshot_bound(static_cast<int>(si))) {
            return SettlementResult();
        }
    }
    std::set<int> unique_bindings;
    for (size_t si = 0; si < snapshot_count; ++si) {
        if (!unique_bindings.insert(bindings[si].item_id).second) return SettlementResult();
    }
    for (std::map<int, InventoryItem>::const_iterator it = planned.begin();
         it != planned.end(); ++it) {
        if (!planned_out[it->first] && it->second.blocker_id >= 0 &&
            planned_out[it->second.blocker_id]) {
            return SettlementResult();
        }
    }

    for (std::map<int, bool>::const_iterator it = planned_out.begin();
         it != planned_out.end(); ++it) {
        if (it->second) planned.erase(it->first);
    }
    inventory_.replace_all(planned, next_id);
    tracks_.clear();
    candidates_.clear();
    operation_pending_ = false;

    SettlementResult result;
    result.happened = !events.empty();
    result.events.swap(events);
    return result;
}

void SessionManager::print_inventory() const {
    const size_t visible_count = inventory_.count_by_status(ItemStatus::VISIBLE);
    const size_t occluded_count = inventory_.count_by_status(ItemStatus::OCCLUDED);
    printf("[INVENTORY] 可见=%zu 遮挡=%zu 总计=%zu\n",
           visible_count, occluded_count, visible_count + occluded_count);
    inventory_.print("  ");
}

FrameProcessResult SessionManager::process_frame(
        const std::vector<Detection>& food_detections,
        const std::vector<BBox>& hand_boxes, int frame_id, long long time_ms) {
    FrameProcessResult output;
    current_time_ms_ = time_ms;
    hand_present_ = !hand_boxes.empty();

    if (hand_present_) {
        operation_pending_ = true;
        no_hand_streak_ = 0;
        no_hand_buffer_.reset();
        update_tracks_while_hand_present(food_detections, hand_boxes, frame_id);
        return output;
    }

    ++no_hand_streak_;
    candidates_.clear();
    no_hand_buffer_.push(food_detections, frame_id);
    if (!no_hand_buffer_.full()) return output;

    Snapshot snapshot = no_hand_buffer_.take_snapshot();
    output.stable_snapshot_generated = true;
    if (!snapshot.valid) return output;
    if (init_state_ == InitState::WAIT_BACKEND) return output;
    if (init_state_ == InitState::WAIT_FIRST_STABLE_SNAPSHOT) {
        if (snapshot.items.empty() &&
            current_time_ms_ - session_start_time_ms_ < FIRST_SNAPSHOT_EMPTY_GRACE_MS) {
            return output;
        }
        initialize_from_snapshot(snapshot);
        return output;
    }

    if (operation_pending_) {
        output.settlement = settle_snapshot(snapshot);
    } else {
        refresh_visible_items_without_operation(snapshot);
    }
    return output;
}

}  // namespace fridge
