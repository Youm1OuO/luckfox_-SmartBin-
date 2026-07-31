// ============================================================================
//  session.cc
//  2.0：持久库存 + 手后 Track + 稳定快照工作副本结算
// ============================================================================
#include "session.h"
#include "fridge_config.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
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

bool strict_match(const InventoryItem& item, const VotingItem& observed) {
    return item.cls_id == observed.cls_id && item.box.area() > 0.0f &&
           observed.box.area() > 0.0f &&
           normalized_nearby_distance(item.box, observed.box)
               <= INVENTORY_STRICT_CENTER_NORM &&
           ratio_difference(item.box.w(), observed.box.w())
               <= INVENTORY_STRICT_WIDTH_RATIO &&
           ratio_difference(item.box.h(), observed.box.h())
               <= INVENTORY_STRICT_HEIGHT_RATIO;
}

bool partial_match(const InventoryItem& item, const VotingItem& observed) {
    return item.cls_id == observed.cls_id && item.box.area() > 0.0f &&
           observed.box.area() > 0.0f &&
           iom(item.box, observed.box) >= INVENTORY_PARTIAL_IOM;
}

bool base_box_contains(const BBox& outer, const BBox& inner) {
    return outer.x1 <= inner.x1 + BASE_BOX_CONTAIN_EPS &&
           outer.y1 <= inner.y1 + BASE_BOX_CONTAIN_EPS &&
           outer.x2 >= inner.x2 - BASE_BOX_CONTAIN_EPS &&
           outer.y2 >= inner.y2 - BASE_BOX_CONTAIN_EPS;
}

bool visible_item_may_grow(const InventoryItem& item, const VotingItem& observed) {
    return item.block_ids.empty() &&
           item.box.area() * (1.0f + VISIBLE_AREA_GROWTH_RATIO_EPS) < observed.box.area();
}

bool track_box_match(const BBox& a, const BBox& b,
                     float center_norm_limit = TRACK_ASSOCIATE_CENTER_NORM,
                     float width_ratio_limit = TRACK_ASSOCIATE_WIDTH_RATIO,
                     float height_ratio_limit = TRACK_ASSOCIATE_HEIGHT_RATIO) {
    return a.area() > 0.0f && b.area() > 0.0f &&
           normalized_nearby_distance(a, b) <= center_norm_limit &&
           ratio_difference(a.w(), b.w()) <= width_ratio_limit &&
           ratio_difference(a.h(), b.h()) <= height_ratio_limit;
}

// 用和动态遮挡结算相同的面积误差，判断 cover 是否已经实际盖住 target。
// 这里不能只用“有重叠”或较低的覆盖比例，否则会把仍有可见部分的后方物品
// 错当成前景物品后面的遮挡物。
bool effectively_covers_entire_box(const BBox& target, const BBox& cover) {
    return target.area() > 0.0f &&
           target.area() - intersection_area(target, cover) <= COVER_REMAINING_AREA_EPS;
}

// 检测框不可能每帧都像真实物体轮廓一样严丝合缝。对已经被 Track 确认移动到
// 前方的物品，允许一定 bbox 边缘误差；但这不是普通静态缺失的兜底，调用方必须
// 额外确认 blocker 是本轮 MOVED 的前景物品。
bool sufficiently_covers_box_for_tracked_occlusion(const BBox& target, const BBox& cover) {
    return effectively_covers_entire_box(target, cover) ||
           cover_ratio(target, cover) >= TRACKED_BLOCKER_OCCLUSION_COVER_RATIO;
}

// 若某条 Track 对应的库存物品当前正挡住一件 OCCLUDED 物品，则当手把它移开时，
// 原位置出现的同类 Detection 更可能是后方物品“露出”，而不是这条 Track 突然
// 回到了起点。该关系只来自上一轮已提交的 block_ids，因此可作为关联时的辅助证据。
bool track_is_front_of_occluded_inventory_item(
        const OperationTrack& track,
        const std::map<int, InventoryItem*>& items) {
    for (std::map<int, InventoryItem*>::const_iterator it = items.begin();
         it != items.end(); ++it) {
        const InventoryItem* item = it->second;
        if (!item || item->status != ItemStatus::OCCLUDED) continue;
        if (item->block_ids.count(track.bound_item_id)) return true;
    }
    return false;
}

bool hand_overlaps_box(const BBox& hand, const BBox& box) {
    return overlap_ratio_of_smaller(hand, box) >= TRACK_HAND_OVERLAP;
}

bool hand_near_box(const BBox& hand, const BBox& box) {
    return hand_overlaps_box(hand, box) ||
           normalized_nearby_distance(hand, box) <= TRACK_HAND_NEAR_NORM;
}

bool hand_fully_covers_box(const BBox& hand, const BBox& box) {
    return cover_ratio(box, hand) >= TRACK_FULL_OCCLUSION_OVERLAP;
}

bool hand_can_carry_box(const BBox& hand, const BBox& box) {
    return cover_ratio(box, hand) >= TRACK_CARRY_OVERLAP || hand_near_box(hand, box);
}

BBox move_box(const BBox& box, float dx, float dy) {
    return BBox(box.x1 + dx, box.y1 + dy, box.x2 + dx, box.y2 + dy);
}

bool vectors_move_together(float hand_dx, float hand_dy,
                           float item_dx, float item_dy) {
    const float hand_len = std::sqrt(hand_dx * hand_dx + hand_dy * hand_dy);
    const float item_len = std::sqrt(item_dx * item_dx + item_dy * item_dy);
    if (hand_len <= TRACK_HAND_MOVE_EPS || item_len <= TRACK_OBJECT_MOVE_EPS) return false;
    const float cosine = (hand_dx * item_dx + hand_dy * item_dy) / (hand_len * item_len);
    const float ratio = item_len / hand_len;
    return cosine >= TRACK_MOVE_DIRECTION_COS_MIN &&
           ratio >= TRACK_MOVE_MAGNITUDE_RATIO_MIN &&
           ratio <= TRACK_MOVE_MAGNITUDE_RATIO_MAX;
}

bool detection_is_at_start(const Detection& detection, const OperationTrack& track) {
    return detection.cls_id == track.cls_id &&
           normalized_nearby_distance(detection.box, track.start_box)
               <= TRACK_START_REAPPEAR_CENTER_NORM;
}

bool any_detection_at_start(const std::vector<Detection>& detections,
                            const OperationTrack& track) {
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detection_is_at_start(detections[i], track)) return true;
    }
    return false;
}

int unique_detection_for_item(const std::vector<Detection>& detections,
                              const InventoryItem& item) {
    int result = -1;
    for (size_t i = 0; i < detections.size(); ++i) {
        if (detections[i].cls_id != item.cls_id) continue;
        const bool matches_current = track_box_match(item.box, detections[i].box,
                                                      TRACK_START_REAPPEAR_CENTER_NORM,
                                                      1.0f, 1.0f);
        const bool matches_base = track_box_match(item.base_box, detections[i].box,
                                                   TRACK_START_REAPPEAR_CENTER_NORM,
                                                   1.0f, 1.0f);
        if (!matches_current && !matches_base) continue;
        if (result >= 0) return -1;
        result = static_cast<int>(i);
    }
    return result;
}

InventoryItem make_inventory_item(int item_id, const VotingItem& observed,
                                  int frame_id, long long time_ms) {
    InventoryItem item;
    item.item_id = item_id;
    item.cls_id = observed.cls_id;
    item.box = observed.box;
    item.base_box = observed.box;
    item.score = observed.best_score;
    item.status = ItemStatus::VISIBLE;
    item.block_ids.clear();
    item.created_frame = frame_id;
    item.updated_frame = frame_id;
    item.created_time_ms = time_ms;
    return item;
}

void set_seen_box(InventoryItem& item, const VotingItem& observed, int frame_id) {
    item.box = observed.box;
    item.score = observed.best_score;
    item.updated_frame = frame_id;
}

InventoryEvent make_event(EventKind kind, const InventoryItem& item,
                          const BBox& before = BBox(), const BBox& after = BBox()) {
    InventoryEvent event;
    event.kind = kind;
    event.item_id = item.item_id;
    event.cls_id = item.cls_id;
    event.box = (kind == EventKind::MOVED) ? after : item.box;
    event.before_box = before;
    event.after_box = after;
    event.score = item.score;
    return event;
}

// 返回 piece \ cover 后至多四个互不重叠的矩形。仅保留正面积碎片。
void subtract_cover_from_piece(const BBox& piece, const BBox& cover,
                               std::vector<BBox>& output) {
    const float ix1 = std::max(piece.x1, cover.x1);
    const float iy1 = std::max(piece.y1, cover.y1);
    const float ix2 = std::min(piece.x2, cover.x2);
    const float iy2 = std::min(piece.y2, cover.y2);
    if (ix2 <= ix1 || iy2 <= iy1) {
        output.push_back(piece);
        return;
    }

    const BBox candidates[] = {
        BBox(piece.x1, piece.y1, piece.x2, iy1),
        BBox(piece.x1, iy2, piece.x2, piece.y2),
        BBox(piece.x1, iy1, ix1, iy2),
        BBox(ix2, iy1, piece.x2, iy2),
    };
    for (size_t i = 0; i < sizeof(candidates) / sizeof(candidates[0]); ++i) {
        if (candidates[i].area() > COVER_REMAINING_AREA_EPS) {
            output.push_back(candidates[i]);
        }
    }
}

void apply_cover(std::vector<BBox>& remaining, const BBox& new_cover) {
    std::vector<BBox> next;
    for (size_t i = 0; i < remaining.size(); ++i) {
        subtract_cover_from_piece(remaining[i], new_cover, next);
    }
    remaining.swap(next);
}

float region_area(const std::vector<BBox>& remaining) {
    float total = 0.0f;
    for (size_t i = 0; i < remaining.size(); ++i) total += remaining[i].area();
    return total;
}

void remove_from_working_inventory(std::map<int, InventoryItem>& working,
                                   int removed_item_id) {
    for (std::map<int, InventoryItem>::iterator it = working.begin();
         it != working.end(); ++it) {
        it->second.block_ids.erase(removed_item_id);
    }
    working.erase(removed_item_id);
}

// F 已被确认位于其他物品前方。按照 F 的 base_box 刷新所有旧物品的 block_ids。
void update_block_ids_as_front(std::map<int, InventoryItem>& working, int front_item_id) {
    std::map<int, InventoryItem>::iterator front_it = working.find(front_item_id);
    if (front_it == working.end()) return;
    const BBox front_box = front_it->second.base_box;
    for (std::map<int, InventoryItem>::iterator it = working.begin();
         it != working.end(); ++it) {
        if (it->first == front_item_id) continue;
        if (intersection_area(front_box, it->second.base_box) > BLOCK_OVERLAP_AREA_EPS) {
            it->second.block_ids.insert(front_item_id);
        } else {
            it->second.block_ids.erase(front_item_id);
        }
    }
}

bool covered_by_blockers_and_snapshot(const InventoryItem& item,
                                      const std::map<int, InventoryItem>& working,
                                      const std::vector<VotingItem>& snapshot_items,
                                      const std::vector<int>& temporary_block_indices) {
    std::vector<BBox> remaining;
    remaining.push_back(item.base_box);
    for (std::set<int>::const_iterator it = item.block_ids.begin();
         it != item.block_ids.end(); ++it) {
        std::map<int, InventoryItem>::const_iterator blocker = working.find(*it);
        if (blocker != working.end()) apply_cover(remaining, blocker->second.base_box);
    }
    for (size_t i = 0; i < temporary_block_indices.size(); ++i) {
        apply_cover(remaining, snapshot_items[temporary_block_indices[i]].box);
    }
    return region_area(remaining) <= COVER_REMAINING_AREA_EPS;
}

// 仅用于“本轮已由 Track 确认 MOVED 的前景物品”。当稳定快照中旧 A 完全消失、
// F 又已经被确认移到 A 前方时，允许 bbox 的边缘存在有限误差。没有 Track 的
// 普通静态缺失不会走这里，仍必须通过严格的矩形差集覆盖，否则不猜测 OUT/遮挡。
bool covered_by_recently_moved_blocker(const InventoryItem& item,
                                       const std::map<int, InventoryItem>& working,
                                       const std::set<int>& moved_front_item_ids) {
    for (std::set<int>::const_iterator it = item.block_ids.begin();
         it != item.block_ids.end(); ++it) {
        if (!moved_front_item_ids.count(*it)) continue;
        std::map<int, InventoryItem>::const_iterator blocker = working.find(*it);
        if (blocker == working.end()) continue;
        if (sufficiently_covers_box_for_tracked_occlusion(item.base_box,
                                                           blocker->second.base_box)) {
            return true;
        }
    }
    return false;
}

bool still_has_visible_area(const InventoryItem& item,
                            const std::map<int, InventoryItem>& working) {
    std::vector<BBox> remaining;
    remaining.push_back(item.base_box);
    for (std::set<int>::const_iterator it = item.block_ids.begin();
         it != item.block_ids.end(); ++it) {
        std::map<int, InventoryItem>::const_iterator blocker = working.find(*it);
        if (blocker != working.end()) apply_cover(remaining, blocker->second.base_box);
    }
    return region_area(remaining) > COVER_REMAINING_AREA_EPS;
}

int add_snapshot_to_working_inventory(std::map<int, InventoryItem>& working,
                                      int& working_next_item_id, VotingItem& observed,
                                      int frame_id, long long time_ms,
                                      std::vector<InventoryEvent>& events) {
    const int item_id = working_next_item_id++;
    observed.item_id = item_id;
    working[item_id] = make_inventory_item(item_id, observed, frame_id, time_ms);
    update_block_ids_as_front(working, item_id);
    events.push_back(make_event(EventKind::IN, working[item_id]));
    return item_id;
}

struct TrackSettlementResult {
    std::vector<std::pair<int, int> > move_pairs;  // item_id, snapshot index
    std::set<int> out_item_ids;
    std::set<int> ambiguous_item_ids;
    // 仅用于失败诊断：每条参与结算的 Track 在最终快照中有多少终点候选。
    std::map<int, int> endpoint_candidate_counts;
};

TrackSettlementResult get_track_settlement_result(
        const std::map<int, InventoryItem>& working,
        const std::map<int, bool>& affirm,
        const std::vector<VotingItem>& snapshot_items,
        const std::vector<OperationTrack>& frozen_tracks,
        bool track_session_is_ambiguous) {
    TrackSettlementResult result;
    if (track_session_is_ambiguous) return result;

    std::map<int, std::vector<int> > track_to_snapshot;
    std::vector<std::vector<int> > snapshot_to_track(snapshot_items.size());
    std::vector<int> moved_item_ids;

    for (size_t ti = 0; ti < frozen_tracks.size(); ++ti) {
        const OperationTrack& track = frozen_tracks[ti];
        if (!track.frozen || track.state == OperationTrackState::INVALID ||
            !track.hold_and_move) {
            continue;
        }
        std::map<int, InventoryItem>::const_iterator item_it = working.find(track.bound_item_id);
        if (item_it == working.end() || item_it->second.status != ItemStatus::VISIBLE) continue;
        std::map<int, bool>::const_iterator affirm_it = affirm.find(track.bound_item_id);
        if (affirm_it != affirm.end() && affirm_it->second) continue;

        moved_item_ids.push_back(track.bound_item_id);
        const BBox expected_end = track.has_release_box ? track.release_box : track.proxy_box;
        for (size_t si = 0; si < snapshot_items.size(); ++si) {
            const VotingItem& observed = snapshot_items[si];
            if (observed.item_id != -1 || observed.cls_id != track.cls_id) continue;
            if (std::fabs(observed.box.w() - expected_end.w()) > TRACK_SETTLEMENT_WIDTH_EPS ||
                std::fabs(observed.box.h() - expected_end.h()) > TRACK_SETTLEMENT_HEIGHT_EPS ||
                center_distance(observed.box, expected_end) > TRACK_SETTLEMENT_CENTER_EPS) {
                continue;
            }
            track_to_snapshot[track.bound_item_id].push_back(static_cast<int>(si));
            snapshot_to_track[si].push_back(track.bound_item_id);
        }
    }

    for (size_t i = 0; i < moved_item_ids.size(); ++i) {
        const int item_id = moved_item_ids[i];
        const std::vector<int>& candidates = track_to_snapshot[item_id];
        result.endpoint_candidate_counts[item_id] = static_cast<int>(candidates.size());
        if (candidates.empty()) {
            result.out_item_ids.insert(item_id);
            continue;
        }
        if (candidates.size() != 1 || snapshot_to_track[candidates.front()].size() != 1) {
            result.ambiguous_item_ids.insert(item_id);
            continue;
        }
        result.move_pairs.push_back(std::make_pair(item_id, candidates.front()));
    }

    if (result.move_pairs.size() + result.out_item_ids.size() > 1) {
        for (size_t i = 0; i < result.move_pairs.size(); ++i) {
            result.ambiguous_item_ids.insert(result.move_pairs[i].first);
        }
        result.ambiguous_item_ids.insert(result.out_item_ids.begin(), result.out_item_ids.end());
        result.move_pairs.clear();
        result.out_item_ids.clear();
    }
    return result;
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

void SessionManager::start_new_session(long long time_ms) {
    no_hand_buffer_.reset();
    track_buffer_.clear();
    frozen_tracks_.clear();
    previous_food_detections_.clear();
    previous_had_hand_ = false;
    track_session_is_ambiguous_ = false;
    operation_pending_ = false;
    hand_present_ = false;
    no_hand_streak_ = 0;
    current_time_ms_ = time_ms;
    session_active_ = true;

    if (has_local_inventory_) {
        initial_check_state_ = InitialCheckState::NOT_NEEDED;
        // 有本地库存时绝不请求后台；保留本地状态就是本次会话的起点。
    } else {
        initial_check_state_ = InitialCheckState::NONE;
        backend_status_ = BackendStatus::UNKNOWN;
    }
}

void SessionManager::init_from_backend(const std::vector<InventoryItem>& items,
                                       bool authoritative_empty) {
    if (!session_active_ || has_local_inventory_ ||
        backend_status_ != BackendStatus::UNKNOWN) {
        return;
    }
    if (items.empty() && !authoritative_empty) {
        mark_backend_unavailable();
        return;
    }

    std::map<int, InventoryItem> loaded;
    int next_id = inventory_.next_item_id();
    for (size_t i = 0; i < items.size(); ++i) {
        InventoryItem item = items[i];
        if (item.item_id <= 0 || loaded.count(item.item_id)) {
            while (loaded.count(next_id)) ++next_id;
            item.item_id = next_id++;
        }
        if (item.base_box.area() <= 0.0f) item.base_box = item.box;
        if (item.box.area() <= 0.0f) item.box = item.base_box;
        if (item.box.area() <= 0.0f || item.base_box.area() <= 0.0f) {
            printf("[BACKEND] 忽略 item#%d：缺少有效 bbox\n", item.item_id);
            continue;
        }
        loaded[item.item_id] = item;
        next_id = std::max(next_id, item.item_id + 1);
    }

    // 若后台能够恢复遮挡关系则保留；只移除指向不存在物品或自身的失效 ID。
    for (std::map<int, InventoryItem>::iterator it = loaded.begin(); it != loaded.end(); ++it) {
        for (std::set<int>::iterator blocker = it->second.block_ids.begin();
             blocker != it->second.block_ids.end();) {
            if (*blocker == it->first || !loaded.count(*blocker)) {
                blocker = it->second.block_ids.erase(blocker);
            } else {
                ++blocker;
            }
        }
    }

    inventory_.replace_all(loaded, next_id);
    rebuild_persistent_item_index_();
    has_local_inventory_ = true;
    backend_status_ = BackendStatus::TRUSTED;
    initial_check_state_ = InitialCheckState::WAITING;
    printf("[BACKEND] 已载入 %zu 个初始库存；等待首次只读快照校验\n", loaded.size());
}

void SessionManager::mark_backend_unavailable() {
    if (!session_active_ || has_local_inventory_) return;
    backend_status_ = BackendStatus::NO_TRUSTED_BACKEND;
    if (ALLOW_SNAPSHOT_BOOTSTRAP_WHEN_BACKEND_UNAVAILABLE) {
        // 当前没有后台接入时的明确测试兜底：先等一份无手稳定快照建本地库存。
        // 这条路径与“可信后台 + 首快照只读校验”刻意分开。
        initial_check_state_ = InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT;
        printf("[BACKEND] 初始库存不可用：等待首次无手稳定快照建立本地测试库存\n");
    } else {
        initial_check_state_ = InitialCheckState::NONE;
        printf("[BACKEND] 初始库存不可用：本次不开启快照建库或库存结算\n");
    }
}

void SessionManager::finish_session(long long time_ms) {
    current_time_ms_ = time_ms;
    if (operation_pending_) {
        printf("[SESSION] 关门时仍有未结算手操作；保留最后一次已提交库存，不在下次开门补算\n");
    }
    no_hand_buffer_.reset();
    track_buffer_.clear();
    frozen_tracks_.clear();
    previous_food_detections_.clear();
    previous_had_hand_ = false;
    track_session_is_ambiguous_ = false;
    operation_pending_ = false;
    hand_present_ = false;
    no_hand_streak_ = 0;
    initial_check_state_ = InitialCheckState::NONE;
    session_active_ = false;
}

void SessionManager::begin_closing_guard() {
    no_hand_buffer_.reset();
    // 暂停会打断相邻帧语义，不能让恢复后的手使用旧 YOLO 结果建 Track。
    previous_food_detections_.clear();
    previous_had_hand_ = false;
}

void SessionManager::resume_after_false_closing() {
    no_hand_buffer_.reset();
    if (!track_buffer_.empty() || !frozen_tracks_.empty()) {
        discard_tracks_and_mark_ambiguous_();
    }
}

bool SessionManager::should_collect_snapshot_() const {
    if (!session_active_) return false;
    if (initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT) return true;
    return has_local_inventory_ &&
           (operation_pending_ || initial_check_state_ == InitialCheckState::WAITING);
}

void SessionManager::save_previous_yolo_result_(
        const std::vector<Detection>& food_detections, bool current_has_hand) {
    previous_food_detections_ = food_detections;
    previous_had_hand_ = current_has_hand;
}

void SessionManager::validate_initial_snapshot_(const Snapshot& snapshot) const {
    std::map<int, std::vector<int> > inventory_to_snapshot;
    std::vector<std::vector<int> > snapshot_to_inventory(snapshot.items.size());
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        if (it->second.status != ItemStatus::VISIBLE) continue;
        for (size_t si = 0; si < snapshot.items.size(); ++si) {
            if (!strict_match(it->second, snapshot.items[si]) &&
                !partial_match(it->second, snapshot.items[si])) {
                continue;
            }
            inventory_to_snapshot[it->first].push_back(static_cast<int>(si));
            snapshot_to_inventory[si].push_back(it->first);
        }
    }

    int consistent_count = 0;
    int inconsistent_count = 0;
    for (size_t si = 0; si < snapshot.items.size(); ++si) {
        if (snapshot_to_inventory[si].size() == 1 &&
            inventory_to_snapshot[snapshot_to_inventory[si].front()].size() == 1) {
            ++consistent_count;
        } else {
            ++inconsistent_count;
            printf("[INIT-CHECK] 快照临时物品 %d（cls=%d）无法唯一对应后台可见库存\n",
                   snapshot.items[si].temporary_id, snapshot.items[si].cls_id);
        }
    }
    printf("[INIT-CHECK] 只读校验完成：可唯一对应=%d，不一致=%d；库存未修改\n",
           consistent_count, inconsistent_count);
}

void SessionManager::initialize_from_bootstrap_snapshot_(const Snapshot& snapshot) {
    std::map<int, InventoryItem> loaded;
    int next_item_id = inventory_.next_item_id();
    for (size_t i = 0; i < snapshot.items.size(); ++i) {
        const int item_id = next_item_id++;
        loaded[item_id] = make_inventory_item(item_id, snapshot.items[i],
                                              snapshot.frame_id, current_time_ms_);
    }
    inventory_.replace_all(loaded, next_item_id);
    rebuild_persistent_item_index_();
    has_local_inventory_ = true;
    initial_check_state_ = InitialCheckState::DONE;
    printf("\033[1;32m[EVENT]\033[0m 本地测试快照建库完成：%zu 个物品\n", loaded.size());
    print_inventory();
}

void SessionManager::finalize_initial_check_before_hand_() {
    if (initial_check_state_ != InitialCheckState::WAITING &&
        initial_check_state_ != InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT) {
        return;
    }
    const bool bootstrap = initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT;
    if (no_hand_buffer_.empty()) {
        if (bootstrap) {
            Snapshot empty_snapshot;
            empty_snapshot.valid = true;
            initialize_from_bootstrap_snapshot_(empty_snapshot);
            initial_check_state_ = InitialCheckState::SKIPPED;
            printf("[BOOTSTRAP] 第一帧即有手，先以空本地测试库存开始本次操作\n");
        } else {
            initial_check_state_ = InitialCheckState::SKIPPED;
            printf("[INIT-CHECK] 第一帧即有手，跳过首次快照校验，采用后台库存\n");
        }
        return;
    }
    Snapshot short_snapshot = no_hand_buffer_.take_snapshot();
    if (bootstrap) {
        initialize_from_bootstrap_snapshot_(short_snapshot);
    } else {
        if (short_snapshot.valid) validate_initial_snapshot_(short_snapshot);
        initial_check_state_ = InitialCheckState::DONE;
    }
}

void SessionManager::update_track_by_visible_item_(OperationTrack& track,
                                                   const Detection& detection,
                                                   const BBox& hand_box) {
    if (track.frozen) return;
    const bool hand_contacts_item = hand_near_box(hand_box, detection.box);

    if (track.state == OperationTrackState::PLACED && !hand_contacts_item) {
        track.last_item_box = detection.box;
        track.has_last_item_box = true;
        track.last_hand_box = hand_box;
        track.has_last_hand_box = true;
        track.hand_path.push_back(hand_box);
        return;
    }
    if (track.state == OperationTrackState::PLACED && hand_contacts_item) {
        track.state = OperationTrackState::VISIBLE;
        track.has_release_box = false;
    }

    float hand_dx = 0.0f;
    float hand_dy = 0.0f;
    if (track.has_last_hand_box) {
        hand_dx = hand_box.cx() - track.last_hand_box.cx();
        hand_dy = hand_box.cy() - track.last_hand_box.cy();
    }
    if (hand_contacts_item) track.seen_hand_contact = true;

    if (track.has_last_item_box) {
        const float item_dx = detection.box.cx() - track.last_item_box.cx();
        const float item_dy = detection.box.cy() - track.last_item_box.cy();
        if (hand_contacts_item && vectors_move_together(hand_dx, hand_dy, item_dx, item_dy)) {
            track.proxy_box = move_box(track.proxy_box, item_dx, item_dy);
            track.proxy_path.push_back(track.proxy_box);
            track.seen_effective_move = true;
            track.still_at_start_count = 0;
        } else if (std::sqrt(hand_dx * hand_dx + hand_dy * hand_dy) > TRACK_HAND_MOVE_EPS &&
                   normalized_nearby_distance(detection.box, track.start_box)
                       <= TRACK_START_REAPPEAR_CENTER_NORM) {
            ++track.still_at_start_count;
            if (track.still_at_start_count >= TRACK_STILL_AT_START_FRAME_LIMIT) {
                track.state = OperationTrackState::INVALID;
            }
        } else {
            track.still_at_start_count = 0;
        }
    }

    track.last_item_box = detection.box;
    track.has_last_item_box = true;
    track.last_hand_box = hand_box;
    track.has_last_hand_box = true;
    track.hand_path.push_back(hand_box);
    if (track.state != OperationTrackState::INVALID) {
        track.state = OperationTrackState::VISIBLE;
    }
    if (track.seen_hand_contact && track.seen_effective_move) {
        track.hold_and_move = true;
    }
    if (track.hold_and_move && !hand_overlaps_box(hand_box, detection.box) &&
        track.state != OperationTrackState::INVALID) {
        track.state = OperationTrackState::PLACED;
        track.release_box = track.proxy_box;
        track.has_release_box = true;
    }
}

void SessionManager::update_track_by_hand_or_mark_invalid_(
        OperationTrack& track, const std::vector<Detection>& detections,
        const BBox& hand_box) {
    if (track.frozen) return;
    float hand_dx = 0.0f;
    float hand_dy = 0.0f;
    if (track.has_last_hand_box) {
        hand_dx = hand_box.cx() - track.last_hand_box.cx();
        hand_dy = hand_box.cy() - track.last_hand_box.cy();
    }
    const float hand_move = std::sqrt(hand_dx * hand_dx + hand_dy * hand_dy);

    if (hand_move > TRACK_HAND_MOVE_EPS && any_detection_at_start(detections, track)) {
        track.state = OperationTrackState::INVALID;
        return;
    }

    const BBox candidate_proxy = move_box(track.proxy_box, hand_dx, hand_dy);
    const bool start_is_empty = !any_detection_at_start(detections, track);
    if (hand_move > TRACK_HAND_MOVE_EPS && hand_can_carry_box(hand_box, candidate_proxy) &&
        start_is_empty) {
        track.proxy_box = candidate_proxy;
        track.proxy_path.push_back(track.proxy_box);
        track.seen_hand_contact = true;
        track.seen_effective_move = true;
        track.hold_and_move = true;
        track.missing_without_hand_count = 0;
        track.state = OperationTrackState::HAND_OCCLUDED;
    } else {
        ++track.missing_without_hand_count;
        if (track.missing_without_hand_count > TRACK_LOST_FRAME_LIMIT) {
            track.state = OperationTrackState::INVALID;
        }
    }
    track.last_item_box = BBox();
    track.has_last_item_box = false;
    track.last_hand_box = hand_box;
    track.has_last_hand_box = true;
    track.hand_path.push_back(hand_box);
}

void SessionManager::create_candidate_tracks_(const std::vector<Detection>& detections,
                                              const BBox& hand_box,
                                              bool first_hand_frame) {
    // 2.0 的一轮只接受一个 Track 最终结果。因此一段连续有手操作已经存在候选
    // 时，不能再因为手框靠近相邻物品而叠加新的候选；否则实际只拿 A、再将 A
    // 放到 B 前面时，B 也会被错误地推成第二条“已移动”Track。
    if (!track_buffer_.empty()) return;

    const InventoryItem* best_item = nullptr;
    const Detection* best_observation = nullptr;
    float best_contact_score = -1.0f;

    for (std::map<int, InventoryItem*>::const_iterator it = item_by_id_.begin();
         it != item_by_id_.end(); ++it) {
        const InventoryItem* item = it->second;
        if (!item || item->status != ItemStatus::VISIBLE) continue;

        const int current_detection_index = unique_detection_for_item(detections, *item);
        int previous_detection_index = -1;
        if (first_hand_frame && !previous_had_hand_) {
            previous_detection_index = unique_detection_for_item(previous_food_detections_, *item);
        }

        const Detection* observation = nullptr;
        bool contact = false;
        float contact_score = -1.0f;
        if (current_detection_index >= 0) {
            observation = &detections[current_detection_index];
            contact = hand_near_box(hand_box, observation->box);
            if (contact) {
                // 真实重叠比“仅在附近”更可信；物品被手覆盖得越多，分数越高。
                const float proximity = normalized_nearby_distance(hand_box, observation->box);
                const float proximity_bonus = 0.10f * std::max(
                    0.0f, 1.0f - proximity / TRACK_HAND_NEAR_NORM);
                contact_score = 1.0f + cover_ratio(observation->box, hand_box) +
                                proximity_bonus;
            }
        }
        if (!contact && previous_detection_index >= 0) {
            observation = &previous_food_detections_[previous_detection_index];
            contact = hand_near_box(hand_box, observation->box);
            if (contact) {
                // 前一帧证据只在手刚出现时使用，优先级低于当前帧 Detection。
                const float proximity = normalized_nearby_distance(hand_box, observation->box);
                const float proximity_bonus = 0.10f * std::max(
                    0.0f, 1.0f - proximity / TRACK_HAND_NEAR_NORM);
                contact_score = 0.5f + cover_ratio(observation->box, hand_box) +
                                proximity_bonus;
            }
        }
        const float full_cover = cover_ratio(item->base_box, hand_box);
        if (!contact && full_cover < TRACK_FULL_OCCLUSION_OVERLAP) continue;

        // 手把旧物品完整盖住，是最强的“从这里开始拿起”的候选证据。
        if (full_cover >= TRACK_FULL_OCCLUSION_OVERLAP) {
            contact_score = std::max(contact_score, 2.0f + full_cover);
        }
        if (contact_score < 0.0f) continue;

        if (!best_item || contact_score > best_contact_score + 0.001f ||
            (std::fabs(contact_score - best_contact_score) <= 0.001f &&
             item->item_id < best_item->item_id)) {
            best_item = item;
            best_observation = observation;
            best_contact_score = contact_score;
        }
    }

    if (!best_item) return;

    OperationTrack track;
    track.bound_item_id = best_item->item_id;
    track.cls_id = best_item->cls_id;
    track.start_box = best_item->base_box;
    track.proxy_box = best_item->base_box;
    track.shelter_or_hold = true;
    track.seen_hand_contact = true;
    track.last_hand_box = hand_box;
    track.has_last_hand_box = true;
    track.hand_path.push_back(hand_box);
    track.proxy_path.push_back(track.proxy_box);
    if (best_observation) {
        track.state = OperationTrackState::VISIBLE;
        track.last_item_box = best_observation->box;
        track.has_last_item_box = true;
    } else {
        track.state = OperationTrackState::HAND_OCCLUDED;
    }
    track_buffer_[track.bound_item_id] = track;
}

void SessionManager::update_tracks_while_hand_present_(
        const std::vector<Detection>& detections, const BBox& hand_box, int /*frame_id*/) {
    std::map<int, std::vector<int> > track_to_detection;
    std::vector<std::vector<int> > detection_to_track(detections.size());

    // 先做已有 Track 与本帧 Detection 的双向唯一关联。
    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        const OperationTrack& track = it->second;
        if (track.frozen || track.state == OperationTrackState::INVALID) continue;
        BBox reference = track.has_last_item_box ? track.last_item_box : track.proxy_box;

        // 已经确认被拿起并移动的物品，或当前本来就挡住库存中遮挡物的前景物品，
        // 在手仍接触它且手明显位移时，优先用 proxy_box + hand_delta 预测本帧
        // 位置。否则“前景物品移开、同类后景物品在原位露出”时，原位的后景
        // Detection 会被错误接到前景 Track 上。
        // 若预测位置没有匹配的、且仍与手接触的 Detection，才回退到上一帧位置，
        // 以兼容手离开物品、物品保持不动的正常放下过程。
        if ((track.hold_and_move ||
             track_is_front_of_occluded_inventory_item(track, item_by_id_)) &&
            track.state != OperationTrackState::PLACED &&
            track.has_last_hand_box) {
            const float hand_dx = hand_box.cx() - track.last_hand_box.cx();
            const float hand_dy = hand_box.cy() - track.last_hand_box.cy();
            const float hand_move = std::sqrt(hand_dx * hand_dx + hand_dy * hand_dy);
            if (hand_move > TRACK_HAND_MOVE_EPS) {
                const BBox predicted = move_box(track.proxy_box, hand_dx, hand_dy);
                bool predicted_item_is_visible = false;
                for (size_t di = 0; di < detections.size(); ++di) {
                    if (detections[di].cls_id == track.cls_id &&
                        hand_near_box(hand_box, detections[di].box) &&
                        track_box_match(predicted, detections[di].box)) {
                        predicted_item_is_visible = true;
                        break;
                    }
                }
                if (predicted_item_is_visible) reference = predicted;
            }
        }
        for (size_t di = 0; di < detections.size(); ++di) {
            if (detections[di].cls_id != track.cls_id ||
                !track_box_match(reference, detections[di].box)) {
                continue;
            }
            track_to_detection[track.bound_item_id].push_back(static_cast<int>(di));
            detection_to_track[di].push_back(track.bound_item_id);
        }
    }

    std::set<int> updated_tracks;
    for (std::map<int, std::vector<int> >::const_iterator it = track_to_detection.begin();
         it != track_to_detection.end(); ++it) {
        if (it->second.size() != 1) continue;
        const int di = it->second.front();
        if (detection_to_track[di].size() != 1) continue;
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(it->first);
        if (track == track_buffer_.end()) continue;
        update_track_by_visible_item_(track->second, detections[di], hand_box);
        updated_tracks.insert(it->first);
    }
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        if (it->second.frozen || it->second.state == OperationTrackState::INVALID ||
            updated_tracks.count(it->first)) {
            continue;
        }
        update_track_by_hand_or_mark_invalid_(it->second, detections, hand_box);
    }
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end();) {
        if (it->second.state == OperationTrackState::INVALID) {
            it = track_buffer_.erase(it);
        } else {
            ++it;
        }
    }

    const bool first_hand_frame = !previous_had_hand_;
    create_candidate_tracks_(detections, hand_box, first_hand_frame);
}

void SessionManager::freeze_all_tracks_() {
    frozen_tracks_.clear();
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end();) {
        if (it->second.state == OperationTrackState::INVALID || !it->second.hold_and_move) {
            it = track_buffer_.erase(it);
            continue;
        }
        it->second.frozen = true;
        frozen_tracks_.push_back(it->second);
        ++it;
    }
}

void SessionManager::clear_tracks_after_settlement_() {
    for (size_t i = 0; i < frozen_tracks_.size(); ++i) frozen_tracks_[i].frozen = false;
    frozen_tracks_.clear();
    track_buffer_.clear();
    track_session_is_ambiguous_ = false;
    operation_pending_ = false;
}

void SessionManager::discard_tracks_and_mark_ambiguous_() {
    for (size_t i = 0; i < frozen_tracks_.size(); ++i) frozen_tracks_[i].frozen = false;
    frozen_tracks_.clear();
    track_buffer_.clear();
    track_session_is_ambiguous_ = true;
}

SettlementResult SessionManager::settle_snapshot_(const Snapshot& snapshot) {
    SettlementResult failed;
    if (!operation_pending_ || !snapshot.valid) return failed;

    // 这一轮只能修改 working_inventory；任何失败都会让正式 inventory 保持不变。
    std::map<int, InventoryItem> working = inventory_.items();
    int working_next_item_id = inventory_.next_item_id();
    std::vector<int> original_item_ids;
    std::map<int, bool> affirm;
    for (std::map<int, InventoryItem>::const_iterator it = working.begin(); it != working.end(); ++it) {
        original_item_ids.push_back(it->first);
        affirm[it->first] = false;
    }
    std::vector<VotingItem> observed = snapshot.items;
    for (size_t si = 0; si < observed.size(); ++si) {
        if (observed[si].temporary_id >= 0) observed[si].temporary_id = -1 - static_cast<int>(si);
        observed[si].item_id = -1;
    }
    std::map<int, int> matched_snapshot_of_item;
    std::vector<InventoryEvent> events;
    std::set<int> moved_front_item_ids;

    // 1. 先处理冻结的 hold_and_move Track。Track 是唯一能证明“哪一个物品
    // 实际被移动过”的证据，因此它的唯一终点必须先占用快照 B；否则当前景 F
    // 移到同类后景 A 的原位置时，严格匹配会先把 F 错绑定为 A，进而把 F 误判
    // 为出库。Track 候选不唯一时仍按 2.0 原则拒绝提交。
    const TrackSettlementResult track_result = get_track_settlement_result(
        working, affirm, observed, frozen_tracks_, track_session_is_ambiguous_);
    if (!track_result.ambiguous_item_ids.empty()) {
        printf("[SETTLE] Track 终点不唯一或同轮多个结果，放弃本轮提交\n");
        for (std::map<int, int>::const_iterator it = track_result.endpoint_candidate_counts.begin();
             it != track_result.endpoint_candidate_counts.end(); ++it) {
            printf("[TRACK] item#%d 的稳定快照终点候选数=%d\n", it->first, it->second);
        }
        return failed;
    }
    for (size_t i = 0; i < track_result.move_pairs.size(); ++i) {
        const int item_id = track_result.move_pairs[i].first;
        const int si = track_result.move_pairs[i].second;
        std::map<int, InventoryItem>::iterator item_it = working.find(item_id);
        if (item_it == working.end() || item_it->second.status != ItemStatus::VISIBLE ||
            affirm[item_id] || observed[si].item_id != -1) {
            return failed;
        }
        InventoryItem& item = item_it->second;
        const BBox before = item.box;
        set_seen_box(item, observed[si], snapshot.frame_id);
        item.base_box = observed[si].box;
        item.block_ids.clear();
        update_block_ids_as_front(working, item_id);
        affirm[item_id] = true;
        observed[si].item_id = item_id;
        matched_snapshot_of_item[item_id] = si;
        moved_front_item_ids.insert(item_id);
        events.push_back(make_event(EventKind::MOVED, item, before, item.box));
    }
    for (std::set<int>::const_iterator it = track_result.out_item_ids.begin();
         it != track_result.out_item_ids.end(); ++it) {
        std::map<int, InventoryItem>::iterator item_it = working.find(*it);
        if (item_it == working.end() || item_it->second.status != ItemStatus::VISIBLE ||
            affirm[*it]) {
            return failed;
        }
        events.push_back(make_event(EventKind::OUT, item_it->second));
        remove_from_working_inventory(working, *it);
    }

    // 2. 严格匹配：只处理 Track 未占用快照后的可见 A，并且接受双向唯一关系。
    std::map<int, std::vector<int> > strict_a_to_b;
    std::vector<std::vector<int> > strict_b_to_a(observed.size());
    for (std::map<int, InventoryItem>::const_iterator it = working.begin(); it != working.end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status != ItemStatus::VISIBLE || affirm[item.item_id]) continue;
        for (size_t si = 0; si < observed.size(); ++si) {
            if (observed[si].item_id != -1 || visible_item_may_grow(item, observed[si]) ||
                !strict_match(item, observed[si])) {
                continue;
            }
            strict_a_to_b[item.item_id].push_back(static_cast<int>(si));
            strict_b_to_a[si].push_back(item.item_id);
        }
    }
    for (std::map<int, std::vector<int> >::const_iterator it = strict_a_to_b.begin();
         it != strict_a_to_b.end(); ++it) {
        if (it->second.size() != 1) continue;
        const int si = it->second.front();
        if (strict_b_to_a[si].size() != 1) continue;
        InventoryItem& item = working[it->first];
        if (affirm[item.item_id] || observed[si].item_id != -1) continue;
        set_seen_box(item, observed[si], snapshot.frame_id);
        affirm[item.item_id] = true;
        observed[si].item_id = item.item_id;
        matched_snapshot_of_item[item.item_id] = si;
    }

    // 3.1 可见 A 的局部匹配，仍要求双向唯一；只更新 box，不更新 base_box。
    std::map<int, std::vector<int> > partial_a_to_b;
    std::vector<std::vector<int> > partial_b_to_a(observed.size());
    for (std::map<int, InventoryItem>::const_iterator it = working.begin(); it != working.end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status != ItemStatus::VISIBLE || affirm[item.item_id]) continue;
        for (size_t si = 0; si < observed.size(); ++si) {
            if (observed[si].item_id != -1 || visible_item_may_grow(item, observed[si]) ||
                !partial_match(item, observed[si])) {
                continue;
            }
            partial_a_to_b[item.item_id].push_back(static_cast<int>(si));
            partial_b_to_a[si].push_back(item.item_id);
        }
    }
    for (std::map<int, std::vector<int> >::const_iterator it = partial_a_to_b.begin();
         it != partial_a_to_b.end(); ++it) {
        if (it->second.size() != 1) continue;
        const int si = it->second.front();
        if (partial_b_to_a[si].size() != 1) continue;
        InventoryItem& item = working[it->first];
        if (affirm[item.item_id] || observed[si].item_id != -1) continue;
        set_seen_box(item, observed[si], snapshot.frame_id);
        affirm[item.item_id] = true;
        observed[si].item_id = item.item_id;
        matched_snapshot_of_item[item.item_id] = si;
    }

    // 3.2 再处理遮挡 A 的局部匹配。是否真正露出留到所有 blocker 稳定后判断。
    std::map<int, std::vector<int> > shelter_a_to_b;
    std::vector<std::vector<int> > shelter_b_to_a(observed.size());
    for (std::map<int, InventoryItem>::const_iterator it = working.begin(); it != working.end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status != ItemStatus::OCCLUDED || affirm[item.item_id]) continue;
        for (size_t si = 0; si < observed.size(); ++si) {
            if (observed[si].item_id != -1 || !base_box_contains(item.base_box, observed[si].box) ||
                !partial_match(item, observed[si])) {
                continue;
            }
            shelter_a_to_b[item.item_id].push_back(static_cast<int>(si));
            shelter_b_to_a[si].push_back(item.item_id);
        }
    }
    for (std::map<int, std::vector<int> >::const_iterator it = shelter_a_to_b.begin();
         it != shelter_a_to_b.end(); ++it) {
        if (it->second.size() != 1) continue;
        const int si = it->second.front();
        if (shelter_b_to_a[si].size() != 1) continue;
        InventoryItem& item = working[it->first];
        if (affirm[item.item_id] || observed[si].item_id != -1) continue;
        affirm[item.item_id] = true;
        observed[si].item_id = item.item_id;
        matched_snapshot_of_item[item.item_id] = si;
    }

    // 3.3 原本可见却未匹配的 A：只能由完整遮挡解释，不能静态猜成 OUT。
    for (size_t oi = 0; oi < original_item_ids.size(); ++oi) {
        const int item_id = original_item_ids[oi];
        std::map<int, InventoryItem>::iterator item_it = working.find(item_id);
        if (item_it == working.end()) continue;  // 已由 Track 确认出库。
        InventoryItem& item = item_it->second;
        if (item.status != ItemStatus::VISIBLE || affirm[item_id]) continue;

        std::vector<int> temporary_block_indices;
        for (size_t si = 0; si < observed.size(); ++si) {
            if (observed[si].item_id == -1 &&
                intersection_area(item.base_box, observed[si].box) > BLOCK_OVERLAP_AREA_EPS) {
                temporary_block_indices.push_back(static_cast<int>(si));
            }
        }
        const bool exactly_covered = covered_by_blockers_and_snapshot(
            item, working, observed, temporary_block_indices);
        const bool covered_by_tracked_front = covered_by_recently_moved_blocker(
            item, working, moved_front_item_ids);
        if (!exactly_covered && !covered_by_tracked_front) {
            printf("[SETTLE] item#%d 未匹配且没有完整遮挡或 Track 出库证据，放弃提交\n",
                   item_id);
            return failed;
        }

        item.status = ItemStatus::OCCLUDED;
        affirm[item_id] = true;
        events.push_back(make_event(EventKind::OCCLUDED, item));
        for (size_t i = 0; i < temporary_block_indices.size(); ++i) {
            const int si = temporary_block_indices[i];
            if (observed[si].item_id == -1) {
                add_snapshot_to_working_inventory(working, working_next_item_id,
                                                  observed[si], snapshot.frame_id,
                                                  current_time_ms_, events);
            }
        }
    }

    // 3.4 仍未绑定的快照 B 就是新入库物品。
    for (size_t si = 0; si < observed.size(); ++si) {
        if (observed[si].item_id == -1) {
            add_snapshot_to_working_inventory(working, working_next_item_id,
                                              observed[si], snapshot.frame_id,
                                              current_time_ms_, events);
        }
    }

    // 3.5 原本遮挡的 A：只有 blocker 稳定后仍留有可见区域，才正式露出。
    for (size_t oi = 0; oi < original_item_ids.size(); ++oi) {
        const int item_id = original_item_ids[oi];
        std::map<int, InventoryItem>::iterator item_it = working.find(item_id);
        if (item_it == working.end()) continue;
        InventoryItem& item = item_it->second;
        if (item.status != ItemStatus::OCCLUDED || !affirm[item_id]) continue;
        std::map<int, int>::iterator matched = matched_snapshot_of_item.find(item_id);
        if (matched == matched_snapshot_of_item.end()) continue;
        const int si = matched->second;
        bool revealed = false;
        if (si >= 0 && si < static_cast<int>(observed.size()) &&
            base_box_contains(item.base_box, observed[si].box) &&
            still_has_visible_area(item, working)) {
            const BBox before = item.box;
            item.status = ItemStatus::VISIBLE;
            set_seen_box(item, observed[si], snapshot.frame_id);
            events.push_back(make_event(EventKind::REVEALED, item, before, item.box));
            revealed = true;
        }
        if (!revealed && si >= 0 && si < static_cast<int>(observed.size())) {
            // 该 B 不是真正露出的 A；解除临时绑定，作为新物品保留。
            observed[si].item_id = -1;
            matched_snapshot_of_item.erase(matched);
            add_snapshot_to_working_inventory(working, working_next_item_id,
                                              observed[si], snapshot.frame_id,
                                              current_time_ms_, events);
        }
    }

    // 最终一致性检查：所有 B 必须唯一绑定；剩余可见原物品不能处于未解释状态。
    std::set<int> seen_snapshot_item_ids;
    for (size_t si = 0; si < observed.size(); ++si) {
        if (observed[si].item_id < 0 || !seen_snapshot_item_ids.insert(observed[si].item_id).second) {
            return failed;
        }
    }
    for (size_t oi = 0; oi < original_item_ids.size(); ++oi) {
        std::map<int, InventoryItem>::const_iterator item = working.find(original_item_ids[oi]);
        if (item != working.end() && item->second.status == ItemStatus::VISIBLE &&
            !affirm[original_item_ids[oi]]) {
            return failed;
        }
    }
    for (std::map<int, InventoryItem>::iterator it = working.begin(); it != working.end(); ++it) {
        for (std::set<int>::iterator blocker = it->second.block_ids.begin();
             blocker != it->second.block_ids.end();) {
            std::map<int, InventoryItem>::const_iterator blocker_item = working.find(*blocker);
            if (*blocker == it->first || blocker_item == working.end() ||
                intersection_area(it->second.base_box, blocker_item->second.base_box)
                    <= BLOCK_OVERLAP_AREA_EPS) {
                blocker = it->second.block_ids.erase(blocker);
            } else {
                ++blocker;
            }
        }
    }

    inventory_.replace_all(working, working_next_item_id);
    rebuild_persistent_item_index_();
    clear_tracks_after_settlement_();

    SettlementResult result;
    result.committed = true;
    result.happened = !events.empty();
    result.events.swap(events);
    return result;
}

void SessionManager::print_inventory() const {
    const size_t visible_count = inventory_.count_by_status(ItemStatus::VISIBLE);
    const size_t occluded_count = inventory_.count_by_status(ItemStatus::OCCLUDED);
    const size_t total_count = visible_count + occluded_count;

    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────┐\n");
    printf("  │  在库清单 │ 可见: %-3zu │ 遮挡: %-3zu │ 共: %-3zu            │\n",
           visible_count, occluded_count, total_count);
    printf("  ├────┬──────────────┬────────┬────────┬──────────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 遮挡数 │ 位置（当前可见框中心） │\n");
    printf("  ├────┼──────────────┼────────┼────────┼──────────────────────────┤\n");
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        const InventoryItem& item = it->second;
        const char* status = item.status == ItemStatus::VISIBLE ? "可见" : "遮挡";
        printf("  │ %-2d │ %-12s │ %-6s │ %-6zu │ (%4.0f,%4.0f)             │\n",
               item.item_id, cls_id_to_chinese(item.cls_id), status,
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
    if (!session_active_) {
        save_previous_yolo_result_(food_detections, hand_present_);
        return output;
    }

    if (hand_present_) {
        // 冷启动的短帧校验必须在清空缓存前完成。
        finalize_initial_check_before_hand_();
        operation_pending_ = true;  // 只有手会打开正式结算窗口。
        no_hand_streak_ = 0;
        no_hand_buffer_.reset();

        if (hand_boxes.size() != 1) {
            track_session_is_ambiguous_ = true;
            track_buffer_.clear();
            frozen_tracks_.clear();
        } else if (has_local_inventory_ && !track_session_is_ambiguous_) {
            update_tracks_while_hand_present_(food_detections, hand_boxes.front(), frame_id);
        }
        save_previous_yolo_result_(food_detections, true);
        return output;
    }

    ++no_hand_streak_;
    if (!should_collect_snapshot_()) {
        save_previous_yolo_result_(food_detections, false);
        return output;
    }

    no_hand_buffer_.push(food_detections, frame_id);
    if (!no_hand_buffer_.full()) {
        save_previous_yolo_result_(food_detections, false);
        return output;
    }

    Snapshot snapshot = no_hand_buffer_.take_snapshot();
    output.stable_snapshot_generated = true;
    if (initial_check_state_ == InitialCheckState::WAITING) {
        validate_initial_snapshot_(snapshot);
        initial_check_state_ = InitialCheckState::DONE;
        printf("\033[1;32m[EVENT]\033[0m 后台初始库存只读校验完成\n");
        print_inventory();
    } else if (initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_SNAPSHOT) {
        initialize_from_bootstrap_snapshot_(snapshot);
    } else if (operation_pending_) {
        freeze_all_tracks_();
        output.settlement = settle_snapshot_(snapshot);
        if (!output.settlement.committed) {
            // 下一组完整 N 帧可以继续尝试静态库存结算，但不能再使用断开的 Track 证据。
            discard_tracks_and_mark_ambiguous_();
        }
    }
    save_previous_yolo_result_(food_detections, false);
    return output;
}

}  // namespace fridge
