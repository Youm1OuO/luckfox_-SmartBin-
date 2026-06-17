// ============================================================================
//  session.cc
//  会话管理器 — 新业务流程6：统一快照裁决 + OperationContext 辅助证据
//
//  关键约束：
//    1. 手不直接产生库存事件，只记录 OperationContext 证据。
//    2. 库存事件只在 baseline_snapshot vs stable_snapshot 的一次 diff 中裁决。
//    3. 整理/移动优先于普通出库/入库，避免被拆成 "OUT + IN"。
//    4. reid_match 只是外观输入，confirmed_relocation 必须结合移动/手/HELD证据。
//    5. low_confidence_relocation 进入跨快照 PendingRelocation，本轮冻结 A/B。
// ============================================================================
#include "session.h"
#include "fridge_config.h"
#include "yolov5.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <opencv2/imgproc.hpp>

namespace fridge {

namespace {

constexpr float RELOCATION_OCCLUSION_OVERLAP_THRESH = 0.35f;
constexpr float RELOCATION_OCCLUSION_IOU_THRESH = 0.18f;

float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

float aspect_ratio(const BBox& b) {
    float h = std::max(1.0f, b.h());
    return std::max(1.0f, b.w()) / h;
}

bool strict_box_match(const BBox& a, int cls_a, const BBox& b, int cls_b) {
    if (cls_a != cls_b) return false;
    if (center_distance(a, b) >= IDENTITY_CENTER_DIST) return false;
    if (area_ratio_diff(a, b) >= IDENTITY_AREA_RATIO) return false;
    if (iou(a, b) < IDENTITY_IOU_THRESH) return false;
    return true;
}

bool relaxed_box_match(const BBox& a, int cls_a, const BBox& b, int cls_b) {
    if (cls_a != cls_b) return false;
    if (area_ratio_diff(a, b) >= 0.40f) return false;
    if (normalized_nearby_distance(a, b) > NEARBY_DISTANCE_THRESH) return false;
    return true;
}

int find_in_snapshot_strict(const Snapshot& snap,
                            int cls_id,
                            const BBox& box,
                            const std::set<int>& used_indices) {
    for (int i = 0; i < (int)snap.items.size(); ++i) {
        if (used_indices.count(i)) continue;
        const VotingItem& vi = snap.items[i];
        if (strict_box_match(box, cls_id, vi.box, vi.cls_id)) {
            return i;
        }
    }
    return -1;
}

int find_in_snapshot_near(const Snapshot& snap,
                          int cls_id,
                          const BBox& box,
                          const std::set<int>& used_indices) {
    int best_idx = -1;
    float best_dist = 999.0f;
    for (int i = 0; i < (int)snap.items.size(); ++i) {
        if (used_indices.count(i)) continue;
        const VotingItem& vi = snap.items[i];
        if (vi.cls_id != cls_id) continue;
        if (area_ratio_diff(box, vi.box) >= 0.45f) continue;
        float d = normalized_nearby_distance(box, vi.box);
        if (d < best_dist && d <= 0.75f) {
            best_dist = d;
            best_idx = i;
        }
    }
    return best_idx;
}

}  // namespace

void OperationContext::reset(int new_context_id,
                             int new_baseline_snapshot_id,
                             int frame_id) {
    *this = OperationContext();
    context_id = new_context_id;
    baseline_snapshot_id = new_baseline_snapshot_id;
    created_frame_id = frame_id;
    updated_frame_id = frame_id;
}

// ============================================================================
//  构造 / 初始化
// ============================================================================

SessionManager::SessionManager()
    : snap_state_(SnapState::IDLE),
      has_snap1_(false),
      hand_present_(false),
      no_hand_streak_(0),
      next_context_id_(1),
      next_pending_id_(1),
      backend_status_(BackendStatus::UNKNOWN),
      init_state_(InitState::WAIT_FIRST_STABLE_SNAPSHOT),
      inventory_initialized_(false),
      current_time_ms_(0),
      session_start_time_ms_(0),
      first_empty_grace_logged_(false) {
    snap1_ = {{}, 0, false, false};
    contrast_ = {{}, 0, false, false};
    current_ = {{}, 0, false, false};
    operation_context_.reset(next_context_id_++, 0, 0);
}

void SessionManager::start_new_session(long long time_ms) {
    current_time_ms_ = time_ms;
    session_start_time_ms_ = time_ms;
    first_empty_grace_logged_ = false;
    inventory_ = InventoryDB();
    backend_items_.clear();
    backend_status_ = BackendStatus::UNKNOWN;
    init_state_ = InitState::WAIT_BACKEND;
    inventory_initialized_ = false;
    pending_relocations_.clear();
    hand_present_ = false;
    no_hand_streak_ = 0;
    reset_snap_state();
    printf("[SESSION] 新开门会话：等待后台库存或首个无手稳定快照\n");
}

void SessionManager::init_from_backend(const std::vector<InventoryItem>& items,
                                       bool authoritative_empty) {
    if (init_state_ == InitState::READY) {
        printf("[SESSION] 后台库存返回过晚，当前会话已 READY，忽略本次后台初始化\n");
        return;
    }

    backend_items_ = items;

    if (!items.empty() || authoritative_empty) {
        backend_status_ = BackendStatus::TRUSTED;
        init_state_ = InitState::WAIT_FIRST_STABLE_SNAPSHOT;
        printf("[SESSION] 后台库存可信：%zu 件，等待首个无手稳定快照对齐\n",
               items.size());
        return;
    }

    backend_items_.clear();
    backend_status_ = BackendStatus::NO_TRUSTED_BACKEND;
    init_state_ = InitState::WAIT_FIRST_STABLE_SNAPSHOT;
    printf("[SESSION] 后台返回空列表但不具权威性，改用首个无手稳定快照初始化\n");
}

void SessionManager::mark_backend_unavailable() {
    if (init_state_ == InitState::READY) {
        return;
    }

    backend_items_.clear();
    backend_status_ = BackendStatus::NO_TRUSTED_BACKEND;
    init_state_ = InitState::WAIT_FIRST_STABLE_SNAPSHOT;
    printf("[SESSION] 无可信后台库存，等待首个无手稳定快照初始化\n");
}

void SessionManager::finish_session(long long time_ms) {
    current_time_ms_ = time_ms;

    if (!pending_relocations_.empty()) {
        printf("[SESSION] 关门：清理 %zu 个未确认整理候选，不强行提交 MOVED\n",
               pending_relocations_.size());
        pending_relocations_.clear();
    }

    hand_present_ = false;
    no_hand_streak_ = 0;

    printf("[SESSION] 会话结束：最终库存 可见=%zu 遮挡=%zu 出库历史=%zu\n",
           inventory_.count_by_status(ItemStatus::VISIBLE),
           inventory_.count_by_status(ItemStatus::OCCLUDED),
           inventory_.count_by_status(ItemStatus::OUT));
}

// ============================================================================
//  基础匹配 / 辅助函数
// ============================================================================

float SessionManager::color_diff(const cv::Mat& frame, const BBox& a, const BBox& b) {
    if (frame.empty()) return 0.0f;
    int ax1 = std::max(0, (int)a.x1), ay1 = std::max(0, (int)a.y1);
    int ax2 = std::min(frame.cols, (int)a.x2), ay2 = std::min(frame.rows, (int)a.y2);
    int bx1 = std::max(0, (int)b.x1), by1 = std::max(0, (int)b.y1);
    int bx2 = std::min(frame.cols, (int)b.x2), by2 = std::min(frame.rows, (int)b.y2);
    if (ax2 <= ax1 || ay2 <= ay1 || bx2 <= bx1 || by2 <= by1) return 999.0f;

    cv::Mat a_small, b_small;
    cv::resize(frame(cv::Rect(ax1, ay1, ax2 - ax1, ay2 - ay1)), a_small, cv::Size(16, 16));
    cv::resize(frame(cv::Rect(bx1, by1, bx2 - bx1, by2 - by1)), b_small, cv::Size(16, 16));
    cv::Scalar ma = cv::mean(a_small), mb = cv::mean(b_small);
    return (std::abs(ma[0] - mb[0]) + std::abs(ma[1] - mb[1]) + std::abs(ma[2] - mb[2])) / 3.0f;
}

bool SessionManager::match_strict(const BBox& a, int cls_a,
                                  const BBox& b, int cls_b,
                                  const cv::Mat& frame) {
    if (!strict_box_match(a, cls_a, b, cls_b)) return false;
    if (!frame.empty() && color_diff(frame, a, b) >= IDENTITY_COLOR_DIFF) return false;
    return true;
}

int SessionManager::find_inventory_item_strict(const BBox& box,
                                               int cls_id,
                                               bool include_out) const {
    for (const auto& kv : inventory_.items()) {
        const InventoryItem& it = kv.second;
        if (!include_out && it.status == ItemStatus::OUT) continue;
        if (strict_box_match(box, cls_id, it.box, it.cls_id)) {
            return kv.first;
        }
    }
    return -1;
}

int SessionManager::find_inventory_item_relaxed(const BBox& box,
                                                int cls_id,
                                                bool include_out) const {
    int best_id = -1;
    float best_dist = 999.0f;
    for (const auto& kv : inventory_.items()) {
        const InventoryItem& it = kv.second;
        if (!include_out && it.status == ItemStatus::OUT) continue;
        if (it.cls_id != cls_id) continue;
        if (!relaxed_box_match(box, cls_id, it.box, it.cls_id)) continue;
        float d = normalized_nearby_distance(box, it.box);
        if (d < best_dist) {
            best_dist = d;
            best_id = kv.first;
        }
    }
    return best_id;
}

int SessionManager::find_inventory_item_for_track(const Track& track) const {
    for (const auto& kv : inventory_.items()) {
        const InventoryItem& it = kv.second;
        if (it.status == ItemStatus::OUT) continue;
        if (it.track_id == track.track_id && it.cls_id == track.cls_id) {
            return kv.first;
        }
    }
    return find_inventory_item_relaxed(track.box, track.cls_id, false);
}

int SessionManager::find_best_hand_track_id(const BBox& hand_box,
                                            const std::vector<Track>& tracks) const {
    int best_id = -1;
    float best_overlap = 0.0f;
    for (const auto& t : tracks) {
        if (!is_hand(t.cls_id)) continue;
        float v = overlap_ratio_of_smaller(hand_box, t.box);
        if (v > best_overlap) {
            best_overlap = v;
            best_id = t.track_id;
        }
    }
    return best_id;
}

int SessionManager::find_original_object_track_id(int item_id) const {
    for (const auto& kv : operation_context_.active_track_evidences) {
        const TrackEvidence& ev = kv.second;
        if (ev.associated_item_id == item_id) {
            return ev.track_id;
        }
    }
    const InventoryItem* item = inventory_.find_by_item(item_id);
    return item ? item->track_id : -1;
}

float SessionManager::reid_score(const VotingItem& a, const VotingItem& b) const {
    if (a.cls_id != b.cls_id) return 0.0f;

    float area_score = 1.0f - area_ratio_diff(a.box, b.box);
    float ar_a = aspect_ratio(a.box);
    float ar_b = aspect_ratio(b.box);
    float ar_score = 1.0f - std::min(std::abs(ar_a - ar_b) / std::max(ar_a, ar_b), 1.0f);
    float score_score = 1.0f - std::min(std::abs(a.best_score - b.best_score), 1.0f);

    // 同类别是基础条件，但不能让它单独决定整理。
    float score = 0.40f + 0.30f * area_score + 0.20f * ar_score + 0.10f * score_score;
    return clamp01(score);
}

float SessionManager::relocation_evidence_score(int item_id,
                                                const BBox& from,
                                                const BBox& to) const {
    float score = 0.0f;

    if (operation_context_.moved_item_candidates.count(item_id)) {
        score = std::max(score, 0.35f);
    }

    for (const auto& kv : operation_context_.active_track_evidences) {
        const TrackEvidence& ev = kv.second;
        if (ev.associated_item_id != item_id) continue;
        float start_near = normalized_nearby_distance(from, ev.start_box);
        float end_near = normalized_nearby_distance(to, ev.end_box);
        if (start_near <= 0.75f && end_near <= 0.75f) {
            score = std::max(score, ev.occluded_by_hand ? 0.80f : 0.75f);
        } else if (start_near <= 0.75f && center_distance(ev.start_box, ev.end_box) > diagonal(from) * TRACK_MOVE_DISTANCE_RATIO) {
            score = std::max(score, 0.45f);
        }
    }

    for (const auto& ev : operation_context_.held_proxy_evidences) {
        if (ev.item_id != item_id || ev.proxy_boxes.empty()) continue;
        const BBox& last_proxy = ev.proxy_boxes.back();
        float proxy_near = normalized_nearby_distance(to, last_proxy);
        if (proxy_near <= 0.75f) {
            score = std::max(score, 0.85f);
        } else if (proxy_near <= NEARBY_DISTANCE_THRESH) {
            score = std::max(score, 0.55f);
        }
    }

    if (operation_context_.confirmed_held_items.count(item_id)) {
        score = std::max(score, operation_context_.hand_long_present ? 0.55f : 0.40f);
    } else if (operation_context_.candidate_held_items.count(item_id)) {
        score = std::max(score, 0.20f);
    }

    return clamp01(score);
}

RelocationDecision SessionManager::relocation_match(int item_id,
                                                    const VotingItem& a,
                                                    const VotingItem& b,
                                                    float reid,
                                                    float second_reid,
                                                    float* evidence_score) const {
    float evidence = relocation_evidence_score(item_id, a.box, b.box);
    if (evidence_score) *evidence_score = evidence;

    if (reid < RELOCATION_REID_MIN) {
        return RelocationDecision::NO_RELOCATION;
    }

    bool unique = (second_reid < 0.0f) || ((reid - second_reid) >= RELOCATION_REID_MARGIN);

    if (unique && reid >= RELOCATION_REID_STRONG && evidence >= RELOCATION_EVIDENCE_STRONG) {
        return RelocationDecision::CONFIRMED;
    }

    if (reid >= RELOCATION_REID_MIN && evidence >= RELOCATION_EVIDENCE_WEAK) {
        return RelocationDecision::LOW_CONFIDENCE;
    }

    return RelocationDecision::NO_RELOCATION;
}

// ============================================================================
//  OperationContext：每帧只收集证据，不直接改库存状态
// ============================================================================

SettlementResult SessionManager::update_hand(const std::vector<BBox>& hand_boxes,
                                             const std::vector<Track>& tracks,
                                             int frame_id,
                                             long long time_ms) {
    current_time_ms_ = time_ms;
    update_operation_context(hand_boxes, tracks, frame_id);
    return SettlementResult();
}

void SessionManager::update_operation_context(const std::vector<BBox>& hand_boxes,
                                              const std::vector<Track>& tracks,
                                              int frame_id) {
    operation_context_.updated_frame_id = frame_id;

    bool has_hand = !hand_boxes.empty();
    if (has_hand) {
        hand_present_ = true;
        no_hand_streak_ = 0;
        operation_context_.hand_seen = true;
        operation_context_.hand_frame_count++;
        if (operation_context_.hand_frame_count >= HAND_LONG_PRESENT_FRAMES) {
            operation_context_.hand_long_present = true;
        }
    } else {
        hand_present_ = false;
        no_hand_streak_++;
    }

    std::set<int> current_track_ids;
    std::set<int> current_visible_item_ids;

    for (const auto& t : tracks) {
        current_track_ids.insert(t.track_id);

        if (is_hand(t.cls_id)) {
            operation_context_.hand_track_ids.insert(t.track_id);
            continue;
        }
        if (!is_food(t.cls_id) || t.score < SNAPSHOT_MIN_SCORE) continue;

        auto active_it = operation_context_.active_track_evidences.find(t.track_id);
        int item_id = active_it == operation_context_.active_track_evidences.end()
                    ? find_inventory_item_for_track(t)
                    : active_it->second.associated_item_id;
        if (item_id < 0) continue;
        current_visible_item_ids.insert(item_id);

        auto it = operation_context_.active_track_evidences.find(t.track_id);
        if (it == operation_context_.active_track_evidences.end()) {
            TrackEvidence ev;
            ev.track_id = t.track_id;
            ev.cls_id = t.cls_id;
            ev.associated_item_id = item_id;
            ev.start_box = t.box;
            ev.end_box = t.box;
            ev.start_frame_id = frame_id;
            ev.end_frame_id = frame_id;
            ev.sample_count = 1;
            operation_context_.active_track_evidences[t.track_id] = ev;
            it = operation_context_.active_track_evidences.find(t.track_id);
        } else {
            TrackEvidence& ev = it->second;
            ev.end_box = t.box;
            ev.end_frame_id = frame_id;
            ev.sample_count++;
        }

        TrackEvidence& ev = it->second;
        for (const auto& hb : hand_boxes) {
            if (overlap_ratio_of_smaller(hb, t.box) >= HELD_HAND_OVERLAP_THRESH) {
                ev.occluded_by_hand = true;
                break;
            }
        }

        float move_dist = center_distance(ev.start_box, ev.end_box);
        float move_ratio = diagonal(ev.start_box) <= 0.0f ? 0.0f : move_dist / diagonal(ev.start_box);
        if (move_ratio >= TRACK_MOVE_DISTANCE_RATIO) {
            ev.confidence = std::max(ev.confidence, clamp01(move_ratio));
            operation_context_.moving_tracks.insert(t.track_id);
            operation_context_.moved_item_candidates.insert(item_id);
        }
    }

    std::set<int> touched_candidate_items;
    std::map<int, BBox> candidate_hand_box;
    std::map<int, int> candidate_hand_track_id;

    if (has_hand) {
        for (const auto& kv : inventory_.items()) {
            const InventoryItem& item = kv.second;
            if (item.status != ItemStatus::VISIBLE) continue;

            float best_overlap = 0.0f;
            BBox best_hand_box;
            int best_hand_track = -1;
            for (const auto& hb : hand_boxes) {
                float v = overlap_ratio_of_smaller(hb, item.box);
                if (v > best_overlap) {
                    best_overlap = v;
                    best_hand_box = hb;
                    best_hand_track = find_best_hand_track_id(hb, tracks);
                }
            }

            if (best_overlap >= HELD_HAND_OVERLAP_THRESH) {
                operation_context_.candidate_held_items[item.item_id]++;
                touched_candidate_items.insert(item.item_id);
                candidate_hand_box[item.item_id] = best_hand_box;
                candidate_hand_track_id[item.item_id] = best_hand_track;
            }
        }

        std::vector<int> candidate_ids;
        for (const auto& kv : operation_context_.candidate_held_items) {
            candidate_ids.push_back(kv.first);
        }
        for (int item_id : candidate_ids) {
            if (touched_candidate_items.count(item_id)) continue;
            if (current_visible_item_ids.count(item_id)) continue;
            if (hand_boxes.empty()) continue;

            operation_context_.candidate_held_items[item_id]++;
            touched_candidate_items.insert(item_id);
            candidate_hand_box[item_id] = hand_boxes[0];
            candidate_hand_track_id[item_id] = find_best_hand_track_id(hand_boxes[0], tracks);
        }

        for (int item_id : touched_candidate_items) {
            const InventoryItem* item = inventory_.find_by_item(item_id);
            if (!item) continue;
            if (current_visible_item_ids.count(item_id)) continue;
            if (operation_context_.candidate_held_items[item_id] < HELD_CONFIRM_FRAMES) continue;

            operation_context_.confirmed_held_items.insert(item_id);

            HeldProxyEvidence* proxy = nullptr;
            for (auto& ev : operation_context_.held_proxy_evidences) {
                if (ev.item_id == item_id) {
                    proxy = &ev;
                    break;
                }
            }
            if (!proxy) {
                HeldProxyEvidence ev;
                ev.item_id = item_id;
                ev.original_object_track_id = find_original_object_track_id(item_id);
                ev.held_by_hand_track_id = candidate_hand_track_id[item_id];
                ev.last_visible_box = item->box;
                ev.hand_bbox_at_hold_start = candidate_hand_box[item_id];
                ev.held_start_frame_id = frame_id;
                ev.confidence = 0.60f;
                operation_context_.held_proxy_evidences.push_back(ev);
                proxy = &operation_context_.held_proxy_evidences.back();
                printf("\033[1;33m[HELD]\033[0m item#%d (%s) 建立手代理轨迹证据\n",
                       item_id, coco_cls_to_name(item->cls_id));
            }
            proxy->held_end_frame_id = frame_id;
            proxy->proxy_boxes.push_back(candidate_hand_box[item_id]);
            if (operation_context_.hand_long_present) {
                proxy->confidence = std::max(proxy->confidence, 0.75f);
            }
        }
    }

    std::vector<int> erase_candidates;
    for (const auto& kv : operation_context_.candidate_held_items) {
        int item_id = kv.first;
        if (operation_context_.confirmed_held_items.count(item_id)) continue;
        if (!has_hand || !touched_candidate_items.count(item_id)) {
            erase_candidates.push_back(item_id);
        }
    }
    for (int item_id : erase_candidates) {
        operation_context_.candidate_held_items.erase(item_id);
    }
}

// ============================================================================
//  初始化 / 快照状态机
// ============================================================================

void SessionManager::init_snapshot(const Snapshot& snap, const cv::Mat& frame) {
    inventory_ = InventoryDB();

    if (backend_status_ == BackendStatus::TRUSTED) {
        printf("[SESSION] 初始化：匹配后台库存 (%zu 件) 与快照 (%zu 件)\n",
               backend_items_.size(), snap.items.size());

        std::set<int> matched_snap_indices;
        for (const auto& bi : backend_items_) {
            bool matched = false;
            for (int vi_idx = 0; vi_idx < (int)snap.items.size(); vi_idx++) {
                if (matched_snap_indices.count(vi_idx)) continue;
                const auto& vi = snap.items[vi_idx];
                if (match_strict(bi.box, bi.cls_id, vi.box, vi.cls_id, frame)) {
                    int new_id = inventory_.add_item(-1, vi.cls_id, vi.box, vi.best_score,
                                                     snap.frame_id, current_time_ms_);
                    matched_snap_indices.insert(vi_idx);
                    matched = true;
                    printf("[SESSION]   item#%d %s → 可见\n", new_id, coco_cls_to_name(vi.cls_id));
                    break;
                }
            }
            if (!matched) {
                int new_id = inventory_.add_item(-1, bi.cls_id, bi.box, bi.score,
                                                 snap.frame_id, current_time_ms_);
                inventory_.set_status(new_id, ItemStatus::OCCLUDED, current_time_ms_);
                printf("[SESSION]   item#%d %s → 遮挡（后台有但YOLO未识别到）\n",
                       new_id, coco_cls_to_name(bi.cls_id));
            }
        }

        for (int vi_idx = 0; vi_idx < (int)snap.items.size(); vi_idx++) {
            if (matched_snap_indices.count(vi_idx)) continue;
            const auto& vi = snap.items[vi_idx];
            int new_id = inventory_.add_item(-1, vi.cls_id, vi.box, vi.best_score,
                                             snap.frame_id, current_time_ms_);
            printf("[SESSION]   item#%d %s → 可见（初始化发现后台未记录物品）\n",
                   new_id, coco_cls_to_name(vi.cls_id));
        }
    } else {
        printf("[SESSION] 初始化：无后台库存，用快照初始化 (%zu 件)\n", snap.items.size());
        for (const auto& vi : snap.items) {
            inventory_.add_item(-1, vi.cls_id, vi.box, vi.best_score,
                                snap.frame_id, current_time_ms_);
        }
    }

    init_state_ = InitState::READY;
    inventory_initialized_ = true;
}

bool SessionManager::snapshots_similar(const Snapshot& a, const Snapshot& b) {
    if (!a.valid || !b.valid) return false;

    std::set<int> used_b;
    int matched_count = 0;
    for (const auto& vi_a : a.items) {
        int idx = find_in_snapshot_strict(b, vi_a.cls_id, vi_a.box, used_b);
        if (idx >= 0) {
            used_b.insert(idx);
            matched_count++;
        }
    }

    int denom = std::max((int)a.items.size(), (int)b.items.size());
    float match_ratio = denom == 0 ? 1.0f : (float)matched_count / (float)denom;
    int count_diff = std::abs((int)a.items.size() - (int)b.items.size());

    return match_ratio >= 0.7f && count_diff <= 2;
}

SettlementResult SessionManager::push_snapshot(const Snapshot& snap, const cv::Mat& frame) {
    if (!snap.valid) {
        return SettlementResult();
    }

    // 带手快照只提供 OperationContext 证据，不进入库存 diff。
    if (snap.has_hand) {
        return SettlementResult();
    }

    SettlementResult result;

    switch (snap_state_) {
        case SnapState::IDLE: {
            if (init_state_ == InitState::WAIT_BACKEND ||
                backend_status_ == BackendStatus::UNKNOWN) {
                mark_backend_unavailable();
            }

            if (!inventory_initialized_ &&
                backend_status_ != BackendStatus::TRUSTED &&
                snap.items.empty() &&
                session_start_time_ms_ > 0 &&
                current_time_ms_ - session_start_time_ms_ < FIRST_SNAPSHOT_EMPTY_GRACE_MS) {
                if (!first_empty_grace_logged_) {
                    printf("[SESSION] 首个无手稳定快照为 0 件，等待曝光稳定窗口 (%lldms)\n",
                           FIRST_SNAPSHOT_EMPTY_GRACE_MS);
                    first_empty_grace_logged_ = true;
                }
                break;
            }

            if (!inventory_initialized_) {
                init_snapshot(snap, frame);
            }
            snap1_ = snap;
            contrast_ = snap;
            has_snap1_ = true;
            snap_state_ = SnapState::COMPARE;
            reset_operation_context(snap.frame_id, snap.frame_id);
            result.happened = true;
            printf("[SESSION] 快照已就绪，进入统一对比模式 (库存=%zu 件)\n",
                   inventory_.count_by_status(ItemStatus::VISIBLE) +
                   inventory_.count_by_status(ItemStatus::OCCLUDED));
            print_inventory();
            break;
        }

        case SnapState::COMPARE: {
            current_ = snap;
            if (snapshots_similar(current_, contrast_)) {
                result = compare_snapshots(current_, frame);
                snap1_ = current_;
                contrast_ = current_;
                reset_operation_context(snap1_.frame_id, snap1_.frame_id);

                if (result.happened) {
                    print_inventory();
                }
            } else {
                contrast_ = current_;
                operation_context_.unstable_frame_count++;
            }
            break;
        }
    }

    return result;
}

void SessionManager::reset_snap_state() {
    snap_state_ = SnapState::IDLE;
    has_snap1_ = false;
    snap1_ = {{}, 0, false, false};
    contrast_ = {{}, 0, false, false};
    current_ = {{}, 0, false, false};
    reset_operation_context(0, 0);
    printf("[SESSION] 快照状态机已重置，等待新快照\n");
}

void SessionManager::reset_operation_context(int baseline_snapshot_id, int frame_id) {
    operation_context_.reset(next_context_id_++, baseline_snapshot_id, frame_id);
}

// ============================================================================
//  PendingRelocation 跨快照确认
// ============================================================================

void SessionManager::process_pending_relocations(const Snapshot& snap2,
                                                 SettlementResult& result,
                                                 std::set<int>& reserved_snap2_indices,
                                                 std::set<int>& protected_occluded_item_ids) {
    if (pending_relocations_.empty()) return;

    std::vector<PendingRelocation> kept;
    for (auto p : pending_relocations_) {
        std::set<int> no_used;
        int old_idx = find_in_snapshot_near(snap2, p.class_id, p.old_bbox, no_used);
        if (old_idx >= 0) {
            printf("[SESSION] pending#%d 取消：item#%d 回到旧位置\n",
                   p.pending_id, p.item_id_A);
            continue;
        }

        int new_idx = find_in_snapshot_near(snap2, p.class_id, p.candidate_new_bbox, no_used);
        if (new_idx < 0) {
            printf("[SESSION] pending#%d 取消：候选新位置不再稳定\n", p.pending_id);
            continue;
        }

        reserved_snap2_indices.insert(new_idx);
        p.stable_count++;
        p.last_checked_snapshot_id = snap2.frame_id;

        const VotingItem& B = snap2.items[new_idx];
        if (p.stable_count >= PENDING_RELOCATION_CONFIRM_FRAMES) {
            InventoryItem* item = inventory_.find_by_item(p.item_id_A);
            if (item && item->status != ItemStatus::OUT) {
                inventory_.set_status(p.item_id_A, ItemStatus::VISIBLE, current_time_ms_);
                inventory_.update_item(p.item_id_A, -1, B.box, B.best_score, snap2.frame_id);
                for (const auto& kv : inventory_.items()) {
                    if (kv.first == p.item_id_A) continue;
                    const InventoryItem& maybe_hidden = kv.second;
                    if (maybe_hidden.status != ItemStatus::OCCLUDED) continue;
                    float overlap = overlap_ratio_of_smaller(B.box, maybe_hidden.box);
                    float box_iou = iou(B.box, maybe_hidden.box);
                    if (overlap >= RELOCATION_OCCLUSION_OVERLAP_THRESH ||
                        box_iou >= RELOCATION_OCCLUSION_IOU_THRESH) {
                        protected_occluded_item_ids.insert(kv.first);
                    }
                }
                printf("\033[1;32m[EVENT]\033[0m 整理确认: item#%d %s "
                       "(pending#%d, reid=%.2f evidence=%.2f)\n",
                       p.item_id_A, coco_cls_to_name(p.class_id), p.pending_id,
                       p.reid_score, p.evidence_score);
                result.events.push_back({EventKind::MOVED, p.item_id_A,
                                          p.class_id, B.box,
                                          p.old_bbox, B.box, B.best_score});
                result.happened = true;
            }
            continue;
        }

        if (p.stable_count > p.expire_after_stable_count) {
            printf("[SESSION] pending#%d 过期：不再强行判整理\n", p.pending_id);
            continue;
        }

        kept.push_back(p);
    }

    pending_relocations_.swap(kept);
}

bool SessionManager::apply_relocation_visibility_changes(
        int moved_item_id,
        const BBox& old_box,
        const VotingItem& new_object,
        const Snapshot& snap2,
        const std::vector<int>& disappeared_indices,
        const std::vector<int>& appeared_indices,
        const std::vector<int>& baseline_item_ids,
        std::set<int>& consumed_disappeared_indices,
        std::set<int>& consumed_appeared_indices,
        const std::set<int>& reserved_snap2_indices,
        std::set<int>& protected_occluded_item_ids,
        bool allow_reveal) {
    bool changed = false;

    // A 移到 B 后，如果 B 覆盖了某个原本可见且本轮消失的 C，
    // C 应进入遮挡，而不是继续落入 10.3 被判出库。
    for (int c_idx : disappeared_indices) {
        if (consumed_disappeared_indices.count(c_idx)) continue;
        if (c_idx < 0 || c_idx >= (int)baseline_item_ids.size()) continue;

        int c_item_id = baseline_item_ids[c_idx];
        if (c_item_id < 0 || c_item_id == moved_item_id) continue;

        InventoryItem* c_item = inventory_.find_by_item(c_item_id);
        if (!c_item || c_item->status != ItemStatus::VISIBLE) continue;

        const VotingItem& C = snap1_.items[c_idx];
        float overlap = overlap_ratio_of_smaller(new_object.box, C.box);
        float box_iou = iou(new_object.box, C.box);
        if (overlap < RELOCATION_OCCLUSION_OVERLAP_THRESH &&
            box_iou < RELOCATION_OCCLUSION_IOU_THRESH) {
            continue;
        }

        inventory_.set_status(c_item_id, ItemStatus::OCCLUDED, current_time_ms_);
        consumed_disappeared_indices.insert(c_idx);
        protected_occluded_item_ids.insert(c_item_id);
        printf("\033[1;32m[EVENT]\033[0m 遮挡: item#%d %s "
               "(被整理 item#%d 覆盖, overlap=%.2f iou=%.2f)\n",
               c_item_id, coco_cls_to_name(c_item->cls_id), moved_item_id,
               overlap, box_iou);
        changed = true;
    }

    if (!allow_reveal) {
        return changed;
    }

    // A 离开旧位置后，如果旧位置附近出现了库存中原本遮挡的 D，
    // D 应从遮挡恢复为可见。
    for (int d_idx : appeared_indices) {
        if (consumed_appeared_indices.count(d_idx)) continue;
        if (reserved_snap2_indices.count(d_idx)) continue;

        const VotingItem& D = snap2.items[d_idx];
        if (normalized_nearby_distance(old_box, D.box) > NEARBY_DISTANCE_THRESH) {
            continue;
        }

        int best_id = -1;
        float best_dist = 999.0f;
        for (const auto& kv : inventory_.items()) {
            if (kv.first == moved_item_id) continue;
            const InventoryItem& inv = kv.second;
            if (inv.status != ItemStatus::OCCLUDED) continue;
            if (inv.cls_id != D.cls_id) continue;
            bool matched =
                relaxed_box_match(D.box, D.cls_id, inv.box, inv.cls_id) ||
                iou(D.box, inv.box) >= RELOCATION_OCCLUSION_IOU_THRESH;
            if (!matched) continue;
            float dist = normalized_nearby_distance(D.box, inv.box);
            if (dist < best_dist) {
                best_dist = dist;
                best_id = kv.first;
            }
        }

        if (best_id < 0) continue;

        InventoryItem* revealed = inventory_.find_by_item(best_id);
        if (!revealed) continue;

        inventory_.set_status(best_id, ItemStatus::VISIBLE, current_time_ms_);
        inventory_.update_item(best_id, -1, D.box, D.best_score, snap2.frame_id);
        consumed_appeared_indices.insert(d_idx);
        printf("\033[1;32m[EVENT]\033[0m 露出: item#%d %s "
               "(整理 item#%d 离开旧位置)\n",
               best_id, coco_cls_to_name(D.cls_id), moved_item_id);
        changed = true;
    }

    return changed;
}

// ============================================================================
//  baseline vs stable 统一差异裁决
// ============================================================================

SettlementResult SessionManager::compare_snapshots(const Snapshot& snap2,
                                                   const cv::Mat& frame) {
    (void)frame;
    SettlementResult result;

    std::vector<std::pair<int, int>> same_position_pairs;
    std::vector<int> disappeared_indices;
    std::vector<int> appeared_indices;
    std::set<int> used_snap2_origin_indices;

    for (int i = 0; i < (int)snap1_.items.size(); ++i) {
        const VotingItem& A = snap1_.items[i];
        int b_idx = find_in_snapshot_strict(snap2, A.cls_id, A.box, used_snap2_origin_indices);
        if (b_idx >= 0) {
            same_position_pairs.push_back({i, b_idx});
            used_snap2_origin_indices.insert(b_idx);
        } else {
            disappeared_indices.push_back(i);
        }
    }

    for (int j = 0; j < (int)snap2.items.size(); ++j) {
        if (!used_snap2_origin_indices.count(j)) {
            appeared_indices.push_back(j);
        }
    }

    std::vector<int> baseline_item_ids(snap1_.items.size(), -1);
    for (int i = 0; i < (int)snap1_.items.size(); ++i) {
        const VotingItem& A = snap1_.items[i];
        baseline_item_ids[i] = find_inventory_item_strict(A.box, A.cls_id, false);
    }

    for (const auto& p : same_position_pairs) {
        const VotingItem& B = snap2.items[p.second];
        int item_id = find_inventory_item_strict(B.box, B.cls_id, false);
        if (item_id < 0) continue;

        const InventoryItem* item = inventory_.find_by_item(item_id);
        if (!item) continue;
        if (item->status == ItemStatus::OCCLUDED) {
            inventory_.set_status(item_id, ItemStatus::VISIBLE, current_time_ms_);
        }
        inventory_.update_item(item_id, -1, B.box, B.best_score, snap2.frame_id);
    }

    std::set<int> consumed_disappeared_indices;
    std::set<int> consumed_appeared_indices;
    std::set<int> reserved_snap2_indices;
    std::set<int> blocked_appeared_inventory_ids;
    std::set<int> protected_occluded_item_ids;

    process_pending_relocations(snap2, result, reserved_snap2_indices,
                                protected_occluded_item_ids);

    // ---------------------------------------------------------------------
    //  10.2 优先处理整理 / 移动
    // ---------------------------------------------------------------------
    for (int a_idx : disappeared_indices) {
        if (consumed_disappeared_indices.count(a_idx)) continue;
        const VotingItem& A = snap1_.items[a_idx];

        int item_id = find_inventory_item_strict(A.box, A.cls_id, false);
        if (item_id < 0) continue;
        const InventoryItem* item = inventory_.find_by_item(item_id);
        if (!item || item->status != ItemStatus::VISIBLE) continue;

        struct Candidate {
            int appeared_idx;
            float reid;
        };
        std::vector<Candidate> candidates;

        for (int b_idx : appeared_indices) {
            if (consumed_appeared_indices.count(b_idx)) continue;
            if (reserved_snap2_indices.count(b_idx)) continue;
            const VotingItem& B = snap2.items[b_idx];
            if (A.cls_id != B.cls_id) continue;
            float r = reid_score(A, B);
            if (r >= RELOCATION_REID_MIN) {
                candidates.push_back({b_idx, r});
            }
        }

        if (candidates.empty()) continue;
        std::sort(candidates.begin(), candidates.end(),
                  [](const Candidate& x, const Candidate& y){ return x.reid > y.reid; });

        int best_idx = candidates[0].appeared_idx;
        float best_reid = candidates[0].reid;
        float second_reid = candidates.size() > 1 ? candidates[1].reid : -1.0f;
        const VotingItem& B = snap2.items[best_idx];

        float evidence_score = 0.0f;
        RelocationDecision decision = relocation_match(item_id, A, B, best_reid,
                                                       second_reid, &evidence_score);
        if (decision == RelocationDecision::CONFIRMED) {
            inventory_.set_status(item_id, ItemStatus::VISIBLE, current_time_ms_);
            inventory_.update_item(item_id, -1, B.box, B.best_score, snap2.frame_id);
            consumed_disappeared_indices.insert(a_idx);
            consumed_appeared_indices.insert(best_idx);
            apply_relocation_visibility_changes(
                item_id, A.box, B, snap2,
                disappeared_indices, appeared_indices, baseline_item_ids,
                consumed_disappeared_indices, consumed_appeared_indices,
                reserved_snap2_indices, protected_occluded_item_ids, true);
            printf("\033[1;32m[EVENT]\033[0m 整理: item#%d %s "
                   "(reid=%.2f evidence=%.2f)\n",
                   item_id, coco_cls_to_name(A.cls_id), best_reid, evidence_score);
            result.events.push_back({EventKind::MOVED, item_id, A.cls_id,
                                     B.box, A.box, B.box, B.best_score});
            result.happened = true;
        } else if (decision == RelocationDecision::LOW_CONFIDENCE) {
            bool exists = false;
            for (auto& p : pending_relocations_) {
                if (p.item_id_A == item_id) {
                    p.candidate_new_bbox = B.box;
                    p.snapshot_object_B = B;
                    p.reid_score = best_reid;
                    p.evidence_score = evidence_score;
                    p.last_checked_snapshot_id = snap2.frame_id;
                    exists = true;
                    break;
                }
            }
            if (!exists) {
                PendingRelocation p;
                p.pending_id = next_pending_id_++;
                p.item_id_A = item_id;
                p.class_id = A.cls_id;
                p.old_bbox = A.box;
                p.candidate_new_bbox = B.box;
                p.snapshot_object_B = B;
                p.reid_score = best_reid;
                p.evidence_score = evidence_score;
                p.created_snapshot_id = snap2.frame_id;
                p.last_checked_snapshot_id = snap2.frame_id;
                p.expire_after_stable_count = PENDING_RELOCATION_EXPIRE_FRAMES;
                pending_relocations_.push_back(p);
                printf("[SESSION] pending#%d: item#%d %s 可能整理 "
                       "(reid=%.2f evidence=%.2f)，本轮冻结\n",
                       p.pending_id, item_id, coco_cls_to_name(A.cls_id),
                       best_reid, evidence_score);
            }
            consumed_disappeared_indices.insert(a_idx);
            consumed_appeared_indices.insert(best_idx);
            reserved_snap2_indices.insert(best_idx);
            if (apply_relocation_visibility_changes(
                    item_id, A.box, B, snap2,
                    disappeared_indices, appeared_indices, baseline_item_ids,
                    consumed_disappeared_indices, consumed_appeared_indices,
                    reserved_snap2_indices, protected_occluded_item_ids, false)) {
                result.happened = true;
            }
        }
    }

    // ---------------------------------------------------------------------
    //  10.3 处理剩余消失物品
    // ---------------------------------------------------------------------
    for (int a_idx : disappeared_indices) {
        if (consumed_disappeared_indices.count(a_idx)) continue;

        const VotingItem& A = snap1_.items[a_idx];
        int item_id = find_inventory_item_strict(A.box, A.cls_id, false);
        if (item_id < 0) continue;

        const InventoryItem* item = inventory_.find_by_item(item_id);
        if (!item) continue;

        if (item->status == ItemStatus::OCCLUDED) {
            continue;
        }
        if (item->status != ItemStatus::VISIBLE) {
            continue;
        }

        bool found_reason = false;
        for (int c_idx : appeared_indices) {
            if (consumed_appeared_indices.count(c_idx)) continue;
            if (reserved_snap2_indices.count(c_idx)) continue;

            const VotingItem& C = snap2.items[c_idx];
            float nearby_dist = normalized_nearby_distance(A.box, C.box);
            if (nearby_dist >= NEARBY_DISTANCE_THRESH) continue;

            int inventory_c_id = -1;
            float best_c_dist = 999.0f;
            for (const auto& kv : inventory_.items()) {
                if (kv.first == item_id) continue;
                const InventoryItem& inv = kv.second;
                if (inv.status == ItemStatus::OUT) continue;
                bool matched_inventory_c =
                    inv.status == ItemStatus::OCCLUDED
                        ? relaxed_box_match(C.box, C.cls_id, inv.box, inv.cls_id)
                        : strict_box_match(C.box, C.cls_id, inv.box, inv.cls_id);
                if (!matched_inventory_c) continue;
                float d = normalized_nearby_distance(C.box, inv.box);
                if (d < best_c_dist) {
                    best_c_dist = d;
                    inventory_c_id = kv.first;
                }
            }
            const InventoryItem* inventory_c =
                inventory_c_id >= 0 ? inventory_.find_by_item(inventory_c_id) : nullptr;

            if (inventory_c && inventory_c->status == ItemStatus::OCCLUDED) {
                inventory_.set_status(item_id, ItemStatus::OUT, current_time_ms_);
                inventory_.set_status(inventory_c_id, ItemStatus::VISIBLE, current_time_ms_);
                inventory_.update_item(inventory_c_id, -1, C.box, C.best_score, snap2.frame_id);
                consumed_appeared_indices.insert(c_idx);
                printf("\033[1;32m[EVENT]\033[0m 取出: item#%d %s "
                       "(旧物品 item#%d 露出)\n",
                       item_id, coco_cls_to_name(A.cls_id), inventory_c_id);
                result.events.push_back({EventKind::OUT, item_id, A.cls_id,
                                         A.box, A.box, A.box, A.best_score});
                result.happened = true;
                found_reason = true;
                break;
            }

            if (inventory_c && inventory_c->status == ItemStatus::VISIBLE) {
                // A 被拿走，C 是原本就存在的物品
                inventory_.set_status(item_id, ItemStatus::OUT, current_time_ms_);
                printf("\033[1;32m[EVENT]\033[0m 取出: item#%d %s (附近有可见物品 item#%d)\n",
                       item_id, coco_cls_to_name(A.cls_id), inventory_c_id);
                result.events.push_back({EventKind::OUT, item_id, A.cls_id,
                                         A.box, A.box, A.box, A.best_score});
                result.happened = true;
                found_reason = true;
                break;
            }

            inventory_.set_status(item_id, ItemStatus::OCCLUDED, current_time_ms_);
            blocked_appeared_inventory_ids.insert(item_id);
            protected_occluded_item_ids.insert(item_id);
            printf("[SESSION] item#%d (%s) 可能被新物品遮挡，C 留给新物品流程处理\n",
                   item_id, coco_cls_to_name(A.cls_id));
            found_reason = true;
            break;
        }

        if (!found_reason) {
            inventory_.set_status(item_id, ItemStatus::OUT, current_time_ms_);
            printf("\033[1;32m[EVENT]\033[0m 取出: item#%d %s (位置空了)\n",
                   item_id, coco_cls_to_name(A.cls_id));
            result.events.push_back({EventKind::OUT, item_id, A.cls_id,
                                     A.box, A.box, A.box, A.best_score});
            result.happened = true;
        }
    }

    // ---------------------------------------------------------------------
    //  10.4 处理剩余新出现物品
    // ---------------------------------------------------------------------
    for (int b_idx : appeared_indices) {
        if (consumed_appeared_indices.count(b_idx)) continue;
        if (reserved_snap2_indices.count(b_idx)) continue;

        const VotingItem& B = snap2.items[b_idx];

        int local_id = find_inventory_item_strict(B.box, B.cls_id, false);
        if (blocked_appeared_inventory_ids.count(local_id)) {
            local_id = -1;
        }
        if (local_id >= 0) {
            const InventoryItem* item = inventory_.find_by_item(local_id);
            if (item && item->status == ItemStatus::OCCLUDED) {
                inventory_.set_status(local_id, ItemStatus::VISIBLE, current_time_ms_);
                printf("[SESSION] item#%d (%s) 从遮挡恢复为可见\n",
                       local_id, coco_cls_to_name(B.cls_id));
            }
            inventory_.update_item(local_id, -1, B.box, B.best_score, snap2.frame_id);
            continue;
        }

        int out_id = find_inventory_item_strict(B.box, B.cls_id, true);
        if (out_id >= 0) {
            const InventoryItem* out_item = inventory_.find_by_item(out_id);
            if (out_item && out_item->status == ItemStatus::OUT) {
                inventory_.set_status(out_id, ItemStatus::VISIBLE, current_time_ms_);
                inventory_.update_item(out_id, -1, B.box, B.best_score, snap2.frame_id);
                printf("[SESSION] item#%d (%s) 从出库恢复为可见\n",
                       out_id, coco_cls_to_name(B.cls_id));
                continue;
            }
        }

        int new_id = inventory_.add_item(-1, B.cls_id, B.box, B.best_score,
                                         snap2.frame_id, current_time_ms_);
        printf("\033[1;32m[EVENT]\033[0m 放入: item#%d %s (置信度 %.0f%%) "
               "位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
               new_id, coco_cls_to_name(B.cls_id), B.best_score * 100,
               B.box.x1, B.box.y1, B.box.x2, B.box.y2);
        result.events.push_back({EventKind::IN, new_id, B.cls_id,
                                 B.box, B.box, B.box, B.best_score});
        result.happened = true;
    }

    // ---------------------------------------------------------------------
    //  10.5 例行清理
    // ---------------------------------------------------------------------
    std::vector<int> out_to_remove;
    for (const auto& kv : inventory_.items()) {
        if (kv.second.status != ItemStatus::OUT) continue;
        for (const auto& kv2 : inventory_.items()) {
            if (kv2.second.status == ItemStatus::OUT) continue;
            if (strict_box_match(kv.second.box, kv.second.cls_id,
                                 kv2.second.box, kv2.second.cls_id)) {
                out_to_remove.push_back(kv.first);
                break;
            }
        }
    }
    for (int id : out_to_remove) {
        printf("[SESSION] 清理冗余 OUT item#%d\n", id);
        inventory_.remove_item(id);
    }

    std::vector<int> occluded_to_remove;
    for (const auto& kv : inventory_.items()) {
        if (kv.second.status != ItemStatus::OCCLUDED) continue;
        if (protected_occluded_item_ids.count(kv.first)) continue;
        for (const auto& kv2 : inventory_.items()) {
            if (kv2.second.status != ItemStatus::VISIBLE) continue;
            if (strict_box_match(kv.second.box, kv.second.cls_id,
                                 kv2.second.box, kv2.second.cls_id)) {
                occluded_to_remove.push_back(kv.first);
                break;
            }
        }
    }
    for (int id : occluded_to_remove) {
        printf("[SESSION] 清理冗余 OCCLUDED item#%d\n", id);
        inventory_.remove_item(id);
    }

    inventory_.cleanup_expired(current_time_ms_);
    return result;
}

// ============================================================================
//  打印库存
// ============================================================================

void SessionManager::print_inventory() {
    size_t n_visible = inventory_.count_by_status(ItemStatus::VISIBLE);
    size_t n_occluded = inventory_.count_by_status(ItemStatus::OCCLUDED);
    size_t n_out = inventory_.count_by_status(ItemStatus::OUT);
    size_t n_in = n_visible + n_occluded;

    printf("\n");
    printf("  ┌──────────────────────────────────────────────────┐\n");
    printf("  │  在库清单 │ 可见: %-3zu │ 遮挡: %-3zu │ 共: %-3zu    │\n",
           n_visible, n_occluded, n_in);
    printf("  ├────┬──────────────┬────────┬───────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 位置 (中心)           │\n");
    printf("  ├────┼──────────────┼────────┼───────────────────────┤\n");

    for (const auto& kv : inventory_.items()) {
        const auto& it = kv.second;
        if (it.status == ItemStatus::OUT) continue;
        const char* status_str = (it.status == ItemStatus::VISIBLE) ? "可见" : "遮挡";
        printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)          │\n",
               it.item_id, coco_cls_to_name(it.cls_id), status_str,
               it.box.cx(), it.box.cy());
    }

    printf("  └────┴──────────────┴────────┴───────────────────────┘\n");

    if (n_out > 0) {
        printf("\n");
        printf("  ┌──────────────────────────────────────────────────┐\n");
        printf("  │  出库记录 │ 共: %-3zu                               │\n", n_out);
        printf("  ├────┬──────────────┬───────────────────────────────┤\n");
        printf("  │ #  │ 类别         │ 原位置 (中心)                 │\n");
        printf("  ├────┼──────────────┼───────────────────────────────┤\n");

        for (const auto& kv : inventory_.items()) {
            const auto& it = kv.second;
            if (it.status != ItemStatus::OUT) continue;
            printf("  │ %-2d │ %-12s │ (%4.0f,%4.0f)                │\n",
                   it.item_id, coco_cls_to_name(it.cls_id),
                   it.box.cx(), it.box.cy());
        }

        printf("  └────┴──────────────┴───────────────────────────────┘\n");
    }
    printf("\n");
}

}  // namespace fridge
