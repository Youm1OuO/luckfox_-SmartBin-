// ============================================================================
//  session.cc
//  会话管理器 — 新业务流程5：纯快照对比
//    - 时间段划分只靠快照对比（不依赖手识别的速度）
//    - 手识别保留用于 HELD 和整理（手快就用，手慢不影响主流程）
//    - 物品状态：VISIBLE（可见）、OCCLUDED（遮挡）、TAKEN（拿走）
//    - 先减后加：先处理消失的物品（拿走/遮挡），再处理新增的物品（放入）
//    - 身份匹配：严格（5条件）用于状态变更，宽松（2条件）用于N帧计数
//    - 一个时间段至少维持 MIN_SEGMENT_SNAPSHOTS 个微快照
// ============================================================================
#include "session.h"
#include "fridge_config.h"
#include "yolov5.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <set>
#include <opencv2/imgproc.hpp>

namespace fridge {

SessionManager::SessionManager(int segment_frames)
    : baseline_initialized_(false),
      current_state_(SystemState::STABLE),
      backend_initialized_(false),
      phase_(SessionPhase::IDLE),
      has_prev_snapshot_(false),
      no_hand_streak_(0),
      segment_frames_(segment_frames),
      last_hand_frame_(-1000),
      current_time_ms_(0),
      snapshot_count_(0),
      n_frame_counter_(0),
      has_last_n_snapshot_(false) {
    current_segment_ = {0, 0, {}, false};
    prev_snapshot_ = {0, 0, {}, false};
    last_n_snapshot_ = {0, 0, {}, false};
}

void SessionManager::init_from_backend(const std::vector<InventoryItem>& backend_items,
                                        const cv::Mat& frame) {
    backend_inventory_ = backend_items;
    backend_initialized_ = true;
    printf("[SESSION] backend inventory received: %zu items\n", backend_items.size());
}


// ============================================================================
//  辅助函数
// ============================================================================

static BBox expand_hand_to_arm(const BBox& hand, float W, float H) {
    float cx = hand.cx(), cy = hand.cy();
    float d_left = cx, d_right = W - cx, d_top = cy, d_bottom = H - cy;
    float dmin = std::min(std::min(d_left, d_right), std::min(d_top, d_bottom));
    BBox arm = hand;
    if (dmin == d_top) arm.y1 = 0;
    else if (dmin == d_bottom) arm.y2 = H;
    else if (dmin == d_left) arm.x1 = 0;
    else arm.x2 = W;
    return arm;
}

static bool hand_covers(const Track* h, const BBox& box, float W, float H) {
    bool center_in = point_in_box(box.cx(), box.cy(), h->box);
    bool covered = overlap_ratio_of_smaller(h->box, box) >= 0.5f;
    if (center_in || covered) return true;
    BBox arm = expand_hand_to_arm(h->box, W, H);
    return point_in_box(box.cx(), box.cy(), arm) || overlap_ratio_of_smaller(arm, box) >= 0.5f;
}

float SessionManager::color_diff_crop(const cv::Mat& frame, const BBox& a, const BBox& b) {
    int ax1 = std::max(0, (int)a.x1), ay1 = std::max(0, (int)a.y1);
    int ax2 = std::min(frame.cols, (int)a.x2), ay2 = std::min(frame.rows, (int)a.y2);
    int bx1 = std::max(0, (int)b.x1), by1 = std::max(0, (int)b.y1);
    int bx2 = std::min(frame.cols, (int)b.x2), by2 = std::min(frame.rows, (int)b.y2);
    if (ax2 <= ax1 || ay2 <= ay1 || bx2 <= bx1 || by2 <= by1) return 999.0f;
    cv::Mat a_small, b_small;
    cv::resize(frame(cv::Rect(ax1, ay1, ax2-ax1, ay2-ay1)), a_small, cv::Size(16, 16));
    cv::resize(frame(cv::Rect(bx1, by1, bx2-bx1, by2-by1)), b_small, cv::Size(16, 16));
    cv::Scalar ma = cv::mean(a_small), mb = cv::mean(b_small);
    return (std::abs(ma[0]-mb[0]) + std::abs(ma[1]-mb[1]) + std::abs(ma[2]-mb[2])) / 3.0f;
}

bool SessionManager::is_same_item_strict(const BBox& box_a, int cls_a,
                                          const BBox& box_b, int cls_b,
                                          const cv::Mat& frame) {
    // 严格身份匹配（5条件）：类别 + 中心距离 + IoU + 面积比 + 颜色
    if (cls_a != cls_b) return false;
    if (center_distance(box_a, box_b) >= IDENTITY_CENTER_DIST) return false;
    if (iou(box_a, box_b) < IDENTITY_IOU_THRESH) return false;
    if (area_ratio(box_a, box_b) < IDENTITY_AREA_RATIO) return false;
    if (!frame.empty() && color_diff_crop(frame, box_a, box_b) >= IDENTITY_COLOR_DIFF) return false;
    return true;
}

// 宽松身份匹配（2条件）：只需类别相同 + 位置大致相同
// 用途：N帧计数中的帧间匹配、方法B的快照对比
bool SessionManager::is_same_item_loose(const BBox& box_a, int cls_a,
                                         const BBox& box_b, int cls_b) {
    if (cls_a != cls_b) return false;
    float dist_thresh = loose_dist_thresh(box_a);
    return center_distance(box_a, box_b) < dist_thresh;
}

float SessionManager::loose_dist_thresh(const BBox& box) {
    return bbox_diagonal(box) * IDENTITY_LOOSE_DIAG_RATIO;
}


// ============================================================================
//  时间段聚合
// ============================================================================
void SessionManager::aggregate_to_segment(const std::vector<const Track*>& foods, int frame_id) {
    if (!current_segment_.valid) {
        current_segment_.start_frame = frame_id;
        current_segment_.valid = true;
    }
    current_segment_.end_frame = frame_id;

    for (const Track* t : foods) {
        auto it = current_segment_.items.find(t->track_id);
        if (it == current_segment_.items.end()) {
            SegmentItem si;
            si.track_id = t->track_id;
            si.cls_id = t->cls_id;
            si.box = t->box;
            si.best_score = t->score;
            si.seen_frames = 1;
            current_segment_.items[t->track_id] = si;
        } else {
            it->second.seen_frames++;
            if (t->score > it->second.best_score) {
                it->second.best_score = t->score;
                it->second.box = t->box;
            }
        }
    }
}


// ============================================================================
//  检测 HELD 物品
// ============================================================================
void SessionManager::detect_held_items(const std::vector<const Track*>& hands,
                                        const std::vector<const Track*>& foods,
                                        int frame_id) {
    std::set<int> seen_tids;
    for (const Track* t : foods) seen_tids.insert(t->track_id);

    for (const auto& kv : inventory_.items()) {
        if (kv.second.status == ItemStatus::TAKEN) continue;
        if (held_items_.count(kv.first)) continue;
        if (kv.second.track_id >= 0 && seen_tids.count(kv.second.track_id)) continue;

        int cover_hand = -1;
        for (const Track* h : hands) {
            if (hand_covers(h, kv.second.box, FRAME_W, FRAME_H)) {
                cover_hand = h->track_id;
                break;
            }
        }
        if (cover_hand >= 0) {
            HeldItem hi;
            hi.item_id = kv.first;
            hi.cls_id = kv.second.cls_id;
            hi.original_pos = kv.second.box;
            hi.score = kv.second.score;
            hi.hand_track_id = cover_hand;
            hi.pickup_frame = frame_id;
            held_items_[kv.first] = hi;
            // 注意：不改变物品的 VISIBLE/OCCLUDED 状态，只在 held_items_ 中记录
            printf("\033[1;33m[DBG]\033[0m item#%d (%s) 被手盖住 → HELD\n",
                   kv.first, coco_cls_to_name(kv.second.cls_id));
        }
    }
}


// ============================================================================
//  处理 HELD 物品（整理 + 取出确认）
// ============================================================================
SettlementResult SessionManager::process_held_items(
        const std::vector<const Track*>& hands,
        const std::vector<const Track*>& foods,
        bool hand_effective,
        int frame_id,
        const cv::Mat& frame) {
    SettlementResult res;
    bool has_frame = !frame.empty();
    std::set<int> consumed_tids;
    std::vector<int> held_done;

    for (auto& kv : held_items_) {
        int item_id = kv.first;
        HeldItem& hi = kv.second;
        InventoryItem* item = inventory_.find_by_item(item_id);
        if (!item) { held_done.push_back(item_id); continue; }

        // 4a) 物品在原位置附近重新出现 → 没被拿走
        const Track* reappear = nullptr;
        for (const Track* f : foods) {
            if (f->cls_id != hi.cls_id || consumed_tids.count(f->track_id)) continue;
            if (has_frame) {
                if (is_same_item_strict(f->box, f->cls_id, hi.original_pos, hi.cls_id, frame))
                    { reappear = f; break; }
            } else {
                if (center_distance(f->box, hi.original_pos) < SMOOTH_RELOCATE_PIX)
                    { reappear = f; break; }
            }
        }
        if (reappear) {
            inventory_.relocate_item(item_id, reappear->track_id, reappear->box,
                                     reappear->score, frame_id);
            inventory_.set_status(item_id, ItemStatus::VISIBLE);  // 恢复为可见
            consumed_tids.insert(reappear->track_id);
            held_done.push_back(item_id);
            printf("\033[1;33m[DBG]\033[0m item#%d 在原位附近重现 → 取消 HELD\n", item_id);
            continue;
        }

        // 4b) 手还在 → 继续 held
        if (hand_effective) {
            const Track* nearest = nullptr;
            float best_d = 1e9f;
            for (const Track* h : hands) {
                float d = center_distance(h->box, item->box);
                if (d < best_d) { best_d = d; nearest = h; }
            }
            if (nearest) {
                item->box = nearest->box;
                item->last_seen_frame = frame_id;
            }
            continue;
        }

        // 4c) 同类新 track 远处出现 → 整理
        const Track* putdown = nullptr;
        for (const Track* f : foods) {
            if (f->cls_id != hi.cls_id || consumed_tids.count(f->track_id)) continue;
            if (inventory_.find_by_track(f->track_id) != nullptr) continue;
            if (center_distance(f->box, hi.original_pos) >= SMOOTH_RELOCATE_PIX) {
                putdown = f; break;
            }
        }
        if (putdown) {
            printf("\n\033[1;32m[EVENT]\033[0m 整理: item#%d %s "
                   "从(%.0f,%.0f)~(%.0f,%.0f) → (%.0f,%.0f)~(%.0f,%.0f)\n",
                   item_id, coco_cls_to_name(hi.cls_id),
                   hi.original_pos.x1, hi.original_pos.y1, hi.original_pos.x2, hi.original_pos.y2,
                   putdown->box.x1, putdown->box.y1, putdown->box.x2, putdown->box.y2);
            inventory_.relocate_item(item_id, putdown->track_id, putdown->box,
                                     putdown->score, frame_id);
            inventory_.set_status(item_id, ItemStatus::VISIBLE);
            consumed_tids.insert(putdown->track_id);
            held_done.push_back(item_id);
            res.happened = true;
            res.events.push_back({EventKind::MOVED, item_id, hi.cls_id,
                                  putdown->box, putdown->score, true});
            continue;
        }

        // 4d) 手不在 + 没重现 + 没放下 → 恢复为可见，交给快照对比判断是否真的被拿走
        //     不直接判 TAKEN，因为物品可能只是被其他物品遮挡了
        inventory_.set_status(item_id, ItemStatus::VISIBLE);
        held_done.push_back(item_id);
        printf("\033[1;33m[DBG]\033[0m item#%d (%s) 手离开，恢复为可见（等快照对比确认）\n",
               item_id, coco_cls_to_name(item->cls_id));
    }

    for (int id : held_done) held_items_.erase(id);
    return res;
}


// ============================================================================
//  N帧阈值过滤：移除时间段中出现次数低于阈值的物品
// ============================================================================
void SessionManager::filter_segment_by_threshold(TimeSegment& seg) {
    if (!seg.valid) return;
    int total_frames = seg.end_frame - seg.start_frame + 1;
    if (total_frames <= 0) return;
    int min_seen = (int)(total_frames * N_SNAPSHOT_THRESHOLD);
    if (min_seen < 1) min_seen = 1;

    for (auto it = seg.items.begin(); it != seg.items.end(); ) {
        if (it->second.seen_frames < min_seen) {
            it = seg.items.erase(it);
        } else {
            ++it;
        }
    }
}


// ============================================================================
//  方法B：检查N帧微快照变化，触发时间段边界
// ============================================================================
bool SessionManager::check_method_b_boundary(
        const std::vector<const Track*>& foods, int frame_id) {
    // 将当前帧的食物检测结果聚合到N帧计数表中
    for (const Track* t : foods) {
        bool found = false;
        for (auto& si : n_frame_items_) {
            if (is_same_item_loose(t->box, t->cls_id, si.box, si.cls_id)) {
                si.seen_frames++;
                if (t->score > si.best_score) {
                    si.best_score = t->score;
                    si.box = t->box;
                }
                found = true;
                break;
            }
        }
        if (!found) {
            SegmentItem si;
            si.track_id = -1;
            si.cls_id = t->cls_id;
            si.box = t->box;
            si.best_score = t->score;
            si.seen_frames = 1;
            n_frame_items_.push_back(si);
        }
    }
    n_frame_counter_++;

    // N帧还没收集完
    if (n_frame_counter_ < N_SNAPSHOT_FRAMES) return false;

    // N帧结束，构建微快照
    int min_count = (int)(N_SNAPSHOT_FRAMES * N_SNAPSHOT_THRESHOLD);
    if (min_count < 1) min_count = 1;

    TimeSegment n_snap;
    n_snap.start_frame = frame_id - N_SNAPSHOT_FRAMES + 1;
    n_snap.end_frame = frame_id;
    n_snap.valid = true;
    int key = 0;
    for (const auto& si : n_frame_items_) {
        if (si.seen_frames >= min_count) {
            n_snap.items[key++] = si;
        }
    }

    // 与上一个N帧微快照比较（用宽松匹配）
    bool boundary = false;
    if (has_last_n_snapshot_) {
        size_t prev_count = last_n_snapshot_.items.size();
        size_t curr_count = n_snap.items.size();

        // 数量变化 → 触发
        if (curr_count != prev_count) {
            boundary = true;
        } else {
            // 数量相同，检查每个物品是否都匹配上
            std::set<int> matched_prev;
            for (const auto& kv_curr : n_snap.items) {
                bool matched = false;
                for (const auto& kv_prev : last_n_snapshot_.items) {
                    if (matched_prev.count(kv_prev.first)) continue;
                    if (is_same_item_loose(kv_curr.second.box, kv_curr.second.cls_id,
                                            kv_prev.second.box, kv_prev.second.cls_id)) {
                        matched_prev.insert(kv_prev.first);
                        matched = true;
                        break;
                    }
                }
                if (!matched) {
                    boundary = true;
                    break;
                }
            }
        }
    }

    // 更新状态
    n_frame_counter_ = 0;
    n_frame_items_.clear();
    last_n_snapshot_ = n_snap;
    has_last_n_snapshot_ = true;

    return boundary;
}


// ============================================================================
//  三步处理：时间段快照对比
//    第一步：先处理减少（消失的物品 → 拿走或遮挡）
//    第二步：处理增加（新出现的物品 → 放入或遮挡恢复）
//    第三步：例行检查（状态一致性校验）
// ============================================================================
SettlementResult SessionManager::process_segment_comparison(
        const TimeSegment& new_seg,
        int frame_id,
        const cv::Mat& frame) {
    SettlementResult res;
    bool has_frame = !frame.empty();

    // ==== 第一步：找消失的物品（先处理减少） ====
    // 遍历库存中所有非TAKEN的物品，看是否在新快照中出现
    std::vector<int> disappeared_items;
    for (const auto& inv_kv : inventory_.items()) {
        if (inv_kv.second.status == ItemStatus::TAKEN) continue;

        // 在新快照中用严格匹配找
        bool found_in_snapshot = false;
        for (const auto& kv : new_seg.items) {
            if (has_frame) {
                if (is_same_item_strict(kv.second.box, kv.second.cls_id,
                                        inv_kv.second.box, inv_kv.second.cls_id, frame)) {
                    found_in_snapshot = true;
                    break;
                }
            } else {
                if (is_same_position(kv.second.box, inv_kv.second.box,
                                     IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH)) {
                    found_in_snapshot = true;
                    break;
                }
            }
        }
        if (found_in_snapshot) continue;

        // 没找到 → 消失了，检查附近有没有新物品C
        disappeared_items.push_back(inv_kv.first);
    }

    for (int item_id : disappeared_items) {
        auto* item = inventory_.find_by_item(item_id);
        if (!item || item->status == ItemStatus::TAKEN) continue;

        bool handled = false;

        // 检查新快照中的物品是否在A的原位置附近
        for (const auto& kv : new_seg.items) {
            float overlap = overlap_ratio_of_smaller(item->box, kv.second.box);
            if (overlap <= NEARBY_OVERLAP_THRESH) continue;

            // C在A的原位置附近
            // 检查C是否在库存中（且是OCCLUDED状态）→ C是旧物品露出来
            bool c_in_inventory = false;
            for (auto& inv_kv2 : inventory_.items()) {
                if (inv_kv2.first == item_id) continue;
                if (inv_kv2.second.status != ItemStatus::OCCLUDED) continue;
                if (has_frame) {
                    if (is_same_item_strict(kv.second.box, kv.second.cls_id,
                                            inv_kv2.second.box, inv_kv2.second.cls_id, frame)) {
                        c_in_inventory = true;
                        // C是旧物品露出来，A确实被拿走
                        printf("\n\033[1;32m[EVENT]\033[0m 取出: item#%d %s "
                               "(置信度 %.0f%%) 原位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
                               item_id, coco_cls_to_name(item->cls_id), item->score * 100,
                               item->box.x1, item->box.y1, item->box.x2, item->box.y2);
                        inventory_.set_status(item_id, ItemStatus::TAKEN);
                        inventory_.set_status(inv_kv2.first, ItemStatus::VISIBLE);
                        res.events.push_back({EventKind::OUT, item_id, item->cls_id,
                                               item->box, item->score, false});
                        res.happened = true;
                        handled = true;
                        break;
                    }
                }
            }
            if (handled) break;

            if (!c_in_inventory) {
                // C不在库存中 → C是新物品压住了A，A变OCCLUDED
                printf("\033[1;33m[DBG]\033[0m item#%d (%s) 被新物品 (cls=%s, 位置=%.0f,%.0f) 压住 → 遮挡\n",
                       item_id, coco_cls_to_name(item->cls_id),
                       coco_cls_to_name(kv.second.cls_id), kv.second.box.cx(), kv.second.box.cy());
                inventory_.set_status(item_id, ItemStatus::OCCLUDED);
                res.happened = true;
                handled = true;
                break;
            }
        }

        if (!handled) {
            // 附近没有新物品 → A被拿走
            printf("\n\033[1;32m[EVENT]\033[0m 取出: item#%d %s "
                   "(置信度 %.0f%%) 原位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
                   item_id, coco_cls_to_name(item->cls_id), item->score * 100,
                   item->box.x1, item->box.y1, item->box.x2, item->box.y2);
            inventory_.set_status(item_id, ItemStatus::TAKEN);
            res.events.push_back({EventKind::OUT, item_id, item->cls_id,
                                   item->box, item->score, false});
            res.happened = true;
        }
    }

    // ==== 第二步：处理增加（后处理增加） ====
    for (const auto& kv : new_seg.items) {
        // 先跟TAKEN状态的物品做严格身份匹配
        bool found_taken = false;
        for (auto& inv_kv : inventory_.items()) {
            if (inv_kv.second.status != ItemStatus::TAKEN) continue;
            if (has_frame) {
                if (!is_same_item_strict(kv.second.box, kv.second.cls_id,
                                         inv_kv.second.box, inv_kv.second.cls_id, frame)) continue;
            } else {
                if (!is_same_position(kv.second.box, inv_kv.second.box,
                                      IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH)) continue;
            }
            // 匹配上 → TAKEN恢复为VISIBLE
            inventory_.relocate_item(inv_kv.first, kv.second.track_id,
                                     kv.second.box, kv.second.best_score, frame_id);
            inventory_.set_status(inv_kv.first, ItemStatus::VISIBLE);
            printf("\033[1;33m[DBG]\033[0m item#%d TAKEN 恢复 → 可见\n", inv_kv.first);
            res.happened = true;
            found_taken = true;
            break;
        }
        if (found_taken) continue;

        // 跟库存做严格身份匹配
        bool found = false;
        for (auto& inv_kv : inventory_.items()) {
            if (inv_kv.second.status == ItemStatus::TAKEN) continue;
            bool matched;
            if (has_frame) {
                matched = is_same_item_strict(kv.second.box, kv.second.cls_id,
                                              inv_kv.second.box, inv_kv.second.cls_id, frame);
            } else {
                matched = is_same_position(kv.second.box, inv_kv.second.box,
                                           IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH);
            }
            if (!matched) continue;

            if (inv_kv.second.status == ItemStatus::OCCLUDED) {
                // 原先是OCCLUDED → 恢复为VISIBLE（遮挡恢复）
                inventory_.relocate_item(inv_kv.first, kv.second.track_id,
                                         kv.second.box, kv.second.best_score, frame_id);
                inventory_.set_status(inv_kv.first, ItemStatus::VISIBLE);
                printf("\033[1;33m[DBG]\033[0m item#%d 遮挡恢复 → 可见\n", inv_kv.first);
                res.happened = true;
            } else {
                // 已经是VISIBLE → 更新位置
                inventory_.relocate_item(inv_kv.first, kv.second.track_id,
                                         kv.second.box, kv.second.best_score, frame_id);
            }
            found = true;
            break;
        }
        if (!found) {
            // 新物品入库
            int new_id = inventory_.add_item(kv.second.track_id, kv.second.cls_id,
                                             kv.second.box, kv.second.best_score,
                                             frame_id, current_time_ms_);
            printf("\n\033[1;32m[EVENT]\033[0m 放入: item#%d %s (置信度 %.0f%%) "
                   "位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
                   new_id, coco_cls_to_name(kv.second.cls_id), kv.second.best_score * 100,
                   kv.second.box.x1, kv.second.box.y1, kv.second.box.x2, kv.second.box.y2);
            res.events.push_back({EventKind::IN, new_id, kv.second.cls_id,
                                   kv.second.box, kv.second.best_score, true});
            res.happened = true;
        }
    }

    // ==== 第三步：例行检查（状态一致性校验） ====
    // 检查TAKEN物品是否与库存中的VISIBLE/OCCLUDED物品重复
    std::vector<int> taken_to_remove;
    for (const auto& kv : new_seg.items) {
        for (auto& inv_kv : inventory_.items()) {
            if (inv_kv.second.status != ItemStatus::TAKEN) continue;
            // 检查是否与库存中其他非TAKEN物品匹配
            for (auto& inv_kv2 : inventory_.items()) {
                if (inv_kv2.second.status == ItemStatus::TAKEN) continue;
                if (has_frame) {
                    if (!is_same_item_strict(inv_kv.second.box, inv_kv.second.cls_id,
                                             inv_kv2.second.box, inv_kv2.second.cls_id, frame)) continue;
                } else {
                    if (!is_same_position(inv_kv.second.box, inv_kv2.second.box,
                                          IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH)) continue;
                }
                // TAKEN物品与库存中物品重复 → 删除TAKEN记录
                taken_to_remove.push_back(inv_kv.first);
                printf("\033[1;33m[DBG]\033[0m item#%d TAKEN与库存中item#%d重复 → 删除TAKEN记录\n",
                       inv_kv.first, inv_kv2.first);
                break;
            }
        }
    }
    for (int id : taken_to_remove) inventory_.remove_item(id);

    // 检查OCCLUDED物品是否与VISIBLE物品重复
    std::vector<int> occluded_to_remove;
    for (auto& inv_kv : inventory_.items()) {
        if (inv_kv.second.status != ItemStatus::OCCLUDED) continue;
        for (auto& inv_kv2 : inventory_.items()) {
            if (inv_kv2.first == inv_kv.first) continue;
            if (inv_kv2.second.status != ItemStatus::VISIBLE) continue;
            if (has_frame) {
                if (!is_same_item_strict(inv_kv.second.box, inv_kv.second.cls_id,
                                         inv_kv2.second.box, inv_kv2.second.cls_id, frame)) continue;
            } else {
                if (!is_same_position(inv_kv.second.box, inv_kv2.second.box,
                                      IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH)) continue;
            }
            occluded_to_remove.push_back(inv_kv.first);
            printf("\033[1;33m[DBG]\033[0m item#%d OCCLUDED与VISIBLE的item#%d重复 → 删除OCCLUDED记录\n",
                   inv_kv.first, inv_kv2.first);
            break;
        }
    }
    for (int id : occluded_to_remove) inventory_.remove_item(id);

    return res;
}


// ============================================================================
//  打印库存（表格格式）
// ============================================================================
void SessionManager::print_inventory() {
    size_t n_total = inventory_.size();
    size_t n_taken = inventory_.count_by_status(ItemStatus::TAKEN);
    size_t n_visible = inventory_.count_by_status(ItemStatus::VISIBLE);
    size_t n_occluded = inventory_.count_by_status(ItemStatus::OCCLUDED);
    size_t n_in = n_visible + n_occluded;

    printf("\n");
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  库存清单 │ 可见: %-3zu │ 遮挡: %-3zu │ 拿走: %-3zu │ 总计: %-3zu │\n",
           n_visible, n_occluded, n_taken, n_total);
    printf("  ├────┬──────────────┬────────┬────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 位置 (中心)        │\n");
    printf("  ├────┼──────────────┼────────┼────────────────────┤\n");

    // 打印可见物品
    for (const auto& kv : inventory_.items()) {
        const auto& it = kv.second;
        if (it.status != ItemStatus::VISIBLE) continue;
        printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)         │\n",
               it.item_id, coco_cls_to_name(it.cls_id), "可见",
               it.box.cx(), it.box.cy());
    }

    // 打印遮挡物品
    for (const auto& kv : inventory_.items()) {
        if (kv.second.status != ItemStatus::OCCLUDED) continue;
        printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)         │\n",
               kv.second.item_id, coco_cls_to_name(kv.second.cls_id), "遮挡",
               kv.second.box.cx(), kv.second.box.cy());
    }

    // 分隔线 + TAKEN 物品
    if (n_taken > 0) {
        printf("  ├────┴──────────────┴────────┴────────────────────┤\n");
        printf("  │  被拿走（临时记录，关门时不上传后台）            │\n");
        printf("  ├────┬──────────────┬────────┬────────────────────┤\n");
        for (const auto& kv : inventory_.items()) {
            if (kv.second.status != ItemStatus::TAKEN) continue;
            printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)         │\n",
                   kv.second.item_id, coco_cls_to_name(kv.second.cls_id), "拿走",
                   kv.second.box.cx(), kv.second.box.cy());
        }
    }

    printf("  └────┴──────────────┴────────┴────────────────────┘\n\n");
}


// ============================================================================
//  旧接口
// ============================================================================
SettlementResult SessionManager::update(const std::vector<Track>& tracks, int frame_id) {
    cv::Mat empty;
    return update(tracks, frame_id, empty, 0);
}


// ============================================================================
//  主 update 接口 — 新业务流程5：纯快照对比
//    - 只有两个阶段：IDLE（初始化）和 COLLECTING（收集+对比+HELD）
//    - 时间段划分只靠快照对比，不依赖手识别
//    - 手在时做 HELD 和整理，但不影响时间段划分
// ============================================================================
SettlementResult SessionManager::update(const std::vector<Track>& tracks,
                                        int frame_id,
                                        const cv::Mat& frame,
                                        long long time_ms) {
    current_time_ms_ = time_ms;
    SettlementResult total_res;
    bool has_frame = !frame.empty();

    // ---- 分类 tracks ----
    std::vector<const Track*> hands, foods;
    for (const auto& t : tracks) {
        if (is_hand(t.cls_id)) hands.push_back(&t);
        else if (is_food(t.cls_id) && t.score >= SNAPSHOT_MIN_SCORE) foods.push_back(&t);
    }
    bool hand_present = !hands.empty();
    if (hand_present) last_hand_frame_ = frame_id;
    bool hand_effective = hand_present || (frame_id - last_hand_frame_ <= HAND_INERTIA_FRAMES);
    current_state_ = hand_present ? SystemState::DISTURBED : SystemState::STABLE;


    // ================================================================
    //  阶段 1：IDLE → 收集初始快照，够了就初始化
    // ================================================================
    if (phase_ == SessionPhase::IDLE) {
        aggregate_to_segment(foods, frame_id);

        // 每N帧检查一次快照边界（微快照变化）
        bool snapshot_boundary = check_method_b_boundary(foods, frame_id);
        if (snapshot_boundary) {
            snapshot_count_++;
        }

        // 收集够了 → 初始化基线
        if (snapshot_count_ >= MIN_SEGMENT_SNAPSHOTS) {
            prev_snapshot_ = current_segment_;
            prev_snapshot_.end_frame = frame_id;
            has_prev_snapshot_ = true;

            if (backend_initialized_) {
                for (const auto& bi : backend_inventory_) {
                    bool matched = false;
                    for (const auto& kv : prev_snapshot_.items) {
                        bool m = has_frame
                            ? is_same_item_strict(kv.second.box, kv.second.cls_id, bi.box, bi.cls_id, frame)
                            : is_same_position(kv.second.box, bi.box, IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH);
                        if (m) {
                            int new_id = inventory_.add_item(kv.second.track_id, kv.second.cls_id,
                                                             kv.second.box, kv.second.best_score,
                                                             frame_id, current_time_ms_);
                            inventory_.set_status(new_id, ItemStatus::VISIBLE);
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        int new_id = inventory_.add_item(-1, bi.cls_id, bi.box, bi.score,
                                                         frame_id, current_time_ms_);
                        inventory_.set_status(new_id, ItemStatus::OCCLUDED);
                    }
                }
            } else {
                for (const auto& kv : prev_snapshot_.items) {
                    inventory_.add_item(kv.second.track_id, kv.second.cls_id,
                                        kv.second.box, kv.second.best_score,
                                        frame_id, current_time_ms_);
                }
            }
            baseline_initialized_ = true;
            printf("[SESSION] 初始化完成: %zu 件 (快照数=%d, 帧=%d)\n",
                   inventory_.size(), snapshot_count_, frame_id);
            total_res.happened = true;

            // 进入 COLLECTING
            phase_ = SessionPhase::COLLECTING;
            current_segment_ = {frame_id, frame_id, {}, false};
            snapshot_count_ = 0;
            n_frame_counter_ = 0;
            n_frame_items_.clear();
            has_last_n_snapshot_ = false;

            // 如果手正好在，做一次 HELD 检测
            if (hand_present) {
                detect_held_items(hands, foods, frame_id);
            }
        }
        return total_res;
    }


    // ================================================================
    //  阶段 2：COLLECTING → 收集快照 + HELD处理 + 快照对比
    //    手在时也继续收集（不切换阶段），HELD 在内部处理
    // ================================================================
    if (phase_ == SessionPhase::COLLECTING) {
        // ---- 始终聚合食物检测到时间段 ----
        aggregate_to_segment(foods, frame_id);

        // ---- 更新库存中物品的位置 ----
        for (const Track* t : foods) {
            InventoryItem* item = inventory_.find_by_track(t->track_id);
            if (!item || item->status == ItemStatus::TAKEN) continue;
            item->last_bbox = item->box;
            item->box = t->box;
            item->score = t->score;
            item->last_seen_frame = frame_id;
            item->stable_frames++;
        }

        // ---- HELD 处理（手在时） ----
        if (hand_present) {
            // 检测新的 HELD 物品
            detect_held_items(hands, foods, frame_id);
            // 处理已有的 HELD 物品
            SettlementResult held_res = process_held_items(hands, foods, hand_effective, frame_id, frame);
            if (held_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        held_res.events.begin(), held_res.events.end());
            }
        } else if (hand_effective) {
            // 手刚离开（惯性期内）→ 处理剩余 HELD
            SettlementResult held_res = process_held_items(hands, foods, hand_effective, frame_id, frame);
            if (held_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        held_res.events.begin(), held_res.events.end());
            }
        }

        // ---- 图片更新检查 ----
        for (const Track* t : foods) {
            InventoryItem* item = inventory_.find_by_track(t->track_id);
            if (!item || item->status != ItemStatus::VISIBLE) continue;
            if (image_updated_.count(item->item_id)) continue;
            if (frame_id - item->created_frame < IMAGE_UPDATE_DELAY_FRAMES) continue;
            auto lb = item_last_box_.find(item->item_id);
            if (lb != item_last_box_.end() &&
                center_distance(t->box, lb->second) >= SMOOTH_SETTLE_PIX) {
                item_last_box_[item->item_id] = t->box;
                continue;
            }
            image_updated_.insert(item->item_id);
            printf("\033[1;33m[DBG]\033[0m item#%d (%s) 稳定截图更新 @frame=%d\n",
                   item->item_id, coco_cls_to_name(item->cls_id), frame_id);
            total_res.happened = true;
            total_res.events.push_back({EventKind::IMAGE_UPDATE, item->item_id,
                                        item->cls_id, item->box, item->score, true});
            item_last_box_[item->item_id] = t->box;
        }

        // ---- 快照边界检测（时间段划分的核心） ----
        bool snapshot_boundary = check_method_b_boundary(foods, frame_id);
        if (snapshot_boundary) {
            snapshot_count_++;
            printf("\033[1;36m[DBG]\033[0m 快照边界 @frame=%d (snapshot_count=%d)\n",
                   frame_id, snapshot_count_);
        }

        // ---- 时间段结束条件：快照变化 + 最小持续时间 ----
        if (snapshot_boundary && snapshot_count_ >= MIN_SEGMENT_SNAPSHOTS) {
            // 时间段结束，做快照对比
            TimeSegment new_snapshot = current_segment_;
            new_snapshot.end_frame = frame_id;

            // N帧阈值过滤
            filter_segment_by_threshold(new_snapshot);

            // 三步处理：先减后加 + 例行检查
            SettlementResult seg_res = process_segment_comparison(
                new_snapshot, frame_id, frame);
            if (seg_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        seg_res.events.begin(), seg_res.events.end());
            }

            // 更新快照，重新开始收集
            prev_snapshot_ = new_snapshot;
            has_prev_snapshot_ = true;
            current_segment_ = {frame_id, frame_id, {}, false};
            snapshot_count_ = 0;
            // 保持 COLLECTING 状态

            if (total_res.happened) {
                print_inventory();
            }
        }

        // ---- HELD 处理后打印库存 ----
        if (total_res.happened && !snapshot_boundary) {
            print_inventory();
        }

        return total_res;
    }


    // ================================================================
    //  兜底
    // ================================================================
    if (total_res.happened) {
        print_inventory();
    }

    return total_res;
}

}  // namespace fridge
