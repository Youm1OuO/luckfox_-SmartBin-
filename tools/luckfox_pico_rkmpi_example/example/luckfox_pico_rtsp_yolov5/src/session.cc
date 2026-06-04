// ============================================================================
//  session.cc
//  会话管理器 — 新业务流程4：时间段快照对比
//    - 不需要【被遮挡】状态：物品只要在库存中且未被确认拿走，就算在库
//    - 先减后加：先处理消失的物品（拿走），再处理新增的物品（放入）
//    - HELD 优先：手在时先处理 HELD，手离开后再做快照对比
// ============================================================================
#include "session.h"
#include "fridge_config.h"
#include "yolov5.h"

#include <cstdio>
#include <cstdlib>
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
      current_time_ms_(0) {
    current_segment_ = {0, 0, {}, false};
    prev_snapshot_ = {0, 0, {}, false};
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
    if (cls_a != cls_b) return false;
    if (center_distance(box_a, box_b) >= IDENTITY_CENTER_DIST) return false;
    if (iou(box_a, box_b) < IDENTITY_IOU_THRESH) return false;
    // 面积比较（面积比在 0.8~1.2 之间）
    float area_a = box_a.area(), area_b = box_b.area();
    if (area_a > 0 && area_b > 0) {
        float ratio = area_a / area_b;
        if (ratio < 0.8f || ratio > 1.25f) return false;
    }
    if (!frame.empty() && color_diff_crop(frame, box_a, box_b) >= IDENTITY_COLOR_DIFF) return false;
    return true;
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
            inventory_.set_status(kv.first, ItemStatus::HELD);
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
            inventory_.set_status(item_id, ItemStatus::INVENTORY);  // 恢复为在库
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
            inventory_.set_status(item_id, ItemStatus::INVENTORY);
            consumed_tids.insert(putdown->track_id);
            held_done.push_back(item_id);
            res.happened = true;
            res.events.push_back({EventKind::MOVED, item_id, hi.cls_id,
                                  putdown->box, putdown->score, true});
            continue;
        }

        // 4d) 手不在 + 没重现 + 没放下 → 恢复为在库，交给快照对比判断是否真的被拿走
        //     不直接判 TAKEN，因为物品可能只是被其他物品遮挡了
        inventory_.set_status(item_id, ItemStatus::INVENTORY);
        held_done.push_back(item_id);
        printf("\033[1;33m[DBG]\033[0m item#%d (%s) 手离开，恢复为在库（等快照对比确认）\n",
               item_id, coco_cls_to_name(item->cls_id));
    }

    for (int id : held_done) held_items_.erase(id);
    return res;
}


// ============================================================================
//  打印库存（表格格式）
// ============================================================================
void SessionManager::print_inventory() {
    size_t n_total = inventory_.size();
    size_t n_taken = inventory_.count_by_status(ItemStatus::TAKEN);
    size_t n_in = n_total - n_taken;

    printf("\n");
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  库存清单 │ 实际在库: %-3zu │ 被拿走: %-3zu │ 总计: %-3zu │\n", n_in, n_taken, n_total);
    printf("  ├────┬──────────────┬────────┬────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 位置 (中心)        │\n");
    printf("  ├────┼──────────────┼────────┼────────────────────┤\n");

    // 先打印在库物品
    for (const auto& kv : inventory_.items()) {
        const auto& it = kv.second;
        if (it.status == ItemStatus::TAKEN || it.status == ItemStatus::HELD) continue;
        printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)         │\n",
               it.item_id, coco_cls_to_name(it.cls_id), "在库",
               it.box.cx(), it.box.cy());
    }

    // 打印 HELD 物品（显示为"遮挡"）
    for (const auto& kv : inventory_.items()) {
        if (kv.second.status != ItemStatus::HELD) continue;
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
//  主 update 接口
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
    //  阶段 1：IDLE → 开机后立刻收集，手出现时初始化基线
    //    不等"无手时段"，直接用"手出现前的检测结果"作为基线
    // ================================================================
    if (phase_ == SessionPhase::IDLE) {
        // 不管有没有手，都聚合到初始段
        aggregate_to_segment(foods, frame_id);
        no_hand_streak_++;

        // 触发条件：手第一次出现，或者已经收集了足够多帧（兜底）
        bool should_init = hand_present || (no_hand_streak_ >= segment_frames_);

        if (should_init && !baseline_initialized_) {
            // 用当前收集到的段作为基线
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
                            inventory_.set_status(new_id, ItemStatus::INVENTORY);
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        int new_id = inventory_.add_item(-1, bi.cls_id, bi.box, bi.score,
                                                         frame_id, current_time_ms_);
                        inventory_.set_status(new_id, ItemStatus::INVENTORY);
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
            printf("[SESSION] 初始化完成: %zu 件 (手出现=%s, 帧数=%d)\n",
                   inventory_.size(), hand_present ? "是" : "否", no_hand_streak_);
            total_res.happened = true;  // 触发库存打印

            // 如果手已经出现了，直接进 HAND_ACTIVE
            if (hand_present) {
                phase_ = SessionPhase::HAND_ACTIVE;
                current_segment_ = {frame_id, frame_id, {}, false};
                detect_held_items(hands, foods, frame_id);
            } else {
                phase_ = SessionPhase::COLLECTING;
                current_segment_ = {frame_id, frame_id, {}, false};
            }
            no_hand_streak_ = 0;
        }
        return total_res;
    }


    // ================================================================
    //  阶段 2：COLLECTING → 收集无手时间段
    // ================================================================
    if (phase_ == SessionPhase::COLLECTING) {
        if (hand_present) {
            // 手出现了 → 切换到 HAND_ACTIVE
            current_segment_.end_frame = frame_id;
            phase_ = SessionPhase::HAND_ACTIVE;
            no_hand_streak_ = 0;
            detect_held_items(hands, foods, frame_id);
        } else {
            aggregate_to_segment(foods, frame_id);
            no_hand_streak_++;

            // 更新库存中物品的位置
            for (const Track* t : foods) {
                InventoryItem* item = inventory_.find_by_track(t->track_id);
                if (!item || item->status == ItemStatus::TAKEN) continue;
                item->last_bbox = item->box;
                item->box = t->box;
                item->score = t->score;
                item->last_seen_frame = frame_id;
                item->stable_frames++;
            }

            // 图片更新检查
            for (const Track* t : foods) {
                InventoryItem* item = inventory_.find_by_track(t->track_id);
                if (!item || item->status != ItemStatus::INVENTORY) continue;
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
        }
        return total_res;
    }


    // ================================================================
    //  阶段 3：HAND_ACTIVE → 手在画面中
    // ================================================================
    if (phase_ == SessionPhase::HAND_ACTIVE) {
        if (hand_present) {
            SettlementResult held_res = process_held_items(hands, foods, hand_effective, frame_id, frame);
            if (held_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        held_res.events.begin(), held_res.events.end());
            }
        } else {
            // 手离开了 → 处理剩余 HELD
            SettlementResult held_res = process_held_items(hands, foods, hand_effective, frame_id, frame);
            if (held_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        held_res.events.begin(), held_res.events.end());
            }
            no_hand_streak_++;
            if (no_hand_streak_ >= 3) {
                // 手确认离开 → 切换到 COMPARING，开始收集新时间段
                phase_ = SessionPhase::COMPARING;
                current_segment_ = {frame_id, frame_id, {}, false};
                aggregate_to_segment(foods, frame_id);
            }
        }
        // HELD 处理后打印库存
        if (total_res.happened) {
            print_inventory();
        }
        return total_res;
    }


    // ================================================================
    //  阶段 4：COMPARING → 收集新时间段 → 对比快照
    // ================================================================
    if (phase_ == SessionPhase::COMPARING) {
        if (hand_present) {
            // 手又回来了 → 切回 HAND_ACTIVE
            current_segment_.end_frame = frame_id;
            phase_ = SessionPhase::HAND_ACTIVE;
            no_hand_streak_ = 0;
            detect_held_items(hands, foods, frame_id);
        } else {
            aggregate_to_segment(foods, frame_id);
            no_hand_streak_++;

            // 更新库存中物品的位置
            for (const Track* t : foods) {
                InventoryItem* item = inventory_.find_by_track(t->track_id);
                if (!item || item->status == ItemStatus::TAKEN) continue;
                item->last_bbox = item->box;
                item->box = t->box;
                item->score = t->score;
                item->last_seen_frame = frame_id;
                item->stable_frames++;
            }

            if (no_hand_streak_ >= segment_frames_) {
                // ---- 新时间段收集完成，做快照对比 ----
                TimeSegment new_snapshot = current_segment_;
                new_snapshot.end_frame = frame_id;

                // ==== 第一步：先处理减少（拿走） ====
                // 找出库存中有但新快照中没有的物品
                for (auto& inv_kv : inventory_.items()) {
                    if (inv_kv.second.status == ItemStatus::TAKEN) continue;

                    // 在新快照中找匹配
                    bool found_in_snapshot = false;
                    for (const auto& kv : new_snapshot.items) {
                        if (is_same_item_strict(kv.second.box, kv.second.cls_id,
                                                inv_kv.second.box, inv_kv.second.cls_id, frame)) {
                            found_in_snapshot = true;
                            break;
                        }
                    }
                    if (found_in_snapshot) continue;

                    // 没找到 → 检查原位置是否有其他物品覆盖住了它
                    // 先查快照中的物品，再查库存中但不在快照中的物品（遮挡物可能自己也被遮挡了）
                    bool covered = false;
                    // 检查快照中的物品
                    for (const auto& kv : new_snapshot.items) {
                        if (kv.first == inv_kv.second.track_id) continue;  // 跳过自己
                        float overlap = overlap_ratio_of_smaller(inv_kv.second.box, kv.second.box);
                        float dist = center_distance(inv_kv.second.box, kv.second.box);
                        if (overlap > 0.3f || dist < 40.0f) {
                            covered = true;
                            printf("\033[1;33m[DBG]\033[0m item#%d (%s) 被快照中的 item(track=%d, %s) 覆盖 "
                                   "(overlap=%.2f, dist=%.0f) → 留在库存中\n",
                                   inv_kv.first, coco_cls_to_name(inv_kv.second.cls_id),
                                   kv.second.track_id, coco_cls_to_name(kv.second.cls_id),
                                   overlap, dist);
                            break;
                        }
                    }
                    // 如果快照中没找到覆盖物，再查库存中但不在快照中的物品
                    // （覆盖物可能自己也被遮挡了，不在快照中）
                    if (!covered) {
                        for (const auto& inv_kv2 : inventory_.items()) {
                            if (inv_kv2.first == inv_kv.first) continue;  // 跳过自己
                            if (inv_kv2.second.status == ItemStatus::TAKEN) continue;
                            // 这个物品不在快照中（被遮挡了）
                            if (new_snapshot.items.count(inv_kv2.second.track_id)) continue;
                            float overlap = overlap_ratio_of_smaller(inv_kv.second.box, inv_kv2.second.box);
                            float dist = center_distance(inv_kv.second.box, inv_kv2.second.box);
                            if (overlap > 0.3f || dist < 40.0f) {
                                covered = true;
                                printf("\033[1;33m[DBG]\033[0m item#%d (%s) 被库存中的 item#%d (%s) 覆盖 "
                                       "(overlap=%.2f, dist=%.0f) → 留在库存中\n",
                                       inv_kv.first, coco_cls_to_name(inv_kv.second.cls_id),
                                       inv_kv2.first, coco_cls_to_name(inv_kv2.second.cls_id),
                                       overlap, dist);
                                break;
                            }
                        }
                    }
                    if (covered) continue;

                    // 确认拿走
                    printf("\n\033[1;32m[EVENT]\033[0m 取出: item#%d %s (置信度 %.0f%%) "
                           "原位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
                           inv_kv.first, coco_cls_to_name(inv_kv.second.cls_id),
                           inv_kv.second.score * 100,
                           inv_kv.second.box.x1, inv_kv.second.box.y1,
                           inv_kv.second.box.x2, inv_kv.second.box.y2);
                    inventory_.set_status(inv_kv.first, ItemStatus::TAKEN);
                    total_res.events.push_back({EventKind::OUT, inv_kv.first,
                                                 inv_kv.second.cls_id, inv_kv.second.box,
                                                 inv_kv.second.score, false});
                    total_res.happened = true;
                }

                // ==== 第二步：处理增加（放入） ====
                for (const auto& kv : new_snapshot.items) {
                    // 跟库存做严格身份匹配
                    bool found = false;
                    for (auto& inv_kv : inventory_.items()) {
                        if (inv_kv.second.status == ItemStatus::TAKEN) continue;
                        if (is_same_item_strict(kv.second.box, kv.second.cls_id,
                                                inv_kv.second.box, inv_kv.second.cls_id, frame)) {
                            // 重新绑定 track
                            inventory_.relocate_item(inv_kv.first, kv.second.track_id,
                                                     kv.second.box, kv.second.best_score, frame_id);
                            found = true;
                            break;
                        }
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
                        total_res.events.push_back({EventKind::IN, new_id, kv.second.cls_id,
                                                     kv.second.box, kv.second.best_score, true});
                        total_res.happened = true;
                    }
                }

                // ==== 第三步：TAKEN 恢复检查 ====
                for (const auto& kv : new_snapshot.items) {
                    for (auto& inv_kv : inventory_.items()) {
                        if (inv_kv.second.status != ItemStatus::TAKEN) continue;
                        if (is_same_item_strict(kv.second.box, kv.second.cls_id,
                                                inv_kv.second.last_bbox, inv_kv.second.cls_id, frame)) {
                            inventory_.relocate_item(inv_kv.first, kv.second.track_id,
                                                     kv.second.box, kv.second.best_score, frame_id);
                            inventory_.set_status(inv_kv.first, ItemStatus::INVENTORY);
                            printf("\033[1;33m[DBG]\033[0m item#%d TAKEN 恢复 → 在库 (YOLO 抖动误判)\n", inv_kv.first);
                            total_res.happened = true;
                            break;
                        }
                    }
                }

                // 更新快照，切换回 COLLECTING
                prev_snapshot_ = new_snapshot;
                has_prev_snapshot_ = true;
                current_segment_ = {frame_id, frame_id, {}, false};
                phase_ = SessionPhase::COLLECTING;
                no_hand_streak_ = 0;

                // 快照对比后打印库存
                if (total_res.happened) {
                    print_inventory();
                }
            }
        }
        return total_res;
    }


    // ================================================================
    //  兜底：其他阶段有事件时也打印库存
    // ================================================================
    if (total_res.happened) {
        print_inventory();
    }

    return total_res;
}

}  // namespace fridge
