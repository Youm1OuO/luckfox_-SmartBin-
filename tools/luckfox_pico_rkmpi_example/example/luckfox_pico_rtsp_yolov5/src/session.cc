// ============================================================================
//  session.cc
//  会话管理器 — 新业务流程5：逐帧对比库存（track_id 优先匹配）
//
//  匹配策略：
//    1. track_id 匹配（ByteTrack 保证同一物体同一 id）→ 最可靠
//    2. 宽松几何匹配（类别 + 中心距离 < 30px）→ track_id 变了时兜底
//    3. 严格匹配（5 条件全满足）→ 仅用于 TAKEN 恢复等高置信场景
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

SessionManager::SessionManager(int stable_frames)
    : baseline_initialized_(false),
      current_state_(SystemState::STABLE),
      backend_initialized_(false),
      phase_(SessionPhase::IDLE),
      stable_streak_(0),
      stable_frames_(stable_frames),
      current_frame_id_(0),
      last_hand_frame_(-1000),
      current_time_ms_(0) {}

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
    float area_a = box_a.area(), area_b = box_b.area();
    if (area_a > 0 && area_b > 0) {
        float ratio = area_a / area_b;
        if (ratio < 0.8f || ratio > 1.25f) return false;
    }
    if (!frame.empty() && color_diff_crop(frame, box_a, box_b) >= IDENTITY_COLOR_DIFF) return false;
    return true;
}

// 宽松几何匹配：类别 + 中心距离（用于移动物品的兜底匹配）
static bool is_same_item_relaxed(const BBox& box_a, int cls_a,
                                  const BBox& box_b, int cls_b) {
    if (cls_a != cls_b) return false;
    return center_distance(box_a, box_b) < 30.0f;
}


// ============================================================================
//  核心：逐帧对比库存（先减后加 + 恢复检查）
//  匹配策略：track_id 优先 → 宽松几何 → 严格匹配
// ============================================================================
SettlementResult SessionManager::compare_frame_to_inventory(
        const std::vector<const Track*>& foods,
        const std::vector<const Track*>& hands,
        const cv::Mat& frame,
        int frame_id) {
    SettlementResult res;
    bool has_frame = !frame.empty();

    // 建立 track_id → Track* 的查找表
    std::map<int, const Track*> track_map;
    for (const Track* t : foods) track_map[t->track_id] = t;

    // 记录已被匹配的 track_id（防止同一 track 被匹配多次）
    std::set<int> consumed_tids;

    // ==== 第一步：减少（消失的物品 → 拿走 或 遮挡）====
    for (auto& kv : inventory_.items()) {
        InventoryItem& item = kv.second;
        if (item.status == ItemStatus::TAKEN || item.status == ItemStatus::HELD) continue;
        if (item.status == ItemStatus::OCCLUDED) continue;

        // 优先：track_id 匹配
        auto it = track_map.find(item.track_id);
        if (it != track_map.end() && !consumed_tids.count(item.track_id)) {
            // track 还在 → 物品还在，更新位置
            const Track* t = it->second;
            item.last_bbox = item.box;
            item.box = t->box;
            item.score = t->score;
            item.last_seen_frame = frame_id;
            item.stable_frames++;
            consumed_tids.insert(item.track_id);
            continue;
        }

        // track_id 匹配不上 → 尝试宽松几何匹配（track_id 可能变了）
        bool found = false;
        for (const Track* t : foods) {
            if (consumed_tids.count(t->track_id)) continue;
            if (is_same_item_relaxed(t->box, t->cls_id, item.box, item.cls_id)) {
                // 匹配上 → track_id 变了，更新绑定
                item.last_bbox = item.box;
                item.track_id = t->track_id;
                item.box = t->box;
                item.score = t->score;
                item.last_seen_frame = frame_id;
                item.stable_frames++;
                consumed_tids.insert(t->track_id);
                found = true;
                break;
            }
        }
        if (found) continue;

        // 消失了 → 检查是否被其他物品遮挡
        bool covered = false;
        for (const Track* t : foods) {
            float cov = 0.0f;
            float a_area = item.box.area();
            if (a_area > 0) {
                float ix1 = std::max(item.box.x1, t->box.x1);
                float iy1 = std::max(item.box.y1, t->box.y1);
                float ix2 = std::min(item.box.x2, t->box.x2);
                float iy2 = std::min(item.box.y2, t->box.y2);
                float iw = std::max(0.0f, ix2 - ix1);
                float ih = std::max(0.0f, iy2 - iy1);
                cov = (iw * ih) / a_area;
            }
            float dist = center_distance(item.box, t->box);
            if (cov > OCCLUDE_OVERLAP_THRESH || dist < OCCLUDE_CENTER_DIST_THRESH) {
                covered = true;
                printf("\033[1;33m[DBG]\033[0m item#%d (%s) 被遮挡 "
                       "(cov=%.2f, dist=%.0f) → OCCLUDED\n",
                       kv.first, coco_cls_to_name(item.cls_id), cov, dist);
                break;
            }
        }
        if (covered) {
            inventory_.set_status(kv.first, ItemStatus::OCCLUDED);
        } else if (!hands.empty()) {
            // 手在画面中 → 检查是否被手拿着
            // 先用严格检查（手框覆盖物品），再用宽松检查（手在物品附近）
            bool hand_near = false;
            for (const Track* h : hands) {
                if (hand_covers(h, item.box, FRAME_W, FRAME_H)) {
                    hand_near = true;
                    break;
                }
            }
            // 宽松检查：手中心离物品中心 < 150px（手移动过程中位置会变）
            if (!hand_near) {
                for (const Track* h : hands) {
                    if (center_distance(h->box, item.box) < 150.0f) {
                        hand_near = true;
                        break;
                    }
                }
            }
            if (hand_near) {
                HeldItem hi;
                hi.item_id = kv.first;
                hi.cls_id = item.cls_id;
                hi.original_pos = item.box;
                hi.score = item.score;
                hi.hand_track_id = hands[0]->track_id;
                hi.pickup_frame = frame_id;
                held_items_[kv.first] = hi;
                inventory_.set_status(kv.first, ItemStatus::HELD);
                printf("\033[1;33m[DBG]\033[0m item#%d (%s) 消失但被手遮挡 → HELD\n",
                       kv.first, coco_cls_to_name(item.cls_id));
            } else {
                printf("\n\033[1;32m[EVENT]\033[0m 取出: item#%d %s (置信度 %.0f%%) "
                       "原位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
                       kv.first, coco_cls_to_name(item.cls_id),
                       item.score * 100,
                       item.box.x1, item.box.y1, item.box.x2, item.box.y2);
                inventory_.set_status(kv.first, ItemStatus::TAKEN);
                item.updated_frame = frame_id;  // 记录 TAKEN 时间
                res.events.push_back({EventKind::OUT, kv.first,
                                     item.cls_id, item.box, item.score, false});
                res.happened = true;
            }
        } else {
            printf("\n\033[1;32m[EVENT]\033[0m 取出: item#%d %s (置信度 %.0f%%) "
                   "原位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
                   kv.first, coco_cls_to_name(item.cls_id),
                   item.score * 100,
                   item.box.x1, item.box.y1, item.box.x2, item.box.y2);
            inventory_.set_status(kv.first, ItemStatus::TAKEN);
            item.updated_frame = frame_id;
            res.events.push_back({EventKind::OUT, kv.first,
                                 item.cls_id, item.box, item.score, false});
            res.happened = true;
        }
    }

    // ==== 第二步：增加（新出现的物品）====
    for (const Track* t : foods) {
        if (consumed_tids.count(t->track_id)) continue;

        // 优先：track_id 匹配库存
        InventoryItem* by_track = inventory_.find_by_track(t->track_id);
        if (by_track && by_track->status != ItemStatus::TAKEN
                     && by_track->status != ItemStatus::HELD) {
            by_track->last_bbox = by_track->box;
            by_track->box = t->box;
            by_track->score = t->score;
            by_track->last_seen_frame = frame_id;
            by_track->stable_frames++;
            consumed_tids.insert(t->track_id);
            continue;
        }

        // 尝试匹配 TAKEN 物品（严格匹配 → 高置信恢复）
        bool matched_taken = false;
        for (auto& kv : inventory_.items()) {
            if (kv.second.status != ItemStatus::TAKEN) continue;
            // 恢复窗口内才尝试恢复
            if (frame_id - kv.second.updated_frame > TAKEN_RECOVERY_FRAMES) continue;
            if (has_frame) {
                if (!is_same_item_strict(t->box, t->cls_id,
                                         kv.second.last_bbox, kv.second.cls_id, frame))
                    continue;
            } else {
                if (!is_same_position(t->box, kv.second.last_bbox,
                                      IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH))
                    continue;
            }
            // 匹配上 → 误判恢复
            inventory_.relocate_item(kv.first, t->track_id, t->box, t->score, frame_id);
            inventory_.set_status(kv.first, ItemStatus::INVENTORY);
            consumed_tids.insert(t->track_id);
            printf("\033[1;33m[DBG]\033[0m item#%d TAKEN 恢复 → 在库\n", kv.first);
            matched_taken = true;
            res.happened = true;
            break;
        }
        if (matched_taken) continue;

        // 尝试匹配 TAKEN 物品（宽松匹配 → 移动物品恢复）
        for (auto& kv : inventory_.items()) {
            if (kv.second.status != ItemStatus::TAKEN) continue;
            if (frame_id - kv.second.updated_frame > TAKEN_RECOVERY_FRAMES) continue;
            if (is_same_item_relaxed(t->box, t->cls_id, kv.second.last_bbox, kv.second.cls_id)) {
                inventory_.relocate_item(kv.first, t->track_id, t->box, t->score, frame_id);
                inventory_.set_status(kv.first, ItemStatus::INVENTORY);
                consumed_tids.insert(t->track_id);
                printf("\033[1;33m[DBG]\033[0m item#%d TAKEN 恢复(宽松) → 在库 "
                       "(移动物品恢复)\n", kv.first);
                matched_taken = true;
                res.happened = true;
                break;
            }
        }
        if (matched_taken) continue;

        // 尝试匹配现有库存（宽松匹配 → 防止重复添加）
        bool matched_inv = false;
        for (auto& kv : inventory_.items()) {
            if (kv.second.status == ItemStatus::TAKEN || kv.second.status == ItemStatus::HELD) continue;
            const BBox& ref_box = (kv.second.status == ItemStatus::OCCLUDED)
                                   ? kv.second.last_bbox : kv.second.box;
            if (!is_same_item_relaxed(t->box, t->cls_id, ref_box, kv.second.cls_id))
                continue;
            inventory_.relocate_item(kv.first, t->track_id, t->box, t->score, frame_id);
            inventory_.set_status(kv.first, ItemStatus::INVENTORY);
            consumed_tids.insert(t->track_id);
            matched_inv = true;
            break;
        }
        if (matched_inv) continue;

        // 都没匹配上 → 新物品，入库
        int new_id = inventory_.add_item(t->track_id, t->cls_id, t->box, t->score,
                                          frame_id, current_time_ms_);
        item_anchor_[new_id] = t->box;
        item_last_box_[new_id] = t->box;
        consumed_tids.insert(t->track_id);
        printf("\n\033[1;32m[EVENT]\033[0m 放入: item#%d %s (置信度 %.0f%%) "
               "位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
               new_id, coco_cls_to_name(t->cls_id), t->score * 100,
               t->box.x1, t->box.y1, t->box.x2, t->box.y2);
        res.events.push_back({EventKind::IN, new_id, t->cls_id,
                             t->box, t->score, true});
        res.happened = true;
    }

    // ==== 第三步：TAKEN 恢复检查（对当前帧所有物品再扫一遍）====
    for (const Track* t : foods) {
        for (auto& kv : inventory_.items()) {
            if (kv.second.status != ItemStatus::TAKEN) continue;
            if (frame_id - kv.second.updated_frame > TAKEN_RECOVERY_FRAMES) continue;
            if (is_same_item_relaxed(t->box, t->cls_id, kv.second.last_bbox, kv.second.cls_id)) {
                inventory_.relocate_item(kv.first, t->track_id, t->box, t->score, frame_id);
                inventory_.set_status(kv.first, ItemStatus::INVENTORY);
                printf("\033[1;33m[DBG]\033[0m item#%d TAKEN 恢复(恢复检查) → 在库\n", kv.first);
                res.happened = true;
                break;
            }
        }
    }

    return res;
}


// ============================================================================
//  遮挡恢复检查
//  1. YOLO 重新检测到 OCCLUDED 物品 → 恢复 INVENTORY
//  2. OCCLUDED 物品附近没有遮挡物了 → 恢复 INVENTORY
// ============================================================================
void SessionManager::occlusion_check(const std::vector<const Track*>& foods,
                                      const cv::Mat& frame,
                                      int frame_id) {
    // 先收集需要恢复的 item_id（不能在遍历中直接修改 status）
    std::vector<int> to_restore;

    for (auto& kv : inventory_.items()) {
        if (kv.second.status != ItemStatus::OCCLUDED) continue;

        // ---- 检查 1：YOLO 重新检测到 ----
        bool redetected = false;
        for (const Track* t : foods) {
            if (t->track_id == kv.second.track_id) {
                inventory_.relocate_item(kv.first, t->track_id, t->box, t->score, frame_id);
                item_anchor_[kv.first] = t->box;
                item_last_box_[kv.first] = t->box;
                printf("\033[1;36m[DBG]\033[0m item#%d (%s) 遮挡恢复(track_id) → 在库\n",
                       kv.first, coco_cls_to_name(kv.second.cls_id));
                redetected = true;
                break;
            }
            if (is_same_item_relaxed(t->box, t->cls_id, kv.second.last_bbox, kv.second.cls_id)) {
                inventory_.relocate_item(kv.first, t->track_id, t->box, t->score, frame_id);
                item_anchor_[kv.first] = t->box;
                item_last_box_[kv.first] = t->box;
                printf("\033[1;36m[DBG]\033[0m item#%d (%s) 遮挡恢复(几何) → 在库\n",
                       kv.first, coco_cls_to_name(kv.second.cls_id));
                redetected = true;
                break;
            }
        }
        if (redetected) {
            inventory_.set_status(kv.first, ItemStatus::INVENTORY);
            continue;
        }

        // ---- 检查 2：遮挡物是否还在 ----
        // 先查当前帧中的物品（YOLO 能看到的）
        bool still_covered = false;
        for (const Track* t : foods) {
            if (t->track_id == kv.second.track_id) continue;  // 跳过自己
            float cov = 0.0f;
            float a_area = kv.second.last_bbox.area();
            if (a_area > 0) {
                float ix1 = std::max(kv.second.last_bbox.x1, t->box.x1);
                float iy1 = std::max(kv.second.last_bbox.y1, t->box.y1);
                float ix2 = std::min(kv.second.last_bbox.x2, t->box.x2);
                float iy2 = std::min(kv.second.last_bbox.y2, t->box.y2);
                float iw = std::max(0.0f, ix2 - ix1);
                float ih = std::max(0.0f, iy2 - iy1);
                cov = (iw * ih) / a_area;
            }
            float dist = center_distance(kv.second.last_bbox, t->box);
            if (cov > OCCLUDE_OVERLAP_THRESH || dist < OCCLUDE_CENTER_DIST_THRESH) {
                still_covered = true;
                break;
            }
        }
        // 再查库存中但不在当前帧的物品（覆盖物可能自己也被遮挡了）
        if (!still_covered) {
            for (const auto& kv2 : inventory_.items()) {
                if (kv2.first == kv.first) continue;
                if (kv2.second.status == ItemStatus::TAKEN || kv2.second.status == ItemStatus::HELD) continue;
                float cov = 0.0f;
                float a_area = kv.second.last_bbox.area();
                if (a_area > 0) {
                    float ix1 = std::max(kv.second.last_bbox.x1, kv2.second.box.x1);
                    float iy1 = std::max(kv.second.last_bbox.y1, kv2.second.box.y1);
                    float ix2 = std::min(kv.second.last_bbox.x2, kv2.second.box.x2);
                    float iy2 = std::min(kv.second.last_bbox.y2, kv2.second.box.y2);
                    float iw = std::max(0.0f, ix2 - ix1);
                    float ih = std::max(0.0f, iy2 - iy1);
                    cov = (iw * ih) / a_area;
                }
                float dist = center_distance(kv.second.last_bbox, kv2.second.box);
                if (cov > OCCLUDE_OVERLAP_THRESH || dist < OCCLUDE_CENTER_DIST_THRESH) {
                    still_covered = true;
                    break;
                }
            }
        }

        if (!still_covered) {
            // 遮挡物不在了 → 恢复在库
            to_restore.push_back(kv.first);
            printf("\033[1;36m[DBG]\033[0m item#%d (%s) 遮挡物移走 → 恢复在库\n",
                   kv.first, coco_cls_to_name(kv.second.cls_id));
        }
    }

    // 批量恢复
    for (int iid : to_restore) {
        inventory_.set_status(iid, ItemStatus::INVENTORY);
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
//  处理 HELD 物品
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

        // 4a) 原位重现 → 取消 HELD
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
            inventory_.set_status(item_id, ItemStatus::INVENTORY);
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

        // 4c) 远处新 track → 整理
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
            item_anchor_[item_id] = putdown->box;
            item_last_box_[item_id] = putdown->box;
            continue;
        }

        // 4d) 手不在 + 没重现 + 没放下 → 恢复在库
        inventory_.set_status(item_id, ItemStatus::INVENTORY);
        held_done.push_back(item_id);
        printf("\033[1;33m[DBG]\033[0m item#%d (%s) 手离开，恢复为在库\n",
               item_id, coco_cls_to_name(item->cls_id));
    }
    for (int id : held_done) held_items_.erase(id);
    return res;
}


// ============================================================================
//  平滑移动整理检测
// ============================================================================
void SessionManager::detect_smooth_relocate(const std::vector<const Track*>& foods,
                                             int frame_id,
                                             const cv::Mat& frame,
                                             SettlementResult& res) {
    for (const Track* t : foods) {
        InventoryItem* item = inventory_.find_by_track(t->track_id);
        if (!item || item->status == ItemStatus::TAKEN || item->status == ItemStatus::HELD) continue;

        int iid = item->item_id;

        if (item_anchor_.find(iid) == item_anchor_.end()) {
            item_anchor_[iid] = item->box;
        }

        bool settled = true;
        auto lb = item_last_box_.find(iid);
        if (lb != item_last_box_.end()) {
            settled = center_distance(item->box, lb->second) < SMOOTH_SETTLE_PIX;
        } else {
            settled = false;
        }

        float move = center_distance(t->box, item_anchor_[iid]);
        if (settled && move >= SMOOTH_RELOCATE_PIX) {
            BBox old_pos = item_anchor_[iid];
            printf("\n\033[1;32m[EVENT]\033[0m 整理(平滑): item#%d %s "
                   "从(%.0f,%.0f)~(%.0f,%.0f) → (%.0f,%.0f)~(%.0f,%.0f)\n",
                   iid, coco_cls_to_name(item->cls_id),
                   old_pos.x1, old_pos.y1, old_pos.x2, old_pos.y2,
                   t->box.x1, t->box.y1, t->box.x2, t->box.y2);
            item_anchor_[iid] = t->box;
            res.happened = true;
            res.events.push_back({EventKind::MOVED, iid, item->cls_id,
                                  t->box, t->score, true});

            for (auto& kv2 : inventory_.items()) {
                if (kv2.first == iid) continue;
                if (kv2.second.status != ItemStatus::INVENTORY) continue;
                float cov = 0.0f;
                float b_area = kv2.second.box.area();
                if (b_area > 0) {
                    float ix1 = std::max(t->box.x1, kv2.second.box.x1);
                    float iy1 = std::max(t->box.y1, kv2.second.box.y1);
                    float ix2 = std::min(t->box.x2, kv2.second.box.x2);
                    float iy2 = std::min(t->box.y2, kv2.second.box.y2);
                    float iw = std::max(0.0f, ix2 - ix1);
                    float ih = std::max(0.0f, iy2 - iy1);
                    cov = (iw * ih) / b_area;
                }
                float dist = center_distance(t->box, kv2.second.box);
                if (cov > OCCLUDE_OVERLAP_THRESH || dist < OCCLUDE_CENTER_DIST_THRESH) {
                    inventory_.set_status(kv2.first, ItemStatus::OCCLUDED);
                    printf("\033[1;33m[DBG]\033[0m item#%d (%s) 被整理后的 item#%d 遮挡 → OCCLUDED\n",
                           kv2.first, coco_cls_to_name(kv2.second.cls_id), iid);
                }
            }
            for (auto& kv2 : inventory_.items()) {
                if (kv2.second.status != ItemStatus::OCCLUDED) continue;
                float cov = 0.0f;
                float b_area = kv2.second.last_bbox.area();
                if (b_area > 0) {
                    float ix1 = std::max(old_pos.x1, kv2.second.last_bbox.x1);
                    float iy1 = std::max(old_pos.y1, kv2.second.last_bbox.y1);
                    float ix2 = std::min(old_pos.x2, kv2.second.last_bbox.x2);
                    float iy2 = std::min(old_pos.y2, kv2.second.last_bbox.y2);
                    float iw = std::max(0.0f, ix2 - ix1);
                    float ih = std::max(0.0f, iy2 - iy1);
                    cov = (iw * ih) / b_area;
                }
                float dist = center_distance(old_pos, kv2.second.last_bbox);
                if (cov > OCCLUDE_OVERLAP_THRESH || dist < OCCLUDE_CENTER_DIST_THRESH) {
                    inventory_.set_status(kv2.first, ItemStatus::INVENTORY);
                    printf("\033[1;36m[DBG]\033[0m item#%d (%s) 原位置物品移走 → 恢复在库\n",
                           kv2.first, coco_cls_to_name(kv2.second.cls_id));
                }
            }
        }

        item_last_box_[iid] = t->box;
    }
}


// ============================================================================
//  打印库存
// ============================================================================
void SessionManager::print_inventory() {
    size_t n_total = inventory_.size();
    size_t n_taken = inventory_.count_by_status(ItemStatus::TAKEN);
    size_t n_occluded = inventory_.count_by_status(ItemStatus::OCCLUDED);
    size_t n_in = n_total - n_taken - inventory_.count_by_status(ItemStatus::HELD);

    printf("\n");
    printf("  ┌─────────────────────────────────────────────────────────┐\n");
    printf("  │  库存清单 │ 在库: %-3zu │ 遮挡: %-3zu │ 拿走: %-3zu │ 总: %-3zu │\n",
           n_in, n_occluded, n_taken, n_total);
    printf("  ├────┬──────────────┬────────┬────────────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 位置 (中心)                │\n");
    printf("  ├────┼──────────────┼────────┼────────────────────────────┤\n");
    for (const auto& kv : inventory_.items()) {
        const auto& it = kv.second;
        if (it.status == ItemStatus::TAKEN || it.status == ItemStatus::HELD) continue;
        const char* status_str = (it.status == ItemStatus::OCCLUDED) ? "遮挡" : "在库";
        printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)                 │\n",
               it.item_id, coco_cls_to_name(it.cls_id), status_str,
               it.box.cx(), it.box.cy());
    }
    if (n_taken > 0) {
        printf("  ├────┴──────────────┴────────┴────────────────────────────┤\n");
        printf("  │  被拿走（临时记录，关门时不上传后台）                    │\n");
        printf("  ├────┬──────────────┬────────┬────────────────────────────┤\n");
        for (const auto& kv : inventory_.items()) {
            if (kv.second.status != ItemStatus::TAKEN) continue;
            printf("  │ %-2d │ %-12s │ %-6s │ (%4.0f,%4.0f)                 │\n",
                   kv.second.item_id, coco_cls_to_name(kv.second.cls_id), "拿走",
                   kv.second.box.cx(), kv.second.box.cy());
        }
    }
    printf("  └────┴──────────────┴────────┴────────────────────────────┘\n\n");
}


// ============================================================================
//  旧接口
// ============================================================================
SettlementResult SessionManager::update(const std::vector<Track>& tracks, int frame_id) {
    cv::Mat empty;
    return update(tracks, frame_id, empty, 0);
}


// ============================================================================
//  主 update 接口（v5：逐帧对比，track_id 优先）
// ============================================================================
SettlementResult SessionManager::update(const std::vector<Track>& tracks,
                                        int frame_id,
                                        const cv::Mat& frame,
                                        long long time_ms) {
    current_time_ms_ = time_ms;
    current_frame_id_ = frame_id;
    SettlementResult total_res;
    bool has_frame = !frame.empty();

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
    //  IDLE → 开机初始化
    // ================================================================
    if (phase_ == SessionPhase::IDLE) {
        stable_streak_++;
        bool should_init = hand_present || (stable_streak_ >= stable_frames_);
        if (should_init && !baseline_initialized_) {
            if (backend_initialized_) {
                for (const auto& bi : backend_inventory_) {
                    bool matched = false;
                    for (const Track* t : foods) {
                        bool m = has_frame
                            ? is_same_item_strict(t->box, t->cls_id, bi.box, bi.cls_id, frame)
                            : is_same_position(t->box, bi.box, IDENTITY_CENTER_DIST, IDENTITY_IOU_THRESH);
                        if (m) {
                            int new_id = inventory_.add_item(t->track_id, t->cls_id,
                                                             t->box, t->score,
                                                             frame_id, current_time_ms_);
                            item_anchor_[new_id] = t->box;
                            item_last_box_[new_id] = t->box;
                            matched = true;
                            break;
                        }
                    }
                    if (!matched) {
                        int new_id = inventory_.add_item(-1, bi.cls_id, bi.box, bi.score,
                                                         frame_id, current_time_ms_);
                        item_anchor_[new_id] = bi.box;
                        item_last_box_[new_id] = bi.box;
                    }
                }
            } else {
                for (const Track* t : foods) {
                    int new_id = inventory_.add_item(t->track_id, t->cls_id,
                                                     t->box, t->score,
                                                     frame_id, current_time_ms_);
                    item_anchor_[new_id] = t->box;
                    item_last_box_[new_id] = t->box;
                }
            }
            baseline_initialized_ = true;
            printf("[SESSION] 初始化完成: %zu 件 (手出现=%s, 帧数=%d)\n",
                   inventory_.size(), hand_present ? "是" : "否", stable_streak_);
            total_res.happened = true;

            if (hand_present) {
                phase_ = SessionPhase::HAND_ACTIVE;
                detect_held_items(hands, foods, frame_id);
            } else {
                phase_ = SessionPhase::ACTIVE;
            }
            stable_streak_ = 0;
        }
        return total_res;
    }


    // ================================================================
    //  ACTIVE → 逐帧对比库存（带帧跳过）
    // ================================================================
    if (phase_ == SessionPhase::ACTIVE) {
        // 帧跳过：只在手出现时立即处理，否则每隔 COMPARE_INTERVAL 帧处理一次
        compare_counter_++;
        bool should_compare = hand_present || (compare_counter_ >= COMPARE_INTERVAL);
        if (should_compare) compare_counter_ = 0;

        if (hand_present && should_compare) {
            SettlementResult cmp_res = compare_frame_to_inventory(foods, hands, frame, frame_id);
            if (cmp_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        cmp_res.events.begin(), cmp_res.events.end());
            }
            phase_ = SessionPhase::HAND_ACTIVE;
            stable_streak_ = 0;
            detect_held_items(hands, foods, frame_id);
        } else if (should_compare) {
            SettlementResult cmp_res = compare_frame_to_inventory(foods, hands, frame, frame_id);
            if (cmp_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        cmp_res.events.begin(), cmp_res.events.end());
            }

            occlusion_check(foods, frame, frame_id);

            // 平滑移动整理检测
            detect_smooth_relocate(foods, frame_id, frame, total_res);
        }

        // 图片更新检查（每帧都跑，不受帧跳过影响）
        for (const Track* t : foods) {
            InventoryItem* item = inventory_.find_by_track(t->track_id);
            if (!item || item->status != ItemStatus::INVENTORY) continue;
            if (image_updated_.count(item->item_id)) continue;
            if (frame_id - item->created_frame < IMAGE_UPDATE_DELAY_FRAMES) continue;
            auto lb_it = item_last_box_.find(item->item_id);
            if (lb_it != item_last_box_.end() &&
                center_distance(t->box, lb_it->second) >= SMOOTH_SETTLE_PIX) {
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

        // 只在有真正的业务事件（IN/OUT/MOVED）时才打印库存
        bool has_business_event = false;
        for (const auto& ev : total_res.events) {
            if (ev.kind == EventKind::IN || ev.kind == EventKind::OUT || ev.kind == EventKind::MOVED) {
                has_business_event = true;
                break;
            }
        }
        if (has_business_event) print_inventory();
        return total_res;
    }


    // ================================================================
    //  HAND_ACTIVE → 手在画面中
    // ================================================================
    if (phase_ == SessionPhase::HAND_ACTIVE) {
        SettlementResult cmp_res = compare_frame_to_inventory(foods, hands, frame, frame_id);
        if (cmp_res.happened) {
            total_res.happened = true;
            total_res.events.insert(total_res.events.end(),
                                    cmp_res.events.begin(), cmp_res.events.end());
        }

        if (hand_present) {
            stable_streak_ = 0;
            SettlementResult held_res = process_held_items(hands, foods, hand_effective, frame_id, frame);
            if (held_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        held_res.events.begin(), held_res.events.end());
            }
        } else {
            SettlementResult held_res = process_held_items(hands, foods, hand_effective, frame_id, frame);
            if (held_res.happened) {
                total_res.happened = true;
                total_res.events.insert(total_res.events.end(),
                                        held_res.events.begin(), held_res.events.end());
            }
            stable_streak_++;
            if (stable_streak_ >= HAND_CONFIRM_LEAVE) {
                phase_ = SessionPhase::ACTIVE;
                stable_streak_ = 0;
            }
        }
        bool has_business_event = false;
        for (const auto& ev : total_res.events) {
            if (ev.kind == EventKind::IN || ev.kind == EventKind::OUT || ev.kind == EventKind::MOVED) {
                has_business_event = true;
                break;
            }
        }
        if (has_business_event) print_inventory();
        return total_res;
    }

    return total_res;
}

}  // namespace fridge
