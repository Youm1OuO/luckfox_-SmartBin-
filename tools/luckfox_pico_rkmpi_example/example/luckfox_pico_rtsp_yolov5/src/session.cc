// ============================================================================
//  session.cc
//  会话管理器 — item_id 身份追踪 + 抗抖动 + 整理识别
// ============================================================================
#include "session.h"
#include "fridge_config.h"
#include "yolov5.h"

#include <cstdio>
#include <algorithm>
#include <set>

namespace fridge {

SessionManager::SessionManager(int occluded_grace_frames)
    : baseline_initialized_(false),
      occluded_grace_frames_(occluded_grace_frames),
      current_state_(SystemState::STABLE),
      last_hand_frame_(-1000) {}

// 方案 A：把手框沿"中心→最近画面边缘"方向延伸到边缘，得到"手+手臂影响区"
// 手臂总是从画面边缘伸进来的，所以手掌框和它进来的那条边之间的区域都算手臂
static BBox expand_hand_to_arm(const BBox& hand, float W, float H) {
    float cx = hand.cx();
    float cy = hand.cy();
    // 算手框中心到四条边的距离，找最近的边
    float d_left   = cx;
    float d_right  = W - cx;
    float d_top    = cy;
    float d_bottom = H - cy;
    float dmin = std::min(std::min(d_left, d_right), std::min(d_top, d_bottom));

    BBox arm = hand;   // 从手掌框开始扩展
    if (dmin == d_top) {
        // 手从顶部进来 → 把框向上延伸到顶边
        arm.y1 = 0;
    } else if (dmin == d_bottom) {
        arm.y2 = H;
    } else if (dmin == d_left) {
        arm.x1 = 0;
    } else {
        arm.x2 = W;
    }
    return arm;
}

// 判断手 h（含手臂延伸区）是否盖住了物品框 box
static bool hand_covers(const Track* h, const BBox& box, float W, float H) {
    // 先用手掌框判（精确）
    bool center_in = point_in_box(box.cx(), box.cy(), h->box);
    bool covered = overlap_ratio_of_smaller(h->box, box) >= 0.5f;
    if (center_in || covered) return true;

    // 再用"手+手臂影响区"判（宽松，覆盖手臂遮挡）
    BBox arm = expand_hand_to_arm(h->box, W, H);
    bool center_in_arm = point_in_box(box.cx(), box.cy(), arm);
    bool covered_arm = overlap_ratio_of_smaller(arm, box) >= 0.5f;
    return center_in_arm || covered_arm;
}

SettlementResult SessionManager::update(const std::vector<Track>& tracks,
                                        int frame_id) {
    SettlementResult res;

    // ================================================================
    //  Step 1: 分类 tracks
    // ================================================================
    std::vector<const Track*> hands;
    std::vector<const Track*> foods;
    for (const auto& t : tracks) {
        if (is_hand(t.cls_id)) hands.push_back(&t);
        else if (is_food(t.cls_id) && t.score >= SNAPSHOT_MIN_SCORE) foods.push_back(&t);
    }
    bool hand_present = !hands.empty();
    if (hand_present) last_hand_frame_ = frame_id;

    // 方案 B：手消失惯性 — 手 track 刚消失的 HAND_INERTIA_FRAMES 帧内,
    // 仍然认为"可能有手"。用于抗"手识别不稳"导致的物品误判拿走。
    bool hand_effective = hand_present ||
        (frame_id - last_hand_frame_ <= HAND_INERTIA_FRAMES);

    current_state_ = hand_present ? SystemState::DISTURBED : SystemState::STABLE;

    // 当前帧能看到的 food track_id 集合
    std::set<int> seen_tids;
    for (const Track* t : foods) seen_tids.insert(t->track_id);

    // ================================================================
    //  Step 2: 初始化基线（没手 + 有物品时）
    // ================================================================
    if (!baseline_initialized_) {
        if (hand_present || foods.empty()) return res;
        for (const Track* t : foods) {
            inventory_.add_item(t->track_id, t->cls_id, t->box, t->score, frame_id);
        }
        baseline_initialized_ = true;
        printf("[SESSION] baseline initialized with %zu items\n", inventory_.size());
        inventory_.print("  ");
        return res;
    }

    // ================================================================
    //  Step 3: 刷新库存里"还能看到"的物品（按 track_id 匹配）
    //          + 平滑移动整理检测（物品 track 没被手遮挡, 自己挪了位置）
    // ================================================================
    for (const Track* t : foods) {
        InventoryItem* item = inventory_.find_by_track(t->track_id);
        if (!item) continue;

        int iid = item->item_id;

        // --- 平滑移动整理检测 ---
        // 锚点不存在 → 用当前位置初始化锚点
        if (item_anchor_.find(iid) == item_anchor_.end()) {
            item_anchor_[iid] = item->box;
        }

        // 判断物品这一帧是否"停下来了"（相对上一帧位移很小）
        bool settled = true;
        auto lb = item_last_box_.find(iid);
        if (lb != item_last_box_.end()) {
            settled = center_distance(item->box, lb->second) < SMOOTH_SETTLE_PIX;
        } else {
            settled = false;  // 还没有上一帧记录，先不判
        }

        // 物品停下来了 + 离锚点足够远 → 平滑移动整理
        float move = center_distance(t->box, item_anchor_[iid]);
        if (settled && move >= SMOOTH_RELOCATE_PIX) {
            printf("\n\033[1;32m[EVENT]\033[0m 整理: item#%d %s (置信度 %.0f%%) "
                   "从(%.0f,%.0f)~(%.0f,%.0f) → (%.0f,%.0f)~(%.0f,%.0f)\n",
                   iid, coco_cls_to_name(item->cls_id), t->score * 100,
                   item_anchor_[iid].x1, item_anchor_[iid].y1,
                   item_anchor_[iid].x2, item_anchor_[iid].y2,
                   t->box.x1, t->box.y1, t->box.x2, t->box.y2);
            item_anchor_[iid] = t->box;   // 更新锚点到新位置
            res.happened = true;
        }

        // 记录这一帧位置（给下一帧判 settled 用）
        item_last_box_[iid] = t->box;

        // 正常刷新库存（位置、状态、last_seen）
        inventory_.update_seen(iid, t->track_id, t->box, t->score, frame_id);
    }

    // ================================================================
    //  Step 4: 处理 held 物品（被手拿着的）— 按"物品 track 跟手走"的思路
    //    核心原则：只要画面里还有手（含惯性），held 物品就一直 held，
    //              绝不判取出。只有手真的离开画面才考虑取出。
    // ================================================================
    std::set<int> consumed_tids;   // 被整理"消费"的新 track，Step 6 不当新物品
    std::vector<int> held_done;    // 处理完的 held（待移除）

    for (auto& kv : held_items_) {
        int item_id = kv.first;
        HeldItem& hi = kv.second;

        InventoryItem* item = inventory_.find_by_item(item_id);
        if (!item) { held_done.push_back(item_id); continue; }

        // 4a) 这个 item 又在它"原位置附近"重新出现了 → 没被拿走，恢复正常
        //     注意：要求新出现的同类 track 跟原位置接近，才算"放回原处/没拿"
        const Track* reappear_at_origin = nullptr;
        for (const Track* f : foods) {
            if (f->cls_id != hi.cls_id) continue;
            if (consumed_tids.count(f->track_id)) continue;
            // 跟原位置中心距离很近 → 认为是原物品又被看到了
            if (center_distance(f->box, hi.original_pos) < 60.0f) {
                reappear_at_origin = f;
                break;
            }
        }
        if (reappear_at_origin) {
            // 恢复：item 绑回这个 track，位置还在原处附近
            inventory_.relocate_item(item_id, reappear_at_origin->track_id,
                                     reappear_at_origin->box,
                                     reappear_at_origin->score, frame_id);
            consumed_tids.insert(reappear_at_origin->track_id);
            held_done.push_back(item_id);
            printf("[DBG] item#%d 在原位附近重现 → 取消 held (没被拿走)\n", item_id);
            continue;
        }

        // 4b) 手还在画面里吗？（任何一只手都算，含惯性期，抗手识别不稳）
        bool any_hand = hand_effective;

        // 4c) 找有没有同类新 track 出现在"远离原位置"的地方 → 放下了 → 整理
        const Track* putdown = nullptr;
        for (const Track* f : foods) {
            if (f->cls_id != hi.cls_id) continue;
            if (inventory_.find_by_track(f->track_id) != nullptr) continue;  // 已在库存
            if (consumed_tids.count(f->track_id)) continue;
            // 远离原位置才算"挪到新地方"
            if (center_distance(f->box, hi.original_pos) >= 60.0f) {
                putdown = f;
                break;
            }
        }

        if (putdown) {
            // ★ 整理：同一个 item_id 重新绑定到新 track + 新位置（身份不变）
            printf("\n\033[1;32m[EVENT]\033[0m 整理: item#%d %s (置信度 %.0f%%) "
                   "从(%.0f,%.0f)~(%.0f,%.0f) → (%.0f,%.0f)~(%.0f,%.0f)\n",
                   item_id, coco_cls_to_name(hi.cls_id), putdown->score * 100,
                   hi.original_pos.x1, hi.original_pos.y1,
                   hi.original_pos.x2, hi.original_pos.y2,
                   putdown->box.x1, putdown->box.y1, putdown->box.x2, putdown->box.y2);
            inventory_.relocate_item(item_id, putdown->track_id, putdown->box,
                                     putdown->score, frame_id);
            consumed_tids.insert(putdown->track_id);
            // 同步平滑路径的锚点，避免下一帧重复报整理
            item_anchor_[item_id] = putdown->box;
            item_last_box_[item_id] = putdown->box;
            held_done.push_back(item_id);
            res.happened = true;
            continue;
        }

        // 4d) 没放下、没重现
        if (any_hand) {
            // ★ 手还在画面 → 物品"跟着手走"，继续 held，绝不判取出
            //   把 held 物品的位置更新成最近的手的位置（视觉上跟着手）
            //   找离原位置最近的手
            const Track* nearest_hand = nullptr;
            float best_d = 1e9f;
            for (const Track* h : hands) {
                float d = center_distance(h->box, item->box);
                if (d < best_d) { best_d = d; nearest_hand = h; }
            }
            if (nearest_hand) {
                // 物品位置 = 手的位置（库存里更新，但不报事件）
                item->box = nearest_hand->box;
                item->last_seen_frame = frame_id;  // 防止被 Step 7 超时删
                hi.hand_track_id = nearest_hand->track_id;
            }
            // 继续 held，等放下或手离开
        } else {
            // 手不在画面（含惯性也过了）→ 物品被带走 → 取出
            printf("\n\033[1;32m[EVENT]\033[0m 取出: item#%d %s (置信度 %.0f%%) "
                   "原位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
                   item_id, coco_cls_to_name(hi.cls_id), hi.score * 100,
                   hi.original_pos.x1, hi.original_pos.y1,
                   hi.original_pos.x2, hi.original_pos.y2);
            inventory_.remove_item(item_id);
            held_done.push_back(item_id);
            res.happened = true;
        }
    }

    for (int id : held_done) held_items_.erase(id);


    // ================================================================
    //  Step 5: 库存里"看不到 + 不在 held"的 VISIBLE 物品 → 判 held
    // ================================================================
    for (const auto& kv : inventory_.items()) {
        const InventoryItem& item = kv.second;
        if (item.status != ItemStatus::VISIBLE) continue;
        if (held_items_.find(item.item_id) != held_items_.end()) continue;
        // 当前帧还能看到（track 匹配）→ 跳过
        if (item.track_id >= 0 && seen_tids.count(item.track_id)) continue;

        // 看不到了，检查有没有手盖住它（含手臂延伸区）
        int cover_hand = -1;
        for (const Track* h : hands) {
            if (hand_covers(h, item.box, FRAME_W, FRAME_H)) { cover_hand = h->track_id; break; }
        }
        if (cover_hand >= 0) {
            HeldItem hi;
            hi.item_id = item.item_id;
            hi.cls_id = item.cls_id;
            hi.original_pos = item.box;
            hi.score = item.score;
            hi.hand_track_id = cover_hand;
            hi.pickup_frame = frame_id;
            held_items_[item.item_id] = hi;
            printf("[DBG] item#%d (%s) 被手盖住 → held @frame=%d\n",
                   item.item_id, coco_cls_to_name(item.cls_id), frame_id);
        }
        // 没手盖住 → 什么都不做（保持 VISIBLE，靠 Step 7 超时统一处理）
        // 这是抗抖动的关键：单帧没看到不立刻改状态
    }

    // ================================================================
    //  Step 6: 新 track → 候选新物品（连续 N 帧确认才算 IN，抗抖动）
    // ================================================================
    // 6a) 先更新/新增候选
    for (const Track* t : foods) {
        if (inventory_.find_by_track(t->track_id) != nullptr) continue;  // 已在库存
        if (consumed_tids.count(t->track_id)) continue;                  // 被整理消费

        auto it = pending_news_.find(t->track_id);
        if (it == pending_news_.end()) {
            PendingNew pn;
            pn.track_id = t->track_id;
            pn.cls_id = t->cls_id;
            pn.box = t->box;
            pn.score = t->score;
            pn.first_seen_frame = frame_id;
            pn.seen_count = 1;
            pending_news_[t->track_id] = pn;
        } else {
            it->second.seen_count++;
            it->second.box = t->box;
            it->second.score = t->score;
        }
    }

    // 6b) 清理：当前帧没看到的候选直接丢弃（抖动）
    for (auto it = pending_news_.begin(); it != pending_news_.end(); ) {
        if (!seen_tids.count(it->first)) {
            it = pending_news_.erase(it);
        } else {
            ++it;
        }
    }

    // 6c) 确认：连续够 N 帧的候选 → 正式 IN
    std::vector<int> confirmed;
    for (auto& kv : pending_news_) {
        if (kv.second.seen_count >= NEW_CONFIRM_FRAMES) {
            confirmed.push_back(kv.first);
        }
    }
    for (int tid : confirmed) {
        const PendingNew& pn = pending_news_[tid];
        int new_id = inventory_.add_item(pn.track_id, pn.cls_id, pn.box, pn.score, frame_id);
        printf("\n\033[1;32m[EVENT]\033[0m 放入: item#%d %s (置信度 %.0f%%) "
               "位置=(%.0f,%.0f)~(%.0f,%.0f)\n",
               new_id, coco_cls_to_name(pn.cls_id), pn.score * 100,
               pn.box.x1, pn.box.y1, pn.box.x2, pn.box.y2);
        pending_news_.erase(tid);
        res.happened = true;
    }

    // ================================================================
    //  Step 7: 物品长时间消失（无手、非 held）→ 标记并超时删除 → OUT
    //    抗抖动：必须连续 occluded_grace_frames 帧没看到才删
    // ================================================================
    std::vector<int> to_remove;
    for (const auto& kv : inventory_.items()) {
        const InventoryItem& item = kv.second;
        if (held_items_.find(item.item_id) != held_items_.end()) continue;
        // 当前帧还能看到 → 跳过
        if (item.track_id >= 0 && seen_tids.count(item.track_id)) continue;
        // 有手在画面（含惯性期）→ 绝不删（可能遮挡 / 手识别不稳）
        if (hand_effective) continue;
        // 没手 + 超时没看到 → 真的没了
        if (frame_id - item.last_seen_frame > occluded_grace_frames_) {
            printf("\n\033[1;32m[EVENT]\033[0m 取出: item#%d %s (置信度 %.0f%%) "
                   "原位置=(%.0f,%.0f)~(%.0f,%.0f) [超时]\n",
                   item.item_id, coco_cls_to_name(item.cls_id), item.score * 100,
                   item.box.x1, item.box.y1, item.box.x2, item.box.y2);
            to_remove.push_back(item.item_id);
            res.happened = true;
        }
    }
    for (int id : to_remove) inventory_.remove_item(id);

    // ================================================================
    //  Step 7.5: 清理已不在库存的 item 的锚点记录（防止 item_id 复用时残留）
    // ================================================================
    for (auto it = item_anchor_.begin(); it != item_anchor_.end(); ) {
        if (inventory_.find_by_item(it->first) == nullptr) it = item_anchor_.erase(it);
        else ++it;
    }
    for (auto it = item_last_box_.begin(); it != item_last_box_.end(); ) {
        if (inventory_.find_by_item(it->first) == nullptr) it = item_last_box_.erase(it);
        else ++it;
    }

    // ================================================================
    //  Step 8: 有变化时打印库存
    // ================================================================
    if (res.happened) {
        printf("[SESSION] inventory now %zu items\n", inventory_.size());
        inventory_.print("  ");
    }

    return res;
}

}  // namespace fridge
