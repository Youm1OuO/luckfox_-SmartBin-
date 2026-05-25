// ============================================================================
//  hand_state.cc
//  手部状态机实现 + 原始事件配对
//
//  关键逻辑回顾（同时也是答辩讲解的故事）：
//
//    1) 我们不直接判断"冰箱里多了/少了什么"，而是盯着每只手，
//       看它从进画面到出画面之间，发生了哪些"拿起"和"放下"的原始事件。
//
//    2) 每只手维护一个 4 状态机（EMPTY / SUS_HOLD / HOLD / SUS_RELEASE）。
//       SUS_*（疑似）状态需要连续 N 帧确认才升级，避免单帧抖动。
//
//    3) 状态转移瞬间产生 RawEvent (PICKUP/PUTDOWN)。
//       - EMPTY → HOLD     的瞬间产生 PICKUP（拿起）
//       - HOLD  → EMPTY    的瞬间产生 PUTDOWN（放下）
//
//    4) 当一只手 track 销毁（手离开画面）时，把它积累的 RawEvent 配对：
//       - 若手是"持物进 + 空手出"，第一个 PUTDOWN 是真"放入冰箱"，
//         之后的 PICKUP+PUTDOWN 配对都是"整理"
//       - 若手是"空手进 + 持物出"，最后一个 PICKUP 是真"取出冰箱"
//       - 若手是"空手进 + 空手出"，全部 PICKUP+PUTDOWN 配对都是"整理"
// ============================================================================
#include "hand_state.h"
#include "fridge_config.h"

#include <algorithm>

namespace fridge {

const char* hand_state_to_str(HandState s) {
    switch (s) {
        case HandState::EMPTY:       return "EMPTY";
        case HandState::SUS_HOLD:    return "SUS_HOLD";
        case HandState::HOLD:        return "HOLD";
        case HandState::SUS_RELEASE: return "SUS_RELEASE";
    }
    return "?";
}

const char* final_event_type_to_str(FinalEventType t) {
    switch (t) {
        case FinalEventType::IN:       return "IN";
        case FinalEventType::OUT:      return "OUT";
        case FinalEventType::RELOCATE: return "RELOCATE";
    }
    return "?";
}

// ----------- 工具函数 -----------

void HandStateManager::split_tracks(const std::vector<Track>& tracks,
                                    std::vector<const Track*>& hands,
                                    std::vector<const Track*>& objects) {
    hands.clear();
    objects.clear();
    for (const auto& t : tracks) {
        if (is_hand(t.cls_id))      hands.push_back(&t);
        else if (is_food(t.cls_id)) objects.push_back(&t);
    }
}

const Track* HandStateManager::find_top_overlapping_object(
        const Track& hand,
        const std::vector<const Track*>& objects,
        float iou_thresh) {
    const Track* best = nullptr;
    float best_iou = iou_thresh;       // 起始为阈值，低于此值的不要
    for (const Track* o : objects) {
        float v = iou(hand.box, o->box);
        if (v > best_iou) {
            best_iou = v;
            best = o;
        }
    }
    return best;
}

// ----------- 主流程 -----------

std::vector<FinalEvent> HandStateManager::update(
        const std::vector<Track>& tracks, int frame_id) {

    std::vector<FinalEvent> out_events;

    // ===== Step 1：拆分 hands / objects =====
    std::vector<const Track*> hands, objects;
    split_tracks(tracks, hands, objects);

    // ===== Step 2：处理本帧每一只手 =====
    // 顺便把出现的 hand_track_id 记下来，等下用于检测哪些手"消失了"
    std::vector<int> seen_hand_ids;
    seen_hand_ids.reserve(hands.size());

    for (const Track* h : hands) {
        seen_hand_ids.push_back(h->track_id);

        // 确保这只手的 log 存在；不存在就建一个
        auto it = hand_logs_.find(h->track_id);
        if (it == hand_logs_.end()) {
            // 新出现的手：初始化 log
            HandLog log;
            log.hand_track_id = h->track_id;
            log.state = HandState::EMPTY;
            log.candidate_cls_id = -1;
            log.candidate_obj_track_id = -1;
            log.sus_counter = 0;
            log.held_cls_id = -1;
            log.held_obj_track_id = -1;
            log.last_seen_frame = frame_id;

            // 关键：判断这只手"出生时"是否已经持物
            // 做法：检查 hand 的 bbox 是否和某个物品 bbox IoU >= HIGH
            const Track* coexist = find_top_overlapping_object(
                *h, objects, HAND_OBJ_IOU_HIGH);
            log.entered_holding = (coexist != nullptr);
            if (log.entered_holding) {
                // 直接进入 HOLD 状态（不走 SUS_HOLD，因为已经"既成事实"）
                log.state = HandState::HOLD;
                log.held_cls_id = coexist->cls_id;
                log.held_obj_track_id = coexist->track_id;
                // 注意：这里不产生 PICKUP 事件，因为这只手"带物入场"，
                //       这个物品本身就是要"放入冰箱"的，等出场时再统一结算
            }
            hand_logs_[h->track_id] = log;
        } else {
            // 已存在：跑状态机
            process_hand(it->second, *h, objects, frame_id);
            it->second.last_seen_frame = frame_id;
        }
    }

    // ===== Step 3：检测哪些手 track 消失了 =====
    // 思路：log 里有但本帧 seen_hand_ids 里没有 → 这只手刚离开
    std::vector<int> disappeared;
    for (auto& kv : hand_logs_) {
        int hid = kv.first;
        if (std::find(seen_hand_ids.begin(), seen_hand_ids.end(), hid)
            == seen_hand_ids.end()) {
            disappeared.push_back(hid);
        }
    }

    // 对每一只消失的手做最终事件配对
    for (int hid : disappeared) {
        auto it = hand_logs_.find(hid);
        if (it == hand_logs_.end()) continue;
        std::vector<FinalEvent> ev = finalize_hand(it->second);
        for (auto& e : ev) {
            out_events.push_back(std::move(e));
        }
        hand_logs_.erase(it);
    }

    return out_events;
}

// 处理某只手的本帧逻辑（状态机核心）
void HandStateManager::process_hand(HandLog& log,
                                    const Track& hand,
                                    const std::vector<const Track*>& objects,
                                    int frame_id) {
    switch (log.state) {

    // -----------------------------------------------------------------
    // 当前是空手 EMPTY：
    //   找有没有物品和手 IoU 跃升到 HIGH
    //   → 进入 SUS_HOLD，锁定该物品作为候选
    // -----------------------------------------------------------------
    case HandState::EMPTY: {
        const Track* candidate = find_top_overlapping_object(
            hand, objects, HAND_OBJ_IOU_HIGH);
        if (candidate != nullptr) {
            log.state = HandState::SUS_HOLD;
            log.candidate_cls_id = candidate->cls_id;
            log.candidate_obj_track_id = candidate->track_id;
            log.candidate_box = candidate->box;
            log.sus_counter = 1;     // 已经持续 1 帧
        }
        // 没找到 → 保持 EMPTY
        break;
    }

    // -----------------------------------------------------------------
    // 疑似拿起 SUS_HOLD：
    //   - 候选物品仍然 IoU 高 → 计数 +1，到 CONFIRM_FRAMES 升级 HOLD
    //   - 候选消失或 IoU 掉了 → 回 EMPTY（候选作废）
    // -----------------------------------------------------------------
    case HandState::SUS_HOLD: {
        // 锁定的候选物品还存在吗？找它的 track
        const Track* same_obj = nullptr;
        for (const Track* o : objects) {
            if (o->track_id == log.candidate_obj_track_id) {
                same_obj = o;
                break;
            }
        }

        bool keep = false;
        if (same_obj) {
            float v = iou(hand.box, same_obj->box);
            if (v >= HAND_OBJ_IOU_HIGH) {
                keep = true;
                log.candidate_box = same_obj->box;
            }
            // 灰区 (LOW < v < HIGH)：不升级也不降级，直接维持
            else if (v >= HAND_OBJ_IOU_LOW) {
                keep = true;          // 维持 SUS_HOLD 但不增加 counter
                // 不增加 counter，等明确再说
                break;                // 直接 return 这一帧
            }
        }

        if (keep) {
            log.sus_counter++;
            if (log.sus_counter >= CONFIRM_FRAMES) {
                // ★ 升级为 HOLD，触发 PICKUP 原始事件
                log.state = HandState::HOLD;
                log.held_cls_id = log.candidate_cls_id;
                log.held_obj_track_id = log.candidate_obj_track_id;

                RawEvent e;
                e.type = RawEventType::PICKUP;
                e.cls_id = log.candidate_cls_id;
                e.pos = log.candidate_box;        // 物品被拿起的位置
                e.frame_id = frame_id;
                log.raw_events.push_back(e);

                log.sus_counter = 0;
            }
        } else {
            // 候选作废，回 EMPTY
            log.state = HandState::EMPTY;
            log.candidate_cls_id = -1;
            log.candidate_obj_track_id = -1;
            log.sus_counter = 0;
        }
        break;
    }

    // -----------------------------------------------------------------
    // 已确认持物 HOLD：
    //   只检查"持有物品"是否还和手 IoU 高
    //   - IoU 还高 → 维持 HOLD
    //   - IoU 掉到 LOW 以下 → 进入 SUS_RELEASE
    //   不再去识别"是不是抓了别的东西"——一只手同时只持一个物品
    // -----------------------------------------------------------------
    case HandState::HOLD: {
        const Track* held = nullptr;
        for (const Track* o : objects) {
            if (o->track_id == log.held_obj_track_id) {
                held = o;
                break;
            }
        }

        if (!held) {
            // 持有物品 track 没了（可能被遮、可能丢失）
            // 进入 SUS_RELEASE，记录最后已知位置
            log.state = HandState::SUS_RELEASE;
            log.candidate_cls_id = log.held_cls_id;
            log.candidate_obj_track_id = log.held_obj_track_id;
            // candidate_box 用上次记录的（即手的当前位置最近似估计）
            log.candidate_box = hand.box;
            log.sus_counter = 1;
        } else {
            float v = iou(hand.box, held->box);
            if (v >= HAND_OBJ_IOU_LOW) {
                // 维持 HOLD，更新位置以备 SUS_RELEASE 使用
                log.candidate_box = held->box;
            } else {
                // 进入 SUS_RELEASE
                log.state = HandState::SUS_RELEASE;
                log.candidate_box = held->box;
                log.sus_counter = 1;
            }
        }
        break;
    }

    // -----------------------------------------------------------------
    // 疑似放下 SUS_RELEASE：
    //   - 持有物品仍 IoU 低 → 计数 +1，到 CONFIRM_FRAMES 真放下，回 EMPTY
    //   - 持有物品 IoU 又升回去 → 回 HOLD（假警报）
    // -----------------------------------------------------------------
    case HandState::SUS_RELEASE: {
        const Track* obj = nullptr;
        for (const Track* o : objects) {
            if (o->track_id == log.held_obj_track_id) {
                obj = o;
                break;
            }
        }

        bool still_low = true;        // 默认假定还低（持有物品都丢失也算"低"）
        if (obj) {
            float v = iou(hand.box, obj->box);
            if (v >= HAND_OBJ_IOU_HIGH) {
                // 又抓回去了 → 回 HOLD
                log.state = HandState::HOLD;
                log.candidate_box = obj->box;
                log.sus_counter = 0;
                break;
            }
            if (v >= HAND_OBJ_IOU_LOW) {
                // 灰区：维持 SUS_RELEASE，不增加 counter
                log.candidate_box = obj->box;
                break;
            }
            // 否则 v < LOW，still_low = true，下面继续走
            log.candidate_box = obj->box;
        }

        if (still_low) {
            log.sus_counter++;
            if (log.sus_counter >= CONFIRM_FRAMES) {
                // ★ 确认放下，触发 PUTDOWN 原始事件
                RawEvent e;
                e.type = RawEventType::PUTDOWN;
                e.cls_id = log.held_cls_id;
                // 放下位置：当前候选 box 的位置（最后看到物品的地方）
                e.pos = log.candidate_box;
                e.frame_id = frame_id;
                log.raw_events.push_back(e);

                // 状态归零
                log.state = HandState::EMPTY;
                log.held_cls_id = -1;
                log.held_obj_track_id = -1;
                log.candidate_cls_id = -1;
                log.candidate_obj_track_id = -1;
                log.sus_counter = 0;
            }
        }
        break;
    }
    }
}

// ----------- 一只手离开画面后的事件配对 -----------

std::vector<FinalEvent> HandStateManager::finalize_hand(HandLog& log) {
    std::vector<FinalEvent> result;

    // 把 raw_events 分成 PICKUP 队列和 PUTDOWN 队列（按发生顺序）
    std::vector<RawEvent> pickups;
    std::vector<RawEvent> putdowns;
    for (const auto& e : log.raw_events) {
        if (e.type == RawEventType::PICKUP)        pickups.push_back(e);
        else if (e.type == RawEventType::PUTDOWN)  putdowns.push_back(e);
    }

    // ----- 情况 1：手是"持物入场" (entered_holding=true) -----
    // 第一个 PUTDOWN 是真"放入冰箱"
    if (log.entered_holding && !putdowns.empty()) {
        const auto& put = putdowns.front();
        FinalEvent fe;
        fe.type = FinalEventType::IN;
        fe.cls_id = put.cls_id;
        fe.to_pos = put.pos;
        fe.frame_begin = log.last_seen_frame;   // 近似
        fe.frame_end = put.frame_id;
        fe.hand_track_id = log.hand_track_id;
        result.push_back(fe);
        putdowns.erase(putdowns.begin());
    }

    // ----- 情况 2：剩余的 pickups 和 putdowns 顺序两两配对 → RELOCATE -----
    while (!pickups.empty() && !putdowns.empty()) {
        const auto& pu = pickups.front();
        const auto& pd = putdowns.front();
        FinalEvent fe;
        fe.type = FinalEventType::RELOCATE;
        fe.cls_id = pu.cls_id;
        fe.from_pos = pu.pos;
        fe.to_pos = pd.pos;
        fe.frame_begin = pu.frame_id;
        fe.frame_end = pd.frame_id;
        fe.hand_track_id = log.hand_track_id;
        result.push_back(fe);
        pickups.erase(pickups.begin());
        putdowns.erase(putdowns.begin());
    }

    // ----- 情况 3：剩余的 pickup（没对应 putdown）→ OUT 取出 -----
    // 说明：手最终带着这个东西离开了画面
    for (const auto& pu : pickups) {
        FinalEvent fe;
        fe.type = FinalEventType::OUT;
        fe.cls_id = pu.cls_id;
        fe.from_pos = pu.pos;
        fe.frame_begin = pu.frame_id;
        fe.frame_end = log.last_seen_frame;
        fe.hand_track_id = log.hand_track_id;
        result.push_back(fe);
    }

    // ----- 情况 4：剩余的 putdown（没对应 pickup）-----
    // 一般不出现：如果出现，说明 entered_holding 判定漏了，补救一下当作 IN
    for (const auto& pd : putdowns) {
        FinalEvent fe;
        fe.type = FinalEventType::IN;
        fe.cls_id = pd.cls_id;
        fe.to_pos = pd.pos;
        fe.frame_begin = log.last_seen_frame;
        fe.frame_end = pd.frame_id;
        fe.hand_track_id = log.hand_track_id;
        result.push_back(fe);
    }

    return result;
}

}  // namespace fridge
