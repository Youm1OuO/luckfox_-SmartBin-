// ============================================================================
//  session_settlement.cc
//  3.0 session no-hand settlement, occlusion, and events
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

using namespace session_internal;

void SessionManager::mark_pending_out_(int item_id) {
    pending_out_ids_.insert(item_id);
    OperationTrack* track = find_runtime_for_item_(item_id);
    if (track && !track->is_suspect_new) {
        track->resolution = ExistingItemResolution::OUT_CONFIRMED;
        track->release_reason = ReleaseReason::NONE;
        track->needs_no_hand_settlement = false;
        track->stable_near_original_no_hand_count = 0;
        track->has_stable_near_original_box = false;
        track->stable_near_original_box = BBox();
        trace_track_("NO-HAND", *track, "confirm-out-after-direct-missing-frames");
    }
}

void SessionManager::clear_visible_count_settlement_(bool restore_uncommitted_outs) {
    for (std::map<int, int>::const_iterator it = visible_count_missing_counts_.begin();
         it != visible_count_missing_counts_.end(); ++it) {
        OperationTrack* track = find_runtime_for_item_(it->first);
        if (track && !track->is_suspect_new) {
            track->no_hand_missing_count = 0;
        }
    }
    if (restore_uncommitted_outs) {
        for (std::set<int>::const_iterator it = visible_count_confirmed_out_ids_.begin();
             it != visible_count_confirmed_out_ids_.end(); ++it) {
            pending_out_ids_.erase(*it);
            OperationTrack* track = find_runtime_for_item_(*it);
            if (track && !track->is_suspect_new &&
                track->resolution == ExistingItemResolution::OUT_CONFIRMED) {
                track->resolution = ExistingItemResolution::NONE;
                track->needs_no_hand_settlement = true;
                track->release_reason = ReleaseReason::NONE;
                trace_track_("VISIBLE-COUNT", *track,
                             "retract-uncommitted-visible-count-out");
            }
        }
    }
    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    visible_count_out_candidate_ids_.clear();
    visible_count_missing_counts_.clear();
    visible_count_confirmed_out_ids_.clear();
    visible_count_prior_survivors_by_cls_.clear();
    visible_count_prior_survivor_boxes_by_cls_.clear();
}

void SessionManager::prepare_visible_count_settlement_(
        const std::vector<Detection>& detections) {
    std::map<int, std::vector<int> > detection_indices_by_cls;
    for (size_t di = 0; di < detections.size(); ++di) {
        if (detections[di].cls_id >= 0 && detections[di].box.area() > 0.0f) {
            detection_indices_by_cls[detections[di].cls_id].push_back(
                static_cast<int>(di));
        }
    }

    std::map<int, std::vector<int> > old_visible_ids_by_cls;
    std::set<int> classes_with_runtime;
    for (std::map<int, InventoryItem>::const_iterator original =
             operation_start_inventory_.begin();
         original != operation_start_inventory_.end(); ++original) {
        std::map<int, InventoryItem>::const_iterator current =
            working_inventory_.find(original->first);
        if (current == working_inventory_.end() ||
            original->second.status == ItemStatus::OCCLUDED ||
            current->second.status == ItemStatus::OCCLUDED) {
            continue;
        }
        const bool confirmed_by_visible_count =
            visible_count_confirmed_out_ids_.count(original->first) > 0;
        if (pending_out_ids_.count(original->first) && !confirmed_by_visible_count) {
            continue;
        }
        const OperationTrack* runtime = find_runtime_for_item_(original->first);
        if (runtime && runtime->is_suspect_new) continue;
        if (runtime && !confirmed_by_visible_count &&
            (runtime->resolution == ExistingItemResolution::OUT_CONFIRMED ||
             runtime->resolution == ExistingItemResolution::OCCLUDED_CONFIRMED)) {
            continue;
        }
        old_visible_ids_by_cls[original->second.cls_id].push_back(original->first);
        if (runtime && (is_active_runtime_track(*runtime) ||
                        existing_item_needs_settlement(*runtime) ||
                        confirmed_by_visible_count)) {
            classes_with_runtime.insert(original->second.cls_id);
        }
    }

    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    std::set<int> next_out_candidates;
    std::set<int> active_deficit_classes;
    std::set<int> continuity_reset_item_ids;

    for (std::map<int, std::vector<int> >::const_iterator group =
             old_visible_ids_by_cls.begin();
         group != old_visible_ids_by_cls.end(); ++group) {
        const int cls_id = group->first;
        const std::vector<int>& old_ids = group->second;
        std::map<int, std::vector<int> >::const_iterator detections_it =
            detection_indices_by_cls.find(cls_id);
        const std::vector<int> empty_detection_indices;
        const std::vector<int>& detection_indices =
            detections_it == detection_indices_by_cls.end()
                ? empty_detection_indices : detections_it->second;

        bool has_protected_old_track = false;
        for (size_t oi = 0; oi < old_ids.size(); ++oi) {
            const OperationTrack* runtime = find_runtime_for_item_(old_ids[oi]);
            if (runtime && is_claim_protected(*runtime)) {
                has_protected_old_track = true;
                break;
            }
        }
        if (old_ids.size() < 2 || !classes_with_runtime.count(cls_id) ||
            has_protected_old_track || detection_indices.size() >= old_ids.size()) {
            continue;
        }

        active_deficit_classes.insert(cls_id);
        std::set<int> remaining_items(old_ids.begin(), old_ids.end());
        std::set<int> remaining_detections;
        std::set<int> confirmed_d_detection_indices;
        for (size_t di = 0; di < detection_indices.size(); ++di) {
            const int detection_index = detection_indices[di];
            const Detection& detection = detections[detection_index];
            const bool owned_by_confirmed_d = detection_matches_confirmed_suspect_d(
                detection, track_buffer_);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d input box=(%.1f,%.1f,%.1f,%.1f) score=%.3f confirmed-d=%d",
                   cls_id, detection_index, detection.box.x1, detection.box.y1,
                   detection.box.x2, detection.box.y2, detection.score,
                   owned_by_confirmed_d ? 1 : 0);
            if (!owned_by_confirmed_d) {
                remaining_detections.insert(detection_index);
            } else {
                confirmed_d_detection_indices.insert(detection_index);
            }
        }
        std::set<int> survivors;
        std::map<int, const char*> assignment_reasons;

        // 先用双方都唯一的严格原位框锁住静止邻居。这样 B 的独立框不会
        // 被 A 的宽松路径再次解释为歧义。
        bool made_progress = true;
        while (made_progress) {
            made_progress = false;
            std::map<int, std::vector<int> > item_candidates;
            std::map<int, std::vector<int> > detection_candidates;
            for (std::set<int>::const_iterator item_id = remaining_items.begin();
                 item_id != remaining_items.end(); ++item_id) {
                std::map<int, InventoryItem>::const_iterator original =
                    operation_start_inventory_.find(*item_id);
                if (original == operation_start_inventory_.end()) continue;
                const BBox reference = original->second.base_box.area() > 0.0f
                    ? original->second.base_box : original->second.box;
                for (std::set<int>::const_iterator detection_index =
                         remaining_detections.begin();
                     detection_index != remaining_detections.end(); ++detection_index) {
                    if (!strict_match_box(original->second.cls_id, reference,
                                          detections[*detection_index].cls_id,
                                          detections[*detection_index].box)) {
                        continue;
                    }
                    item_candidates[*item_id].push_back(*detection_index);
                    detection_candidates[*detection_index].push_back(*item_id);
                }
            }
            for (std::map<int, std::vector<int> >::const_iterator item =
                     item_candidates.begin();
                 item != item_candidates.end(); ++item) {
                if (item->second.size() != 1) continue;
                const int detection_index = item->second.front();
                if (detection_candidates[detection_index].size() != 1 ||
                    !remaining_items.count(item->first) ||
                    !remaining_detections.count(detection_index)) {
                    continue;
                }
                visible_count_detection_owner_[detection_index] = item->first;
                visible_count_survivor_ids_.insert(item->first);
                survivors.insert(item->first);
                assignment_reasons[detection_index] = "unique-direct-static";
                remaining_items.erase(item->first);
                remaining_detections.erase(detection_index);
                trace_("VISIBLE-COUNT",
                       "cls=%d detection=%d reserved-for-item=%d reason=unique-direct-static",
                       cls_id, detection_index, item->first);
                made_progress = true;
            }
        }

        // 只有画面真的不可区分时才使用上一张无手帧的存活 id 作为稳定决胜；
        // 它不是 map 遍历顺序，也不会单独证明 MOVED。
        const std::map<int, std::set<int> >::const_iterator previous_survivors_it =
            visible_count_prior_survivors_by_cls_.find(cls_id);
        const std::set<int> empty_previous_survivors;
        const std::set<int>& previous_survivors =
            previous_survivors_it == visible_count_prior_survivors_by_cls_.end()
                ? empty_previous_survivors : previous_survivors_it->second;
        const std::map<int, std::map<int, BBox> >::const_iterator previous_boxes_it =
            visible_count_prior_survivor_boxes_by_cls_.find(cls_id);
        const std::map<int, BBox> empty_previous_boxes;
        const std::map<int, BBox>& previous_survivor_boxes =
            previous_boxes_it == visible_count_prior_survivor_boxes_by_cls_.end()
                ? empty_previous_boxes : previous_boxes_it->second;
        for (std::set<int>::const_iterator prior = previous_survivors.begin();
             prior != previous_survivors.end(); ++prior) {
            if (!remaining_items.count(*prior) || remaining_detections.empty()) continue;
            const std::map<int, BBox>::const_iterator previous_box =
                previous_survivor_boxes.find(*prior);
            if (previous_box == previous_survivor_boxes.end()) continue;
            const OperationTrack* runtime = find_runtime_for_item_(*prior);
            std::map<int, InventoryItem>::const_iterator original =
                operation_start_inventory_.find(*prior);
            if (original == operation_start_inventory_.end()) continue;
            int best_detection = -1;
            float best_cost = std::numeric_limits<float>::infinity();
            for (std::set<int>::const_iterator detection_index =
                     remaining_detections.begin();
                 detection_index != remaining_detections.end(); ++detection_index) {
                if (!visible_count_survivor_box_is_continuous(
                        cls_id, previous_box->second,
                        detections[*detection_index].box)) {
                    trace_("VISIBLE-COUNT",
                           "cls=%d detection=%d prior-survivor=%d action=reject-prior reason=box-discontinuous",
                           cls_id, *detection_index, *prior);
                    continue;
                }
                const float cost = visible_count_owner_cost(
                    original->second, runtime, detections[*detection_index]);
                if (cost + 0.0001f < best_cost ||
                    (std::fabs(cost - best_cost) <= 0.0001f &&
                     (best_detection < 0 || *detection_index < best_detection))) {
                    best_cost = cost;
                    best_detection = *detection_index;
                }
            }
            if (best_detection < 0) continue;
            visible_count_detection_owner_[best_detection] = *prior;
            visible_count_survivor_ids_.insert(*prior);
            survivors.insert(*prior);
            assignment_reasons[best_detection] = "previous-no-hand-survivor";
            remaining_items.erase(*prior);
            remaining_detections.erase(best_detection);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d reserved-for-item=%d reason=previous-no-hand-survivor",
                   cls_id, best_detection, *prior);
        }

        // 仍不可区分时，选择全局最小成本 pair；完全相同才按 item_id /
        // detection index 决胜，保证两张连续无手帧不会因遍历顺序换身份。
        while (!remaining_items.empty() && !remaining_detections.empty()) {
            int best_item = -1;
            int best_detection = -1;
            float best_cost = std::numeric_limits<float>::infinity();
            for (std::set<int>::const_iterator item_id = remaining_items.begin();
                 item_id != remaining_items.end(); ++item_id) {
                const OperationTrack* runtime = find_runtime_for_item_(*item_id);
                std::map<int, InventoryItem>::const_iterator original =
                    operation_start_inventory_.find(*item_id);
                if (original == operation_start_inventory_.end()) continue;
                for (std::set<int>::const_iterator detection_index =
                         remaining_detections.begin();
                     detection_index != remaining_detections.end(); ++detection_index) {
                    const float cost = visible_count_owner_cost(
                        original->second, runtime, detections[*detection_index]);
                    const bool better_cost = cost + 0.0001f < best_cost;
                    const bool equal_cost = std::fabs(cost - best_cost) <= 0.0001f;
                    const bool better_tie = equal_cost &&
                        (best_item < 0 || *item_id < best_item ||
                         (*item_id == best_item && *detection_index < best_detection));
                    if (better_cost || better_tie) {
                        best_cost = cost;
                        best_item = *item_id;
                        best_detection = *detection_index;
                    }
                }
            }
            if (best_item < 0 || best_detection < 0) break;
            visible_count_detection_owner_[best_detection] = best_item;
            visible_count_survivor_ids_.insert(best_item);
            survivors.insert(best_item);
            assignment_reasons[best_detection] = "min-cost";
            remaining_items.erase(best_item);
            remaining_detections.erase(best_detection);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d reserved-for-item=%d reason=min-cost",
                   cls_id, best_detection, best_item);
        }

        std::map<int, BBox> current_survivor_boxes;
        for (std::map<int, int>::const_iterator assignment =
                 visible_count_detection_owner_.begin();
             assignment != visible_count_detection_owner_.end(); ++assignment) {
            if (!survivors.count(assignment->second) ||
                detections[assignment->first].cls_id != cls_id) {
                continue;
            }
            current_survivor_boxes[assignment->second] = detections[assignment->first].box;
        }
        bool survivor_box_discontinuous = false;
        for (std::set<int>::const_iterator prior = previous_survivors.begin();
             prior != previous_survivors.end(); ++prior) {
            if (!survivors.count(*prior)) continue;
            const std::map<int, BBox>::const_iterator previous_box =
                previous_survivor_boxes.find(*prior);
            const std::map<int, BBox>::const_iterator current_box =
                current_survivor_boxes.find(*prior);
            if (previous_box == previous_survivor_boxes.end() ||
                current_box == current_survivor_boxes.end() ||
                !visible_count_survivor_box_is_continuous(
                    cls_id, previous_box->second, current_box->second)) {
                survivor_box_discontinuous = true;
                trace_("VISIBLE-COUNT",
                       "cls=%d survivor=%d action=clear-deficit reason=survivor-box-discontinuous "
                       "previous=(%.1f,%.1f,%.1f,%.1f) current=(%.1f,%.1f,%.1f,%.1f)",
                       cls_id, *prior,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.x1,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.y1,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.x2,
                       previous_box == previous_survivor_boxes.end() ? 0.0f : previous_box->second.y2,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.x1,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.y1,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.x2,
                       current_box == current_survivor_boxes.end() ? 0.0f : current_box->second.y2);
            }
        }
        if (survivor_box_discontinuous) {
            continuity_reset_item_ids.insert(old_ids.begin(), old_ids.end());
        }
        visible_count_prior_survivors_by_cls_[cls_id] = survivors;
        visible_count_prior_survivor_boxes_by_cls_[cls_id] = current_survivor_boxes;
        for (size_t oi = 0; oi < old_ids.size(); ++oi) {
            if (!survivors.count(old_ids[oi])) {
                next_out_candidates.insert(old_ids[oi]);
            }
        }
        for (size_t di = 0; di < detection_indices.size(); ++di) {
            const int detection_index = detection_indices[di];
            const Detection& detection = detections[detection_index];
            if (confirmed_d_detection_indices.count(detection_index)) {
                trace_("VISIBLE-COUNT",
                       "cls=%d detection=%d decision=confirmed-d box=(%.1f,%.1f,%.1f,%.1f) score=%.3f",
                       cls_id, detection_index, detection.box.x1, detection.box.y1,
                       detection.box.x2, detection.box.y2, detection.score);
                continue;
            }
            const std::map<int, int>::const_iterator owner =
                visible_count_detection_owner_.find(detection_index);
            if (owner == visible_count_detection_owner_.end()) {
                trace_("VISIBLE-COUNT",
                       "cls=%d detection=%d decision=unassigned box=(%.1f,%.1f,%.1f,%.1f) score=%.3f",
                       cls_id, detection_index, detection.box.x1, detection.box.y1,
                       detection.box.x2, detection.box.y2, detection.score);
                continue;
            }
            const std::map<int, const char*>::const_iterator reason =
                assignment_reasons.find(detection_index);
            trace_("VISIBLE-COUNT",
                   "cls=%d detection=%d decision=reserve item=%d reason=%s box=(%.1f,%.1f,%.1f,%.1f) score=%.3f",
                   cls_id, detection_index, owner->second,
                   reason == assignment_reasons.end() ? "unknown" : reason->second,
                   detection.box.x1, detection.box.y1, detection.box.x2, detection.box.y2,
                   detection.score);
        }
        for (size_t oi = 0; oi < old_ids.size(); ++oi) {
            if (survivors.count(old_ids[oi])) continue;
            for (std::map<int, int>::const_iterator assignment =
                     visible_count_detection_owner_.begin();
                 assignment != visible_count_detection_owner_.end(); ++assignment) {
                if (detections[assignment->first].cls_id != cls_id) continue;
                trace_("VISIBLE-COUNT",
                       "detection=%d reserved_for_item=%d excluded_from_item=%d "
                       "reason=independent-same-class-survivor",
                       assignment->first, assignment->second, old_ids[oi]);
            }
        }
        trace_("VISIBLE-COUNT",
               "cls=%d old-visible=%zu raw-detections=%zu retained-old=%zu deficit=%zu",
               cls_id, old_ids.size(), detection_indices.size(), survivors.size(),
               old_ids.size() - survivors.size());
    }

    for (std::map<int, std::set<int> >::iterator prior =
             visible_count_prior_survivors_by_cls_.begin();
         prior != visible_count_prior_survivors_by_cls_.end();) {
        if (!active_deficit_classes.count(prior->first)) {
            visible_count_prior_survivors_by_cls_.erase(prior++);
        } else {
            ++prior;
        }
    }
    for (std::map<int, std::map<int, BBox> >::iterator prior =
             visible_count_prior_survivor_boxes_by_cls_.begin();
         prior != visible_count_prior_survivor_boxes_by_cls_.end();) {
        if (!active_deficit_classes.count(prior->first)) {
            visible_count_prior_survivor_boxes_by_cls_.erase(prior++);
        } else {
            ++prior;
        }
    }

    // 数量恢复、类别不再短缺或手再次出现之前，都不能让上一轮缺失证据在
    // 随机帧继续累积；已标记但尚未提交的 OUT 也要一并撤销。
    for (std::map<int, int>::iterator missing = visible_count_missing_counts_.begin();
         missing != visible_count_missing_counts_.end();) {
        const bool reset_for_discontinuity =
            continuity_reset_item_ids.count(missing->first) > 0;
        if (next_out_candidates.count(missing->first) && !reset_for_discontinuity) {
            ++missing;
            continue;
        }
        const int item_id = missing->first;
        if (visible_count_confirmed_out_ids_.count(item_id)) {
            pending_out_ids_.erase(item_id);
            visible_count_confirmed_out_ids_.erase(item_id);
            OperationTrack* track = find_runtime_for_item_(item_id);
            if (track && !track->is_suspect_new &&
                track->resolution == ExistingItemResolution::OUT_CONFIRMED) {
                track->resolution = ExistingItemResolution::NONE;
                track->needs_no_hand_settlement = true;
                track->release_reason = ReleaseReason::NONE;
            }
            trace_("VISIBLE-COUNT",
                   "item=%d action=retract-confirmed-out reason=%s",
                   item_id, reset_for_discontinuity
                       ? "survivor-box-discontinuous" : "visible-count-recovered");
        }
        OperationTrack* track = find_runtime_for_item_(item_id);
        if (track && !track->is_suspect_new) track->no_hand_missing_count = 0;
        missing = visible_count_missing_counts_.erase(missing);
    }

    visible_count_out_candidate_ids_.swap(next_out_candidates);
    for (std::set<int>::const_iterator candidate =
             visible_count_out_candidate_ids_.begin();
         candidate != visible_count_out_candidate_ids_.end(); ++candidate) {
        const int old_count = visible_count_missing_counts_[*candidate];
        const int new_count = old_count + 1;
        visible_count_missing_counts_[*candidate] = new_count;
        OperationTrack* track = find_runtime_for_item_(*candidate);
        if (track && !track->is_suspect_new) {
            track->no_hand_missing_count = new_count;
        }
        trace_("VISIBLE-COUNT",
               "item=%d action=count-deficit old=%d new=%d threshold=%d",
               *candidate, old_count, new_count, FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        if (new_count >= FLOW3_NO_HAND_OUT_MISSING_FRAMES &&
            !pending_out_ids_.count(*candidate)) {
            mark_pending_out_(*candidate);
            visible_count_confirmed_out_ids_.insert(*candidate);
            if (track && !track->is_suspect_new) {
                trace_track_("VISIBLE-COUNT", *track,
                             "confirm-out-visible-count-deficit");
            } else {
                trace_("VISIBLE-COUNT",
                       "item=%d action=confirm-out reason=visible-count-deficit",
                       *candidate);
            }
        }
    }
}

void SessionManager::refresh_confirmed_blockers_(
        std::map<int, InventoryItem>* final_items,
        const std::set<int>& observed_working_ids,
        const std::set<int>& confirmed_front_ids,
        std::set<int>* fully_occluded_item_ids) {
    if (!final_items || !fully_occluded_item_ids) return;
    fully_occluded_item_ids->clear();

    // 所有身份、最终可靠框和 pending IN / MOVED 都已经在调用前准备完毕。
    // 这里统一读取上一轮已提交关系和本轮已确认前景，绝不把 hand/suspect D
    // 的临时框直接写成正式 blocker。
    for (std::map<int, InventoryItem>::iterator target = final_items->begin();
         target != final_items->end(); ++target) {
        const int target_id = target->first;
        const std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(target_id);
        const InventoryItem& relationship_source =
            original == operation_start_inventory_.end() ? target->second : original->second;
        // 已确认 MOVED / IN 的 target 已在 final_items 中刷新为本轮可靠最终框；
        // 未观察到的旧 C 则仍保留操作开始时的完整参考框。
        const BBox target_box = target->second.base_box.area() > 0.0f
            ? target->second.base_box : target->second.box;

        std::set<int> effective_block_ids;
        std::vector<BBox> cover_boxes;
        const std::set<int> historical_block_ids = relationship_source.block_ids;

        for (int source = 0; source < 2; ++source) {
            const std::set<int>& candidate_ids = source == 0
                ? historical_block_ids : confirmed_front_ids;
            for (std::set<int>::const_iterator blocker = candidate_ids.begin();
                 blocker != candidate_ids.end(); ++blocker) {
                if (*blocker == target_id || pending_out_ids_.count(*blocker)) continue;
                std::map<int, InventoryItem>::const_iterator front =
                    final_items->find(*blocker);
                if (front == final_items->end()) continue;
                const BBox front_box = front->second.base_box.area() > 0.0f
                    ? front->second.base_box : front->second.box;
                if (target_box.area() <= 0.0f || front_box.area() <= 0.0f ||
                    intersection_area(target_box, front_box) <= BLOCK_OVERLAP_AREA_EPS) {
                    continue;
                }
                if (effective_block_ids.insert(*blocker).second) {
                    cover_boxes.push_back(front_box);
                }
            }
        }

        if (observed_working_ids.count(target_id)) {
            // 直接看见 C 时状态必须为 VISIBLE；仍相交的 blocker 是前景关系，
            // 不是“继续完全遮挡”的状态，因此不能被无条件清空。
            target->second.status = ItemStatus::VISIBLE;
            target->second.block_ids.swap(effective_block_ids);
            continue;
        }

        if (original == operation_start_inventory_.end()) {
            // 本轮新 D 没有直接无手身份时不会成为正式库存目标，更不能凭几何
            // 推导出新的遮挡关系。
            target->second.block_ids.swap(effective_block_ids);
            continue;
        }

        // 未直接观察到的旧 C 继续使用操作开始时的可靠身份和 base_box，丢弃
        // 有手阶段可能留下的临时 status / block_ids，只保留有效历史和新前景。
        InventoryItem retained = original->second;
        retained.block_ids.swap(effective_block_ids);
        if (fully_covered_by(target_box, cover_boxes)) {
            retained.status = ItemStatus::OCCLUDED;
            fully_occluded_item_ids->insert(target_id);
        }
        // 覆盖并集不完整时不得因“没看到”直接改为 VISIBLE 或 REVEALED；保留
        // 上一轮状态并让既有 OUT 证据或下面的遮挡失效计数继续处理。
        target->second = retained;
    }
}

bool SessionManager::advance_occlusion_loss_out_evidence_(
        const std::map<int, InventoryItem>& final_items,
        const std::set<int>& observed_working_ids,
        const std::set<int>& fully_occluded_item_ids) {
    bool unresolved = false;
    for (std::map<int, InventoryItem>::const_iterator original =
             operation_start_inventory_.begin();
         original != operation_start_inventory_.end(); ++original) {
        const int item_id = original->first;
        // 专用的“遮挡解释失效”缺失链只适用于上一轮已经确认完全遮挡的 C。
        // VISIBLE + block_ids 仅表示局部前景关系，不能凭它绕过旧 C 自己的
        // OUT 证据而把一次漏检当成出库。
        if (original->second.status != ItemStatus::OCCLUDED) {
            occlusion_loss_missing_counts_.erase(item_id);
            continue;
        }
        if (!final_items.count(item_id) || pending_out_ids_.count(item_id) ||
            observed_working_ids.count(item_id) || fully_occluded_item_ids.count(item_id)) {
            occlusion_loss_missing_counts_.erase(item_id);
            continue;
        }

        // 有活动旧 C 轨迹时，既有无手路径负责缺失计数；这里只补“历史被遮挡、
        // 本轮没有自己的 runtime”的空洞，避免与原有 OUT 证据重复计数。
        const OperationTrack* runtime = find_runtime_for_item_(item_id);
        if (runtime && !runtime->is_suspect_new && existing_item_needs_settlement(*runtime)) {
            occlusion_loss_missing_counts_.erase(item_id);
            continue;
        }

        const int previous = occlusion_loss_missing_counts_[item_id];
        const int next = previous + 1;
        occlusion_loss_missing_counts_[item_id] = next;
        trace_("OCCLUSION",
               "item=%d action=coverage-lost-direct-missing old=%d new=%d threshold=%d",
               item_id, previous, next, FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        if (next >= FLOW3_NO_HAND_OUT_MISSING_FRAMES) {
            mark_pending_out_(item_id);
            occlusion_loss_missing_counts_.erase(item_id);
            trace_("OCCLUSION", "item=%d action=confirm-out-after-coverage-loss", item_id);
        } else {
            unresolved = true;
        }
    }
    return unresolved;
}

SettlementResult SessionManager::settle_no_hand_frame_(
        const std::vector<Detection>& detections, int frame_id) {
    SettlementResult result;
    if (!working_inventory_active_) return result;

    // 细节9：先得到本帧同类“一框一物品”的唯一保留关系。它只在可见数量
    // 已不足时填补单摄像头无法观察个体身份的空白，普通 C->B/D 仲裁不变。
    prepare_visible_count_settlement_(detections);

    std::map<int, InventoryItem> final_items = working_inventory_;
    const std::vector<Detection>& observed = detections;
    std::vector<int> observation_owner(observed.size(), -1);
    std::map<int, int> item_to_observation;
    std::map<int, BBox> references;
    std::set<int> track_priority_ids;
    std::set<int> ambiguous_ids;
    std::set<int> all_ids;
    std::map<int, BBox> original_references;
    std::set<int> original_position_ids;

    for (std::map<int, InventoryItem>::const_iterator it = final_items.begin();
         it != final_items.end(); ++it) {
        all_ids.insert(it->first);
        references[it->first] = it->second.base_box.area() > 0.0f
            ? it->second.base_box : it->second.box;
        const OperationTrack* runtime = find_runtime_for_item_(it->first);
        const bool runtime_in_claim_grace = runtime && is_claim_protected(*runtime);
        const bool runtime_has_ambiguous_b = runtime &&
            (runtime->b_claim_ambiguous || runtime->contact_path_ambiguous);
        if (runtime_has_ambiguous_b) ambiguous_ids.insert(it->first);
        if (!runtime_in_claim_grace && runtime && runtime->has_placed_box) {
            references[it->first] = runtime->placed_box;
        } else if (!runtime_in_claim_grace && runtime && runtime->is_suspect_new &&
                   runtime->has_last_seen_box) {
            references[it->first] = runtime->last_seen_box;
        }
        else if (!runtime_in_claim_grace && runtime && !runtime->observed_track.empty()) {
            references[it->first] = runtime->observed_track.back();
        }
        else if (!runtime_in_claim_grace && runtime && !runtime->is_suspect_new &&
                 reappear_candidate_is_confirmed(*runtime)) {
            references[it->first] = runtime->reappear_candidate_box;
        }
        else if (!runtime_in_claim_grace && runtime && !runtime->track.empty()) {
            references[it->first] = runtime->track.back();
        }
        if (!runtime_has_ambiguous_b &&
            (pending_in_ids_.count(it->first) || confirmed_moved_ids_.count(it->first) ||
            (!runtime_in_claim_grace && runtime && (runtime->hold_and_move ||
                         runtime->contact_state != ContactState::NONE ||
                         (has_meaningful_hand_move(*runtime) &&
                          runtime->state != OperationTrackState::NORMAL))))) {
            track_priority_ids.insert(it->first);
        }
    }

    // 同类 B 仍有多个成熟来源时，暂时把这些 C 从所有“动态/宽松”绑定
    // 阶段移出；后面的原位置/缺失处理会保留旧库存，不按 item_id 强选一个。
    for (std::set<int>::const_iterator it = ambiguous_ids.begin();
         it != ambiguous_ids.end(); ++it) {
        all_ids.erase(*it);
        track_priority_ids.erase(*it);
    }

    // 已由可见数量结算保留给某个旧 C 的框必须先占用，不能让同一个框随后
    // 又被另一个同类 C 的宽松轨迹绑定。即使旧 C 的运行时路径有歧义，这个
    // 强制绑定也只表示“保留一个可见实例”，不单独生成 MOVED。
    for (std::map<int, int>::const_iterator assignment =
             visible_count_detection_owner_.begin();
         assignment != visible_count_detection_owner_.end(); ++assignment) {
        const int detection_index = assignment->first;
        const int item_id = assignment->second;
        if (detection_index < 0 ||
            static_cast<size_t>(detection_index) >= observed.size() ||
            !final_items.count(item_id) || pending_out_ids_.count(item_id) ||
            observation_owner[detection_index] >= 0 ||
            item_to_observation.count(item_id)) {
            continue;
        }
        if (final_items[item_id].cls_id != observed[detection_index].cls_id) continue;
        item_to_observation[item_id] = detection_index;
        observation_owner[detection_index] = item_id;
        trace_("VISIBLE-COUNT",
               "detection=%d reserved-for-item=%d action=force-no-hand-observation",
               detection_index, item_id);
    }

    // 动态轨迹没有唯一终点时，不能因为当前框暂时漏检就直接把旧物品
    // 当成 OUT。把本次操作开始前的原位置作为第二条独立证据源；它只
    // 用于旧库存，不允许刚提升的 D 抢回自己的终点框。
    for (std::map<int, InventoryItem>::const_iterator it =
             operation_start_inventory_.begin();
         it != operation_start_inventory_.end(); ++it) {
        if (!final_items.count(it->first) || pending_in_ids_.count(it->first)) continue;
        original_position_ids.insert(it->first);
        original_references[it->first] = it->second.base_box.area() > 0.0f
            ? it->second.base_box : it->second.box;
    }
    for (std::set<int>::const_iterator it = ambiguous_ids.begin();
         it != ambiguous_ids.end(); ++it) {
        original_position_ids.erase(*it);
    }

    // 绑定顺序：已确认移动/新 D 的终点 -> 普通严格匹配 -> 局部匹配。
    bind_mutually_unique(final_items, track_priority_ids, observed,
                         &references, &item_to_observation, &observation_owner, true, false);
    bind_mutually_unique(final_items, all_ids, observed,
                         &references, &item_to_observation, &observation_owner, false, false);
    // 若物品在手尚未离开时已经掉队并停在候选路径中段，终点框不会命中；
    // 在原位置回查前再做一次整条 path 的唯一最近绑定，避免这类物品被 OUT。
    bind_mutually_unique_track_paths(final_items, track_priority_ids, observed,
                                     track_buffer_, &item_to_observation, &observation_owner);
    // 再回查旧位置。这样 hold_and_move 尚未凑满、或只被手短暂擦过的
    // 物品，只要在原位重新出现，就不会因为轨迹参考框漂移而被误判出库。
    bind_mutually_unique(final_items, original_position_ids, observed,
                         &original_references, &item_to_observation, &observation_owner,
                         false, false);
    bind_mutually_unique(final_items, original_position_ids, observed,
                         &original_references, &item_to_observation, &observation_owner,
                         false, true);
    bind_mutually_unique(final_items, all_ids, observed,
                         &references, &item_to_observation, &observation_owner, false, true);

    // 文档允许 HAND_* 下尚未来得及凑满两次证据的物品，在手离开后的直接
    // 无手帧中由自己的候选轨迹补确认整理。这里仍要求它命中了自己的轨迹参考框，
    // 不把普通未匹配框当作移动终点。
    for (std::map<int, int>::const_iterator it = item_to_observation.begin();
         it != item_to_observation.end(); ++it) {
        const OperationTrack* runtime = find_runtime_for_item_(it->first);
        std::map<int, InventoryItem>::const_iterator original =
            operation_start_inventory_.find(it->first);
        if (!runtime || original == operation_start_inventory_.end() ||
            !is_active_runtime_track(*runtime) || is_claim_protected(*runtime)) {
            continue;
        }
        // CONTACT_CANDIDATE 必须先凑够两次有效 B 才能确认整理；仅有一条
        // observed_move_values 不能绕过 CONTACT_MOVING 门槛。普通 HAND_*
        // 仍保留原有的 move_values/hold_and_move 收尾规则。
        const bool has_contact_confirmation =
            runtime->contact_state == ContactState::CONTACT_MOVING &&
            runtime->hold_and_move;
        const bool has_hand_confirmation = runtime->contact_state == ContactState::NONE &&
            (runtime->hold_and_move || has_meaningful_hand_move(*runtime));
        if (!has_contact_confirmation && !has_hand_confirmation &&
            runtime->observed_move_values.empty()) {
            continue;
        }
        if (runtime->contact_state != ContactState::NONE &&
            !has_contact_confirmation) {
            continue;
        }
        const bool matches_endpoint = track_match_box(
            original->second.cls_id, references[it->first],
            observed[it->second].cls_id, observed[it->second].box);
        const bool matches_any_path = track_path_match_cost(
            original->second, *runtime, observed[it->second]) <
            std::numeric_limits<float>::infinity();
        if ((matches_endpoint || matches_any_path) &&
            boxes_differ_as_move(original->second.base_box, observed[it->second].box)) {
            confirmed_moved_ids_.insert(it->first);
        }
    }

    std::set<int> observed_ids;
    for (std::map<int, int>::const_iterator it = item_to_observation.begin();
         it != item_to_observation.end(); ++it) {
        observed_ids.insert(it->first);
        InventoryItem& item = final_items[it->first];
        update_seen(item, observed[it->second], frame_id);
        if (pending_in_ids_.count(it->first) || confirmed_moved_ids_.count(it->first)) {
            item.base_box = observed[it->second].box;
        }
        item.status = ItemStatus::VISIBLE;
    }

    // 仅“本轮确认移动/确认入库，且在当前无手帧直接出现”的物品可成为前景遮挡物。
    std::set<int> confirmed_front_ids;
    for (std::set<int>::const_iterator it = observed_ids.begin(); it != observed_ids.end(); ++it) {
        if (pending_in_ids_.count(*it) || confirmed_moved_ids_.count(*it)) {
            confirmed_front_ids.insert(*it);
        }
    }

    std::set<int> fully_occluded_ids;
    refresh_confirmed_blockers_(&final_items, observed_ids, confirmed_front_ids,
                                &fully_occluded_ids);

    // OUT 判定会在本帧把 blocker 加入 pending_out_ids_。若这改变了 blocker
    // 集合，必须在删除 final_items 前重算一次，不能让已正式 OUT 的 A 继续
    // 遮挡 C，也不能把 C 直接伪造为 REVEALED。
    const std::set<int> pending_out_before_no_hand = pending_out_ids_;
    bool has_unresolved_state = has_unresolved_no_hand_state_(
        observed_ids, fully_occluded_ids);
    if (pending_out_before_no_hand != pending_out_ids_) {
        refresh_confirmed_blockers_(&final_items, observed_ids, confirmed_front_ids,
                                    &fully_occluded_ids);
    }

    const std::set<int> pending_out_before_occlusion_loss = pending_out_ids_;
    if (advance_occlusion_loss_out_evidence_(final_items, observed_ids,
                                             fully_occluded_ids)) {
        has_unresolved_state = true;
    }
    if (pending_out_before_occlusion_loss != pending_out_ids_) {
        refresh_confirmed_blockers_(&final_items, observed_ids, confirmed_front_ids,
                                    &fully_occluded_ids);
    }

    for (std::set<int>::const_iterator out = pending_out_ids_.begin();
         out != pending_out_ids_.end(); ++out) {
        final_items.erase(*out);
    }

    if (has_unresolved_state) {
        trace_("SETTLE",
               "defer-commit inventory=%zu tracks=%zu pending-in=%zu pending-out=%zu",
               final_items.size(), track_buffer_.size(), pending_in_ids_.size(),
               pending_out_ids_.size());
        return result;
    }

    // 最后才生成正式事件；整个过程中没有任何“未绑定无手框自动 IN”。
    std::vector<InventoryEvent> events;
    for (std::map<int, InventoryItem>::const_iterator old = operation_start_inventory_.begin();
         old != operation_start_inventory_.end(); ++old) {
        std::map<int, InventoryItem>::const_iterator now = final_items.find(old->first);
        if (now == final_items.end()) {
            if (pending_out_ids_.count(old->first)) {
                events.push_back(make_event(EventKind::OUT, old->second));
            }
            continue;
        }
        if (confirmed_moved_ids_.count(old->first) &&
            boxes_differ_as_move(old->second.base_box, now->second.base_box)) {
            events.push_back(make_event(EventKind::MOVED, now->second,
                                        old->second.base_box, now->second.base_box));
        }
        if (old->second.status != now->second.status) {
            if (now->second.status == ItemStatus::OCCLUDED) {
                events.push_back(make_event(EventKind::OCCLUDED, now->second));
            } else if (old->second.status == ItemStatus::OCCLUDED &&
                       now->second.status == ItemStatus::VISIBLE) {
                events.push_back(make_event(EventKind::REVEALED, now->second));
            }
        }
    }
    for (std::set<int>::const_iterator in = pending_in_ids_.begin();
         in != pending_in_ids_.end(); ++in) {
        std::map<int, InventoryItem>::const_iterator item = final_items.find(*in);
        if (item != final_items.end() && observed_ids.count(*in)) {
            events.push_back(make_event(EventKind::IN, item->second));
        }
    }

    for (size_t si = 0; si < observation_owner.size(); ++si) {
        if (observation_owner[si] < 0) {
            printf("[3.0] 无手直接帧出现未绑定检测 cls=%d；没有 D 证据链，不自动 IN\n",
                   observed[si].cls_id);
            trace_("SETTLE", "unbound-detection=%zu cls=%d result=no-auto-in",
                   si, observed[si].cls_id);
        }
    }

    trace_("SETTLE",
           "commit inventory-before=%zu inventory-after=%zu events=%zu pending-in=%zu pending-out=%zu",
           inventory_.size(), final_items.size(), events.size(), pending_in_ids_.size(),
           pending_out_ids_.size());
    for (std::map<int, OperationTrack>::const_iterator track = track_buffer_.begin();
         track != track_buffer_.end(); ++track) {
        trace_track_("SETTLE", track->second, "commit-final-track");
    }
    for (size_t event_index = 0; event_index < events.size(); ++event_index) {
        const InventoryEvent& event = events[event_index];
        trace_("SETTLE",
               "event=%s item=%d cls=%d box=(%.1f,%.1f,%.1f,%.1f) "
               "before=(%.1f,%.1f,%.1f,%.1f) after=(%.1f,%.1f,%.1f,%.1f)",
               event_kind_name(event.kind), event.item_id, event.cls_id,
               event.box.x1, event.box.y1, event.box.x2, event.box.y2,
               event.before_box.x1, event.before_box.y1,
               event.before_box.x2, event.before_box.y2,
               event.after_box.x1, event.after_box.y1,
               event.after_box.x2, event.after_box.y2);
    }

    inventory_.replace_all(final_items, working_next_item_id_);
    rebuild_persistent_item_index_();
    result.committed = true;
    result.happened = !events.empty();
    result.events.swap(events);
    clear_runtime_after_commit_();
    return result;
}

}  // namespace fridge
