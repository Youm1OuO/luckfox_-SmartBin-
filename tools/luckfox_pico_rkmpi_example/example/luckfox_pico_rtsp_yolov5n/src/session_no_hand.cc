// ============================================================================
//  session_no_hand.cc
//  3.0 session direct no-hand observation and recovery
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
#include <sstream>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fridge {

using namespace session_internal;

namespace {

std::string no_hand_candidate_context_ids(const std::set<int>& item_ids) {
    std::ostringstream stream;
    stream << "[";
    for (std::set<int>::const_iterator it = item_ids.begin(); it != item_ids.end(); ++it) {
        if (it != item_ids.begin()) stream << ",";
        stream << *it;
    }
    stream << "]";
    return stream.str();
}

const char* no_hand_candidate_context_decision(const SameClassCandidateContext& context) {
    if (context.viable_unresolved_old_item_ids.empty()) {
        return "independent-d-candidate";
    }
    return context.viable_unresolved_old_item_ids.size() == 1
        ? "unique-unresolved-old-c"
        : "multiple-unresolved-old-c";
}

bool cross_class_track_has_move_evidence(const OperationTrack& track) {
    return track.hold_and_move || has_meaningful_hand_move(track) ||
        (track.contact_state == ContactState::CONTACT_MOVING &&
         !track.observed_move_values.empty()) ||
        (track.has_reappear_candidate_box &&
         reappear_candidate_is_confirmed(track));
}

bool cross_class_endpoint_has_higher_owner(
        int detection_index, int current_item_id,
        const std::vector<Detection>& detections,
        const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks) {
    if (detection_index < 0 ||
        static_cast<size_t>(detection_index) >= detections.size()) {
        return true;
    }
    const Detection& detection = detections[static_cast<size_t>(detection_index)];
    const StrictDetectionOwner strict_owner = strict_owner_for_detection(
        detection, current_item_id, working, operation_start, pending_in_ids, tracks);
    // 对替代终点采取保守规则：任何已经有严格/运行时 owner 的框都不作为
    // “A 的另一条可靠路径”，避免用身份提示抢走更高等级的旧 C 或 D。
    if (strict_owner.kind != StrictOwnerKind::NONE) return true;
    for (std::map<int, OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        const OperationTrack& other = it->second;
        if (other.is_suspect_new && is_active_runtime_track(other) &&
            detection_can_belong_to_active_track(detection, other)) {
            return true;
        }
    }
    const std::map<int, int> no_known_owner;
    const int mature_owner = unique_c_reappear_owner_for_detection(
        detection, tracks, no_known_owner);
    return mature_owner >= 0 && mature_owner != current_item_id;
}

bool cross_class_endpoint_matches_track(const OperationTrack& track,
                                        const InventoryItem& original,
                                        const Detection& detection) {
    if (track.cls_id != detection.cls_id ||
        !boxes_differ_as_move(original.base_box.area() > 0.0f
                                  ? original.base_box : original.box,
                              detection.box)) {
        return false;
    }
    if (track.has_reappear_candidate_box &&
        track_match_box(track.cls_id, track.reappear_candidate_box,
                        detection.cls_id, detection.box)) {
        return true;
    }
    if (track.has_placed_box &&
        track_match_box(track.cls_id, track.placed_box,
                        detection.cls_id, detection.box)) {
        return true;
    }
    if (track_path_match_cost(original, track, detection) <
        std::numeric_limits<float>::infinity()) {
        return true;
    }
    return reappear_candidate_path_matches(track, detection);
}

int find_cross_class_alternative_endpoint(
        const std::vector<Detection>& detections, const std::set<int>& claimed,
        const OperationTrack& track, const std::map<int, InventoryItem>& working,
        const std::map<int, InventoryItem>& operation_start,
        const std::set<int>& pending_in_ids,
        const std::map<int, OperationTrack>& tracks, int excluded_detection) {
    if (track.is_suspect_new || track.item_id <= 0 ||
        !cross_class_track_has_move_evidence(track)) {
        return -1;
    }
    const std::map<int, InventoryItem>::const_iterator original =
        operation_start.find(track.item_id);
    if (original == operation_start.end()) return -1;

    std::set<int> candidate_claimed = claimed;
    candidate_claimed.insert(excluded_detection);
    std::vector<int> candidates;
    if (track.has_reappear_candidate_box) {
        const int candidate = unique_detection_for_box(
            detections, candidate_claimed, track.cls_id,
            track.reappear_candidate_box, true, true);
        if (candidate >= 0) candidates.push_back(candidate);
    }
    const BBox reference = track.has_placed_box ? track.placed_box : estimated_box(track);
    const int reference_candidate = unique_detection_for_box(
        detections, candidate_claimed, track.cls_id, reference, true, true);
    if (reference_candidate >= 0) candidates.push_back(reference_candidate);
    const int path_candidate = unique_no_hand_reappear_detection_for_track(
        detections, candidate_claimed, track.item_id, track,
        working, operation_start, pending_in_ids, tracks);
    if (path_candidate >= 0) candidates.push_back(path_candidate);

    int result = -1;
    for (size_t i = 0; i < candidates.size(); ++i) {
        const int candidate = candidates[i];
        if (candidate == excluded_detection ||
            !cross_class_endpoint_matches_track(track, original->second,
                                                detections[candidate]) ||
            cross_class_endpoint_has_higher_owner(
                candidate, track.item_id, detections, working,
                operation_start, pending_in_ids, tracks)) {
            continue;
        }
        if (result >= 0 && result != candidate) return -1;
        result = candidate;
    }
    return result;
}

}  // namespace

void SessionManager::register_post_hand_reveal_suspects_(
        const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices) {
    // 首次完整 B 必须紧接着本次手操作出现。若允许很久之后的任意无手框
    // 触发，就会把普通漏检重现或静态误检错误写成 IN。
    if (no_hand_streak_ <= 0 ||
        no_hand_streak_ > FLOW3_POST_HAND_REVEAL_WINDOW_FRAMES) {
        return;
    }

    for (size_t di = 0; di < detections.size(); ++di) {
        const int detection_index = static_cast<int>(di);
        if (claimed_detection_indices->count(detection_index)) continue;
        const Detection& d = detections[di];

        // 先完成旧 C、已有 D 的认领。剩余 B 若能合理属于任意旧库存或
        // 活动轨迹，也必须保持未决，不能另建一个“全程被挡住的新 D”。
        const bool matches_old_inventory = detection_matches_old_working_inventory(
            d, working_inventory_, operation_start_inventory_);
        const bool conflicts_active_track =
            detection_conflicts_with_active_track(d, track_buffer_);
        const bool blocked_by_protected_track =
            protected_existing_track_blocks_post_hand_d(d, track_buffer_);
        if (matches_old_inventory || conflicts_active_track ||
            blocked_by_protected_track) {
            trace_("D-GUARD",
                   "candidate=%d cls=%d reject=known-old:%d active-track:%d protected:%d",
                   detection_index, d.cls_id, matches_old_inventory ? 1 : 0,
                   conflicts_active_track ? 1 : 0,
                   blocked_by_protected_track ? 1 : 0);
            continue;
        }
        if (!hand_track_touches_detection(hand_track_, d)) {
            trace_("D-GUARD", "candidate=%d cls=%d reject=outside-hand-track",
                   detection_index, d.cls_id);
            continue;
        }

        // 同类旧 C 若在本轮仍没有自己的独立检测/轨迹结论，手离开后
        // 首次出现的 B 也只能保持未决。否则 C 的一次漏检会被误写成
        // POST_HAND_REVEAL_D，库存数量会逐次膨胀。
        bool unresolved_same_class_old = false;
        for (std::map<int, InventoryItem>::const_iterator old =
                 working_inventory_.begin(); old != working_inventory_.end(); ++old) {
            if (old->second.cls_id != d.cls_id ||
                pending_in_ids_.count(old->first) ||
                old->second.status == ItemStatus::OCCLUDED) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(old->first);
            if (runtime && runtime->is_suspect_new) continue;
            if (existing_item_resolved_without_current_detection(runtime)) {
                trace_("D-GUARD",
                       "candidate=%d cls=%d old=%d disposition=terminal-%s",
                       detection_index, d.cls_id, old->first,
                       existing_resolution_name(runtime->resolution));
                continue;
            }
            bool independent_detection = false;
            for (size_t oi = 0; oi < detections.size(); ++oi) {
                if (detections[oi].cls_id != old->second.cls_id ||
                    detections[oi].box.area() <= 0.0f ||
                    center_distance(detections[oi].box, d.box) <= 4.0f) {
                    continue;
                }
                if (strict_match_box(old->second.cls_id, old->second.box,
                                     detections[oi].cls_id, detections[oi].box) ||
                    partial_match_box(old->second.cls_id, old->second.box,
                                      detections[oi].cls_id, detections[oi].box)) {
                    independent_detection = true;
                    break;
                }
            }
            const bool runtime_ambiguous = runtime &&
                (runtime->b_claim_ambiguous || runtime->contact_path_ambiguous ||
                 runtime->no_hand_candidate_ambiguous);
            // 即使 runtime 指针存在，只要它只是 NORMAL 且没有独立静态框，
            // 也不能把它当成已解决。STATIC_CONFIRMED 只对当前确有独立框的
            // 情形成立；离开当前帧后缺少该框仍要阻止同类 D。
            const bool old_still_unresolved = !independent_detection &&
                !existing_item_resolved_without_current_detection(runtime);
            if (runtime_ambiguous || old_still_unresolved) {
                trace_("D-GUARD",
                       "candidate=%d cls=%d blocked-by-old=%d independent=%d "
                       "runtime=%s resolution=%s ambiguous=%d",
                       detection_index, d.cls_id, old->first,
                       independent_detection ? 1 : 0,
                       runtime ? operation_track_state_name(runtime->state) : "NONE",
                       runtime ? existing_resolution_name(runtime->resolution) : "NONE",
                       runtime_ambiguous ? 1 : 0);
                unresolved_same_class_old = true;
                break;
            }
            trace_("D-GUARD",
                   "candidate=%d cls=%d old=%d disposition=independent-static runtime=%s resolution=%s",
                   detection_index, d.cls_id, old->first,
                   runtime ? operation_track_state_name(runtime->state) : "NONE",
                   runtime ? existing_resolution_name(runtime->resolution) : "NONE");
        }
        if (unresolved_same_class_old) {
            printf("[3.0] 手离开后同类旧 C 尚未解决，B cls=%d 保持未决\n",
                   d.cls_id);
            trace_("D-GUARD", "candidate=%d cls=%d result=keep-unresolved",
                   detection_index, d.cls_id);
            continue;
        }
        trace_("D-GUARD",
               "candidate=%d cls=%d result=create-d all-same-class-old-settled",
               detection_index, d.cls_id);

        OperationTrack track;
        const int key = new_suspect_id_();
        track.suspect_id = key;
        track.is_suspect_new = true;
        track.suspect_source = SuspectSource::POST_HAND_REVEAL_D;
        track.cls_id = d.cls_id;
        // 这类 D 的 B 已经是完整框；它没有也不应伪造 first_hand_block_box。
        track.original_box = d.box;
        track.last_seen_box = d.box;
        track.has_last_seen_box = true;
        track.placed_box = d.box;
        track.has_placed_box = true;
        track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
        track.shelter_or_hold = true;
        track.self_match_count = 1;
        track.no_hand_self_match_count = 1;
        track.hand_track_start_index = 0;
        track.post_hand_reveal_no_hand_streak = no_hand_streak_;
        track.track.push_back(d.box);
        track_buffer_[key] = track;
        claimed_detection_indices->insert(detection_index);
        printf("[3.0] 手离开后预登记疑似新物品 D suspect#%d cls=%d source=%s\n",
               key, d.cls_id, suspect_source_name(track.suspect_source));
        trace_track_("D-GUARD", track_buffer_[key], "create-post-hand-reveal-d");
    }
}

void SessionManager::observe_no_hand_frame_(const std::vector<Detection>& detections) {
    std::set<int> claimed;
    std::vector<int> promote_keys;
    std::set<int> discard_keys;
    std::set<int> discard_settled_old_c_duplicate_keys;
    std::set<int> discard_stale_alias_keys;
    // 每张无手帧重新建立身份计划；它不能把上一帧的临时排除带入下一帧。
    cross_class_duplicate_identity_exclusions_.clear();
    const std::map<int, int> independent_static_owner_by_detection =
        build_independent_no_hand_static_owner_by_detection(
            detections, working_inventory_, operation_start_inventory_, track_buffer_);
    std::map<int, int> independent_static_owner_by_item;
    for (std::map<int, int>::const_iterator owner =
             independent_static_owner_by_detection.begin();
         owner != independent_static_owner_by_detection.end(); ++owner) {
        independent_static_owner_by_item[owner->second] = owner->first;
    }
    // 先隔离同类/异类的低分重复观测，再开始 C、D 的逐对象扫描。shadow 只
    // 占用本帧临时 claimed 集合，弱框不会成为 D、alias 或某个 C 的替代终点；
    // 下一帧完全重算，空间分离后的真实物品仍可进入原有 D 仲裁。
    const DetectionShadowPlan shadow_plan = build_detection_shadow_plan(
        detections, working_inventory_, operation_start_inventory_, pending_in_ids_,
        track_buffer_, independent_static_owner_by_item, trace_frame_id_);
    shadow_detection_indices_ = shadow_plan.detection_indices;
    shadow_owner_by_detection_ = shadow_plan.owner_item_by_detection;
    shadow_hint_by_detection_ = shadow_plan.hint_by_detection;
    claimed.insert(shadow_detection_indices_.begin(), shadow_detection_indices_.end());
    for (std::map<int, DetectionShadowHint>::const_iterator hint =
             shadow_hint_by_detection_.begin();
         hint != shadow_hint_by_detection_.end(); ++hint) {
        trace_("SHADOW",
               "phase=NO_HAND detection=%d owner-detection=%d owner=%d runtime=%d "
               "score=%.3f owner-score=%.3f iom=%.3f center-norm=%.3f action=shadow",
               hint->second.detection_index, hint->second.owner_detection_index,
               hint->second.owner_item_id, hint->second.owner_runtime_key,
               hint->second.score, hint->second.owner_score, hint->second.iom_value,
               hint->second.center_norm);
    }

    // “本帧存在无法唯一归属的路径候选”是瞬时证据；每张直接无手帧都重新
    // 计算，不能让上一帧歧义永久阻塞，也不能把本帧歧义拿去累计 OUT。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        it->second.no_hand_detection_index = -1;
        if (!it->second.is_suspect_new) {
            it->second.no_hand_candidate_ambiguous = false;
            it->second.no_hand_candidate_reserved_by_stronger_owner = false;
        } else if (it->second.pending_d_quarantined_by_old_c) {
            it->second.alias_no_hand_matched_this_frame = false;
        }
    }

    // 临时 D 在有手阶段可能早于某个旧 C 的完整路径证据出现。无手阶段再次
    // 看到同一个候选框时，先补全 D -> old C 的冲突边，再让旧 C 优先处理；
    // 绝不能让一个“不认识 A”的 pending D 排除 A 的最终检测框。
    const std::map<int, int> no_known_item_owner;
    for (size_t di = 0; di < detections.size(); ++di) {
        const SameClassCandidateContext context = build_same_class_candidate_context(
            detections[di], static_cast<int>(di), working_inventory_,
            operation_start_inventory_, pending_in_ids_, track_buffer_,
            no_known_item_owner);
        const std::string direct_old_ids =
            no_hand_candidate_context_ids(context.direct_old_item_ids);
        const std::string viable_old_ids =
            no_hand_candidate_context_ids(context.viable_unresolved_old_item_ids);
        const std::string matching_suspect_ids =
            no_hand_candidate_context_ids(context.matching_suspect_runtime_keys);
        trace_("CANDIDATE-CONTEXT",
               "candidate=%zu cls=%d direct-old=%s viable-unresolved-old=%s "
               "matching-suspect=%s decision=%s allow-d-promote=%d phase=NO_HAND",
               di, detections[di].cls_id, direct_old_ids.c_str(), viable_old_ids.c_str(),
               matching_suspect_ids.c_str(), no_hand_candidate_context_decision(context),
               context.viable_unresolved_old_item_ids.empty() ? 1 : 0);
        if (context.viable_unresolved_old_item_ids.empty()) continue;
        for (std::set<int>::const_iterator suspect =
                 context.matching_suspect_runtime_keys.begin();
             suspect != context.matching_suspect_runtime_keys.end(); ++suspect) {
            link_suspect_to_conflicting_old_items_(
                *suspect, context.viable_unresolved_old_item_ids,
                "NO_HAND", static_cast<int>(di));
        }
    }

    // 认领顺序必须是：已有 C -> 没有手离开后新建的已有 D -> 剩余 B 的
    // POST_HAND_REVEAL_D。track_buffer_ 的负 id 是 D，若直接按 map 遍历，
    // D 会先于 C 抢框，故显式分两轮处理。
    for (int phase = 0; phase < 2; ++phase) {
        const bool process_suspects = phase == 1;
        for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            OperationTrack& track = it->second;
            const bool existing_needs_settlement =
                existing_item_needs_settlement(track);
            if (track.is_suspect_new != process_suspects ||
                (!is_active_runtime_track(track) && !existing_needs_settlement)) {
                continue;
            }

            // CONTACT_* 在有手阶段保持 state=NORMAL，因此不能落入旧的
            // HAND_* 分支。手离开后只按原位置和真实 observed_track 收尾。
            if (!track.is_suspect_new &&
                track.contact_state != ContactState::NONE) {
                reset_stable_near_original_no_hand_evidence_(
                    &track, "contact-track-active");
                int contact_found = unique_contact_original_detection(
                    detections, claimed, track, working_inventory_);
                if (contact_found >= 0) {
                    const Detection& d = detections[contact_found];
                    claimed.insert(contact_found);
                    track.has_tentative_b_box = false;
                    track.tentative_b_match_count = 0;
                    track.tentative_b_started_touching_hand = false;
                    track.has_reappear_candidate_box = false;
                    track.reappear_candidate_match_count = 0;
                    track.reappear_candidate_started_touching_hand = false;
                    const bool was_moving =
                        track.contact_state == ContactState::CONTACT_MOVING;
                    append_contact_observation(&track, d, false);
                    track.hold_evidence_count = 0;
                    ++track.not_hold_evidence_count;
                    if (track.not_hold_evidence_count >=
                        FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, d, trace_frame_id_);
                        }
                        release_not_held_(track, false,
                                          ReleaseReason::CONTACT_RETURNED_ORIGINAL,
                                          contact_found,
                                          &d.box,
                                          "no-hand-contact-returned-original");
                    } else if (was_moving) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, d, trace_frame_id_);
                        }
                    }
                    continue;
                }

                const int contact_found_on_path = unique_contact_detection_for_track(
                    detections, claimed, track, working_inventory_, track_buffer_);
                if (contact_found_on_path >= 0) {
                    const Detection& d = detections[contact_found_on_path];
                    if (is_claim_protected(track)) {
                        // 无手帧也属于保护期的后续有效帧：允许 C 记录自己
                        // 的本地 B，但不把它从其他 C/D 的候选集合中拿走。
                        record_tentative_b(&track, d, false);
                        append_contact_observation(&track, d, false);
                        continue;
                    }
                    claimed.insert(contact_found_on_path);
                    const bool was_moving =
                        track.contact_state == ContactState::CONTACT_MOVING;
                    if (!was_moving) {
                        const bool candidate_ready = update_reappear_candidate(
                            &track, d, false);
                        if (candidate_ready) {
                            track.contact_state = ContactState::CONTACT_MOVING;
                            track.hold_and_move = true;
                            track.hold_evidence_count = 0;
                            track.not_hold_evidence_count = 0;
                        }
                    }
                    append_contact_observation(&track, d, false);
                    if (track.item_id > 0) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, d, trace_frame_id_);
                        }
                    }
                    continue;
                }
                if (has_contact_path_candidate(detections, claimed, track,
                                               working_inventory_)) {
                    track.contact_path_ambiguous = true;
                    trace_track_("MATCH", track, "contact-path-ambiguous");
                }
                // 没有原位置或唯一轨迹 B 时保持未决；不能仅因一帧缺失 OUT。
                trace_track_("MATCH", track, "contact-no-unique-candidate");
                continue;
            }
            BBox reference = track.has_placed_box ? track.placed_box : estimated_box(track);
            if (track.state == OperationTrackState::PLACED && track.has_placed_box) {
                reference = track.placed_box;
            }
            int found = -1;
            bool found_at_original_position = false;
            bool found_as_reappear_candidate = false;
            std::set<int> excluded_static_neighbor_detections;
            std::set<int> candidate_claimed;
            const char* found_source = "NONE";
            bool stale_reappear_candidate_cleared = false;
            bool stale_tentative_b_cleared = false;
            int cross_class_excluded_detection = -1;

            // 细节10：已有 reappear candidate 仍然优先，但若它被本帧唯一严格
            // 原位框明确证伪为静止邻居 B，就只能说明“不是 A”。跳过该框后要在
            // 同一无手帧继续查找 A 的预计位置/真实路径，不能直接永久未决。
            for (;;) {
                candidate_claimed = claimed;
                candidate_claimed.insert(excluded_static_neighbor_detections.begin(),
                                         excluded_static_neighbor_detections.end());
                found = -1;
                found_at_original_position = false;
                found_as_reappear_candidate = false;
                bool found_from_saved_reappear_candidate = false;
                found_source = "NONE";

                // 细节5的无手收尾顺序：已有 C 必须先回查原位置；若 C 仍在原处，
                // 后面的动态路径和 B 候选都不能把它改判成移动。
                if (!track.is_suspect_new && track.original_box.area() > 0.0f) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                      track.cls_id, track.original_box,
                                                      true, false);
                    if (found < 0) {
                        const int nearest = best_detection_for_box(
                            detections, candidate_claimed, track.cls_id,
                            track.original_box, true, false);
                        if (nearest >= 0 && has_unique_operation_start_owner(
                                detections[nearest], track.item_id,
                                operation_start_inventory_)) {
                            found = nearest;
                        }
                    }
                    if (found >= 0) {
                        const bool original_position_candidate =
                            !track.has_hand_estimate_anchor_box ||
                            contact_detection_is_at_original(track, detections[found]);
                        found_at_original_position = original_position_candidate;
                        // 两阶段身份仲裁：原位框先只作为候选收集。只有低分
                        // 跨类别近同框同时满足真实移动证据和可靠替代终点时，
                        // 才在本帧暂时降低它对 A 原位身份的权威；其他场景
                        // 完全沿用原有 ORIGINAL 路径。
                        if (!track.is_suspect_new && found_at_original_position &&
                            cross_class_excluded_detection < 0) {
                            const CrossClassDuplicateHint duplicate_hint =
                                find_cross_class_duplicate_hint(detections, found);
                            const int alternative_endpoint = duplicate_hint.valid()
                                ? find_cross_class_alternative_endpoint(
                                      detections, candidate_claimed, track,
                                      working_inventory_, operation_start_inventory_,
                                      pending_in_ids_, track_buffer_, found)
                                : -1;
                            trace_(
                                "IDENTITY",
                                "item=%d detection=%d source=ORIGINAL duplicate-valid=%d competitor=%d "
                                "score=%.3f competitor-score=%.3f alternative=%d",
                                track.item_id, found, duplicate_hint.valid() ? 1 : 0,
                                duplicate_hint.competing_index, duplicate_hint.score,
                                duplicate_hint.competing_score, alternative_endpoint);
                            if (duplicate_hint.valid() && alternative_endpoint >= 0) {
                                cross_class_excluded_detection = found;
                                cross_class_duplicate_identity_exclusions_[track.item_id].insert(
                                    found);
                                excluded_static_neighbor_detections.insert(found);
                                trace_(
                                    "IDENTITY",
                                    "item=%d detection=%d source=ORIGINAL action=temporary-exclude-cross-class-duplicate "
                                    "competitor=%d score=%.3f competitor-score=%.3f iom=%.3f iou=%.3f "
                                    "center-norm=%.3f alternative=%d reason=cross-class-duplicate-identity-only",
                                    track.item_id, found, duplicate_hint.competing_index,
                                    duplicate_hint.score, duplicate_hint.competing_score,
                                    duplicate_hint.iom_value, duplicate_hint.iou_value,
                                    duplicate_hint.center_norm, alternative_endpoint);
                                trace_track_(
                                    "IDENTITY", track,
                                    "temporary-exclude-cross-class-duplicate-and-continue-path");
                                continue;
                            }
                        }
                        // 对 HAND/CONTACT 恢复中的 C，严格/局部原框匹配先只
                        // 确定身份。超过 CONTACT 12px 时它不再是“立即原位”
                        // 证据，却仍必须保留给静态灰区或正式移动判断；此前在
                        // 这里清掉 found 会让稳定候选永远进不到 STATIC-SETTLE。
                        // 有 hand_estimate_anchor_box 的 C 已经在有手阶段暂时
                        // 释放过；无手帧即使重新看到严格原框，也要经过连续直接
                        // 观测结算，不能把 anchor 物体的一帧框当成最终结论。
                        // anchor 恢复中的 C 只有在候选超过 CONTACT 的短时
                        // 原位门槛时才需要进入灰区连续结算；真正回到原位的
                        // 框继续沿原有 ORIGINAL 立即释放路径处理，避免手
                        // 随后重新进入时把一张无手静态帧误当成未完成恢复。
                        found_source = found_at_original_position
                            ? "ORIGINAL" : "ORIGINAL_RECOVERY_CANDIDATE";
                    }
                }
                // 已在有手阶段看到过 B 时，真实 B 候选比“旧框 + 手位移”的估计
                // 更可信，但它同样需要连续自匹配，不能在本帧直接确认整理。
                if (found < 0 && !track.is_suspect_new &&
                    track.has_reappear_candidate_box) {
                    found = unique_detection_for_box(
                        detections, candidate_claimed, track.cls_id,
                        track.reappear_candidate_box, true, true);
                    if (found < 0) {
                        found = best_detection_for_box(
                            detections, candidate_claimed, track.cls_id,
                            track.reappear_candidate_box, true, true);
                    }
                    found_as_reappear_candidate = found >= 0;
                    found_from_saved_reappear_candidate = found >= 0;
                    if (found >= 0) found_source = "REAPPEAR_CANDIDATE";
                }
                if (found < 0) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                     track.cls_id, reference, true, true);
                    if (found >= 0) found_source = "REFERENCE";
                }
                // D 刚放下时，完整框可能和“手中局部框 + 手位移”差异很大。它已经
                // 通过前序来源建立了链路，收尾首帧应优先尝试自身最近一次真实观测。
                // 普通旧库存不走这个宽松兜底。
                if (found < 0 && track.is_suspect_new && track.has_last_seen_box) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                      track.cls_id, track.last_seen_box,
                                                      true, true);
                    if (found >= 0) found_source = "SUSPECT_LAST_SEEN";
                }
                if (found < 0 && track.is_suspect_new && track.has_last_hand_block_box) {
                    found = unique_detection_for_box(detections, candidate_claimed,
                                                      track.cls_id, track.last_hand_block_box,
                                                      true, true);
                    if (found >= 0) found_source = "SUSPECT_HAND_BLOCK";
                }
                if (found < 0 && track.is_suspect_new) {
                    found = unique_d_reappearance_detection_for_track(
                        detections, candidate_claimed, it->first, track,
                        working_inventory_, track_buffer_);
                    if (found >= 0) {
                        found_source = "SUSPECT_REAPPEAR";
                        printf("[3.0] suspect#%d 通过局部→完整路径重现匹配到无手检测\n",
                               track.suspect_id);
                    }
                }
                if (found < 0 && !track.is_suspect_new) {
                    found = unique_no_hand_reappear_detection_for_track(
                        detections, candidate_claimed, it->first, track,
                        working_inventory_, operation_start_inventory_,
                        pending_in_ids_, track_buffer_);
                    found_as_reappear_candidate = found >= 0;
                    if (found >= 0) {
                        found_source = "UNIQUE_REAPPEAR";
                        printf("[3.0] item#%d 在无手帧将同类 B 暂记为重新出现候选\n",
                               track.item_id);
                    }
                }

                const StrictDetectionOwner fallback_strict_owner = found >= 0
                    ? strict_owner_for_detection(
                        detections[found], track.item_id, working_inventory_,
                        operation_start_inventory_, pending_in_ids_, track_buffer_)
                    : StrictDetectionOwner();
                if (found < 0 || track.is_suspect_new || is_claim_protected(track) ||
                    found_at_original_position ||
                    contact_detection_is_at_original(track, detections[found]) ||
                    !strict_owner_blocks_old_c(fallback_strict_owner.kind)) {
                    break;
                }

                const std::map<int, int>::const_iterator static_owner =
                    independent_static_owner_by_detection.find(found);
                if (static_owner == independent_static_owner_by_detection.end() ||
                    static_owner->second == track.item_id) {
                    // 另一旧 C 没有本帧独立严格原位证据，仍属于细节8的真实
                    // 同类歧义；交给下面原有的 ambiguous 分支处理。
                    break;
                }

                const bool candidate_matches_static_neighbor =
                    track.has_reappear_candidate_box &&
                    track_match_box(track.cls_id, track.reappear_candidate_box,
                                    detections[found].cls_id, detections[found].box);
                const bool tentative_matches_static_neighbor =
                    track.has_tentative_b_box &&
                    track_match_box(track.cls_id, track.tentative_b_box,
                                    detections[found].cls_id, detections[found].box);
                if (candidate_matches_static_neighbor) {
                    track.has_reappear_candidate_box = false;
                    track.reappear_candidate_match_count = 0;
                    track.reappear_candidate_started_touching_hand = false;
                    track.drop_evidence_count = 0;
                    stale_reappear_candidate_cleared = true;
                }
                if (tentative_matches_static_neighbor) {
                    track.has_tentative_b_box = false;
                    track.tentative_b_match_count = 0;
                    track.tentative_b_started_touching_hand = false;
                    stale_tentative_b_cleared = true;
                }
                // 即使本次候选来自 reference/path 而非保存的 candidate，排除后
                // 的下一个真实框也必须重新走连续确认，不能单帧确认 MOVED。
                track.reappearance_pending = true;
                trace_(
                    "MATCH",
                    "item=%d detection=%d source=%s other-owner=%d "
                    "other-owner-evidence=direct-original-mutually-unique "
                    "action=exclude-independent-static-neighbor-and-fallback "
                    "candidate-cleared=%d tentative-cleared=%d saved-candidate-source=%d "
                    "box=(%.1f,%.1f,%.1f,%.1f)",
                    track.item_id, found, found_source, static_owner->second,
                    candidate_matches_static_neighbor ? 1 : 0,
                    tentative_matches_static_neighbor ? 1 : 0,
                    found_from_saved_reappear_candidate ? 1 : 0,
                    detections[found].box.x1, detections[found].box.y1,
                    detections[found].box.x2, detections[found].box.y2);
                trace_track_("MATCH", track,
                             "exclude-static-neighbor-candidate-continue-fallback");
                excluded_static_neighbor_detections.insert(found);
            }

            if (!track.is_suspect_new && is_claim_protected(track)) {
                reset_stable_near_original_no_hand_evidence_(
                    &track, "active-claim-grace");
                if (found < 0) {
                    // 保护期内的缺失只是未决，不产生 HOLD/OUT 证据；同时
                    // register_post_hand_reveal_suspects_ 会继续屏蔽同类 D。
                    continue;
                }
                const Detection& provisional = detections[found];
                const bool at_original = found_at_original_position ||
                    contact_detection_is_at_original(track, provisional);
                if (at_original) {
                    claimed.insert(found);
                    track.b_claim_ambiguous = false;
                    track.has_tentative_b_box = false;
                    track.tentative_b_match_count = 0;
                    track.tentative_b_started_touching_hand = false;
                    track.has_reappear_candidate_box = false;
                    track.reappear_candidate_match_count = 0;
                    track.reappear_candidate_started_touching_hand = false;
                    ++track.not_hold_evidence_count;
                    track.hold_evidence_count = 0;
                    track.last_hand_block_box = provisional.box;
                    track.has_last_hand_block_box = true;
                    if (track.not_hold_evidence_count >=
                        FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                        std::map<int, InventoryItem>::iterator item =
                            working_inventory_.find(track.item_id);
                        if (item != working_inventory_.end()) {
                            update_seen(item->second, provisional, trace_frame_id_);
                        }
                        release_not_held_(track, false,
                                          ReleaseReason::ORIGINAL_DETECTION,
                                          found,
                                          &provisional.box,
                                          "no-hand-protected-original");
                    }
                } else {
                    record_tentative_b(&track, provisional, false);
                    track.reappearance_pending = true;
                }
                continue;
            }
            if (!track.is_suspect_new && found >= 0 &&
                !found_at_original_position &&
                !contact_detection_is_at_original(track, detections[found])) {
                const StrictDetectionOwner strict_owner = strict_owner_for_detection(
                    detections[found], track.item_id, working_inventory_,
                    operation_start_inventory_, pending_in_ids_, track_buffer_);
                const bool is_current_c_alias =
                    strict_owner_is_quarantined_alias_of_old_c(
                        strict_owner, track.item_id, track_buffer_);
                if ((strict_owner.kind == StrictOwnerKind::PENDING_D ||
                     strict_owner.kind == StrictOwnerKind::QUARANTINED_PENDING_D) &&
                    !is_current_c_alias) {
                    // Pending D 没有 old C 的排他权限，也不能反过来被 C 单帧
                    // 抢走。保留双方独立证据链，日志明确它是 runtime D 而非
                    // "other old item"。
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d defer=pending-runtime-owner "
                           "owner-kind=%s owner-id=%d owner-runtime=%d "
                           "owner-authority=provisional-runtime-only",
                           track.item_id, found,
                           strict_owner_kind_name(strict_owner.kind),
                           strict_owner.item_id, strict_owner.runtime_key);
                    trace_track_("MATCH", track,
                                 "no-hand-candidate-belongs-to-pending-runtime-d");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "pending-runtime-owner");
                    continue;
                }
                if (strict_owner_blocks_old_c(strict_owner.kind)) {
                    // A 的预计轨迹可以碰巧经过 B 的原位，但一个完整框若已经
                    // 严格属于另一件旧库存 B，就不能被 A 先抢走。它既不是 A
                    // 的 MOVED 证据，也不是 A 缺失的 OUT 证据，应保持未决。
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d reject=strict-owner "
                           "owner-kind=%s owner-id=%d owner-runtime=%d "
                           "excluded-static-neighbors=%zu stale-candidate-reset=%d "
                           "tentative-reset=%d fallback-source=%s result=real-ambiguity",
                           track.item_id, found, strict_owner_kind_name(strict_owner.kind),
                           strict_owner.item_id, strict_owner.runtime_key,
                           excluded_static_neighbor_detections.size(),
                           stale_reappear_candidate_cleared ? 1 : 0,
                           stale_tentative_b_cleared ? 1 : 0, found_source);
                    trace_track_("MATCH", track,
                                 "no-hand-candidate-strictly-owned-by-old-or-confirmed-item");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "reappearance-or-claim-ambiguity");
                    continue;
                }
            }
            if (!track.is_suspect_new && found >= 0 &&
                !found_at_original_position &&
                !contact_detection_is_at_original(track, detections[found])) {
                const std::map<int, int> no_known_owner;
                const int mature_owner = unique_c_reappear_owner_for_detection(
                    detections[found], track_buffer_, no_known_owner);
                if (mature_owner == -2) {
                    mark_mature_hand_b_ambiguity(detections[found], &track_buffer_,
                                                  no_known_owner);
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d excluded-static-neighbors=%zu "
                           "stale-candidate-reset=%d tentative-reset=%d fallback-source=%s "
                           "result=real-shared-ambiguity",
                           track.item_id, found,
                           excluded_static_neighbor_detections.size(),
                           stale_reappear_candidate_cleared ? 1 : 0,
                           stale_tentative_b_cleared ? 1 : 0, found_source);
                    trace_track_("MATCH", track, "no-hand-shared-candidate");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "reappearance-or-claim-ambiguity");
                    continue;
                }
                if (mature_owner >= 0 && mature_owner != track.item_id) {
                    const std::map<int, OperationTrack>::const_iterator owner =
                        track_buffer_.find(mature_owner);
                    const int this_strength =
                        reappear_owner_evidence_strength(track, detections[found]);
                    const int owner_strength = owner == track_buffer_.end()
                        ? 0
                        : reappear_owner_evidence_strength(owner->second,
                                                            detections[found]);
                    if (owner_strength > this_strength) {
                        // 更强旧 C 已暂存这张框。当前 C 仍不能把它当作自己
                        // 已缺失；本帧先继续等待自己的终点/下一张证据。
                        track.reappearance_pending = true;
                        track.no_hand_candidate_reserved_by_stronger_owner = true;
                        trace_(
                            "MATCH",
                            "item=%d detection=%d owner=%d owner-evidence=%d "
                            "self-evidence=%d result=reserved-by-stronger-old-c",
                            track.item_id, found, mature_owner, owner_strength,
                            this_strength);
                        trace_track_("MATCH", track,
                                     "no-hand-candidate-reserved-by-stronger-old-c");
                        reset_stable_near_original_no_hand_evidence_(
                            &track, "reserved-by-stronger-old-c");
                        continue;
                    }
                    track.reappearance_pending = true;
                    track.no_hand_candidate_ambiguous = true;
                    trace_("MATCH",
                           "item=%d detection=%d excluded-static-neighbors=%zu "
                           "stale-candidate-reset=%d tentative-reset=%d fallback-source=%s "
                           "result=real-other-c-owner",
                           track.item_id, found,
                           excluded_static_neighbor_detections.size(),
                           stale_reappear_candidate_cleared ? 1 : 0,
                           stale_tentative_b_cleared ? 1 : 0, found_source);
                    trace_track_("MATCH", track, "no-hand-candidate-owned-by-other-c");
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "reappearance-or-claim-ambiguity");
                    continue;
                }
                track.b_claim_ambiguous = false;
            }
            if (found < 0) {
                // 替代路径若在本帧最终没有留下可用终点，临时排除自动
                // 失效，并回退到原有原位/缺失规则，不能永久 defer。
                if (cross_class_excluded_detection >= 0) {
                    cross_class_duplicate_identity_exclusions_[track.item_id].erase(
                        cross_class_excluded_detection);
                    if (cross_class_duplicate_identity_exclusions_[track.item_id].empty()) {
                        cross_class_duplicate_identity_exclusions_.erase(track.item_id);
                    }
                    excluded_static_neighbor_detections.erase(
                        cross_class_excluded_detection);
                    cross_class_excluded_detection = -1;
                    trace_track_("IDENTITY", track,
                                 "cross-class-duplicate-alternative-failed-fallback-original");
                    continue;
                }
                if (!track.is_suspect_new) {
                    reset_stable_near_original_no_hand_evidence_(
                        &track, "no-direct-unique-owner");
                    bool reserved_by_stronger_owner = false;
                    track.no_hand_candidate_ambiguous =
                        has_ambiguous_no_hand_reappear_candidate(
                            detections, candidate_claimed, it->first, track,
                            working_inventory_, operation_start_inventory_,
                            pending_in_ids_, track_buffer_,
                            independent_static_owner_by_detection,
                            &reserved_by_stronger_owner);
                    track.no_hand_candidate_reserved_by_stronger_owner =
                        reserved_by_stronger_owner;
                    trace_track_("MATCH", track,
                                 track.no_hand_candidate_ambiguous
                                     ? "no-hand-path-ambiguous"
                                     : (reserved_by_stronger_owner
                                         ? "no-hand-path-reserved-by-stronger-old-c"
                                         : "no-hand-no-path-candidate"));
                    if (!excluded_static_neighbor_detections.empty()) {
                        trace_("MATCH",
                               "item=%d excluded-static-neighbors=%zu "
                               "stale-candidate-reset=%d tentative-reset=%d "
                               "fallback-source=NONE fallback-detection=-1 "
                               "ambiguous-after-exclusion=%d result=%s",
                               track.item_id,
                               excluded_static_neighbor_detections.size(),
                               stale_reappear_candidate_cleared ? 1 : 0,
                               stale_tentative_b_cleared ? 1 : 0,
                               track.no_hand_candidate_ambiguous ? 1 : 0,
                               track.no_hand_candidate_ambiguous
                                   ? "real-path-ambiguity"
                                   : "excluded-static-neighbor-then-no-a-candidate");
                    }
                }
                // 对 C-D alias：当前框若已被 C 直接认领，D 只是同一框的
                // 重复运行时解释，必须等待第二张直接无手帧再合并；不能按
                // 普通 D 的“本帧未自匹配”规则立刻丢掉，否则 C 无法获得
                // 连续的 alias 仲裁证据。
                if (track.is_suspect_new &&
                    track.pending_d_quarantined_by_old_c) {
                    if (track.alias_no_hand_matched_this_frame) {
                        trace_("C-D-ALIAS",
                               "old-count=%zu suspect=%d phase=NO_HAND relation=shared-detection "
                               "direct-count=%d action=%s",
                               track.conflicting_old_item_ids.size(), track.suspect_id,
                               track.alias_no_hand_match_count,
                               track.alias_no_hand_match_count >=
                                       FLOW3_NO_HAND_D_CONFIRM_FRAMES
                                   ? "discard-duplicate"
                                   : "wait-second-no-hand");
                        if (track.alias_no_hand_match_count >=
                            FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                            discard_keys.insert(it->first);
                        }
                    } else {
                        track.alias_no_hand_match_count = 0;
                        const bool all_old_settled =
                            all_conflicting_old_c_independently_settled(
                                track, track_buffer_);
                        const bool runtime_only =
                            !track.promoted_to_working_inventory && track.item_id <= 0;
                        if (all_old_settled && runtime_only) {
                            ++track.alias_no_hand_missing_count;
                            const bool discard_stale_alias =
                                track.alias_no_hand_missing_count >=
                                FLOW3_NO_HAND_D_CONFIRM_FRAMES;
                            trace_(
                                "C-D-ALIAS",
                                "old-count=%zu suspect=%d phase=NO_HAND relation=no-direct-pair "
                                "all-conflicting-old-settled=1 runtime-only=1 "
                                "missing-count=%d/%d action=%s",
                                track.conflicting_old_item_ids.size(), track.suspect_id,
                                track.alias_no_hand_missing_count,
                                FLOW3_NO_HAND_D_CONFIRM_FRAMES,
                                discard_stale_alias
                                    ? "discard-stale-alias-after-settled-old-c-no-hand-missing"
                                    : "wait-stale-alias-evidence");
                            trace_track_(
                                "C-D-ALIAS", track,
                                discard_stale_alias
                                    ? "stale-alias-no-direct-evidence-confirmed"
                                    : "stale-alias-no-direct-evidence-wait");
                            if (discard_stale_alias) {
                                discard_keys.insert(it->first);
                                discard_stale_alias_keys.insert(it->first);
                            }
                        } else {
                            const int previous_missing_count =
                                track.alias_no_hand_missing_count;
                            track.alias_no_hand_missing_count = 0;
                            trace_(
                                "C-D-ALIAS",
                                "old-count=%zu suspect=%d phase=NO_HAND relation=no-direct-pair "
                                "all-conflicting-old-settled=%d runtime-only=%d "
                                "missing-count=%d/%d action=%s",
                                track.conflicting_old_item_ids.size(), track.suspect_id,
                                all_old_settled ? 1 : 0, runtime_only ? 1 : 0,
                                previous_missing_count,
                                FLOW3_NO_HAND_D_CONFIRM_FRAMES,
                                previous_missing_count > 0
                                    ? "reset-stale-alias-evidence"
                                    : "keep-unresolved");
                            trace_track_("C-D-ALIAS", track,
                                         all_old_settled
                                             ? "quarantined-alias-not-runtime-only"
                                             : "conflicting-old-c-not-yet-settled");
                        }
                    }
                    continue;
                }
                // 普通 D 必须在后续直接无手帧中连续自匹配。未达到门槛就
                // 消失，不保留为悬空 IN，更不能让它在后续随机帧重新凑次数。
                if (track.is_suspect_new &&
                    track.no_hand_self_match_count < FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                    discard_keys.insert(it->first);
                }
                continue;
            }
            const Detection& d = detections[found];
            bool shared_quarantined_alias = false;
            if (!track.is_suspect_new) {
                track.no_hand_detection_index = found;
                for (std::set<int>::const_iterator key =
                         track.conflicting_suspect_keys.begin();
                     key != track.conflicting_suspect_keys.end(); ++key) {
                    std::map<int, OperationTrack>::iterator suspect =
                        track_buffer_.find(*key);
                    if (suspect == track_buffer_.end() ||
                        !suspect->second.pending_d_quarantined_by_old_c ||
                        !suspect->second.conflicting_old_item_ids.count(track.item_id) ||
                        !quarantined_suspect_matches_detection(suspect->second, d)) {
                        continue;
                    }
                    OperationTrack& alias_d = suspect->second;
                    alias_d.alias_no_hand_matched_this_frame = true;
                    ++alias_d.alias_no_hand_match_count;
                    alias_d.alias_no_hand_missing_count = 0;
                    track.alias_no_hand_match_count =
                        alias_d.alias_no_hand_match_count;
                    track.reappearance_pending = true;
                    shared_quarantined_alias = true;
                    trace_("C-D-ALIAS",
                           "old-item=%d suspect=%d detection=%d phase=NO_HAND "
                           "relation=shared-detection direct-count=%d "
                           "action=%s",
                           track.item_id, alias_d.suspect_id, found,
                           alias_d.alias_no_hand_match_count,
                           alias_d.alias_no_hand_match_count >=
                                   FLOW3_NO_HAND_D_CONFIRM_FRAMES
                               ? "assign-to-old-c-discard-duplicate"
                               : "wait-second-no-hand");
                    if (alias_d.alias_no_hand_match_count >=
                        FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                        discard_keys.insert(*key);
                    }
                }
            }
            claimed.insert(found);
            if (!track.is_suspect_new) {
                track.no_hand_candidate_ambiguous = false;
                trace_("MATCH",
                       "item=%d detection=%d source=%s excluded-static-neighbors=%zu "
                       "stale-candidate-reset=%d tentative-reset=%d "
                       "ambiguous-after-exclusion=0 result=fallback-selected "
                       "box=(%.1f,%.1f,%.1f,%.1f)",
                       track.item_id, found,
                       found_source,
                       excluded_static_neighbor_detections.size(),
                       stale_reappear_candidate_cleared ? 1 : 0,
                       stale_tentative_b_cleared ? 1 : 0,
                       d.box.x1, d.box.y1, d.box.x2, d.box.y2);
            }

            if (track.is_suspect_new) {
                if (track.pending_d_quarantined_by_old_c) {
                    // 任意直接无手证据（独立、共享或尚待仲裁）都会中断“D 消失”
                    // 的连续计数。不能把被直接证据打断的缺失片段拼起来。
                    track.alias_no_hand_missing_count = 0;
                }
                if (track.pending_d_quarantined_by_old_c &&
                    quarantined_suspect_detection_is_duplicate_of_settled_old_c(
                        track, track_buffer_, found, d)) {
                    // C 已先独立结算；D 当前又只命中同一个 C 框时，D 不再
                    // 具有独立新增物品的任何证据。立即回收该重复 runtime
                    // identity，避免下一帧把它累积为 IN。
                    discard_keys.insert(it->first);
                    discard_settled_old_c_duplicate_keys.insert(it->first);
                    trace_("C-D-ALIAS",
                           "suspect=%d detection=%d phase=NO_HAND "
                           "relation=duplicate-settled-old-c "
                           "action=discard-duplicate-settled-old-c",
                           track.suspect_id, found);
                    continue;
                }
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                ++track.no_hand_self_match_count;
                track.no_hand_detection_index = found;
                if (track.pending_d_quarantined_by_old_c) {
                    if (!quarantined_suspect_has_distinct_old_c_detections(
                            track, track_buffer_, found, d)) {
                        track.alias_no_hand_match_count = 0;
                        trace_("C-D-ALIAS",
                               "suspect=%d detection=%d phase=NO_HAND relation=not-independent "
                               "action=keep-unresolved",
                               track.suspect_id, found);
                        continue;
                    }
                    track.alias_no_hand_matched_this_frame = true;
                    ++track.alias_no_hand_match_count;
                    trace_("C-D-ALIAS",
                           "suspect=%d detection=%d phase=NO_HAND relation=distinct-detections "
                           "old-count=%zu direct-count=%d action=%s",
                           track.suspect_id, found, track.conflicting_old_item_ids.size(),
                           track.alias_no_hand_match_count,
                           track.alias_no_hand_match_count >=
                                   FLOW3_NO_HAND_D_CONFIRM_FRAMES
                               ? "confirm-independent-d"
                               : "wait-second-no-hand");
                    if (track.alias_no_hand_match_count <
                        FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                        continue;
                    }
                    const int released_key = it->first;
                    unlink_quarantined_suspect_(released_key,
                                                "confirm-independent-d");
                    set_live_state_(&track, LiveObservationState::PROVISIONAL_D,
                                    true, "independent-no-hand-d-confirmed");
                }
            }
            if (track.is_suspect_new && !track.promoted_to_working_inventory) {
                ++track.self_match_count;
                if (track.self_match_count >= NEW_ITEM_CONFIRM_FRAMES) {
                    promote_keys.push_back(it->first);
                }
            }
            if (track.is_suspect_new && track.promoted_to_working_inventory) {
                track.placed_box = d.box;
                track.has_placed_box = true;
                track.drop_confirmed = true;
                std::map<int, InventoryItem>::iterator item =
                    working_inventory_.find(track.item_id);
                if (item != working_inventory_.end()) {
                    update_seen(item->second, d, trace_frame_id_);
                    item->second.base_box = d.box;
                }
                continue;
            }
            if (!track.is_suspect_new) {
                const bool still_at_original = found_at_original_position ||
                    (!track.has_hand_estimate_anchor_box &&
                     partial_match_box(track.cls_id, track.original_box,
                                       d.cls_id, d.box,
                                       FLOW3_TRACK_PARTIAL_IOM));
                if (still_at_original) {
                    std::map<int, InventoryItem>::iterator item =
                        working_inventory_.find(track.item_id);
                    if (item != working_inventory_.end()) {
                        update_seen(item->second, d, trace_frame_id_);
                    }
                    release_not_held_(track, false,
                                      ReleaseReason::ORIGINAL_DETECTION,
                                      found,
                                      &d.box,
                                      "no-hand-original");
                } else {
                    // r16：普通原位 release 已因 CONTACT 的 12px 门槛被
                    // 拒绝时，只有这个旧 C 自己连续两张唯一、尺度一致的
                    // 无手近原位框才可完成静态结算。这里绝不改动 D 的
                    // alias 仲裁；C 结算后，后续 phase=1 仍完全走 r15。
                    const bool may_be_static_gray_zone_old_c =
                        track.state == OperationTrackState::NORMAL &&
                        track.contact_state == ContactState::NONE &&
                        track.resolution == ExistingItemResolution::STATIC_CONFIRMED &&
                        track.needs_no_hand_settlement &&
                        track.has_hand_estimate_anchor_box;
                    if (may_be_static_gray_zone_old_c &&
                        try_release_stable_near_original_no_hand_(
                            &track, found, d,
                            independent_static_owner_by_detection,
                            found_source)) {
                        continue;
                    }
                    const bool should_observe_candidate =
                        found_as_reappear_candidate || track.reappearance_pending ||
                        track.has_reappear_candidate_box;
                    bool candidate_ready = true;
                    if (should_observe_candidate) {
                        const bool had_candidate = track.has_reappear_candidate_box;
                        candidate_ready = update_reappear_candidate(&track, d, false);
                        trace_("MATCH",
                               "item=%d detection=%d source=no-hand-reappear action=%s count=%d ready=%d",
                               track.item_id, found, had_candidate ? "update" : "start",
                               track.reappear_candidate_match_count,
                               candidate_ready ? 1 : 0);
                    }
                    if (!candidate_ready) continue;
                    track.last_seen_box = d.box;
                    track.has_last_seen_box = true;
                    if ((track.hold_and_move || has_meaningful_hand_move(track)) &&
                        boxes_differ_as_move(track.original_box, d.box) &&
                        (!shared_quarantined_alias ||
                         track.alias_no_hand_match_count >=
                             FLOW3_NO_HAND_D_CONFIRM_FRAMES)) {
                        confirm_rearrange_(track, d.box, d.score, trace_frame_id_);
                    }
                }
            }
        }

        // 第一轮 C 处理结束后，正常静态 C 也要先保留自己的严格框，再让 D
        // 用宽松的局部/路径规则查找。这样严格属于旧库存的 B 不会被 D 抢走。
        if (phase == 0) {
            reserve_unique_no_hand_static_inventory_detections(
                detections, working_inventory_, operation_start_inventory_, track_buffer_,
                &claimed);
        }
    }

    for (std::set<int>::const_iterator discard = discard_keys.begin();
         discard != discard_keys.end(); ++discard) {
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(*discard);
        if (track == track_buffer_.end()) continue;
        const bool discard_alias_duplicate =
            track->second.pending_d_quarantined_by_old_c;
        const bool duplicate_of_settled_old_c =
            discard_settled_old_c_duplicate_keys.count(*discard) > 0;
        const bool stale_alias_after_settled_old_c =
            discard_stale_alias_keys.count(*discard) > 0;
        if (discard_alias_duplicate) {
            unlink_quarantined_suspect_(
                *discard,
                stale_alias_after_settled_old_c
                    ? "discard-stale-alias-after-settled-old-c-no-hand-missing"
                    : (duplicate_of_settled_old_c
                        ? "discard-duplicate-settled-old-c"
                        : "discard-duplicate"));
        }
        if (stale_alias_after_settled_old_c) {
            printf("[3.0] suspect#%d 的关联旧 C 已独立结算，且连续无手帧无直接证据，"
                   "按过期 alias 丢弃\n", track->second.suspect_id);
            trace_(
                "C-D-ALIAS",
                "suspect=%d phase=NO_HAND "
                "action=discard-stale-alias-after-settled-old-c-no-hand-missing "
                "working-inventory-write=0 pending-in-write=0 event=none",
                track->second.suspect_id);
            trace_track_("C-D-ALIAS", track->second,
                         "discard-stale-alias-after-settled-old-c-no-hand-missing");
        } else if (duplicate_of_settled_old_c) {
            printf("[3.0] suspect#%d 只命中已独立结算旧 C 的框，"
                   "按重复身份丢弃\n", track->second.suspect_id);
            trace_track_("C-D-ALIAS", track->second,
                         "discard-quarantined-duplicate-of-settled-old-c");
        } else if (discard_alias_duplicate) {
            printf("[3.0] suspect#%d 与旧 C 的共享无手框已连续确认，"
                   "按重复身份丢弃\n", track->second.suspect_id);
            trace_track_("C-D-ALIAS", track->second,
                         "discard-quarantined-duplicate-after-shared-no-hand-confirmation");
        } else {
            printf("[3.0] suspect#%d 的手离开后完整框未连续确认，丢弃候选\n",
                   track->second.suspect_id);
            trace_track_("STATE", track->second,
                         "discard-suspect-missing-continuous-no-hand-confirmation");
        }
        if (track->second.promoted_to_working_inventory && track->second.item_id > 0) {
            const int item_id = track->second.item_id;
            working_inventory_.erase(item_id);
            pending_in_ids_.erase(item_id);
            confirmed_moved_ids_.erase(item_id);
            pending_out_ids_.erase(item_id);
        }
        clear_pending_front_evidence_for_suspect_(
            track->first, "discard-suspect-before-no-hand-commit");
        track_buffer_.erase(track);
    }

    for (size_t i = 0; i < promote_keys.size(); ++i) {
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(promote_keys[i]);
        if (track == track_buffer_.end() || !track->second.has_last_seen_box) continue;
        Detection d;
        d.box = track->second.last_seen_box;
        d.cls_id = track->second.cls_id;
        d.score = 0.0f;
        promote_suspect_(promote_keys[i], d, trace_frame_id_);
        track->second.placed_box = d.box;
        track->second.has_placed_box = true;
        track->second.drop_confirmed = true;
    }

    // 所有旧 C、已有 D 都已优先处理完成；此时剩余 B 才能成为“全程被手
    // 遮挡、手离开后首次显现”的 D。
    register_post_hand_reveal_suspects_(detections, &claimed);
    const std::set<int> no_new_existing_tracks;
    advance_claim_grace_(no_new_existing_tracks);
}

bool SessionManager::defer_direct_missing_out_for_possible_occlusion_(
        int target_item_id, const std::vector<Detection>& detections,
        int* blocker_item_id, int* detection_index) const {
    if (blocker_item_id) *blocker_item_id = -1;
    if (detection_index) *detection_index = -1;
    const std::map<int, InventoryItem>::const_iterator target =
        operation_start_inventory_.find(target_item_id);
    if (target == operation_start_inventory_.end()) return false;
    const BBox target_box = target->second.base_box.area() > 0.0f
        ? target->second.base_box : target->second.box;
    if (target_box.area() <= 0.0f) return false;

    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        const OperationTrack& candidate = it->second;
        if (candidate.is_suspect_new || candidate.item_id <= 0 ||
            candidate.item_id == target_item_id ||
            !operation_start_inventory_.count(candidate.item_id) ||
            pending_out_ids_.count(candidate.item_id) ||
            confirmed_moved_ids_.count(candidate.item_id) ||
            candidate.resolution == ExistingItemResolution::MOVED_CONFIRMED) {
            continue;
        }

        const bool has_real_move_evidence = candidate.hold_and_move ||
            has_meaningful_hand_move(candidate) ||
            (candidate.contact_state == ContactState::CONTACT_MOVING &&
             !candidate.observed_move_values.empty()) ||
            (candidate.has_reappear_candidate_box &&
             reappear_candidate_is_confirmed(candidate));
        if (!has_real_move_evidence || candidate.no_hand_detection_index < 0 ||
            static_cast<size_t>(candidate.no_hand_detection_index) >= detections.size()) {
            continue;
        }

        const Detection& observed = detections[candidate.no_hand_detection_index];
        bool strictly_owned_by_unrelated_item = false;
        for (std::map<int, InventoryItem>::const_iterator item =
                 working_inventory_.begin(); item != working_inventory_.end(); ++item) {
            if (item->first == candidate.item_id || item->first == target_item_id) {
                continue;
            }
            if (strict_match(item->second, observed)) {
                strictly_owned_by_unrelated_item = true;
                break;
            }
        }
        if (observed.cls_id != candidate.cls_id ||
            !boxes_differ_as_move(candidate.original_box, observed.box) ||
            strictly_owned_by_unrelated_item ||
            intersection_area(target_box, observed.box) <= BLOCK_OVERLAP_AREA_EPS) {
            continue;
        }

        if (blocker_item_id) *blocker_item_id = candidate.item_id;
        if (detection_index) *detection_index = candidate.no_hand_detection_index;
        return true;
    }
    return false;
}

bool SessionManager::has_unresolved_no_hand_state_(
        const std::vector<Detection>& detections,
        const std::set<int>& observed_item_ids,
        const std::set<int>& fully_occluded_item_ids,
        const std::map<int, BlockerTransitionPlan>& transition_plans) {
    bool unresolved = false;

    // 数量不足 OUT 候选中可能有“本次手操作没有单独建轨”的旧 C。它们也
    // 必须让整轮操作等待到第二张直接无手帧，不能因为没有 OperationTrack
    // 就在首帧把旧库存原样提交回去。
    for (std::set<int>::const_iterator item =
             visible_count_out_candidate_ids_.begin();
         item != visible_count_out_candidate_ids_.end(); ++item) {
        const std::map<int, BlockerTransitionPlan>::const_iterator plan =
            transition_plans.find(*item);
        if (plan != transition_plans.end() &&
            (plan->second.out == OutDisposition::HOLD_FOR_PENDING_OCCLUSION ||
             plan->second.out == OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION ||
             plan->second.out == OutDisposition::NOT_APPLICABLE)) {
            unresolved = unresolved ||
                plan->second.out == OutDisposition::HOLD_FOR_PENDING_OCCLUSION;
            continue;
        }
        if (observed_item_ids.count(*item) || fully_occluded_item_ids.count(*item)) {
            continue;
        }
        if (pending_out_ids_.count(*item)) {
            trace_("VISIBLE-COUNT",
                   "item=%d settlement=out-confirmed-visible-count-deficit",
                   *item);
        } else {
            unresolved = true;
            trace_("VISIBLE-COUNT",
                   "item=%d settlement=defer-visible-count-deficit count=%d",
                   *item, visible_count_missing_counts_[*item]);
        }
    }

    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;

        const std::map<int, BlockerTransitionPlan>::const_iterator lifecycle_plan =
            transition_plans.find(track.item_id);
        const bool hold_for_pending_occlusion = lifecycle_plan !=
            transition_plans.end() && lifecycle_plan->second.out ==
                OutDisposition::HOLD_FOR_PENDING_OCCLUSION;
        if (hold_for_pending_occlusion) {
            const bool ordinary_direct_pending_out =
                pending_out_ids_.count(track.item_id) &&
                !visible_count_confirmed_out_ids_.count(track.item_id) &&
                track.no_hand_missing_count >= FLOW3_NO_HAND_OUT_MISSING_FRAMES;
            if (ordinary_direct_pending_out) {
                pending_out_ids_.erase(track.item_id);
                if (track.resolution == ExistingItemResolution::OUT_CONFIRMED) {
                    track.resolution = ExistingItemResolution::NONE;
                    track.needs_no_hand_settlement = true;
                    track.release_reason = ReleaseReason::NONE;
                }
                track.no_hand_missing_count = std::max(
                    0, FLOW3_NO_HAND_OUT_MISSING_FRAMES - 1);
            }
            unresolved = true;
            trace_track_("OCCLUSION", track,
                         "hold-out-for-pending-disappearance-supported");
            continue;
        }

        if (track.is_suspect_new) {
            if (!is_active_runtime_track(track)) continue;
            // D 只有在至少两张直接无手帧中连续匹配到自己后才可提交 IN。
            // 这是单对象时序证据，不是多检测框投票。
            if (!track.promoted_to_working_inventory || track.item_id <= 0 ||
                !observed_item_ids.count(track.item_id) ||
                track.no_hand_self_match_count < FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
                unresolved = true;
                trace_track_("NO-HAND", track, "suspect-not-yet-directly-confirmed");
            }
            continue;
        }

        // OCCLUDED_CONFIRMED is also used as a per-frame runtime marker after
        // a newly established causal/full-coverage plan.  A deferred
        // transaction does not write that projection to working_inventory_,
        // so an ordinary VISIBLE operation-start item must not remain
        // terminal when the front relation loses coverage on the next frame.
        // Formal OCCLUDED inventory starts are intentionally left untouched;
        // their dedicated coverage-loss chain handles the historical proof.
        const std::map<int, InventoryItem>::const_iterator original_item =
            operation_start_inventory_.find(track.item_id);
        if (!fully_occluded_item_ids.count(track.item_id) &&
            track.resolution == ExistingItemResolution::OCCLUDED_CONFIRMED &&
            original_item != operation_start_inventory_.end() &&
            original_item->second.status == ItemStatus::VISIBLE) {
            track.resolution = ExistingItemResolution::NONE;
            track.release_reason = ReleaseReason::NONE;
            track.needs_no_hand_settlement = true;
            trace_track_("OCCLUSION", track,
                         "retract-provisional-occlusion-after-coverage-loss");
        }

        if (!existing_item_needs_settlement(track)) continue;

        const bool is_visible_count_out_candidate =
            visible_count_out_candidate_ids_.count(track.item_id) &&
            !observed_item_ids.count(track.item_id) &&
            !fully_occluded_item_ids.count(track.item_id);
        const bool is_visible_count_survivor =
            visible_count_survivor_ids_.count(track.item_id) &&
            observed_item_ids.count(track.item_id);
        if (is_visible_count_out_candidate) {
            // 该 C 已经在本帧同类一对一分配中没有任何可见框。邻居 B 的框
            // 已被独占保留，不能再作为 A 的歧义候选阻止连续 OUT 计数。
            trace_track_("VISIBLE-COUNT", track,
                         pending_out_ids_.count(track.item_id)
                             ? "out-confirmed-visible-count-deficit"
                             : "await-visible-count-deficit-confirmation");
            continue;
        }
        if (is_visible_count_survivor) {
            // 强制保留只代表一个可见实例，不使它绕过本轮“连续无手”要求；
            // 本函数被调用前已由 prepare_visible_count_settlement_ 保证两帧
            // 都会重新建立同一保留关系。
            track.no_hand_missing_count = 0;
            trace_track_("VISIBLE-COUNT", track,
                         "directly-observed-visible-count-survivor");
            continue;
        }

        // 状态被意外变成 NORMAL 绝不是“旧 C 已结案”。此时没有可靠活动
        // 轨迹可安全累计 OUT，必须保持未决并阻止本轮提交。
        if (!is_active_runtime_track(track)) {
            unresolved = true;
            trace_track_("NO-HAND", track, "normal-track-still-needs-settlement");
            continue;
        }

        // Once the blocker lifecycle has produced a formal full-coverage
        // result for this target, same-class reservation ambiguity no longer
        // represents unresolved OUT evidence.  It only affected which visible
        // box could belong to the target; the confirmed front blocker already
        // owns the causal OCCLUDED proof for this transaction.
        if (!fully_occluded_item_ids.count(track.item_id) &&
            (is_claim_protected(track) || track.b_claim_ambiguous ||
             track.contact_path_ambiguous || track.no_hand_candidate_ambiguous ||
             track.no_hand_candidate_reserved_by_stronger_owner)) {
            unresolved = true;
            trace_track_("NO-HAND", track,
                         track.no_hand_candidate_ambiguous
                             ? "ambiguous-no-hand-path-candidate"
                             : (track.no_hand_candidate_reserved_by_stronger_owner
                                 ? "no-hand-path-reserved-by-stronger-old-c"
                                 : "claim-or-contact-ambiguity"));
            continue;
        }
        if (!fully_occluded_item_ids.count(track.item_id) &&
            old_track_has_unresolved_alias_(track)) {
            unresolved = true;
            trace_track_("C-D-ALIAS", track,
                         "old-c-awaits-quarantined-pending-d-no-hand-arbitration");
            continue;
        }
        if (observed_item_ids.count(track.item_id)) {
            // 手离开后首次重新出现的同类 B 仍要在下一张直接无手帧中和
            // 自己连续匹配。不能让本帧的一对一绑定绕过这条身份确认链路。
            if (track.has_reappear_candidate_box &&
                !reappear_candidate_is_confirmed(track)) {
                unresolved = true;
                trace_track_("NO-HAND", track, "reappear-candidate-needs-second-frame");
                continue;
            }
            // 只有本轮因 hand_id 保护而丢失过 delta 的 HAND_* 旧 C，才
            // 必须在这里等直接恢复路径结案，避免同伴的 MOVED/IN 先提交并
            // 清空整轮 operation runtime。CONTACT_*、静态收尾等既有路径
            // 仍按原来的结算时机放行，不能被这个联合提交门永久挡住。
            const bool requires_interrupted_direct_recovery_barrier =
                track.hand_delta_interrupted &&
                track.contact_state == ContactState::NONE &&
                is_active_existing_hand_track(track);
            if (requires_interrupted_direct_recovery_barrier &&
                existing_item_needs_settlement(track)) {
                unresolved = true;
                trace_track_("SETTLE", track,
                             "defer-directly-observed-still-unresolved");
                continue;
            }
            track.no_hand_missing_count = 0;
            trace_track_("NO-HAND", track, "directly-observed");
            continue;
        }

        // 一张普通 direct-missing OUT 已经暂存，但本帧由确认 blocker
        // 完整解释时，必须先撤回这条尚未提交的 OUT。否则 A 在下一帧完成
        // MOVED 时，C 虽然已经可判 OCCLUDED，仍会在 final_items 删除阶段
        // 被错误移除。visible-count OUT 不属于这个窄保护，绝不在此撤回。
        const bool ordinary_direct_pending_out =
            pending_out_ids_.count(track.item_id) &&
            !visible_count_confirmed_out_ids_.count(track.item_id) &&
            track.no_hand_missing_count >= FLOW3_NO_HAND_OUT_MISSING_FRAMES;
        if (fully_occluded_item_ids.count(track.item_id) &&
            ordinary_direct_pending_out) {
            pending_out_ids_.erase(track.item_id);
            if (track.resolution == ExistingItemResolution::OUT_CONFIRMED) {
                track.resolution = ExistingItemResolution::NONE;
                track.release_reason = ReleaseReason::NONE;
                track.needs_no_hand_settlement = true;
                track.no_hand_missing_count =
                    std::max(0, FLOW3_NO_HAND_OUT_MISSING_FRAMES - 1);
                trace_track_("OCCLUSION", track,
                             "retract-direct-out-after-confirmed-occlusion");
            }
        }

        // 这只保护“普通 direct-missing”链路在 A 的已有移动证据尚差最后
        // 一帧确认时不抢先 OUT。A 还不是 blocker，不写 block_ids/status；
        // 若 A 下一帧没有按既有流程确认 MOVED，保护自动消失，C 从阈值-1
        // 继续原有 OUT 链，不会永久冻结。
        int possible_front_id = -1;
        int possible_front_detection = -1;
        if (!fully_occluded_item_ids.count(track.item_id) &&
            defer_direct_missing_out_for_possible_occlusion_(
                track.item_id, detections, &possible_front_id,
                &possible_front_detection)) {
            if (ordinary_direct_pending_out) {
                pending_out_ids_.erase(track.item_id);
                if (track.resolution == ExistingItemResolution::OUT_CONFIRMED) {
                    track.resolution = ExistingItemResolution::NONE;
                    track.release_reason = ReleaseReason::NONE;
                    track.needs_no_hand_settlement = true;
                }
                track.no_hand_missing_count =
                    std::max(0, FLOW3_NO_HAND_OUT_MISSING_FRAMES - 1);
                trace_track_("OCCLUSION", track,
                             "retract-direct-out-for-unconfirmed-moving-front");
            }
            unresolved = true;
            trace_("OCCLUSION",
                   "target=%d action=defer-direct-out possible-front=%d detection=%d "
                   "missing=%d/%d",
                   track.item_id, possible_front_id, possible_front_detection,
                   track.no_hand_missing_count, FLOW3_NO_HAND_OUT_MISSING_FRAMES);
            continue;
        }
        if (fully_occluded_item_ids.count(track.item_id)) {
            track.no_hand_missing_count = 0;
            track.resolution = ExistingItemResolution::OCCLUDED_CONFIRMED;
            track.release_reason = ReleaseReason::FULLY_OCCLUDED;
            track.needs_no_hand_settlement = false;
            track.stable_near_original_no_hand_count = 0;
            track.has_stable_near_original_box = false;
            track.stable_near_original_box = BBox();
            trace_track_("NO-HAND", track, "fully-occluded-by-confirmed-front-item");
            continue;
        }
        if (pending_out_ids_.count(track.item_id)) continue;

        // CONTACT_CANDIDATE 没有足够的真实移动证据，继续保持未决；只有
        // CONTACT_MOVING 才可沿实际观察路径累计无手缺失并最终 OUT。
        const bool contact_out_evidence =
            track.contact_state == ContactState::CONTACT_MOVING &&
            track.hold_and_move && !track.observed_move_values.empty();
        const bool hand_out_evidence =
            track.contact_state == ContactState::NONE &&
            track.state != OperationTrackState::NORMAL &&
            (track.hold_and_move || has_meaningful_hand_move(track));
        const bool interrupted_direct_exit_evidence =
            track.contact_state == ContactState::NONE &&
            track.state != OperationTrackState::NORMAL &&
            track.hand_delta_interrupted && track.has_direct_exit_evidence &&
            track.has_first_hand_block_box && track.direct_exit_frame >= 0 &&
            track.direct_exit_box.area() > 0.0f &&
            boxes_differ_as_move(track.original_box, track.direct_exit_box) &&
            !track.b_claim_ambiguous && !track.contact_path_ambiguous &&
            !track.no_hand_candidate_ambiguous &&
            !track.no_hand_candidate_reserved_by_stronger_owner &&
            !old_track_has_unresolved_alias_(track);
        const bool direct_object_exit_evidence =
            track.contact_state == ContactState::NONE &&
            track.state != OperationTrackState::NORMAL &&
            track.has_direct_object_exit_evidence &&
            track.direct_object_path_streak >= FLOW3_NO_HAND_D_CONFIRM_FRAMES &&
            track.direct_object_last_frame >= 0 &&
            track.has_direct_object_last_box &&
            boxes_differ_as_move(track.original_box, track.direct_object_last_box) &&
            !track.b_claim_ambiguous && !track.contact_path_ambiguous &&
            !track.no_hand_candidate_ambiguous &&
            !track.no_hand_candidate_reserved_by_stronger_owner &&
            !old_track_has_unresolved_alias_(track);
        const bool hand_group_exit_evidence =
            track.contact_state == ContactState::NONE &&
            track.state != OperationTrackState::NORMAL &&
            track.carrier_capture_context && track.capture_was_fully_hidden &&
            !track.hand_group_identity_invalid && track.hand_group_exit_witness &&
            track.hand_group_exit_frame >= 0 &&
            !track.possible_carrier_hand_ids.empty() &&
            old_position_is_clean(detections, track, working_inventory_) &&
            !track.b_claim_ambiguous && !track.contact_path_ambiguous &&
            !track.no_hand_candidate_ambiguous &&
            !track.no_hand_candidate_reserved_by_stronger_owner &&
            !old_track_has_unresolved_alias_(track);
        if (!contact_out_evidence && !hand_out_evidence &&
            !interrupted_direct_exit_evidence && !direct_object_exit_evidence &&
            !hand_group_exit_evidence) {
            unresolved = true;
            trace_track_("NO-HAND", track, "missing-without-sufficient-out-evidence");
            continue;
        }
        if (interrupted_direct_exit_evidence && !contact_out_evidence &&
            !hand_out_evidence) {
            trace_("DIRECT-EXIT",
                   "item=%d action=accept-for-existing-missing-chain evidence-frame=%d "
                   "missing=%d/%d",
                   track.item_id, track.direct_exit_frame, track.no_hand_missing_count,
                   FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        }
        if (direct_object_exit_evidence && !contact_out_evidence &&
            !hand_out_evidence && !interrupted_direct_exit_evidence) {
            trace_("DIRECT-OBJECT",
                   "item=%d action=accept-for-existing-missing-chain path-frame=%d "
                   "streak=%d missing=%d/%d",
                   track.item_id, track.direct_object_last_frame,
                   track.direct_object_path_streak, track.no_hand_missing_count,
                   FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        }
        if (hand_group_exit_evidence && !contact_out_evidence && !hand_out_evidence &&
            !interrupted_direct_exit_evidence && !direct_object_exit_evidence) {
            trace_("HAND-GROUP-EXIT",
                   "item=%d action=accept-for-existing-missing-chain witness-frame=%d "
                   "candidates=%zu missing=%d/%d",
                   track.item_id, track.hand_group_exit_frame,
                   track.possible_carrier_hand_ids.size(), track.no_hand_missing_count,
                   FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        }

        const int old_missing_count = track.no_hand_missing_count;
        ++track.no_hand_missing_count;
        trace_("NO-HAND", "item=%d direct-missing-count=%d->%d threshold=%d",
               track.item_id, old_missing_count, track.no_hand_missing_count,
               FLOW3_NO_HAND_OUT_MISSING_FRAMES);
        trace_track_("NO-HAND", track, "direct-missing-frame");
        if (track.no_hand_missing_count >= FLOW3_NO_HAND_OUT_MISSING_FRAMES) {
            mark_pending_out_(track.item_id);
        } else {
            unresolved = true;
        }
    }

    // 最终保险：即使此前某个旧 C 的运行时记录已经意外失活，只要同类 D
    // 要正式 IN，而该旧 C 在本帧没有独立观测/遮挡/已确认结束，仍拒绝提交。
    for (std::set<int>::const_iterator pending = pending_in_ids_.begin();
         pending != pending_in_ids_.end(); ++pending) {
        std::map<int, InventoryItem>::const_iterator d = working_inventory_.find(*pending);
        if (d == working_inventory_.end()) continue;
        for (std::map<int, InventoryItem>::const_iterator old =
                 operation_start_inventory_.begin();
             old != operation_start_inventory_.end(); ++old) {
            if (old->second.cls_id != d->second.cls_id ||
                pending_out_ids_.count(old->first) ||
                observed_item_ids.count(old->first) ||
                fully_occluded_item_ids.count(old->first)) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(old->first);
            if (existing_item_resolved_without_current_detection(runtime)) continue;
            unresolved = true;
            trace_("SETTLE",
                   "block-commit pending-d=%d cls=%d unresolved-old=%d runtime=%s resolution=%s",
                   *pending, d->second.cls_id, old->first,
                   runtime ? operation_track_state_name(runtime->state) : "NONE",
                   runtime ? existing_resolution_name(runtime->resolution) : "NONE");
            break;
        }
    }
    return unresolved;
}

}  // namespace fridge
