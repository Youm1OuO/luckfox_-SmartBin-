// ============================================================================
//  session_hand.cc
//  3.0 session HAND and CONTACT operation state machine
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

std::string candidate_context_ids(const std::set<int>& item_ids) {
    std::ostringstream stream;
    stream << "[";
    for (std::set<int>::const_iterator it = item_ids.begin(); it != item_ids.end(); ++it) {
        if (it != item_ids.begin()) stream << ",";
        stream << *it;
    }
    stream << "]";
    return stream.str();
}

const char* candidate_context_decision(const SameClassCandidateContext& context) {
    if (context.viable_unresolved_old_item_ids.empty()) {
        return "independent-d-candidate";
    }
    return context.viable_unresolved_old_item_ids.size() == 1
        ? "unique-unresolved-old-c"
        : "multiple-unresolved-old-c";
}

}  // namespace

void SessionManager::begin_working_operation_(const BBox& hand_box,
                                               const std::vector<Detection>& detections) {
    working_inventory_ = inventory_.items();
    operation_start_inventory_ = inventory_.items();
    working_next_item_id_ = inventory_.next_item_id();
    working_inventory_active_ = true;
    track_buffer_.clear();
    pending_in_ids_.clear();
    pending_out_ids_.clear();
    confirmed_moved_ids_.clear();
    released_hand_candidate_ids_.clear();
    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    visible_count_out_candidate_ids_.clear();
    visible_count_missing_counts_.clear();
    visible_count_confirmed_out_ids_.clear();
    visible_count_prior_survivors_by_cls_.clear();
    visible_count_prior_survivor_boxes_by_cls_.clear();
    occlusion_loss_missing_counts_.clear();
    hand_track_.clear();
    hand_track_.push_back(hand_box);
    old_hand_box_ = hand_box;
    has_old_hand_box_ = true;
    no_hand_streak_ = 0;
    active_operation_id_ = next_operation_id_++;
    trace_("STATE", "begin operation inventory=%zu detections=%zu hand=(%.1f,%.1f,%.1f,%.1f)",
           working_inventory_.size(), detections.size(), hand_box.x1, hand_box.y1,
           hand_box.x2, hand_box.y2);
    process_effective_hand_frame_(hand_box, detections, true);
}

void SessionManager::link_suspect_to_conflicting_old_items_(
        int runtime_key, const std::set<int>& old_item_ids,
        const char* phase, int detection_index) {
    if (old_item_ids.empty()) return;
    std::map<int, OperationTrack>::iterator suspect = track_buffer_.find(runtime_key);
    if (suspect == track_buffer_.end() || !suspect->second.is_suspect_new) {
        return;
    }

    // 先过滤掉本帧已经结案的 old C。只有实际会写入的冲突边才能让一个
    // 已暂存的 D 降回 runtime-only alias，避免无效候选意外撤销 D 证据链。
    std::set<int> valid_old_item_ids;
    for (std::set<int>::const_iterator old_id = old_item_ids.begin();
         old_id != old_item_ids.end(); ++old_id) {
        std::map<int, OperationTrack>::iterator old = track_buffer_.find(*old_id);
        if (old == track_buffer_.end() || old->second.is_suspect_new ||
            !is_unresolved_operation_start_old_track(old->second)) {
            continue;
        }
        valid_old_item_ids.insert(*old_id);
    }
    if (valid_old_item_ids.empty()) return;

    OperationTrack& d = suspect->second;
    // 某个旧 C 的可行路径可能在 D 连续自匹配后才显露。此时 D 虽然已经
    // 暂存进 working_inventory_，仍只是未提交候选；必须回收该 staged item
    // 并降回 C-D alias，不能让一个“不认识 A”的 pending D 保留排他资格、
    // 参与 blocker 或先于 A 提交 IN。
    if (d.promoted_to_working_inventory && d.item_id > 0) {
        const int staged_item_id = d.item_id;
        working_inventory_.erase(staged_item_id);
        pending_in_ids_.erase(staged_item_id);
        pending_out_ids_.erase(staged_item_id);
        confirmed_moved_ids_.erase(staged_item_id);
        d.item_id = 0;
        d.promoted_to_working_inventory = false;
        d.drop_confirmed = false;
        d.no_hand_detection_index = -1;
        trace_("C-D-ALIAS",
               "suspect=%d staged-item=%d detection=%d phase=%s "
               "action=retract-pending-d-to-complete-conflict-graph",
               d.suspect_id, staged_item_id, detection_index,
               phase ? phase : "UNKNOWN");
    }
    d.pending_d_quarantined_by_old_c = true;
    for (std::set<int>::const_iterator old_id = valid_old_item_ids.begin();
         old_id != valid_old_item_ids.end(); ++old_id) {
        std::map<int, OperationTrack>::iterator old = track_buffer_.find(*old_id);
        const bool inserted = d.conflicting_old_item_ids.insert(*old_id).second;
        old->second.conflicting_suspect_keys.insert(runtime_key);
        if (inserted) {
            trace_("C-D-ALIAS",
                   "old-item=%d suspect=%d detection=%d phase=%s relation=viable-old-c "
                   "action=link-complete-candidate-context",
                   *old_id, d.suspect_id, detection_index, phase ? phase : "UNKNOWN");
        }
    }
}

void SessionManager::append_move_to_existing_hand_tracks_(const MoveValue& delta) {
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.contact_state != ContactState::NONE) {
            // 仅作调试/未来扩展记录；CONTACT_* 的物品位置不能由手位移推算。
            track.hand_move_values.push_back(delta);
        }
        if (track.state != OperationTrackState::HAND_PARTIAL_BLOCKED &&
            track.state != OperationTrackState::HAND_FULL_BLOCKED) {
            continue;
        }
        track.move_values.push_back(delta);
        track.track.push_back(estimated_box(track));
    }
}

void SessionManager::update_existing_contact_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        const HandDirectOldOwnerPlan& direct_old_owner_plan) {
    std::vector<std::pair<int, int> > release_keys;
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (!is_active_contact_track(track)) continue;

        const int direct_owner_index = direct_old_owner_detection_for_item(
            direct_old_owner_plan, track.item_id);
        const HandDirectOldOwnerStrength direct_owner_strength =
            direct_owner_index >= 0
                ? direct_old_owner_strength_for_detection(
                      direct_old_owner_plan, direct_owner_index)
                : HandDirectOldOwnerStrength::LOCAL_WEAK;
        if (direct_owner_index >= 0 &&
            direct_owner_strength == HandDirectOldOwnerStrength::STRICT) {
            const Detection& direct = detections[direct_owner_index];
            claimed_detection_indices->insert(direct_owner_index);
            (*known_item_owner)[track.item_id] = direct_owner_index;
            // 严格原位框比 CONTACT 的路径候选优先。若之前已暂时进入
            // CONTACT_MOVING，本帧直接原位证据也必须撤销未提交的正向证据。
            if (track.contact_state == ContactState::CONTACT_MOVING) {
                track.contact_state = ContactState::CONTACT_CANDIDATE;
                track.hold_and_move = false;
                track.drop_evidence_count = 0;
            }
            append_contact_observation(&track, direct,
                                       hand_is_near(hand_box, direct.box));
            track.has_tentative_b_box = false;
            track.tentative_b_match_count = 0;
            track.tentative_b_started_touching_hand = false;
            track.has_reappear_candidate_box = false;
            track.reappear_candidate_match_count = 0;
            track.reappear_candidate_started_touching_hand = false;
            ++track.not_hold_evidence_count;
            track.hold_evidence_count = 0;
            if (track.not_hold_evidence_count >= FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                release_keys.push_back(std::make_pair(it->first, direct_owner_index));
            }
            continue;
        }

        // 若 CONTACT_* 已经用真实 B 看见过物品的新位置，覆盖率必须相对
        // 当前真实位置计算；仍拿最初 A.box 计算会使“先推、后握住”永远
        // 停留在 CONTACT_*。
        const BBox reference = track.has_tentative_b_box ? track.tentative_b_box
            : (track.has_last_seen_box ? track.last_seen_box : track.original_box);
        if (reference.area() <= 0.0f) continue;

        // 覆盖率达到 e2 后，接触候选转入已有 HAND_* 流程；同一帧稍后由
        // update_existing_hand_tracks_ 继续处理，避免两套逻辑同时认领。
        if (hand_fully_covers(hand_box, reference) || hand_affects(hand_box, reference)) {
            // 保留 contact 阶段的真实位置作为之后 HAND_* 估计的原点。
            // original_box 不能改，它还要用于判断最终是否发生整理/出库。
            track.hand_estimate_anchor_box = reference;
            track.has_hand_estimate_anchor_box = true;
            track.move_values.clear();
            track.track.clear();
            track.track.push_back(reference);
            track.contact_state = ContactState::NONE;
            track.state = hand_fully_covers(hand_box, reference)
                ? OperationTrackState::HAND_FULL_BLOCKED
                : OperationTrackState::HAND_PARTIAL_BLOCKED;
            if (track.has_last_seen_box) {
                track.first_hand_block_box = track.last_seen_box;
                track.last_hand_block_box = track.last_seen_box;
                track.has_first_hand_block_box = true;
                track.has_last_hand_block_box = true;
            }
            trace_track_("STATE", track, "contact-to-hand-transition");
            continue;
        }

        const std::set<int> candidate_claimed = claimed_with_other_direct_old_owners(
            *claimed_detection_indices, direct_old_owner_plan, track.item_id);
        int observed_index = -1;
        if (direct_owner_index >= 0 &&
            direct_owner_strength == HandDirectOldOwnerStrength::LOCAL_CONTINUOUS) {
            // 连续局部框可以继续服务于它自己的真实 CONTACT 路径，但对其他
            // C/D 已经是排他所有权，不能再按遍历顺序借走。严格原位框仍由
            // 下方的 CONTACT 原位 helper 处理；活动 CONTACT 的严格框若已
            // 超过 12px，会在所有权计划阶段保留为普通路径观察，不能清掉
            // 原有 tentative/hold 连续证据。
            observed_index = direct_owner_index;
        } else {
            observed_index = unique_contact_detection_for_track(
                detections, candidate_claimed, track, working_inventory_, track_buffer_);
        }
        if (observed_index < 0) {
            // 漏检、多个同类候选或与其他库存冲突都不是“未持有”的证据。
            if (has_contact_path_candidate(detections, candidate_claimed,
                                           track, working_inventory_)) {
                track.contact_path_ambiguous = true;
            }
            continue;
        }

        const Detection& detection = detections[observed_index];
        const bool touching_hand = hand_is_near(hand_box, detection.box);
        const bool had_touch_before = track.contact_started_touching_hand;
        const BBox previous = track.has_last_seen_box
            ? track.last_seen_box : track.original_box;
        const bool at_original = contact_detection_is_at_original(track, detection);
        const float object_move = center_distance(previous, detection.box);
        const int matching_tentative_count =
            track.has_tentative_b_box &&
            track_match_box(track.cls_id, track.tentative_b_box,
                            detection.cls_id, detection.box)
                ? track.tentative_b_match_count : 0;

        if (is_claim_protected(track) && !at_original) {
            // 保护期内照常检查这条 C→B 本地路径，但不得把 B 写入 claimed /
            // known_item_owner；同一 B 仍必须对成熟 C 和 D 链路开放仲裁。
            record_tentative_b(&track, detection, touching_hand);
            // observed_track 是 C 自己的本地真实观测，不等于本帧的排他
            // B 认领。保存它可避免 CONTACT→HAND 时丢掉最新可靠物品框。
            append_contact_observation(&track, detection, touching_hand);
            if (object_move >= FLOW3_CONTACT_OBJECT_MOVE_EPS &&
                (had_touch_before || touching_hand)) {
                ++track.hold_evidence_count;
                track.not_hold_evidence_count = 0;
            }
            continue;
        }

        claimed_detection_indices->insert(observed_index);
        if (track.item_id > 0) (*known_item_owner)[track.item_id] = observed_index;
        append_contact_observation(&track, detection, touching_hand);

        if (!at_original) {
            track.has_tentative_b_box = false;
            track.tentative_b_match_count = 0;
            track.tentative_b_started_touching_hand = false;
        }

        if (track.contact_state == ContactState::CONTACT_MOVING) {
            std::map<int, InventoryItem>::iterator item =
                working_inventory_.find(track.item_id);
            if (item != working_inventory_.end()) {
                update_seen(item->second, detection, trace_frame_id_);
            }
            continue;
        }

        if (at_original) {
            track.has_tentative_b_box = false;
            track.tentative_b_match_count = 0;
            track.tentative_b_started_touching_hand = false;
            track.has_reappear_candidate_box = false;
            track.reappear_candidate_match_count = 0;
            track.reappear_candidate_started_touching_hand = false;
            ++track.not_hold_evidence_count;
            track.hold_evidence_count = 0;
        } else if (object_move >= FLOW3_CONTACT_OBJECT_MOVE_EPS &&
                   (had_touch_before || touching_hand)) {
            // 只要第一次有效 B 已经与手相贴，后续 B 可在手离开后继续沿
            // observed_track 确认，不把手框方向当作物品方向。
            ++track.hold_evidence_count;
            if (matching_tentative_count > 0) {
                // t3 的唯一 B 接上 t0~t2 已记录的本地证据时，直接继承
                // 连续性；它仍是在成熟后才变成正式归属。
                track.hold_evidence_count = std::max(
                    track.hold_evidence_count, matching_tentative_count);
            }
            track.not_hold_evidence_count = 0;
        }

        if (track.not_hold_evidence_count >= FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
            release_keys.push_back(std::make_pair(it->first, observed_index));
            continue;
        }
        if (track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
            track.contact_state = ContactState::CONTACT_MOVING;
            track.hold_and_move = true;
            track.hold_evidence_count = 0;
            track.not_hold_evidence_count = 0;
            track.shelter_or_hold = true;
            printf("[3.0] item#%d 确认低覆盖率推/拉，进入 CONTACT_MOVING\n",
                   track.item_id);
        }
    }

    for (size_t i = 0; i < release_keys.size(); ++i) {
        std::map<int, OperationTrack>::iterator it = track_buffer_.find(release_keys[i].first);
        if (it != track_buffer_.end() &&
            it->second.contact_state == ContactState::CONTACT_CANDIDATE) {
            release_not_held_(it->second, false,
                              ReleaseReason::CONTACT_RETURNED_ORIGINAL,
                              release_keys[i].second,
                              &detections[release_keys[i].second].box,
                              "hand-contact-returned-original");
        }
    }
}

void SessionManager::mark_new_contact_candidates_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        std::set<int>* new_existing_track_ids) {
    // 已释放的候选在手仍停留附近时不反复创建；手离开该物品后才允许下一次
    // 接触重新建立候选。
    for (std::set<int>::iterator released = released_hand_candidate_ids_.begin();
         released != released_hand_candidate_ids_.end();) {
        std::map<int, InventoryItem>::const_iterator item =
            working_inventory_.find(*released);
        if (item == working_inventory_.end() ||
            !hand_is_near(hand_box, item->second.base_box.area() > 0.0f
                                      ? item->second.base_box : item->second.box)) {
            released = released_hand_candidate_ids_.erase(released);
        } else {
            ++released;
        }
    }

    for (std::map<int, InventoryItem>::const_iterator it = working_inventory_.begin();
         it != working_inventory_.end(); ++it) {
        const InventoryItem& item = it->second;
        if (item.status == ItemStatus::OCCLUDED ||
            released_hand_candidate_ids_.count(item.item_id)) {
            continue;
        }
        OperationTrack* existing = find_runtime_for_item_(item.item_id);
        if (existing && is_active_runtime_track(*existing)) continue;

        const BBox reference = item.base_box.area() > 0.0f
            ? item.base_box : item.box;
        if (reference.area() <= 0.0f || hand_cover_ratio(hand_box, reference) >=
                FLOW3_HAND_PARTIAL_COVER_RATIO ||
            !hand_is_near(hand_box, reference)) {
            continue;
        }

        OperationTrack track;
        track.item_id = item.item_id;
        track.cls_id = item.cls_id;
        track.original_box = reference;
        track.contact_state = ContactState::CONTACT_CANDIDATE;
        track.needs_no_hand_settlement = true;
        track.shelter_or_hold = true;
        track.claim_grace_remaining = FLOW3_NEW_TRACK_CLAIM_GRACE_FRAMES;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track_buffer_[item.item_id] = track;
        OperationTrack& created = track_buffer_[item.item_id];
        if (new_existing_track_ids) new_existing_track_ids->insert(item.item_id);

        int observed_index = -1;
        std::map<int, int>::const_iterator owner = known_item_owner->find(item.item_id);
        if (owner != known_item_owner->end() &&
            owner->second >= 0 &&
            owner->second < static_cast<int>(detections.size()) &&
            contact_path_match_cost(created, detections[owner->second]) <
                std::numeric_limits<float>::infinity()) {
            observed_index = owner->second;
        } else {
            observed_index = unique_contact_detection_for_track(
                detections, *claimed_detection_indices, created,
                working_inventory_, track_buffer_);
        }
        if (observed_index >= 0) {
            const bool touching = hand_is_near(hand_box, detections[observed_index].box);
            const bool at_original = contact_detection_is_at_original(
                created, detections[observed_index]);
            // 保护期只禁止“移动后的同类 B”的排他归属。C 若确实仍在旧
            // 位置，保留自己的严格框是安全的；否则撤销 reserve 阶段可能
            // 做出的预占，只把它保存为本地 tentative B。
            if (at_original || !is_claim_protected(created)) {
                if (!claimed_detection_indices->count(observed_index)) {
                    claimed_detection_indices->insert(observed_index);
                }
                (*known_item_owner)[item.item_id] = observed_index;
                append_contact_observation(&created, detections[observed_index], touching);
            } else {
                claimed_detection_indices->erase(observed_index);
                known_item_owner->erase(item.item_id);
                record_tentative_b(&created, detections[observed_index], touching);
                append_contact_observation(&created, detections[observed_index], touching);
            }
        }
        printf("[3.0] item#%d 进入 CONTACT_CANDIDATE（低覆盖率且手相贴）\n",
               item.item_id);
        trace_track_("STATE", created, "create-contact-candidate");
    }
}

void SessionManager::promote_suspect_(int runtime_key, const Detection& detection,
                                      int frame_id) {
    std::map<int, OperationTrack>::iterator it = track_buffer_.find(runtime_key);
    if (it == track_buffer_.end()) return;
    OperationTrack& track = it->second;
    if (!track.is_suspect_new || track.promoted_to_working_inventory ||
        track.self_match_count < NEW_ITEM_CONFIRM_FRAMES) {
        return;
    }
    if (track.pending_d_quarantined_by_old_c) {
        trace_("D-GUARD",
               "suspect=%d cls=%d action=keep-quarantined-runtime-only "
               "reason=unresolved-c-d-alias self-match=%d formal-owner-authority=0",
               track.suspect_id, track.cls_id, track.self_match_count);
        return;
    }
    const int item_id = working_next_item_id_++;
    working_inventory_[item_id] = make_inventory_item(item_id, detection,
                                                       frame_id, current_time_ms_);
    // 初次检测可能只是局部框；它只作为临时位置，后续无手直接帧会刷新。
    track.item_id = item_id;
    track.promoted_to_working_inventory = true;
    pending_in_ids_.insert(item_id);
    printf("[3.0] suspect#%d 已提升为工作库存 item#%d（尚未正式 IN）\n",
           track.suspect_id, item_id);
    trace_track_("STATE", track, "promote-suspect-to-working-inventory");
}

void SessionManager::confirm_rearrange_(OperationTrack& track,
                                        const BBox& release_box,
                                        float score, int frame_id) {
    if (track.item_id <= 0) return;
    std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
    if (item == working_inventory_.end()) return;
    item->second.box = release_box;
    item->second.base_box = release_box;
    item->second.score = score;
    item->second.updated_frame = frame_id;
    item->second.status = ItemStatus::VISIBLE;
    item->second.block_ids.clear();
    track.placed_box = release_box;
    track.has_placed_box = true;
    track.drop_confirmed = true;
    track.state = OperationTrackState::PLACED;
    track.contact_state = ContactState::NONE;
    track.resolution = ExistingItemResolution::MOVED_CONFIRMED;
    track.release_reason = ReleaseReason::NONE;
    track.needs_no_hand_settlement = false;
    track.stable_near_original_no_hand_count = 0;
    track.has_stable_near_original_box = false;
    track.stable_near_original_box = BBox();
    track.b_claim_ambiguous = false;
    track.no_hand_candidate_ambiguous = false;
    track.claim_grace_remaining = 0;
    track.no_hand_missing_count = 0;
    track.has_tentative_b_box = false;
    track.tentative_b_match_count = 0;
    track.tentative_b_started_touching_hand = false;
    confirmed_moved_ids_.insert(track.item_id);
    set_live_state_(&track, LiveObservationState::PLACED, false,
                    "moved-confirmed-after-direct-evidence");
    trace_track_("STATE", track, "confirm-rearrange");
}

void SessionManager::rollback_provisional_moved_to_direct_original_(
        OperationTrack* track, int detection_index, const Detection& detection) {
    if (!track || track->is_suspect_new || track->item_id <= 0 ||
        !working_inventory_active_ ||
        !operation_start_inventory_.count(track->item_id) ||
        detection.cls_id != track->cls_id) {
        return;
    }
    if (track->resolution != ExistingItemResolution::MOVED_CONFIRMED &&
        track->state != OperationTrackState::PLACED) {
        return;
    }

    // confirm_rearrange_ 写入的是本次连续手操作的工作状态。唯一的严格
    // 原位置所有权可以在提交前推翻它：必须同时撤掉 MOVED 集合、终点框、
    // drop 标志和它作为前景 blocker 的资格，不能只把 state 改回 NORMAL。
    std::map<int, InventoryItem>::iterator item = working_inventory_.find(track->item_id);
    if (item == working_inventory_.end()) return;
    update_seen(item->second, detection, trace_frame_id_);
    item->second.base_box = detection.box;
    item->second.status = ItemStatus::VISIBLE;
    item->second.block_ids.clear();

    confirmed_moved_ids_.erase(track->item_id);
    pending_out_ids_.erase(track->item_id);
    track->placed_box = BBox();
    track->has_placed_box = false;
    track->drop_confirmed = false;
    track->drop_evidence_count = 0;
    track->last_seen_box = detection.box;
    track->has_last_seen_box = true;
    trace_("ROLLBACK",
           "item=%d action=rollback-provisional-moved-to-direct-original "
           "detection=%d box=(%.1f,%.1f,%.1f,%.1f)",
           track->item_id, detection_index, detection.box.x1, detection.box.y1,
           detection.box.x2, detection.box.y2);
    release_not_held_(*track, false, ReleaseReason::ORIGINAL_DETECTION,
                      detection_index, &detection.box,
                      "direct-original-owner-rolls-back-provisional-moved");
}

void SessionManager::release_not_held_(OperationTrack& track, bool occluded,
                                       ReleaseReason reason,
                                       int evidence_detection_index,
                                       const BBox* evidence_box,
                                       const char* caller) {
    const OperationTrackState old_state = track.state;
    const ContactState old_contact_state = track.contact_state;
    const int old_hold = track.hold_evidence_count;
    const int old_not_hold = track.not_hold_evidence_count;
    const bool had_reappear_candidate = track.has_reappear_candidate_box;
    const int old_stable_near_original_count =
        track.stable_near_original_no_hand_count;
    const bool had_stable_near_original_box =
        track.has_stable_near_original_box;
    // 有手帧里的“仍在原位”只能说明当前这一刻没有移动。手可能在同一
    // 连续操作中随后再次拿起它，因此要保留一个可重新激活的锚点，并等
    // 无手直接帧完成真正静态结算。完整遮挡和无手原位证据才是终态。
    const bool keep_reopen_anchor = !occluded && hand_present_ &&
        (reason == ReleaseReason::ORIGINAL_DETECTION ||
         reason == ReleaseReason::CONTACT_RETURNED_ORIGINAL);
    const BBox reopen_anchor = track.has_last_seen_box
        ? track.last_seen_box : track.original_box;
    if (track.item_id > 0) {
        std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
        if (item != working_inventory_.end()) item->second.status =
            occluded ? ItemStatus::OCCLUDED : ItemStatus::VISIBLE;
        released_hand_candidate_ids_.insert(track.item_id);
    }
    track.state = OperationTrackState::NORMAL;
    track.contact_state = ContactState::NONE;
    track.release_reason = reason;
    track.resolution = reason == ReleaseReason::FULLY_OCCLUDED
        ? ExistingItemResolution::OCCLUDED_CONFIRMED
        : ExistingItemResolution::STATIC_CONFIRMED;
    track.needs_no_hand_settlement = keep_reopen_anchor;
    track.stable_near_original_no_hand_count = 0;
    track.has_stable_near_original_box = false;
    track.stable_near_original_box = BBox();
    track.shelter_or_hold = false;
    track.hold_and_move = false;
    track.hold_evidence_count = 0;
    track.not_hold_evidence_count = 0;
    track.has_reappear_candidate_box = false;
    track.reappear_candidate_match_count = 0;
    track.drop_evidence_count = 0;
    track.reappearance_pending = false;
    track.reappear_candidate_started_touching_hand = false;
    track.b_claim_ambiguous = false;
    track.contact_path_ambiguous = false;
    track.no_hand_candidate_ambiguous = false;
    track.claim_grace_remaining = 0;
    track.has_tentative_b_box = false;
    track.tentative_b_match_count = 0;
    track.tentative_b_started_touching_hand = false;
    track.has_first_hand_block_box = false;
    track.has_last_hand_block_box = false;
    track.move_values.clear();
    track.track.clear();
    track.observed_move_values.clear();
    track.observed_track.clear();
    track.hand_move_values.clear();
    track.contact_started_touching_hand = false;
    track.no_hand_missing_count = 0;
    if (keep_reopen_anchor) {
        track.last_seen_box = reopen_anchor;
        track.has_last_seen_box = reopen_anchor.area() > 0.0f;
        track.hand_estimate_anchor_box = reopen_anchor;
        track.has_hand_estimate_anchor_box = reopen_anchor.area() > 0.0f;
        if (track.has_hand_estimate_anchor_box) track.track.push_back(reopen_anchor);
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
    } else {
        track.has_hand_estimate_anchor_box = false;
        track.hand_track_start_index = -1;
    }
    if (evidence_box) {
        trace_("RELEASE",
               "item=%d old_state=%s old_contact=%s -> NORMAL reason=%s "
               "original_evidence=%d evidence_index=%d evidence_box=(%.1f,%.1f,%.1f,%.1f) "
               "caller=%s hold=%d not_hold=%d candidate_cleared=%d "
               "stable-near-original-before=%d stable-near-original-box=%d provisional=%d",
               track.item_id, operation_track_state_name(old_state),
               contact_state_name(old_contact_state), release_reason_name(reason),
               reason == ReleaseReason::ORIGINAL_DETECTION ||
                       reason == ReleaseReason::CONTACT_RETURNED_ORIGINAL ? 1 : 0,
               evidence_detection_index, evidence_box->x1, evidence_box->y1,
               evidence_box->x2, evidence_box->y2, caller ? caller : "NONE",
               old_hold, old_not_hold,
               had_reappear_candidate ? 1 : 0,
               old_stable_near_original_count,
               had_stable_near_original_box ? 1 : 0,
               keep_reopen_anchor ? 1 : 0);
    } else {
        trace_("RELEASE",
               "item=%d old_state=%s old_contact=%s -> NORMAL reason=%s "
               "original_evidence=0 evidence_index=-1 evidence_box=NONE hold=%d "
               "not_hold=%d caller=%s candidate_cleared=%d "
               "stable-near-original-before=%d stable-near-original-box=%d provisional=%d",
               track.item_id, operation_track_state_name(old_state),
               contact_state_name(old_contact_state), release_reason_name(reason),
               old_hold, old_not_hold, caller ? caller : "NONE",
               had_reappear_candidate ? 1 : 0,
               old_stable_near_original_count,
               had_stable_near_original_box ? 1 : 0,
               keep_reopen_anchor ? 1 : 0);
    }
    trace_track_("STATE", track, keep_reopen_anchor
                 ? "provisional-static-confirmed-awaiting-no-hand"
                 : "released-with-confirmed-resolution");
}

void SessionManager::reset_stable_near_original_no_hand_evidence_(
        OperationTrack* track, const char* reason) {
    if (!track || track->is_suspect_new ||
        (track->stable_near_original_no_hand_count == 0 &&
         !track->has_stable_near_original_box)) {
        return;
    }
    trace_("STATIC-SETTLE",
           "item=%d phase=%s detection=-1 stable-count=%d/%d "
           "action=reset-stable-near-original reason=%s",
           track->item_id, trace_hand_phase_ ? "HAND" : "NO_HAND",
           track->stable_near_original_no_hand_count,
           FLOW3_NO_HAND_D_CONFIRM_FRAMES,
           reason ? reason : "NONE");
    track->stable_near_original_no_hand_count = 0;
    track->has_stable_near_original_box = false;
    track->stable_near_original_box = BBox();
}

bool SessionManager::try_release_stable_near_original_no_hand_(
        OperationTrack* track, int detection_index, const Detection& detection,
        const std::map<int, int>& independent_static_owner_by_detection,
        const char* source) {
    if (!track) return false;

    OperationTrack& c = *track;
    const std::map<int, InventoryItem>::iterator working =
        working_inventory_.find(c.item_id);
    const std::map<int, int>::const_iterator static_owner =
        independent_static_owner_by_detection.find(detection_index);
    const bool operation_start_old_c =
        !c.is_suspect_new && c.item_id > 0 &&
        operation_start_inventory_.count(c.item_id) > 0;
    const bool provisional_static_after_hand = operation_start_old_c &&
        c.state == OperationTrackState::NORMAL &&
        c.contact_state == ContactState::NONE &&
        c.resolution == ExistingItemResolution::STATIC_CONFIRMED &&
        c.needs_no_hand_settlement && c.has_hand_estimate_anchor_box &&
        c.hand_estimate_anchor_box.area() > 0.0f;
    const bool direct_unique_owner = detection_index >= 0 &&
        static_owner != independent_static_owner_by_detection.end() &&
        static_owner->second == c.item_id;
    const bool scale_aware_near_original =
        strict_match_box(c.cls_id, c.original_box,
                         detection.cls_id, detection.box) &&
        partial_match_box(c.cls_id, c.original_box,
                          detection.cls_id, detection.box,
                          FLOW3_TRACK_PARTIAL_IOM);
    const float center_delta = center_distance(c.original_box, detection.box);
    const float normalized_center_delta =
        normalized_final_motion_distance(c.original_box, detection.box);
    const bool formal_move = boxes_differ_as_move(c.original_box, detection.box);
    const bool low_motion_static = normalized_center_delta <=
        FLOW3_STATIC_CENTER_SHIFT_NORM;
    // 有 hand_estimate_anchor_box 的恢复候选无论落在 CONTACT 原位门槛内还是
    // 12px~正式阈值灰区，都必须走同一条连续无手结算；只有正式归一化移动
    // 才能阻止静态释放。low_motion_static 表示“更像 YOLO 抖动”，不是可以
    // 把候选提前丢掉的理由。
    const bool static_gray_zone = !formal_move;
    const char* motion_zone = formal_move
        ? "FORMAL_MOVE"
        : (low_motion_static ? "LOW_MOTION_STATIC" : "STATIC_GRAY_ZONE");
    const bool hand_or_contact_move_evidence =
        c.hold_and_move || has_meaningful_hand_move(c) ||
        c.contact_state != ContactState::NONE;
    const bool reappearance_claim_or_path_ambiguity =
        c.reappearance_pending || c.b_claim_ambiguous ||
        c.contact_path_ambiguous || c.no_hand_candidate_ambiguous ||
        c.claim_grace_remaining > 0;
    const bool has_prior_stable_box = c.has_stable_near_original_box;
    const bool prior_evidence_consistent =
        (c.stable_near_original_no_hand_count == 0) == !has_prior_stable_box;
    const bool previous_stable_match = !has_prior_stable_box ||
        (strict_match_box(c.cls_id, c.stable_near_original_box,
                          detection.cls_id, detection.box) &&
         partial_match_box(c.cls_id, c.stable_near_original_box,
                           detection.cls_id, detection.box,
                           FLOW3_TRACK_PARTIAL_IOM));
    const bool working_item_is_visible =
        working != working_inventory_.end() &&
        working->second.status == ItemStatus::VISIBLE &&
        !pending_out_ids_.count(c.item_id);

    trace_("STATIC-SETTLE",
           "item=%d phase=NO_HAND detection=%d source=%s "
           "motion-reference-diagonal-px=%.3f motion-zone=%s",
           c.item_id, detection_index, source ? source : "NONE",
           final_motion_reference_diagonal(c.original_box), motion_zone);

    const char* reason = "eligible";
    if (!provisional_static_after_hand) {
        reason = "not-operation-start-provisional-static-c";
    } else if (!direct_unique_owner) {
        reason = "no-direct-unique-static-owner";
    } else if (!scale_aware_near_original) {
        reason = "not-scale-aware-near-original";
    } else if (formal_move) {
        reason = "formal-move-threshold-reached";
    } else if (hand_or_contact_move_evidence) {
        reason = "hand-or-contact-move-evidence";
    } else if (reappearance_claim_or_path_ambiguity) {
        reason = "reappearance-or-claim-ambiguity";
    } else if (!working_item_is_visible) {
        reason = "missing-visible-working-inventory-item";
    } else if (!prior_evidence_consistent) {
        reason = "inconsistent-previous-static-evidence";
    } else if (!previous_stable_match) {
        reason = "not-continuous-with-previous-static-frame";
    }
    const bool eligible = provisional_static_after_hand && direct_unique_owner &&
        scale_aware_near_original && static_gray_zone &&
        !hand_or_contact_move_evidence &&
        !reappearance_claim_or_path_ambiguity && working_item_is_visible &&
        prior_evidence_consistent && previous_stable_match;

    if (!eligible) {
        trace_("STATIC-SETTLE",
               "item=%d phase=NO_HAND detection=%d source=%s "
               "center-distance-px=%.3f normalized-center-distance=%.4f iom=%.4f "
               "width-ratio-diff=%.4f height-ratio-diff=%.4f formal-move=%d "
               "has-real-move-evidence=%d low-motion-static=%d unique-static-owner=%d "
               "previous-stable-present=%d previous-stable-match=%d "
               "stable-count=%d/%d action=reset-stable-near-original reason=%s",
               c.item_id, detection_index, source ? source : "NONE",
               center_delta, normalized_center_delta, iom(c.original_box, detection.box),
               ratio_difference(c.original_box.w(), detection.box.w()),
               ratio_difference(c.original_box.h(), detection.box.h()),
               formal_move ? 1 : 0, hand_or_contact_move_evidence ? 1 : 0,
               low_motion_static ? 1 : 0,
               direct_unique_owner ? 1 : 0, has_prior_stable_box ? 1 : 0,
               previous_stable_match ? 1 : 0,
               c.stable_near_original_no_hand_count,
               FLOW3_NO_HAND_D_CONFIRM_FRAMES, reason);
        reset_stable_near_original_no_hand_evidence_(track, reason);
        return false;
    }

    ++c.stable_near_original_no_hand_count;
    c.stable_near_original_box = detection.box;
    c.has_stable_near_original_box = true;
    if (c.stable_near_original_no_hand_count <
        FLOW3_NO_HAND_D_CONFIRM_FRAMES) {
        trace_("STATIC-SETTLE",
               "item=%d phase=NO_HAND detection=%d source=%s "
               "center-distance-px=%.3f normalized-center-distance=%.4f iom=%.4f "
               "width-ratio-diff=%.4f height-ratio-diff=%.4f formal-move=0 "
               "has-real-move-evidence=0 low-motion-static=%d unique-static-owner=1 "
               "previous-stable-present=%d previous-stable-match=%d "
               "stable-count=%d/%d action=wait-stable-near-original reason=eligible",
               c.item_id, detection_index, source ? source : "NONE",
               center_delta, normalized_center_delta, iom(c.original_box, detection.box),
               ratio_difference(c.original_box.w(), detection.box.w()),
               ratio_difference(c.original_box.h(), detection.box.h()),
               low_motion_static ? 1 : 0,
               has_prior_stable_box ? 1 : 0, previous_stable_match ? 1 : 0,
               c.stable_near_original_no_hand_count,
               FLOW3_NO_HAND_D_CONFIRM_FRAMES);
        return false;
    }

    const int stable_count_before_release = c.stable_near_original_no_hand_count;
    const bool needs_settlement_before_release = c.needs_no_hand_settlement;
    // 不把 unresolved C-D alias 当作这里的阻塞条件：正是 C 用自己的直接
    // 证据完成 release 后，r15 才能安全判断 runtime-only D 是否应该回收。
    update_seen(working->second, detection, trace_frame_id_);
    release_not_held_(c, false, ReleaseReason::STABLE_NEAR_ORIGINAL_NO_HAND,
                      detection_index, &detection.box,
                      "no-hand-stable-near-original");
    trace_("STATIC-SETTLE",
           "item=%d phase=NO_HAND detection=%d source=%s "
           "center-distance-px=%.3f normalized-center-distance=%.4f iom=%.4f "
           "width-ratio-diff=%.4f height-ratio-diff=%.4f formal-move=0 "
           "has-real-move-evidence=0 low-motion-static=%d unique-static-owner=1 "
           "previous-stable-present=%d previous-stable-match=%d "
           "stable-count=%d/%d action=release-stable-near-original "
           "release-reason=STABLE_NEAR_ORIGINAL_NO_HAND needs-settle-before=%d "
           "needs-settle-after=%d inventory-update=existing-item-only event=none",
           c.item_id, detection_index, source ? source : "NONE",
           center_delta, normalized_center_delta, iom(c.original_box, detection.box),
           ratio_difference(c.original_box.w(), detection.box.w()),
           ratio_difference(c.original_box.h(), detection.box.h()),
           low_motion_static ? 1 : 0,
           has_prior_stable_box ? 1 : 0, previous_stable_match ? 1 : 0,
           stable_count_before_release, FLOW3_NO_HAND_D_CONFIRM_FRAMES,
           needs_settlement_before_release ? 1 : 0,
           c.needs_no_hand_settlement ? 1 : 0);
    return true;
}

void SessionManager::advance_claim_grace_(
        const std::set<int>& new_existing_track_ids) {
    // “两张后续有效帧”按真正执行逐帧状态机的帧计数。手框微小且整帧被
    // 跳过时不会来到这里；无手收尾帧也算有效帧，避免手离开后保护期永远
    // 不结束。新建于本帧的 C 不递减，故 t0/t1/t2 均受保护，t3 才成熟。
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || track.item_id <= 0 ||
            !is_active_runtime_track(track) ||
            track.claim_grace_remaining <= 0 ||
            new_existing_track_ids.count(track.item_id)) {
            continue;
        }
        --track.claim_grace_remaining;
        if (track.claim_grace_remaining == 0) {
            // 保存下来的本地 B 证据现在可作为正式候选的起点，但真正的
            // 全局认领仍必须由下一张成熟帧的唯一仲裁完成。
            seed_reappear_from_tentative_b(&track);
            printf("[3.0] item#%d 的 B 认领保护期结束，后续可参与唯一仲裁\n",
                   track.item_id);
            trace_track_("STATE", track, "claim-grace-ended");
        }
    }
}

void SessionManager::update_existing_hand_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        const MoveValue& delta, std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        const HandDirectOldOwnerPlan& direct_old_owner_plan) {
    std::vector<int> promote_keys;
    // 对 HAND_* + hold_and_move=false，静止手帧只能更新局部观测，不能把
    // “还在原位”或“跟手移动”当作新的有效证据。
    const bool hand_moved = move_length(delta) >= TRACK_HAND_MOVE_EPS;
    bool has_hand_visible_suspect = false;
    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        if (it->second.is_suspect_new &&
            it->second.suspect_source == SuspectSource::HAND_VISIBLE_D &&
            (it->second.state == OperationTrackState::HAND_PARTIAL_BLOCKED ||
             it->second.state == OperationTrackState::HAND_FULL_BLOCKED)) {
            has_hand_visible_suspect = true;
            break;
        }
    }
    std::map<int, int> static_owner_by_detection;
    if (has_hand_visible_suspect) {
        // reserve_visible_known_detections_() 在本函数之后才实际写入静态 C 的
        // claim。先在副本上得到完全相同的预约计划，避免已有 HAND_VISIBLE_D
        // 按 map 顺序抢走本应唯一属于静态旧 C 的严格框。
        static_owner_by_detection =
            build_mutually_unique_hand_static_owner_by_detection_(
                hand_box, detections, *claimed_detection_indices, *known_item_owner);
    }
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.state != OperationTrackState::HAND_PARTIAL_BLOCKED &&
            track.state != OperationTrackState::HAND_FULL_BLOCKED) {
            continue;
        }

        const BBox expected = estimated_box(track);
        // 已有库存物品后续仍按 e2/e1 更新当前 HAND 状态；这里使用完整
        // 估计框，而不是 last_hand_block_box 这类局部 YOLO 框。若手已
        // 移开到 r < e2，则保留候选状态等待放下/收尾，不凭这一帧直接释放。
        if (!track.is_suspect_new) {
            if (hand_fully_covers(hand_box, expected)) {
                track.state = OperationTrackState::HAND_FULL_BLOCKED;
            } else if (hand_affects(hand_box, expected)) {
                track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
            }
        }
        const int direct_owner_index = track.is_suspect_new ? -1 :
            direct_old_owner_detection_for_item(direct_old_owner_plan, track.item_id);
        const HandDirectOldOwnerStrength direct_owner_strength =
            direct_owner_index >= 0
                ? direct_old_owner_strength_for_detection(
                      direct_old_owner_plan, direct_owner_index)
                : HandDirectOldOwnerStrength::LOCAL_WEAK;
        const bool direct_owner_is_contact_original = direct_owner_index >= 0 &&
            contact_detection_is_at_original(track, detections[direct_owner_index]);
        if (direct_owner_index >= 0 &&
            direct_owner_strength == HandDirectOldOwnerStrength::STRICT &&
            (!track.has_hand_estimate_anchor_box ||
             direct_owner_is_contact_original)) {
            // 自己唯一的严格原位框必须先于任何 estimated/track/reappear 路径。
            // 它只累计既有的“没有拿起”反向证据，不能被同一帧的宽松路径
            // 重新解释成 MOVED，也不能被别的 C/D 借走。
            const Detection& old_d = detections[direct_owner_index];
            claimed_detection_indices->insert(direct_owner_index);
            (*known_item_owner)[track.item_id] = direct_owner_index;
            track.b_claim_ambiguous = false;
            track.has_tentative_b_box = false;
            track.tentative_b_match_count = 0;
            track.tentative_b_started_touching_hand = false;
            track.has_reappear_candidate_box = false;
            track.reappear_candidate_match_count = 0;
            track.reappear_candidate_started_touching_hand = false;
            track.reappearance_pending = false;
            if (hand_moved) {
                ++track.not_hold_evidence_count;
                track.hold_evidence_count = 0;
            }
            track.last_seen_box = old_d.box;
            track.has_last_seen_box = true;
            track.last_hand_block_box = old_d.box;
            track.has_last_hand_block_box = true;
            if (hand_moved && track.not_hold_evidence_count >=
                FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                std::map<int, InventoryItem>::iterator item =
                    working_inventory_.find(track.item_id);
                if (item != working_inventory_.end()) {
                    update_seen(item->second, old_d, trace_frame_id_);
                    item->second.base_box = old_d.box;
                }
                release_not_held_(track, false, ReleaseReason::ORIGINAL_DETECTION,
                                  direct_owner_index, &old_d.box,
                                  "hand-direct-old-owner-strict-original");
            }
            continue;
        }
        const std::set<int> candidate_claimed = claimed_with_other_direct_old_owners(
            *claimed_detection_indices, direct_old_owner_plan,
            track.is_suspect_new ? -1 : track.item_id);
        int observed_index = unique_detection_for_box(
            detections, candidate_claimed, track.cls_id, expected,
            true, true);
        if (observed_index < 0) {
            observed_index = best_detection_for_box(
                detections, candidate_claimed, track.cls_id, expected,
                true, true);
        }
        if (observed_index < 0 && track.has_last_hand_block_box) {
            const BBox local_expected = move_box(track.last_hand_block_box, delta);
            observed_index = unique_detection_for_box(
                detections, candidate_claimed, track.cls_id, local_expected,
                true, true);
            if (observed_index < 0) {
                observed_index = best_detection_for_box(
                    detections, candidate_claimed, track.cls_id, local_expected,
                    true, true);
            }
            if (observed_index < 0) {
                observed_index = unique_hand_affected_detection_for_box(
                    detections, candidate_claimed, track.cls_id,
                    local_expected, hand_box);
            }
        }
        if (observed_index < 0 && track.has_last_seen_box) {
            observed_index = unique_detection_for_box(
                detections, candidate_claimed, track.cls_id,
                track.last_seen_box, true, true);
            if (observed_index < 0) {
                observed_index = best_detection_for_box(
                    detections, candidate_claimed, track.cls_id,
                    track.last_seen_box, true, true);
            }
        }
        bool observed_matches_reappear_candidate = false;
        if (observed_index < 0 && !track.is_suspect_new &&
            track.has_reappear_candidate_box) {
            observed_index = unique_detection_for_box(
                detections, candidate_claimed, track.cls_id,
                track.reappear_candidate_box, true, true);
            if (observed_index < 0) {
                observed_index = best_detection_for_box(
                    detections, candidate_claimed, track.cls_id,
                    track.reappear_candidate_box, true, true);
            }
            observed_matches_reappear_candidate = observed_index >= 0;
        }
        int old_position_index = unique_detection_at_old_position(
            detections, candidate_claimed, track);
        // CONTACT_* 转入 HAND_* 后，B 仍可能与最初 A.box 有较大 IoM。
        // 普通 HAND_* 的“局部重叠即旧位置”规则在这里会把已经推开的 B
        // 误当成 A 原地未动。因此该分支只接受真正仍贴近 original_box 的
        // 真实框；其他框交给 observed_track / 无手逐帧收尾。
        if (track.has_hand_estimate_anchor_box && old_position_index >= 0 &&
            !contact_detection_is_at_original(
                track, detections[old_position_index])) {
            old_position_index = -1;
        }
        const bool old_clean = old_position_is_clean(detections, track, working_inventory_);

        // HAND_* 的普通候选也必须经过同类成熟 C 的一对一仲裁。此前只在
        // scan 阶段检查这一点，导致 update 按 map 顺序先抢走 B，第二个 C
        // 根本没有机会表达“同样合理”的歧义。
        if (!track.is_suspect_new && !is_claim_protected(track) &&
            observed_index >= 0 &&
            !contact_detection_is_at_original(track, detections[observed_index])) {
            const int mature_owner = unique_c_reappear_owner_for_detection(
                detections[observed_index], track_buffer_, *known_item_owner);
            if (mature_owner == -2) {
                mark_mature_hand_b_ambiguity(detections[observed_index],
                                              &track_buffer_, *known_item_owner);
                observed_index = -1;
                observed_matches_reappear_candidate = false;
                track.reappearance_pending = true;
            } else if (mature_owner >= 0 && mature_owner != track.item_id) {
                observed_index = -1;
                observed_matches_reappear_candidate = false;
                track.reappearance_pending = true;
            } else {
                // 当前 B 没有被另一条成熟 HAND_* 轨迹解释；若此前只是
                // 暂时歧义，本帧的唯一本地匹配可以解除它。
                track.b_claim_ambiguous = false;
            }
        }

        if (track.is_suspect_new) {
            if (observed_index >= 0) {
                const Detection& d = detections[observed_index];
                const std::map<int, int>::const_iterator static_owner =
                    static_owner_by_detection.find(observed_index);
                if (track.suspect_source == SuspectSource::HAND_VISIBLE_D &&
                    static_owner != static_owner_by_detection.end() &&
                    operation_start_inventory_.count(static_owner->second)) {
                    trace_("D-GUARD",
                           "suspect=%d cls=%d action=yield-to-mutually-unique-static-old-c "
                           "detection=%d old-item=%d self-match=%d promote=0 claimed-write=0",
                           track.suspect_id, track.cls_id, observed_index,
                           static_owner->second, track.self_match_count);
                    continue;
                }
                claimed_detection_indices->insert(observed_index);
                if (track.item_id > 0) {
                    (*known_item_owner)[track.item_id] = observed_index;
                }
                ++track.self_match_count;
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                if (hand_touches_detection(hand_box, d.box)) {
                    track.last_hand_block_box = d.box;
                    track.has_last_hand_block_box = true;
                }
                if (!track.promoted_to_working_inventory &&
                    track.self_match_count >= NEW_ITEM_CONFIRM_FRAMES) {
                    promote_keys.push_back(it->first);
                }
                if (track.promoted_to_working_inventory &&
                    !hand_touches_detection(hand_box, d.box)) {
                    track.placed_box = d.box;
                    track.has_placed_box = true;
                    track.drop_confirmed = true;
                    track.state = OperationTrackState::PLACED;
                    std::map<int, InventoryItem>::iterator item =
                        working_inventory_.find(track.item_id);
                    if (item != working_inventory_.end()) {
                        update_seen(item->second, d, trace_frame_id_);
                        item->second.base_box = d.box;
                    }
                }
            }
            continue;
        }

        if (is_claim_protected(track)) {
            // 新建 HAND_* C 在保护期内仍按自己的预计框/局部框寻找 B；但
            // 除“明确仍在旧位置”的检测外，不能抢占 B 的全局归属。
            if (old_position_index >= 0) {
                const Detection& old_d = detections[old_position_index];
                claimed_detection_indices->insert(old_position_index);
                (*known_item_owner)[track.item_id] = old_position_index;
                track.b_claim_ambiguous = false;
                track.has_tentative_b_box = false;
                track.tentative_b_match_count = 0;
                track.tentative_b_started_touching_hand = false;
                track.has_reappear_candidate_box = false;
                track.reappear_candidate_match_count = 0;
                track.reappear_candidate_started_touching_hand = false;
                if (hand_moved) {
                    ++track.not_hold_evidence_count;
                    track.hold_evidence_count = 0;
                }
                track.last_hand_block_box = old_d.box;
                track.has_last_hand_block_box = true;
                if (hand_moved && track.not_hold_evidence_count >=
                    FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                    std::map<int, InventoryItem>::iterator item =
                        working_inventory_.find(track.item_id);
                    if (item != working_inventory_.end()) {
                        update_seen(item->second, old_d, trace_frame_id_);
                    }
                    release_not_held_(track, false,
                                      ReleaseReason::ORIGINAL_DETECTION,
                                      old_position_index,
                                      &old_d.box,
                                      "hand-track-protected-original");
                }
            } else if (observed_index >= 0) {
                const bool touching = hand_touches_detection(
                    hand_box, detections[observed_index].box);
                record_tentative_b(&track, detections[observed_index],
                                   touching);
                if (hand_moved && old_clean &&
                    (touching || track.tentative_b_started_touching_hand)) {
                    ++track.hold_evidence_count;
                    track.not_hold_evidence_count = 0;
                }
                track.reappearance_pending = true;
            } else {
                // 没有本地候选也只是未知，不能在保护期内累积 HOLD/OUT。
                track.reappearance_pending = true;
            }
            continue;
        }

        // C 曾经不可见后重新出现的 B，或已经有过首次 B 的 C，必须先走
        // candidate 自匹配。即使当前 B 刚好落回预计框，也不能因为一帧框
        // 跳动而直接把它认成 C / D。
        const bool should_observe_reappear_candidate = observed_index >= 0 &&
            (track.reappearance_pending || track.has_reappear_candidate_box ||
             observed_matches_reappear_candidate);

        if (track.hold_and_move) {
            if (observed_index >= 0) {
                const Detection& d = detections[observed_index];
                claimed_detection_indices->insert(observed_index);
                if (track.item_id > 0) {
                    (*known_item_owner)[track.item_id] = observed_index;
                }
                BBox previous = track.has_last_seen_box ? track.last_seen_box :
                    (track.has_last_hand_block_box ? track.last_hand_block_box : expected);
                bool candidate_ready = true;
                if (should_observe_reappear_candidate) {
                    if (track.has_reappear_candidate_box) {
                        previous = track.reappear_candidate_box;
                    }
                    candidate_ready = update_reappear_candidate(
                        &track, d, hand_touches_detection(hand_box, d.box));
                }
                if (!candidate_ready) {
                    // 首次 B 仅防止它被登记成 D；不能作为放下或整理的单帧证据。
                    continue;
                }
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                if (hand_touches_detection(hand_box, d.box)) {
                    track.last_hand_block_box = d.box;
                    track.has_last_hand_block_box = true;
                }
                const float hand_distance = move_length(delta);
                const float object_distance = center_distance(previous, d.box);
                const bool falls_behind = hand_distance >= TRACK_HAND_MOVE_EPS &&
                    object_distance <= hand_distance * FLOW3_DROP_FALL_BEHIND_RATIO &&
                    iom(previous, d.box) >= FLOW3_TRACK_PARTIAL_IOM;
                const bool detaches_from_hand = !hand_touches_detection(hand_box, d.box);
                const bool becomes_more_complete = becomes_more_like_complete_box(
                    track, previous, d.box);
                if (falls_behind && (detaches_from_hand || becomes_more_complete)) {
                    ++track.drop_evidence_count;
                } else {
                    track.drop_evidence_count = 0;
                }
                if (track.drop_evidence_count >= FLOW3_DROP_EVIDENCE_REQUIRED) {
                    confirm_rearrange_(track, d.box, d.score, trace_frame_id_);
                }
            } else {
                // B 暂时再次被手挡住时不能凭“看不见”判断放下；但连续放下
                // 证据被打断，下一次重新出现仍要从候选开始确认。
                track.reappearance_pending = true;
                track.drop_evidence_count = 0;
            }
            continue;
        }

        // False 代表“待确认”。先看旧位置，再看估计轨迹；模糊帧不改计数。
        if (old_position_index >= 0) {
            const Detection& old_d = detections[old_position_index];
            claimed_detection_indices->insert(old_position_index);
            if (track.item_id > 0) {
                (*known_item_owner)[track.item_id] = old_position_index;
            }
            track.b_claim_ambiguous = false;
            if (hand_moved) {
                ++track.not_hold_evidence_count;
                track.hold_evidence_count = 0;
            }
            track.last_hand_block_box = old_d.box;
            track.has_last_hand_block_box = true;
            if (hand_moved &&
                track.not_hold_evidence_count >= FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                std::map<int, InventoryItem>::iterator item = working_inventory_.find(track.item_id);
                if (item != working_inventory_.end()) {
                    update_seen(item->second, old_d, trace_frame_id_);
                }
                release_not_held_(track, false,
                                  ReleaseReason::ORIGINAL_DETECTION,
                                  old_position_index,
                                  &old_d.box,
                                  "hand-track-original");
            }
            continue;
        }

        if (should_observe_reappear_candidate) {
            const Detection& d = detections[observed_index];
            claimed_detection_indices->insert(observed_index);
            if (track.item_id > 0) {
                (*known_item_owner)[track.item_id] = observed_index;
            }
            const bool candidate_ready = update_reappear_candidate(
                &track, d, hand_touches_detection(hand_box, d.box));
            if (candidate_ready) {
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                if (hand_touches_detection(hand_box, d.box)) {
                    track.last_hand_block_box = d.box;
                    track.has_last_hand_block_box = true;
                }
            }
            // 有手首帧 B 必须贴手才可用这条“候选 B 连续自匹配”链路建立
            // hold；否则保留到无手逐帧收尾按更严格的路径规则处理。
            if (hand_moved && candidate_ready &&
                track.reappear_candidate_started_touching_hand &&
                old_clean && !detection_strictly_matches_other_item(
                    d, track.item_id, working_inventory_)) {
                track.hold_evidence_count = std::max(
                    track.hold_evidence_count + 1,
                    track.reappear_candidate_match_count);
                track.not_hold_evidence_count = 0;
            }
            if (hand_moved &&
                track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
                track.hold_and_move = true;
                track.hold_evidence_count = 0;
                track.not_hold_evidence_count = 0;
            }
            continue;
        }

        if (observed_index >= 0 && old_clean &&
            !detection_strictly_matches_other_item(detections[observed_index], track.item_id,
                                                   working_inventory_)) {
            const Detection& d = detections[observed_index];
            claimed_detection_indices->insert(observed_index);
            if (track.item_id > 0) {
                (*known_item_owner)[track.item_id] = observed_index;
            }
            if (hand_moved) {
                ++track.hold_evidence_count;
                track.not_hold_evidence_count = 0;
            }
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;
            if (hand_touches_detection(hand_box, d.box)) {
                track.last_hand_block_box = d.box;
                track.has_last_hand_block_box = true;
            }
            if (hand_moved &&
                track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
                track.hold_and_move = true;
                track.hold_evidence_count = 0;
                track.not_hold_evidence_count = 0;
            }
        } else if (observed_index >= 0) {
            // 该框不能证明移动，但可作为下一帧局部连续性的参考。
            track.last_hand_block_box = detections[observed_index].box;
            track.has_last_hand_block_box = true;
        } else {
            track.reappearance_pending = true;
            // HAND_FULL + hold_and_move=False 时，没有可直接使用的 B。只有
            // 原位置的 C、已确认遮挡物、手仍覆盖原位置、或干净的旧位置
            // 离开这四种情形能改变证据；其余一律保持模糊。
            if (track.state == OperationTrackState::HAND_FULL_BLOCKED) {
                if (confirmed_blocker_covers_old_c(
                        track, detections, *known_item_owner, track_buffer_)) {
                    if (hand_moved) {
                        ++track.not_hold_evidence_count;
                        track.hold_evidence_count = 0;
                    }
                    if (hand_moved && track.not_hold_evidence_count >=
                        FLOW3_NOT_HOLD_EVIDENCE_REQUIRED) {
                        // 已确认的 blocker 只能说明 C 可能仍在旧位置被遮住，
                        // 不能替代“C 原位置有独立检测”的静态结论。保留 C
                        // 的轨迹和无手结算资格，避免之后把同一实体登记为 D。
                        trace_track_("STATE", track,
                                     "confirmed-blocker-keeps-unresolved");
                    }
                } else if (hand_affects(hand_box, track.original_box)) {
                    // 手仍盖在 C 原位置：既不能说明拿起，也不能说明没拿起。
                } else if (!any_detection_at_old_position(detections, track)) {
                    if (hand_moved) {
                        ++track.hold_evidence_count;
                        track.not_hold_evidence_count = 0;
                    }
                    if (hand_moved &&
                        track.hold_evidence_count >= FLOW3_HOLD_EVIDENCE_REQUIRED) {
                        track.hold_and_move = true;
                        track.hold_evidence_count = 0;
                        track.not_hold_evidence_count = 0;
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < promote_keys.size(); ++i) {
        std::map<int, OperationTrack>::iterator track = track_buffer_.find(promote_keys[i]);
        if (track == track_buffer_.end() || !track->second.has_last_seen_box) continue;
        Detection d;
        d.box = track->second.last_seen_box;
        d.cls_id = track->second.cls_id;
        d.score = 0.0f;
        promote_suspect_(promote_keys[i], d, trace_frame_id_);
    }
}

void SessionManager::reopen_released_static_tracks_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        const MoveValue& delta) {
    // 有手阶段已确认“目前仍在原位”的 C 不是一次操作的最终结论。若同一
    // 只手随后发生有效位移，并且本帧没有 C 自己的唯一原位检测，就把它
    // 恢复为 HAND_*，从最后一个可靠原位框重新开始记录手部路径。
    if (move_length(delta) < TRACK_HAND_MOVE_EPS) return;

    const std::set<int> no_claimed_detections;
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new || track.item_id <= 0 ||
            track.state != OperationTrackState::NORMAL ||
            track.contact_state != ContactState::NONE ||
            !track.needs_no_hand_settlement ||
            track.resolution != ExistingItemResolution::STATIC_CONFIRMED) {
            continue;
        }
        if (unique_detection_at_old_position(detections, no_claimed_detections,
                                             track) >= 0) {
            trace_track_("STATE", track,
                         "provisional-static-kept-by-current-original-detection");
            continue;
        }

        const BBox anchor = track.has_last_seen_box ? track.last_seen_box : track.original_box;
        if (anchor.area() <= 0.0f) {
            trace_track_("STATE", track,
                         "cannot-reopen-static-track-without-valid-anchor");
            continue;
        }

        released_hand_candidate_ids_.erase(track.item_id);
        track.state = hand_fully_covers(hand_box, move_box(anchor, delta))
            ? OperationTrackState::HAND_FULL_BLOCKED
            : OperationTrackState::HAND_PARTIAL_BLOCKED;
        track.contact_state = ContactState::NONE;
        track.resolution = ExistingItemResolution::NONE;
        track.release_reason = ReleaseReason::NONE;
        track.needs_no_hand_settlement = true;
        track.shelter_or_hold = true;
        track.hold_and_move = false;
        track.hold_evidence_count = 0;
        track.not_hold_evidence_count = 0;
        track.drop_confirmed = false;
        track.drop_evidence_count = 0;
        track.claim_grace_remaining = 0;
        track.has_tentative_b_box = false;
        track.tentative_b_match_count = 0;
        track.tentative_b_started_touching_hand = false;
        track.has_reappear_candidate_box = false;
        track.reappear_candidate_match_count = 0;
        track.reappear_candidate_started_touching_hand = false;
        track.reappearance_pending = true;
        track.contact_path_ambiguous = false;
        track.b_claim_ambiguous = false;
        track.no_hand_candidate_ambiguous = false;
        track.no_hand_missing_count = 0;
        track.last_seen_box = anchor;
        track.has_last_seen_box = true;
        track.first_hand_block_box = anchor;
        track.last_hand_block_box = anchor;
        track.has_first_hand_block_box = true;
        track.has_last_hand_block_box = true;
        track.hand_estimate_anchor_box = anchor;
        track.has_hand_estimate_anchor_box = true;
        track.move_values.clear();
        track.track.clear();
        track.observed_move_values.clear();
        track.observed_track.clear();
        track.hand_move_values.clear();
        track.track.push_back(anchor);
        track.move_values.push_back(delta);
        track.track.push_back(estimated_box(track));
        track.hand_track_start_index = std::max(
            0, static_cast<int>(hand_track_.size()) - 2);
        trace_("STATE",
               "item=%d reopen-after-static-release-lost-original anchor=(%.1f,%.1f,%.1f,%.1f) "
               "delta=(%.1f,%.1f) expected=(%.1f,%.1f,%.1f,%.1f)",
               track.item_id, anchor.x1, anchor.y1, anchor.x2, anchor.y2,
               delta.dx, delta.dy, estimated_box(track).x1, estimated_box(track).y1,
               estimated_box(track).x2, estimated_box(track).y2);
        trace_track_("STATE", track, "reopen-after-static-release-lost-original");
    }
}

void SessionManager::scan_or_update_suspects_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        const std::map<int, int>& known_item_owner,
        const HandDirectOldOwnerPlan& direct_old_owner_plan,
        bool /*first_hand_frame*/) {
    // scan 内也会把 B 暂认给 C。复制一份本帧归属，避免同一个尚未确认的 C
    // 在同一帧连续抢走两个同类框；外部在 scan 后不再需要这个临时映射。
    std::map<int, int> effective_known_item_owner = known_item_owner;
    for (size_t di = 0; di < detections.size(); ++di) {
        if (claimed_detection_indices->count(static_cast<int>(di))) continue;
        const int detection_index = static_cast<int>(di);
        std::map<int, int>::const_iterator direct_owner =
            direct_old_owner_plan.owner_by_detection.find(detection_index);
        if (direct_owner != direct_old_owner_plan.owner_by_detection.end() ||
            direct_old_owner_plan.ambiguous_detection_indices.count(detection_index)) {
            trace_("D-GUARD",
                   "candidate=%zu cls=%d action=yield-to-direct-old-owner-plan "
                   "owner=%d ambiguous=%d",
                   di, detections[di].cls_id,
                   direct_owner == direct_old_owner_plan.owner_by_detection.end()
                       ? -1 : direct_owner->second,
                   direct_old_owner_plan.ambiguous_detection_indices.count(detection_index)
                       ? 1 : 0);
            continue;
        }
        const Detection& d = detections[di];
        const bool hand_visible_d = hand_touches_detection(hand_box, d.box);
        // D 已经被放到 C 原位置时，手可能继续移开，因此 D 不一定还贴手。
        // 只要它是唯一一个覆盖“当前看不见的 C”原位置的未认领框，也必须
        // 预登记；否则 C 会在后续无手阶段被错误当成 OUT。
        const int replacement_owner = unique_c_replacement_owner_for_detection(
            d, track_buffer_, effective_known_item_owner);
        if (!hand_visible_d && replacement_owner < 0) {
            continue;
        }

        // 先排除普通静态库存、已放下的旧 C。正在 HAND_* 的旧 C 不在这里
        // 用宽松局部匹配直接吞框：同类贴手 B 应由下面的 candidate 规则
        // 唯一归属，不能因为帧间跳动直接落进 D。
        bool belongs_to_known_item = false;
        for (std::map<int, InventoryItem>::const_iterator it = working_inventory_.begin();
             it != working_inventory_.end(); ++it) {
            // 这个旧 item 已经在本帧认领了另一个检测框。当前 D 即使与
            // 它的旧框局部相似，也不能再次借用同一身份；否则相邻同类
            // 物品会永远进不了疑似 D 链路。
            if (effective_known_item_owner.count(it->first) &&
                effective_known_item_owner.find(it->first)->second !=
                    static_cast<int>(di)) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(it->first);
            if (runtime && (is_active_existing_hand_track(*runtime) ||
                            is_active_contact_track(*runtime))) continue;
            const BBox reference = it->second.base_box.area() > 0.0f
                ? it->second.base_box : it->second.box;
            if (strict_match(it->second, d) || partial_match(it->second, d) ||
                hand_partial_match_box(it->second.cls_id, reference, d.cls_id, d.box)) {
                belongs_to_known_item = true;
                break;
            }
        }
        if (!belongs_to_known_item) {
            for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
                 it != track_buffer_.end(); ++it) {
                const OperationTrack& track = it->second;
                if (!track.is_suspect_new || track.cls_id != d.cls_id ||
                    track.state == OperationTrackState::NORMAL) {
                    continue;
                }
                if (track.item_id > 0 && effective_known_item_owner.count(track.item_id) &&
                    effective_known_item_owner.find(track.item_id)->second !=
                        static_cast<int>(di)) {
                    continue;
                }
                const bool matches_estimate =
                    track_match_box(track.cls_id, estimated_box(track), d.cls_id, d.box);
                const bool matches_last_seen = track.has_last_seen_box &&
                    track_match_box(track.cls_id, track.last_seen_box, d.cls_id, d.box);
                const bool matches_last_hand_part = track.has_last_hand_block_box &&
                    (track_match_box(track.cls_id, track.last_hand_block_box, d.cls_id, d.box) ||
                     hand_partial_match_box(track.cls_id, track.last_hand_block_box,
                                            d.cls_id, d.box));
                if (matches_estimate || matches_last_seen || matches_last_hand_part) {
                    belongs_to_known_item = true;
                    break;
                }
            }
        }
        if (belongs_to_known_item) continue;

        // 细节5的关键优先级：一个手边同类 B 若只有一个无自身检测的活动 C
        // 可以解释，就先记录为 C 的 reappear_candidate，而不是新 D。若多个
        // C 都合理，宁可保持未决，也绝不因为 map 顺序生成一条 D 链路。
        if (hand_visible_d) {
            const int c_owner = unique_c_reappear_owner_for_detection(
                d, track_buffer_, effective_known_item_owner);
            if (c_owner == -2) {
                mark_mature_hand_b_ambiguity(d, &track_buffer_,
                                              effective_known_item_owner);
                printf("[3.0] 同类贴手 B cls=%d 可属于多个 HAND_* C，保持未决\n",
                       d.cls_id);
                continue;
            }
            if (c_owner >= 0) {
                std::map<int, OperationTrack>::iterator c = track_buffer_.find(c_owner);
                if (c != track_buffer_.end()) {
                    c->second.b_claim_ambiguous = false;
                    if (c->second.has_reappear_candidate_box &&
                        track_match_box(c->second.cls_id,
                                        c->second.reappear_candidate_box,
                                        d.cls_id, d.box)) {
                        const bool candidate_ready =
                            update_reappear_candidate(&c->second, d, true);
                        trace_("MATCH",
                               "item=%d detection=%zu source=hand-reappear update=1 count=%d ready=%d",
                               c->second.item_id, di,
                               c->second.reappear_candidate_match_count,
                               candidate_ready ? 1 : 0);
                    } else {
                        start_reappear_candidate(&c->second, d, true);
                        trace_("MATCH",
                               "item=%d detection=%zu source=hand-reappear start=1 count=%d",
                               c->second.item_id, di,
                               c->second.reappear_candidate_match_count);
                    }
                    claimed_detection_indices->insert(static_cast<int>(di));
                    effective_known_item_owner[c->second.item_id] =
                        static_cast<int>(di);
                    printf("[3.0] item#%d 将同类贴手 B 暂记为重新出现候选"
                           "（等待自匹配）\n", c->second.item_id);
                    continue;
                }
            }
        }

        // 保护期内的旧 C 仍可做本地路径匹配，但不能把 B 变成排他归属。
        // 因而只要存在同类、尚未解决的保护期 C，本帧就暂不创建 D；能接上
        // 自己路径的 C 同时保存 tentative B，之后由成熟轨迹重新仲裁。
        bool deferred_by_protected_existing = false;
        for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            OperationTrack& c = it->second;
            if (!is_claim_protected(c) || c.cls_id != d.cls_id ||
                effective_known_item_owner.count(c.item_id)) {
                continue;
            }
            const bool path_matches = detection_can_belong_to_active_track(d, c);
            if (path_matches) {
                record_tentative_b(&c, d, hand_visible_d);
            }
            // hand_visible_d 是“当前手操作正在接触这个同类框”的强上下文；
            // 即使框跳动暂时脱离路径，也先留在未决池，避免 t0/t1 误建 D。
            if (path_matches || hand_visible_d || replacement_owner >= 0) {
                deferred_by_protected_existing = true;
            }
        }
        if (deferred_by_protected_existing) {
            printf("[3.0] 同类 B cls=%d 仍处于旧 C 保护期，暂不建立 D\n",
                   d.cls_id);
            continue;
        }

        // CONTACT_* 的 B 即使本帧因框跳动没有被 update 认领，也不能直接
        // 进入 D。先保留为旧物品的未决解释，等待下一帧真实观测。
        bool belongs_to_contact_track = false;
        for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            if (is_active_contact_track(it->second) &&
                is_claim_mature(it->second) &&
                detection_can_belong_to_active_track(d, it->second)) {
                belongs_to_contact_track = true;
                break;
            }
        }
        if (belongs_to_contact_track) continue;

        // 先完成当前候选框的同类冲突图，再决定它是否能进入 D 链路。旧 C
        // 即使已经临时拥有另一检测框，也不能让“没有直接框、但路径可到达
        // 当前 d”的旧 A 漏出 alias；否则临时 D 会反过来排除 A 的最终框。
        const SameClassCandidateContext candidate_context =
            build_same_class_candidate_context(
                d, detection_index, working_inventory_, operation_start_inventory_,
                pending_in_ids_, track_buffer_, effective_known_item_owner);
        const std::string direct_old_ids =
            candidate_context_ids(candidate_context.direct_old_item_ids);
        const std::string viable_old_ids =
            candidate_context_ids(candidate_context.viable_unresolved_old_item_ids);
        const std::string matching_suspect_ids =
            candidate_context_ids(candidate_context.matching_suspect_runtime_keys);
        trace_("CANDIDATE-CONTEXT",
               "candidate=%d cls=%d direct-old=%s viable-unresolved-old=%s "
               "matching-suspect=%s decision=%s allow-d-promote=%d",
               detection_index, d.cls_id, direct_old_ids.c_str(), viable_old_ids.c_str(),
               matching_suspect_ids.c_str(), candidate_context_decision(candidate_context),
               candidate_context.viable_unresolved_old_item_ids.empty() ? 1 : 0);

        // 保留已验证的“旧 C 另有直接框时也要建立 alias”逻辑，并补上上面
        // 收集到的无直接框可行旧 C。两类关系都只存在于当前操作运行时。
        std::set<int> alias_old_item_ids =
            candidate_context.viable_unresolved_old_item_ids;
        for (std::map<int, InventoryItem>::const_iterator old =
                 operation_start_inventory_.begin();
             old != operation_start_inventory_.end(); ++old) {
            if (old->second.cls_id != d.cls_id ||
                !working_inventory_.count(old->first) ||
                pending_in_ids_.count(old->first)) {
                continue;
            }
            std::map<int, OperationTrack>::const_iterator runtime =
                track_buffer_.find(old->first);
            if (runtime == track_buffer_.end() ||
                !is_unresolved_operation_start_old_track(runtime->second) ||
                !is_active_runtime_track(runtime->second)) {
                continue;
            }
            std::map<int, int>::const_iterator owner =
                effective_known_item_owner.find(old->first);
            // 已经拥有另一直接框的未决 C 仍保留原有 alias 语义；这不是
            // 对它的第二次身份认领，而是 D 独立性检查需要的冲突边。
            if (owner == effective_known_item_owner.end() ||
                owner->second == static_cast<int>(di)) {
                continue;
            }
            alias_old_item_ids.insert(old->first);
        }
        if (!alias_old_item_ids.empty()) {
            OperationTrack track;
            const int key = new_suspect_id_();
            track.suspect_id = key;
            track.is_suspect_new = true;
            track.suspect_source = replacement_owner >= 0
                ? SuspectSource::C_POSITION_REPLACEMENT_D
                : SuspectSource::HAND_VISIBLE_D;
            track.cls_id = d.cls_id;
            track.original_box = d.box;
            track.last_seen_box = d.box;
            track.has_last_seen_box = true;
            if (hand_visible_d) {
                track.first_hand_block_box = d.box;
                track.last_hand_block_box = d.box;
                track.has_first_hand_block_box = true;
                track.has_last_hand_block_box = true;
            }
            track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
            track.shelter_or_hold = true;
            track.self_match_count = 1;
            track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
            track.track.push_back(d.box);
            track_buffer_[key] = track;
            link_suspect_to_conflicting_old_items_(key, alias_old_item_ids,
                                                    "HAND", detection_index);
            claimed_detection_indices->insert(static_cast<int>(di));
            trace_("D-GUARD",
                   "candidate=%zu cls=%d action=quarantine-pending-d suspect=%d "
                   "conflicting-old-count=%zu reason=complete-same-class-candidate-context "
                   "hand-touching=%d formal-owner-authority=0",
                   di, d.cls_id, key, alias_old_item_ids.size(),
                   hand_visible_d ? 1 : 0);
            trace_track_("STATE", track_buffer_[key],
                         "create-quarantined-hand-visible-suspect");
            continue;
        }

        // 即使当前成熟 C 的宽松路径暂时没有命中，其他同类旧 C 仍可能
        // 正在等待自己的原位置/轨迹证据。此时 B 只能进未决池，不能利用
        // “没有候选”这一瞬间伪造 D；只有同类旧 C 都有独立归属或已明确
        // 结束后，才允许开始新 D 链路。
        bool unresolved_same_class_old = false;
        for (std::map<int, InventoryItem>::const_iterator old =
                 working_inventory_.begin(); old != working_inventory_.end(); ++old) {
            if (old->second.cls_id != d.cls_id ||
                pending_in_ids_.count(old->first) ||
                old->second.status == ItemStatus::OCCLUDED ||
                effective_known_item_owner.count(old->first)) {
                continue;
            }
            const OperationTrack* runtime = find_runtime_for_item_(old->first);
            if (runtime && runtime->is_suspect_new) continue;
            if (runtime && runtime->state == OperationTrackState::PLACED) continue;
            if (runtime && is_active_runtime_track(*runtime)) {
                unresolved_same_class_old = true;
                break;
            }
            // 没有运行时轨迹但本轮没有独立 owner，说明普通静态 C 也尚未
            // 解决；保守地等待下一帧，而不是把同类 B 计成数量增长。
            unresolved_same_class_old = true;
            break;
        }
        if (unresolved_same_class_old) {
            printf("[3.0] 同类旧 C 尚未得到独立归属，B cls=%d 进入未决池\n",
                   d.cls_id);
            continue;
        }

        // 如果没有在 update_existing_hand_tracks_ 中找到，是一个真正新的 D。
        OperationTrack track;
        const int key = new_suspect_id_();
        track.suspect_id = key;
        track.is_suspect_new = true;
        track.suspect_source = replacement_owner >= 0
            ? SuspectSource::C_POSITION_REPLACEMENT_D
            : SuspectSource::HAND_VISIBLE_D;
        track.cls_id = d.cls_id;
        track.original_box = d.box;
        track.last_seen_box = d.box;
        track.has_last_seen_box = true;
        // C 原位置替代 D 可能已不贴手。此时它仍有 last_seen_box 和候选
        // 轨迹，但不伪造“被手遮挡时的局部框”。
        if (hand_visible_d) {
            track.first_hand_block_box = d.box;
            track.last_hand_block_box = d.box;
            track.has_first_hand_block_box = true;
            track.has_last_hand_block_box = true;
        }
        track.state = OperationTrackState::HAND_PARTIAL_BLOCKED;
        track.shelter_or_hold = true;
        track.self_match_count = 1;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track.track.push_back(d.box);
        track_buffer_[key] = track;
        claimed_detection_indices->insert(static_cast<int>(di));
        printf("[3.0] 预登记疑似新物品 D suspect#%d cls=%d source=%s\n",
               key, d.cls_id, suspect_source_name(track.suspect_source));
        trace_track_("STATE", track_buffer_[key], "create-hand-visible-suspect");
    }
}

void SessionManager::mark_newly_hand_blocked_items_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        std::set<int>* claimed_detection_indices,
        std::map<int, int>* known_item_owner,
        std::set<int>* new_existing_track_ids) {
    // 已经确认“这次没有被拿起”的物品，直到手真正离开前都不重复入 HAND_*。
    for (std::set<int>::iterator it = released_hand_candidate_ids_.begin();
         it != released_hand_candidate_ids_.end();) {
        std::map<int, InventoryItem>::const_iterator item = working_inventory_.find(*it);
        if (item == working_inventory_.end() || !hand_affects(hand_box, item->second.box)) {
            it = released_hand_candidate_ids_.erase(it);
        } else {
            ++it;
        }
    }

    for (std::map<int, InventoryItem>::iterator it = working_inventory_.begin();
         it != working_inventory_.end(); ++it) {
        InventoryItem& item = it->second;
        OperationTrack* existing = find_runtime_for_item_(item.item_id);
        if (existing && is_active_runtime_track(*existing) &&
            existing->state != OperationTrackState::PLACED) {
            continue;
        }
        if (item.status == ItemStatus::OCCLUDED ||
            released_hand_candidate_ids_.count(item.item_id)) continue;
        // reserve_visible_known_detections_ 已经在本帧为它保留了自己的
        // 严格框。它仍可在后续帧因漏检重新进入 HAND_*，但不能在这一帧
        // 把旁边的新 D 局部框认成自己的遮挡框。
        if (known_item_owner->count(item.item_id)) continue;

        const BBox reference = item.base_box.area() > 0.0f ? item.base_box : item.box;
        const bool full = hand_fully_covers(hand_box, reference);
        const bool partial = !full && hand_affects(hand_box, reference);
        // e2 <= r < e1 即使当前没有 A 的 YOLO 框，也必须进入
        // HAND_PARTIAL；是否有局部框只决定能否更新 last_hand_block_box。
        if (!partial && !full) continue;

        int observed_index = unique_hand_affected_detection_for_box(
            detections, *claimed_detection_indices, item.cls_id, reference, hand_box);
        if (observed_index < 0) {
            observed_index = best_hand_affected_detection_for_box(
                detections, *claimed_detection_indices, item.cls_id, reference, hand_box);
        }
        // 同一局部框若同时合理地属于两个同类旧库存，不能按 map 顺序硬认领。
        // 保持它未决；后续 scan 也会看到它“可能属于旧库存”，因此不会误建 D。
        if (observed_index >= 0 &&
            hand_affected_existing_candidate_count(working_inventory_,
                                                    detections[observed_index], hand_box) != 1) {
            observed_index = -1;
        }

        OperationTrack track;
        track.item_id = item.item_id;
        track.cls_id = item.cls_id;
        track.original_box = reference;
        track.needs_no_hand_settlement = true;
        track.shelter_or_hold = true;
        track.claim_grace_remaining = FLOW3_NEW_TRACK_CLAIM_GRACE_FRAMES;
        track.hand_track_start_index = static_cast<int>(hand_track_.size()) - 1;
        track.track.push_back(track.original_box);
        track.state = full ? OperationTrackState::HAND_FULL_BLOCKED
                           : OperationTrackState::HAND_PARTIAL_BLOCKED;
        // 首帧没有可靠 C 框时，下一次同类 B 即使落回预计位置，也必须先
        // 建立重新出现候选并等待自匹配，不能一帧确认移动。
        track.reappearance_pending = observed_index < 0;
        if (observed_index >= 0) {
            const Detection& d = detections[observed_index];
            if (contact_detection_is_at_original(track, d) ||
                !is_claim_protected(track)) {
                track.last_seen_box = d.box;
                track.has_last_seen_box = true;
                track.first_hand_block_box = d.box;
                track.last_hand_block_box = d.box;
                track.has_first_hand_block_box = true;
                track.has_last_hand_block_box = true;
                claimed_detection_indices->insert(observed_index);
                (*known_item_owner)[item.item_id] = observed_index;
            } else {
                // 首帧已跑离旧位置的同类框只能记作本地 B；两帧保护期内
                // 不得以它阻止成熟 C 或真正 D 的仲裁。
                record_tentative_b(&track, d, hand_touches_detection(hand_box, d.box));
                track.reappearance_pending = true;
            }
        }
        // 已经放下过又再次被手接触时，覆盖旧运行时记录即可。
        for (std::map<int, OperationTrack>::iterator rt = track_buffer_.begin();
             rt != track_buffer_.end();) {
            if (rt->second.item_id == item.item_id) {
                trace_track_("STATE", rt->second,
                             "replace-runtime-with-new-hand-blocked-track");
                rt = track_buffer_.erase(rt);
            } else {
                ++rt;
            }
        }
        // C 重新被手接触时会重建自己的 runtime；仍在隔离中的 D 保留了
        // D -> C 的关系，必须用它恢复新 C -> D 的反向索引。否则后续同一
        // 无手框只会被 C 看到，D 会被错误当作“缺失的 stale alias”。
        for (std::map<int, OperationTrack>::const_iterator suspect =
                 track_buffer_.begin(); suspect != track_buffer_.end(); ++suspect) {
            if (!suspect->second.is_suspect_new ||
                !suspect->second.pending_d_quarantined_by_old_c ||
                !suspect->second.conflicting_old_item_ids.count(item.item_id)) {
                continue;
            }
            track.conflicting_suspect_keys.insert(suspect->first);
            trace_("C-D-ALIAS",
                   "old-item=%d suspect=%d phase=HAND relation=restored "
                   "action=rebind-after-retouch",
                   item.item_id, suspect->first);
        }
        track_buffer_[item.item_id] = track;
        if (new_existing_track_ids) new_existing_track_ids->insert(item.item_id);
        trace_track_("STATE", track_buffer_[item.item_id],
                     "create-hand-blocked-existing-item");
    }
}

void SessionManager::apply_suspect_cover_evidence_(
        const BBox& hand_box, const std::vector<Detection>& /*detections*/,
        bool hand_moved) {
    for (std::map<int, OperationTrack>::const_iterator dit = track_buffer_.begin();
         dit != track_buffer_.end(); ++dit) {
        const OperationTrack& d = dit->second;
        if (!d.is_suspect_new ||
            (!d.has_last_seen_box && !d.has_last_hand_block_box)) {
            continue;
        }
        // C_POSITION_REPLACEMENT_D 可能已经离开手框，只有 last_seen_box，
        // 不能再把 last_hand_block_box 当成 D 唯一可用的遮挡位置。
        const BBox d_box = d.has_placed_box ? d.placed_box :
            (d.has_last_seen_box ? d.last_seen_box : d.last_hand_block_box);
        for (std::map<int, OperationTrack>::iterator cit = track_buffer_.begin();
             cit != track_buffer_.end(); ++cit) {
            OperationTrack& c = cit->second;
            if (c.is_suspect_new || c.state == OperationTrackState::NORMAL ||
                c.state == OperationTrackState::PLACED || c.original_box.area() <= 0.0f) {
                continue;
            }
            std::vector<BBox> covers;
            covers.push_back(hand_box);
            covers.push_back(d_box);
            const bool union_covers_c = fully_covered_by(c.original_box, covers);
            const bool d_covers_c = fully_covered_by(c.original_box,
                                                      std::vector<BBox>(1, d_box));
            if (!union_covers_c) continue;
            // D + 手的并集只能在手实际移动的有效帧中构成 C 的反向证据。
            // 静止手帧仍属于 HAND_* 的模糊帧，不能改变 hold/not_hold 或释放 C。
            if (!hand_moved) continue;
            // 尚未连续确认并放下的 D 只是“可能挡住 C”的解释，不能改变 C
            // 的 hold/not-hold 计数；否则一次误检又会把 C 提前释放。
            if (!d.promoted_to_working_inventory || !d.drop_confirmed) continue;
            c.hold_evidence_count = 0;
            ++c.not_hold_evidence_count;
            if (d_covers_c) {
                std::map<int, InventoryItem>::iterator item = working_inventory_.find(c.item_id);
                if (item != working_inventory_.end()) {
                    item->second.status = ItemStatus::OCCLUDED;
                    item->second.block_ids.insert(d.item_id);
                }
                release_not_held_(c, true, ReleaseReason::FULLY_OCCLUDED,
                                  -1, nullptr,
                                  "hand-confirmed-full-occlusion");
            }
        }
    }
}

void SessionManager::process_effective_hand_frame_(
        const BBox& hand_box, const std::vector<Detection>& detections,
        bool first_hand_frame) {
    MoveValue delta;
    if (!first_hand_frame) {
        delta.dx = hand_box.cx() - old_hand_box_.cx();
        delta.dy = hand_box.cy() - old_hand_box_.cy();
        hand_track_.push_back(hand_box);
        append_move_to_existing_hand_tracks_(delta);
    }
    const bool hand_moved = !first_hand_frame &&
        move_length(delta) >= TRACK_HAND_MOVE_EPS;

    // 先更新已有 HAND_*；随后先为仍稳定可见的旧库存保留本帧自己的
    // 严格框，再处理新进入 HAND_* 的物品。这个先后顺序很关键：若旧
    // 苹果已有完整框，旁边贴手的新苹果不能先被“局部可能属于旧苹果”
    // 的规则抢走。
    std::set<int> claimed;
    std::map<int, int> known_item_owner;
    std::set<int> new_existing_track_ids;
    const HandDirectOldOwnerPlan direct_old_owner_plan =
        build_mutually_unique_hand_direct_old_owner_by_detection_(
            detections, claimed, known_item_owner);
    if (!first_hand_frame) {
        // 已确认 MOVED 仍是本次 operation 的临时工作状态。路径更新之前先
        // 让唯一的严格原位 owner 有机会完整撤销它，避免 PLACED 继续污染
        // 后续无手绑定、前景遮挡物与最终事件。
        for (std::map<int, int>::const_iterator owner =
                 direct_old_owner_plan.owner_by_detection.begin();
             owner != direct_old_owner_plan.owner_by_detection.end(); ++owner) {
            if (direct_old_owner_strength_for_detection(
                    direct_old_owner_plan, owner->first) !=
                HandDirectOldOwnerStrength::STRICT) {
                continue;
            }
            std::map<int, OperationTrack>::iterator track =
                track_buffer_.find(owner->second);
            if (track != track_buffer_.end()) {
                const bool was_provisional_moved =
                    track->second.state == OperationTrackState::PLACED ||
                    track->second.resolution == ExistingItemResolution::MOVED_CONFIRMED;
                rollback_provisional_moved_to_direct_original_(
                    &track->second, owner->first, detections[owner->first]);
                // 回滚后这张严格原位框仍必须保留给旧 C，直到本帧结束。
                // 否则后续的“新接触/新遮挡”扫描会把已经回滚为静态的
                // runtime 当成未认领物品，再次覆盖为 HAND_*。这只是本帧
                // 临时所有权；下一张有手帧仍按原有规则重新评估是否被拿起。
                if (was_provisional_moved &&
                    track->second.state == OperationTrackState::NORMAL &&
                    track->second.resolution == ExistingItemResolution::STATIC_CONFIRMED) {
                    claimed.insert(owner->first);
                    known_item_owner[owner->second] = owner->first;
                }
            }
        }
        // 低覆盖率 CONTACT_* 先用物品真实检测框更新；这里不能使用手位移
        // 推算的 estimated_box。
        update_existing_contact_tracks_(hand_box, detections, &claimed,
                                        &known_item_owner, direct_old_owner_plan);
        update_existing_hand_tracks_(hand_box, detections, delta, &claimed,
                                     &known_item_owner, direct_old_owner_plan);
        // 原位暂时确认后的 C 若在本帧再次随手离开原位置，必须先恢复
        // HAND_* 身份链，再让静态预约或 D 扫描处理剩余检测框。
        reopen_released_static_tracks_(hand_box, detections, delta);
    }
    reserve_visible_known_detections_(hand_box, detections, &claimed,
                                      &known_item_owner);
    mark_new_contact_candidates_(hand_box, detections, &claimed,
                                 &known_item_owner, &new_existing_track_ids);
    mark_newly_hand_blocked_items_(hand_box, detections, &claimed,
                                   &known_item_owner, &new_existing_track_ids);
    scan_or_update_suspects_(hand_box, detections, &claimed, known_item_owner,
                             direct_old_owner_plan,
                             first_hand_frame);
    apply_suspect_cover_evidence_(hand_box, detections, hand_moved);
    advance_claim_grace_(new_existing_track_ids);
    // 有手阶段每张有效帧都必须维护实时观察层。这里不改变库存或事件，
    // 只把 C/D/alias 当前的可撤销结论写入运行时记录和调试日志。
    update_hand_live_states_();
}

}  // namespace fridge
