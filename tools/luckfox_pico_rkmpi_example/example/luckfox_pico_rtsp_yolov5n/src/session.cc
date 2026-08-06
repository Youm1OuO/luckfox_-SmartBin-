// ============================================================================
//  session.cc
//  3.0：工作库存 + HAND_* 候选 + 疑似新物品 D + 无手逐帧条件提交
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

SessionManager::SessionManager() {}

void SessionManager::trace_(const char* tag, const char* format, ...) const {
    if (!FLOW3_DEBUG_TRACE_LOG) return;
    printf("[3.0-TRACE][%s][op=%d][frame=%d][%s] ", tag,
           active_operation_id_, trace_frame_id_,
           trace_hand_phase_ ? "HAND" : "NO_HAND");
    va_list args;
    va_start(args, format);
    vprintf(format, args);
    va_end(args);
    printf("\n");
}

void SessionManager::set_live_state_(OperationTrack* track,
                                     LiveObservationState state,
                                     bool provisional,
                                     const char* reason) {
    if (!track) return;
    const std::string next_reason = reason ? reason : "NONE";
    if (track->live_state == state &&
        track->live_state_provisional == provisional &&
        track->live_state_reason == next_reason) {
        return;
    }
    track->live_state = state;
    track->live_state_provisional = provisional;
    track->live_state_last_changed_frame = trace_frame_id_;
    track->live_state_reason = next_reason;
    const BBox box = track->has_last_seen_box ? track->last_seen_box : estimated_box(*track);
    trace_("LIVE-STATE",
           "owner-kind=%s runtime=%d item=%d cls=%d state=%s provisional=%d "
           "reason=%s box=(%.1f,%.1f,%.1f,%.1f)",
           runtime_owner_kind_name(*track),
           track->is_suspect_new ? track->suspect_id : track->item_id,
           track->item_id, track->cls_id,
           live_observation_state_name(state), provisional ? 1 : 0,
           track->live_state_reason.c_str(),
           box.x1, box.y1, box.x2, box.y2);
}

void SessionManager::update_hand_live_states_() {
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        OperationTrack& track = it->second;
        if (track.is_suspect_new) {
            set_live_state_(&track,
                            track.pending_d_quarantined_by_old_c
                                ? LiveObservationState::C_D_ALIAS
                                : LiveObservationState::PROVISIONAL_D,
                            true,
                            track.pending_d_quarantined_by_old_c
                                ? "same-class-unsettled-old-alias"
                                : "hand-visible-provisional-d");
            continue;
        }
        if (!is_unresolved_operation_start_old_track(track)) continue;
        if (!track.conflicting_suspect_keys.empty()) {
            set_live_state_(&track, LiveObservationState::C_D_ALIAS, true,
                            "same-class-provisional-d-alias");
        } else if (track.contact_state != ContactState::NONE) {
            set_live_state_(&track, LiveObservationState::HAND_CONTACT, true,
                            "contact-track-active");
        } else if (track.hold_and_move || has_meaningful_hand_move(track) ||
                   track.reappearance_pending) {
            set_live_state_(&track, LiveObservationState::POSSIBLE_MOVED, true,
                            "hand-path-or-reappearance-pending");
        } else if (track.state == OperationTrackState::HAND_FULL_BLOCKED) {
            set_live_state_(&track, LiveObservationState::POSSIBLE_OCCLUDED, true,
                            "hand-fully-covers-old-c");
        } else {
            set_live_state_(&track, LiveObservationState::HAND_CONTACT, true,
                            "hand-affects-old-c");
        }
    }
}

bool SessionManager::old_track_has_unresolved_alias_(const OperationTrack& track) const {
    if (track.is_suspect_new) return false;
    for (std::set<int>::const_iterator key = track.conflicting_suspect_keys.begin();
         key != track.conflicting_suspect_keys.end(); ++key) {
        std::map<int, OperationTrack>::const_iterator suspect = track_buffer_.find(*key);
        if (suspect != track_buffer_.end() && suspect->second.is_suspect_new &&
            suspect->second.pending_d_quarantined_by_old_c) {
            return true;
        }
    }
    return false;
}

void SessionManager::unlink_quarantined_suspect_(int runtime_key,
                                                  const char* action) {
    std::map<int, OperationTrack>::iterator suspect = track_buffer_.find(runtime_key);
    if (suspect == track_buffer_.end() || !suspect->second.is_suspect_new) return;
    OperationTrack& d = suspect->second;
    for (std::set<int>::const_iterator old_id = d.conflicting_old_item_ids.begin();
         old_id != d.conflicting_old_item_ids.end(); ++old_id) {
        std::map<int, OperationTrack>::iterator old = track_buffer_.find(*old_id);
        if (old != track_buffer_.end()) {
            old->second.conflicting_suspect_keys.erase(runtime_key);
            trace_("C-D-ALIAS",
                   "old-item=%d suspect=%d phase=%s action=%s relation=closed",
                   *old_id, d.suspect_id,
                   trace_hand_phase_ ? "HAND" : "NO_HAND",
                   action ? action : "unlink");
        }
    }
    d.conflicting_old_item_ids.clear();
    d.pending_d_quarantined_by_old_c = false;
    d.alias_no_hand_match_count = 0;
    d.alias_no_hand_missing_count = 0;
}

void SessionManager::trace_track_(const char* tag, const OperationTrack& track,
                                  const char* reason) const {
    const BBox expected = estimated_box(track);
    trace_(tag,
           "item=%d suspect=%d cls=%d state=%s contact=%s resolution=%s "
           "needs_settle=%d static-near-original=%d/%d grace=%d hold=%d not_hold=%d "
           "missing=%d ambiguous=%d "
           "owner-kind=%s quarantine=%d alias-old=%zu alias-d=%zu alias-no-hand=%d "
           "alias-missing=%d "
           "live=%s provisional=%d "
           "original=(%.1f,%.1f,%.1f,%.1f) expected=(%.1f,%.1f,%.1f,%.1f) "
           "last=%d:(%.1f,%.1f,%.1f,%.1f) reappear=%d/%d:(%.1f,%.1f,%.1f,%.1f) "
           "reason=%s",
           track.item_id, track.suspect_id, track.cls_id,
           operation_track_state_name(track.state),
           contact_state_name(track.contact_state),
           existing_resolution_name(track.resolution),
           track.needs_no_hand_settlement ? 1 : 0,
           track.stable_near_original_no_hand_count,
           track.has_stable_near_original_box ? 1 : 0,
           track.claim_grace_remaining, track.hold_evidence_count,
           track.not_hold_evidence_count, track.no_hand_missing_count,
           (track.b_claim_ambiguous || track.contact_path_ambiguous ||
            track.no_hand_candidate_ambiguous) ? 1 : 0,
           runtime_owner_kind_name(track),
           track.pending_d_quarantined_by_old_c ? 1 : 0,
           track.conflicting_old_item_ids.size(),
           track.conflicting_suspect_keys.size(),
           track.alias_no_hand_match_count,
           track.alias_no_hand_missing_count,
           live_observation_state_name(track.live_state),
           track.live_state_provisional ? 1 : 0,
           track.original_box.x1, track.original_box.y1,
           track.original_box.x2, track.original_box.y2,
           expected.x1, expected.y1, expected.x2, expected.y2,
           track.has_last_seen_box ? 1 : 0,
           track.last_seen_box.x1, track.last_seen_box.y1,
           track.last_seen_box.x2, track.last_seen_box.y2,
           track.has_reappear_candidate_box ? 1 : 0,
           track.reappear_candidate_match_count,
           track.reappear_candidate_box.x1, track.reappear_candidate_box.y1,
           track.reappear_candidate_box.x2, track.reappear_candidate_box.y2,
           reason ? reason : "NONE");
}

void SessionManager::rebuild_persistent_item_index_() {
    item_by_id_.clear();
    std::map<int, InventoryItem>& items = inventory_.mutable_items();
    for (std::map<int, InventoryItem>::iterator it = items.begin(); it != items.end(); ++it) {
        item_by_id_[it->first] = &it->second;
    }
}

void SessionManager::reset_operation_runtime_() {
    working_inventory_.clear();
    operation_start_inventory_.clear();
    working_next_item_id_ = inventory_.next_item_id();
    working_inventory_active_ = false;
    track_buffer_.clear();
    pending_in_ids_.clear();
    pending_out_ids_.clear();
    confirmed_moved_ids_.clear();
    released_hand_candidate_ids_.clear();
    cross_class_duplicate_identity_exclusions_.clear();
    visible_count_detection_owner_.clear();
    visible_count_survivor_ids_.clear();
    visible_count_out_candidate_ids_.clear();
    visible_count_missing_counts_.clear();
    visible_count_confirmed_out_ids_.clear();
    visible_count_prior_survivors_by_cls_.clear();
    visible_count_prior_survivor_boxes_by_cls_.clear();
    occlusion_loss_missing_counts_.clear();
    hand_track_.clear();
    has_old_hand_box_ = false;
    next_suspect_id_ = -1;
    no_hand_streak_ = 0;
    active_operation_id_ = 0;
}

void SessionManager::start_new_session(long long time_ms) {
    reset_operation_runtime_();
    current_time_ms_ = time_ms;
    hand_present_ = false;
    session_active_ = true;
    if (has_local_inventory_) {
        initial_check_state_ = InitialCheckState::NOT_NEEDED;
    } else {
        initial_check_state_ = InitialCheckState::NONE;
        backend_status_ = BackendStatus::UNKNOWN;
    }
}

void SessionManager::init_from_backend(const std::vector<InventoryItem>& items,
                                       bool authoritative_empty) {
    if (!session_active_ || has_local_inventory_ ||
        backend_status_ != BackendStatus::UNKNOWN) return;
    if (items.empty() && !authoritative_empty) {
        mark_backend_unavailable();
        return;
    }

    std::map<int, InventoryItem> loaded;
    int next_id = inventory_.next_item_id();
    for (size_t i = 0; i < items.size(); ++i) {
        InventoryItem item = items[i];
        if (item.item_id <= 0 || loaded.count(item.item_id)) item.item_id = next_id++;
        if (item.base_box.area() <= 0.0f) item.base_box = item.box;
        if (item.box.area() <= 0.0f) item.box = item.base_box;
        if (item.base_box.area() <= 0.0f || item.box.area() <= 0.0f) continue;
        item.status = ItemStatus::VISIBLE;
        item.block_ids.clear();
        loaded[item.item_id] = item;
        next_id = std::max(next_id, item.item_id + 1);
    }
    inventory_.replace_all(loaded, next_id);
    rebuild_persistent_item_index_();
    has_local_inventory_ = true;
    backend_status_ = BackendStatus::TRUSTED;
    initial_check_state_ = InitialCheckState::WAITING;
}

void SessionManager::mark_backend_unavailable() {
    if (!session_active_ || has_local_inventory_) return;
    backend_status_ = BackendStatus::NO_TRUSTED_BACKEND;
    initial_check_state_ = ALLOW_INITIAL_FRAME_BOOTSTRAP_WHEN_BACKEND_UNAVAILABLE
        ? InitialCheckState::BOOTSTRAP_FROM_FRAME : InitialCheckState::NONE;
}

void SessionManager::finish_session(long long time_ms) {
    current_time_ms_ = time_ms;
    // 未完成操作不提交；正式库存保持上一次已确认结果。
    reset_operation_runtime_();
    hand_present_ = false;
    initial_check_state_ = InitialCheckState::NONE;
    session_active_ = false;
}

void SessionManager::begin_closing_guard() {
    has_old_hand_box_ = false;
}

void SessionManager::resume_after_false_closing() {
    // CLOSING 打断了连续证据链，最安全的处理是丢弃工作副本，保留正式库存。
    if (working_inventory_active_) reset_operation_runtime_();
}

void SessionManager::validate_initial_no_hand_frame_(
        const std::vector<Detection>& detections) const {
    int matched = 0;
    for (size_t si = 0; si < detections.size(); ++si) {
        int count = 0;
        for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
             it != inventory_.items().end(); ++it) {
            if (strict_match(it->second, detections[si]) ||
                partial_match(it->second, detections[si])) {
                ++count;
            }
        }
        if (count == 1) ++matched;
    }
    printf("[INIT-CHECK] 首张无手帧校验：%d/%zu 个框可唯一对应；库存未修改\n",
           matched, detections.size());
}

void SessionManager::initialize_from_bootstrap_no_hand_frame_(
        const std::vector<Detection>& detections, int frame_id) {
    std::map<int, InventoryItem> loaded;
    int next_id = inventory_.next_item_id();
    for (size_t i = 0; i < detections.size(); ++i) {
        loaded[next_id] = make_inventory_item(next_id, detections[i], frame_id,
                                               current_time_ms_);
        ++next_id;
    }
    inventory_.replace_all(loaded, next_id);
    rebuild_persistent_item_index_();
    has_local_inventory_ = true;
    initial_check_state_ = InitialCheckState::DONE;
    printf("[INIT] 本地首张无手帧建库：%zu 个物品\n", loaded.size());
}

void SessionManager::finalize_initial_check_before_hand_() {
    if (initial_check_state_ != InitialCheckState::WAITING &&
        initial_check_state_ != InitialCheckState::BOOTSTRAP_FROM_FRAME) return;
    // 手在任何无手帧前就进入画面时，没有可用于首次校验/建库的直接检测。
    if (initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_FRAME) {
        initialize_from_bootstrap_no_hand_frame_(std::vector<Detection>(), 0);
    } else {
        initial_check_state_ = InitialCheckState::SKIPPED;
    }
}

int SessionManager::new_suspect_id_() {
    return next_suspect_id_--;
}

OperationTrack* SessionManager::find_runtime_for_item_(int item_id) {
    for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        if (it->second.item_id == item_id) return &it->second;
    }
    return nullptr;
}

const OperationTrack* SessionManager::find_runtime_for_item_(int item_id) const {
    for (std::map<int, OperationTrack>::const_iterator it = track_buffer_.begin();
         it != track_buffer_.end(); ++it) {
        if (it->second.item_id == item_id) return &it->second;
    }
    return nullptr;
}

void SessionManager::clear_runtime_after_commit_() {
    trace_("STATE", "clear-operation-runtime-after-commit");
    reset_operation_runtime_();
}

void SessionManager::print_inventory() const {
    const size_t visible_count = inventory_.count_by_status(ItemStatus::VISIBLE);
    const size_t occluded_count = inventory_.count_by_status(ItemStatus::OCCLUDED);
    printf("\n");
    printf("  ┌──────────────────────────────────────────────────────────────────┐\n");
    printf("  │  在库清单 │ 可见: %-3zu │ 遮挡: %-3zu │ 共: %-3zu            │\n",
           visible_count, occluded_count, visible_count + occluded_count);
    printf("  ├────┬──────────────┬────────┬────────┬──────────────────────────┤\n");
    printf("  │ #  │ 类别         │ 状态   │ 遮挡数 │ 位置（当前可见框中心） │\n");
    printf("  ├────┼──────────────┼────────┼────────┼──────────────────────────┤\n");
    for (std::map<int, InventoryItem>::const_iterator it = inventory_.items().begin();
         it != inventory_.items().end(); ++it) {
        const InventoryItem& item = it->second;
        printf("  │ %-2d │ %-12s │ %-6s │ %-6zu │ (%4.0f,%4.0f)             │\n",
               item.item_id, cls_id_to_chinese(item.cls_id),
               item.status == ItemStatus::VISIBLE ? "可见" : "遮挡",
               item.block_ids.size(), item.box.cx(), item.box.cy());
    }
    printf("  └────┴──────────────┴────────┴────────┴──────────────────────────┘\n\n");
}

FrameProcessResult SessionManager::process_frame(
        const std::vector<Detection>& food_detections,
        const std::vector<BBox>& hand_boxes, int frame_id, long long time_ms) {
    FrameProcessResult output;
    trace_frame_id_ = frame_id;
    trace_hand_phase_ = !hand_boxes.empty();
    current_time_ms_ = time_ms;
    hand_present_ = !hand_boxes.empty();
    trace_("FRAME", "foods=%zu hands=%zu operation_active=%d tracks=%zu no_hand_streak=%d",
           food_detections.size(), hand_boxes.size(),
           working_inventory_active_ ? 1 : 0, track_buffer_.size(), no_hand_streak_);
    for (size_t i = 0; i < food_detections.size(); ++i) {
        const Detection& detection = food_detections[i];
        trace_("FOOD-DETECTION",
               "input-index=%zu cls=%d score=%.3f box=(%.1f,%.1f,%.1f,%.1f) "
               "business-input=1 osd-business-visible=1",
               i, detection.cls_id, detection.score,
               detection.box.x1, detection.box.y1,
               detection.box.x2, detection.box.y2);
    }

    if (!session_active_) {
        trace_("FRAME", "ignore-inactive-session foods=%zu hands=%zu",
               food_detections.size(), hand_boxes.size());
        return output;
    }

    if (hand_present_) {
        for (std::map<int, OperationTrack>::iterator it = track_buffer_.begin();
             it != track_buffer_.end(); ++it) {
            OperationTrack& track = it->second;
            // r16 的静态近原位证据只允许来自连续无手帧。手重新出现时，
            // 无论手是否接触该 C，都不能把前后两段无手证据拼在一起。
            if (!track.is_suspect_new) {
                reset_stable_near_original_no_hand_evidence_(
                    &track, "hand-returned");
                continue;
            }
            if (!track.pending_d_quarantined_by_old_c ||
                track.alias_no_hand_missing_count == 0) {
                continue;
            }
            trace_("C-D-ALIAS",
                   "old-count=%zu suspect=%d phase=HAND relation=hand-returned "
                   "missing-count=%d/%d action=reset-stale-alias-evidence",
                   track.conflicting_old_item_ids.size(), track.suspect_id,
                   track.alias_no_hand_missing_count,
                   FLOW3_NO_HAND_D_CONFIRM_FRAMES);
            track.alias_no_hand_missing_count = 0;
        }
        // 细节9的数量不足只属于一段连续无手收尾。手重新进入时，上一段
        // 尚未提交的“一框一物品”候选必须撤销，不能跨操作继续凑 OUT 帧数。
        if (!visible_count_missing_counts_.empty() ||
            !visible_count_confirmed_out_ids_.empty()) {
            clear_visible_count_settlement_(true);
        }
        finalize_initial_check_before_hand_();
        no_hand_streak_ = 0;
        if (!has_local_inventory_) return output;
        const BBox hand = choose_primary_hand(hand_boxes);
        if (!working_inventory_active_) {
            begin_working_operation_(hand, food_detections);
            return output;
        }
        // 手框不动时只有在没有任何未决 HAND_* / CONTACT_* 轨迹的情况下
        // 才能整帧跳过。手指可能在手腕不动时推动物品，所以活动轨迹必须
        // 继续逐帧检查真实物品检测框。
        bool has_active_runtime = false;
        for (std::map<int, OperationTrack>::const_iterator it =
                 track_buffer_.begin(); it != track_buffer_.end(); ++it) {
            if (is_active_runtime_track(it->second)) {
                has_active_runtime = true;
                break;
            }
        }
        if (!has_active_runtime && has_old_hand_box_ &&
            hand_boxes_effectively_same(old_hand_box_, hand)) {
            trace_("FRAME", "skip-stationary-hand-no-active-track");
            return output;
        }
        process_effective_hand_frame_(hand, food_detections, false);
        old_hand_box_ = hand;
        has_old_hand_box_ = true;
        return output;
    }

    ++no_hand_streak_;
    output.no_hand_frame_processed = true;
    if (working_inventory_active_) {
        // 手刚离开的每一帧都直接供 D/C 候选完成认领、连续自匹配和缺失证据。
        // 不缓存、不平均框，也不做多数投票。
        observe_no_hand_frame_(food_detections);
        has_old_hand_box_ = false;
        output.settlement = settle_no_hand_frame_(food_detections, frame_id);
        return output;
    }

    if (initial_check_state_ == InitialCheckState::WAITING ||
        initial_check_state_ == InitialCheckState::BOOTSTRAP_FROM_FRAME) {
        if (initial_check_state_ == InitialCheckState::WAITING) {
            validate_initial_no_hand_frame_(food_detections);
            initial_check_state_ = InitialCheckState::DONE;
        } else {
            initialize_from_bootstrap_no_hand_frame_(food_detections, frame_id);
        }
    }
    return output;
}

}  // namespace fridge
