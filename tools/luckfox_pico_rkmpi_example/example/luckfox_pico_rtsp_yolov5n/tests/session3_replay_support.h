#ifndef __SESSION3_REPLAY_SUPPORT_H
#define __SESSION3_REPLAY_SUPPORT_H

#include <vector>

#include "fridge_config.h"
#include "session.h"

namespace session3_replay {

fridge::Detection det(int cls, float x1, float y1, float x2, float y2,
                      float score = 0.9f);
fridge::InventoryItem item(int id, int cls, float x1, float y1,
                           float x2, float y2);
void initial_no_hand_frame(fridge::SessionManager* session,
                           const std::vector<fridge::Detection>& foods,
                           int* frame);
bool has_event(const fridge::SettlementResult& result, fridge::EventKind kind,
               int item_id);
void send_frame(fridge::SessionManager* session,
                const std::vector<fridge::Detection>& foods,
                const std::vector<fridge::BBox>& hands, int* frame);
fridge::SettlementResult settle_after_hand(
        fridge::SessionManager* session,
        const std::vector<fridge::Detection>& no_hand_foods, int* frame);

void test_move_keeps_identity();
void test_drop_at_middle_of_candidate_path_keeps_identity();
void test_out_removes_existing_identity();
void test_unconfirmed_hand_candidate_is_out_after_final_absence();
void test_unconfirmed_hand_candidate_reappears_at_old_position();
void test_fully_hand_hidden_item_can_out();
void test_hand_cover_ratio_controls_hand_state();
void test_new_item_requires_d_chain();
void test_new_item_can_confirm_from_first_no_hand_frame();
void test_partially_seen_new_item_can_reappear_full_on_middle_path();
void test_fully_hidden_new_item_can_enter_from_post_hand_reveal();
void test_new_item_replacing_c_old_position_is_registered_without_hand_contact();
void test_post_hand_reveal_requires_continuous_no_hand_confirmation();
void test_d_reappearance_does_not_claim_strict_existing_inventory();
void test_hand_visible_d_does_not_confirm_from_strict_static_old_c();
void test_new_item_dropped_before_hand_moves_away_keeps_its_identity();
void test_partial_existing_item_is_not_registered_as_new_d();
void test_fast_same_class_b_becomes_c_candidate_not_d();
void test_new_track_claim_grace_defers_same_class_d();
void test_mature_same_class_tracks_keep_shared_b_ambiguous_until_no_hand_count_settlement();
void test_no_hand_frame_respects_new_track_claim_grace();
void test_drop_requires_continuous_evidence();
void test_ambiguous_hand_partial_box_does_not_create_d();
void test_adjacent_same_class_new_item_gets_its_own_d_chain();
void test_unbound_no_hand_box_never_auto_in();
void test_identity_ambiguity_holds_visible_target_without_out_evidence();
void test_identity_ambiguity_does_not_block_confirmed_full_occlusion();
void test_confirmed_target_exit_blocks_causal_front_missing();
void test_confirmed_moved_orange_occludes_reserved_same_class_apple();
void test_visible_count_survivors_preserve_causal_missing_apple();
void test_provisional_causal_occlusion_survives_delayed_no_hand_settlement();
void test_provisional_causal_occlusion_clears_on_target_direct_observation();
void test_causal_front_missing_accepts_substantial_partial_cover();
void test_moved_target_drops_stale_historical_blocker();
void test_inventory_rejects_invalid_formal_occlusion_proof();
void test_pending_out_candidates_converge_monotonically();
void test_moved_front_item_occludes_then_reveals();
void test_disappearance_supported_moved_front_occludes_without_out();
void test_causal_front_missing_releases_hand_target_without_out();
void test_disappearance_evidence_resets_when_hand_returns();
void test_edge_residual_historical_blocker_reveals_after_move();
void test_dense_three_apple_topology_reveals_edge_residual_target();
void test_occluded_target_observation_without_blocker_change_stays_occluded();
void test_out_blocker_reveals_observed_target_in_causal_order();
void test_cross_class_duplicate_original_does_not_create_d();
void test_historical_blocker_out_starts_new_missing_chain();
void test_partial_front_relation_does_not_start_occlusion_loss_out_chain();
void test_remaining_historical_blocker_keeps_c_occluded_after_other_moves();
void test_low_coverage_contact_move_keeps_identity();
void test_low_coverage_contact_without_endpoint_stays_pending();
void test_low_coverage_contact_candidate_releases_at_original();
void test_contact_to_hand_transition_keeps_observed_anchor();
void test_stationary_hand_frame_does_not_change_hand_evidence();
void test_one_hand_moves_two_items_with_one_hand_id();
void test_two_hands_keep_opposite_item_deltas_when_detection_order_changes();
void test_merged_hand_detection_suspends_item_deltas();
void test_skewed_merged_hand_detection_suspends_all_item_deltas();
void test_second_hand_overlap_suspends_existing_item_delta();
void test_temporarily_lost_hand_recovers_without_gap_delta();
void test_post_hand_reveal_commits_on_second_direct_frame();
void test_c_reappear_commits_after_second_direct_frame();
void test_active_a_moves_next_to_static_same_class_b();
void test_stale_same_class_reappear_candidate_falls_back_to_moved_a();
void test_stale_same_class_reappear_candidate_without_a_waits_then_out();
void test_provisional_static_a_reopens_when_moved_later_in_same_operation();
void test_same_class_static_neighbor_box_allows_active_a_out();
void test_adjacent_same_class_real_boxes_out_only_removed_item();
void test_adjacent_same_class_single_frame_deficit_recovers_without_out();
void test_adjacent_same_class_deficit_requires_continuous_survivor_box();
void test_out_requires_two_direct_missing_frames();
void test_no_hand_occlusion_uses_cover_union();
void test_unconfirmed_moving_front_defers_out_until_occlusion();
void test_unconfirmed_moving_front_failure_restores_out_chain();
void test_confirmed_occlusion_edge_residual_geometry();
void test_static_gray_zone_old_c_settles_before_r15_stale_alias_cleanup();
void test_static_gray_zone_evidence_resets_when_hand_returns();
void test_quarantined_same_class_duplicate_merges_back_to_old_c();
void test_quarantined_same_class_stale_alias_disappears_after_old_c_settles();
void test_retouch_old_c_preserves_alias_links_and_shared_resolution();
void test_quarantined_same_class_real_d_confirms_after_distinct_no_hand_boxes();
void test_item7_gray_recovery_does_not_block_item8_move();
void test_cover_union_keeps_multiple_tiny_residuals_until_total_area_check();
void test_hand_affected_old_c_does_not_block_static_owner_reservation();
void test_same_class_old_owner_blocks_path_match_before_identity_swap();
void test_direct_original_evidence_rolls_back_provisional_moved();
void test_hand_front_cover_remains_provisional_until_no_hand_confirmation();
void test_same_grade_local_old_owners_remain_ambiguous_despite_center_distance();
void test_same_class_candidate_context_includes_unowned_viable_old_c();
void test_confirmed_reappear_owner_beats_wide_path_but_keeps_equal_tie();

}  // namespace session3_replay

#endif  // __SESSION3_REPLAY_SUPPORT_H
