// Host-side SessionManager replay scenarios: no_hand.
#include "session3_replay_support.h"
#include "session_internal.h"

#include <algorithm>
#include <assert.h>
#include <map>
#include <vector>

namespace session3_replay {

int event_index(const fridge::SettlementResult& result, fridge::EventKind kind,
                int item_id) {
    for (size_t i = 0; i < result.events.size(); ++i) {
        if (result.events[i].kind == kind && result.events[i].item_id == item_id) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

void assert_event_before(const fridge::SettlementResult& result,
                         fridge::EventKind first_kind, int first_item_id,
                         fridge::EventKind second_kind, int second_item_id) {
    const int first = event_index(result, first_kind, first_item_id);
    const int second = event_index(result, second_kind, second_item_id);
    assert(first >= 0 && second >= 0 && first < second);
}

void test_unbound_no_hand_box_never_auto_in() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    // 手只碰到旧苹果；远处橙子没有“贴手出现”的 D 证据链。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    fridge::SettlementResult result = settle_after_hand(&session, stable, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

// 身份归属不唯一和“目标没有检测框”是两件不同的事。前者必须中断
// disappearance / OUT 证据，而不能被当作无歧义的消失。
void test_identity_ambiguity_holds_visible_target_without_out_evidence() {
    fridge::session_internal::OcclusionDecisionInput input;
    input.previous_status = fridge::ItemStatus::VISIBLE;
    input.observation_conflict = true;
    input.after_geometry.covered_ratio = 0.95f;
    input.after_geometry.residual_is_outer_boundary_only = true;
    input.relation_changed_by_confirmed_front = true;
    input.current_confirmed_front = true;

    const fridge::session_internal::OcclusionDecisionResult result =
        fridge::session_internal::decide_occlusion_lifecycle(input);
    assert(result.visibility ==
           fridge::session_internal::VisibilityDecision::PENDING_OCCLUSION_EVIDENCE);
    assert(result.out ==
           fridge::session_internal::OutDisposition::HOLD_FOR_PENDING_OCCLUSION);
    assert(!result.disappearance_candidate);
}

// 同类 owner reservation 只阻止候选框归属，不会推翻已确认前景给出的
// strict/edge-residual 完整遮挡证明。该证明必须直接提交 OCCLUDED。
void test_identity_ambiguity_does_not_block_confirmed_full_occlusion() {
    fridge::session_internal::OcclusionDecisionInput input;
    input.previous_status = fridge::ItemStatus::VISIBLE;
    input.observation_conflict = true;
    input.relation_changed_by_confirmed_front = true;
    input.current_confirmed_front = true;
    input.after_geometry.strict_full = false;
    input.after_geometry.edge_residual_full = true;
    input.after_geometry.full = true;
    input.after_geometry.covered_ratio = 0.956f;
    input.after_geometry.residual_is_outer_boundary_only = true;
    input.after_witness_blocker_ids.insert(5);

    const fridge::session_internal::OcclusionDecisionResult result =
        fridge::session_internal::decide_occlusion_lifecycle(input);
    assert(result.visibility ==
           fridge::session_internal::VisibilityDecision::ENTER_OCCLUDED);
    assert(result.out ==
           fridge::session_internal::OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION);
    assert(result.allow_occluded_transition);
    assert(result.proposed_proof.kind ==
           fridge::OcclusionProofKind::EDGE_RESIDUAL_UNION);
    assert(result.proposed_proof.witness_blocker_ids.size() == 1);
    assert(result.proposed_proof.witness_blocker_ids.count(5));
}

// A target with its own confirmed exit conclusion must be settled by that
// conclusion; a moved/IN front witness cannot relabel it as causal occlusion.
void test_confirmed_target_exit_blocks_causal_front_missing() {
    fridge::session_internal::OcclusionDecisionInput input;
    input.previous_status = fridge::ItemStatus::VISIBLE;
    input.relation_changed_by_confirmed_front = true;
    input.current_confirmed_front = true;
    input.after_geometry.covered_ratio = 0.90f;
    input.after_geometry.residual_is_outer_boundary_only = true;
    input.target_has_independent_exit_evidence = true;
    input.target_has_confirmed_independent_exit = true;
    input.causal_front_missing_candidate = true;
    input.causal_witness_blocker_ids.insert(7);

    const fridge::session_internal::OcclusionDecisionResult result =
        fridge::session_internal::decide_occlusion_lifecycle(input);
    assert(result.visibility == fridge::session_internal::VisibilityDecision::KEEP_VISIBLE);
    assert(result.out == fridge::session_internal::OutDisposition::NORMAL_OUT_EVIDENCE);
    assert(!result.allow_occluded_transition);
    assert(!result.causal_front_missing_candidate);
}

// 实机回放：三个同类苹果各有自己的静态 owner，橙子 #5 从左侧移到右上角
// 并完整覆盖苹果 #1。#1 的同类候选路径会被更强 owner reservation 排除，
// 但这不能阻止已确认的橙子前景提交正式 OCCLUDED。
void test_confirmed_moved_orange_occludes_reserved_same_class_apple() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 878, 245, 1060, 427));  // upper-right apple
    initial.push_back(item(2, 0, 803, 387, 989, 565));   // lower apple
    initial.push_back(item(3, 0, 727, 247, 901, 423));   // upper-left apple
    initial.push_back(item(4, 25, 96, 332, 374, 580));   // soda can
    initial.push_back(item(5, 2, 461, 270, 656, 461));   // orange
    session.init_from_backend(initial, true);
    int frame = 1;

    std::vector<fridge::Detection> stable;
    stable.push_back(det(25, 96, 332, 374, 583));
    stable.push_back(det(0, 878, 245, 1058, 423));
    stable.push_back(det(0, 805, 387, 987, 565));
    stable.push_back(det(0, 727, 247, 901, 421));
    stable.push_back(det(2, 460, 272, 654, 465));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(301, 354, 641, 716)),
               &frame);

    std::vector<fridge::Detection> hand_middle;
    hand_middle.push_back(det(0, 805, 387, 989, 565));
    hand_middle.push_back(det(0, 880, 245, 1058, 425));
    hand_middle.push_back(det(0, 729, 245, 901, 421));
    hand_middle.push_back(det(25, 103, 330, 372, 580));
    hand_middle.push_back(det(2, 505, 272, 654, 465));
    send_frame(&session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(349, 261, 701, 710)),
               &frame);
    send_frame(&session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(341, 243, 678, 650)),
               &frame);

    std::vector<fridge::Detection> hand_late;
    hand_late.push_back(det(0, 900, 249, 1058, 423));
    hand_late.push_back(det(2, 783, 220, 950, 421));
    hand_late.push_back(det(25, 101, 334, 372, 578));
    hand_late.push_back(det(0, 798, 385, 990, 567));
    send_frame(&session, hand_late,
               std::vector<fridge::BBox>(1, fridge::BBox(456, 154, 958, 634)),
               &frame);

    std::vector<fridge::Detection> hand_near;
    hand_near.push_back(det(25, 96, 332, 380, 580));
    hand_near.push_back(det(2, 945, 260, 1050, 445));
    hand_near.push_back(det(0, 725, 254, 841, 369, 0.308f));
    send_frame(&session, hand_near,
               std::vector<fridge::BBox>(1, fridge::BBox(621, 190, 1036, 621)),
               &frame);

    std::vector<fridge::Detection> hand_end;
    hand_end.push_back(det(2, 940, 245, 1047, 434));
    hand_end.push_back(det(25, 98, 332, 374, 574));
    send_frame(&session, hand_end,
               std::vector<fridge::BBox>(1, fridge::BBox(614, 170, 1045, 610)),
               &frame);

    std::vector<fridge::Detection> final_no_hand;
    final_no_hand.push_back(det(0, 801, 394, 989, 565));
    final_no_hand.push_back(det(0, 727, 247, 887, 425));
    final_no_hand.push_back(det(25, 96, 336, 376, 578));
    final_no_hand.push_back(det(2, 823, 234, 1052, 450));
    const fridge::SettlementResult result = settle_after_hand(
        &session, final_no_hand, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 5);
    assert(has_event(result, fridge::EventKind::MOVED, 5));
    assert(has_event(result, fridge::EventKind::OCCLUDED, 1));
    assert(!has_event(result, fridge::EventKind::OUT, 1));
    assert_event_before(result, fridge::EventKind::MOVED, 5,
                        fridge::EventKind::OCCLUDED, 1);
    const fridge::InventoryItem* hidden = session.inventory().find_by_item(1);
    assert(hidden != 0);
    assert(hidden->status == fridge::ItemStatus::OCCLUDED);
    assert(hidden->block_ids.count(5));
    assert(hidden->occlusion_proof.kind ==
           fridge::OcclusionProofKind::EDGE_RESIDUAL_UNION);
}

// 现场拓扑的主机压缩回放：三只苹果中 #1 被手影响并在收尾无手帧中漏检，
// #3/#5 各自保留自己的直接框；两只橙子中只有 #9 从左侧移动到 #1 的旧位置。
// visible-count 的身份结论不能把 #1 恢复成 observation-conflict，也不能让
// #1 的缺额先走 OUT；已确认的 #9 必须先发布 MOVED，再由因果 proof 保护 #1。
void test_visible_count_survivors_preserve_causal_missing_apple() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 300, 100, 400, 200));  // missing apple
    initial.push_back(item(3, 0, 700, 100, 800, 200));  // visible apple
    initial.push_back(item(5, 0, 700, 300, 800, 400));  // visible apple
    initial.push_back(item(7, 2, 100, 300, 200, 400));  // static orange
    initial.push_back(item(9, 2, 100, 100, 200, 200));  // moved front orange
    session.init_from_backend(initial, true);
    int frame = 1;

    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 300, 100, 400, 200));
    stable.push_back(det(0, 700, 100, 800, 200));
    stable.push_back(det(0, 700, 300, 800, 400));
    stable.push_back(det(2, 100, 300, 200, 400));
    stable.push_back(det(2, 100, 100, 200, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::Detection> hand_start;
    hand_start.push_back(det(0, 700, 100, 800, 200));
    hand_start.push_back(det(0, 700, 300, 800, 400));
    hand_start.push_back(det(2, 100, 300, 200, 400));
    hand_start.push_back(det(2, 100, 100, 200, 200));
    send_frame(&session, hand_start,
               std::vector<fridge::BBox>(1, fridge::BBox(70, 70, 430, 230)),
               &frame);

    std::vector<fridge::Detection> hand_middle = hand_start;
    hand_middle[3] = det(2, 170, 100, 270, 200);
    send_frame(&session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(140, 70, 500, 230)),
               &frame);

    hand_middle[3] = det(2, 240, 105, 340, 205);
    send_frame(&session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(210, 70, 570, 230)),
               &frame);

    hand_middle[3] = det(2, 300, 111, 400, 211);
    send_frame(&session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(270, 70, 430, 240)),
               &frame);

    std::vector<fridge::Detection> final_no_hand;
    final_no_hand.push_back(det(0, 700, 100, 800, 200));
    final_no_hand.push_back(det(0, 700, 300, 800, 400));
    final_no_hand.push_back(det(2, 100, 300, 200, 400));
    final_no_hand.push_back(det(2, 300, 111, 400, 211));
    const fridge::SettlementResult result = settle_after_hand(
        &session, final_no_hand, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 5);
    assert(has_event(result, fridge::EventKind::MOVED, 9));
    assert(has_event(result, fridge::EventKind::OCCLUDED, 1));
    assert(!has_event(result, fridge::EventKind::OUT, 1));
    assert_event_before(result, fridge::EventKind::MOVED, 9,
                        fridge::EventKind::OCCLUDED, 1);
    const fridge::InventoryItem* hidden = session.inventory().find_by_item(1);
    assert(hidden != 0);
    assert(hidden->status == fridge::ItemStatus::OCCLUDED);
    assert(hidden->block_ids.size() == 1);
    assert(hidden->block_ids.count(9));
    assert(hidden->occlusion_proof.kind ==
           fridge::OcclusionProofKind::CAUSAL_FRONT_MISSING);
    assert(hidden->occlusion_proof.witness_blocker_ids.size() == 1);
    assert(hidden->occlusion_proof.witness_blocker_ids.count(9));
}

// 第一张无手帧已经建立 CAUSAL_FRONT_MISSING，但另一个独立的 post-hand D
// 还需要第二张直接无手帧才能结案。期间 #1 不能因为 #3/#5 的强原位 owner
// 暂时预约了宽松候选框，就从已确认遮挡退回 HOLD/OUT 并无限 defer-commit。
static std::vector<fridge::Detection> prepare_delayed_causal_occlusion_session(
        fridge::SessionManager* session, int* frame) {
    assert(session != 0 && frame != 0);
    session->start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 300, 100, 400, 200));  // hidden apple
    initial.push_back(item(3, 0, 700, 100, 800, 200));  // direct apple owner
    initial.push_back(item(5, 0, 700, 300, 800, 400));  // direct apple owner
    initial.push_back(item(7, 2, 100, 300, 200, 400));  // static orange
    initial.push_back(item(9, 2, 100, 100, 200, 200));  // moved front orange
    session->init_from_backend(initial, true);

    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 300, 100, 400, 200));
    stable.push_back(det(0, 700, 100, 800, 200));
    stable.push_back(det(0, 700, 300, 800, 400));
    stable.push_back(det(2, 100, 300, 200, 400));
    stable.push_back(det(2, 100, 100, 200, 200));
    initial_no_hand_frame(session, stable, frame);

    std::vector<fridge::Detection> hand_start;
    hand_start.push_back(det(0, 700, 100, 800, 200));
    hand_start.push_back(det(0, 700, 300, 800, 400));
    hand_start.push_back(det(2, 100, 300, 200, 400));
    hand_start.push_back(det(2, 100, 100, 200, 200));
    send_frame(session, hand_start,
               std::vector<fridge::BBox>(1, fridge::BBox(70, 70, 430, 230)),
               frame);

    std::vector<fridge::Detection> hand_middle = hand_start;
    hand_middle[3] = det(2, 170, 100, 270, 200);
    send_frame(session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(140, 70, 500, 230)),
               frame);
    hand_middle[3] = det(2, 240, 105, 340, 205);
    send_frame(session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(210, 70, 570, 230)),
               frame);
    hand_middle[3] = det(2, 300, 111, 400, 211);
    send_frame(session, hand_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(270, 70, 430, 240)),
               frame);

    std::vector<fridge::Detection> delayed_no_hand;
    delayed_no_hand.push_back(det(0, 700, 100, 800, 200));
    delayed_no_hand.push_back(det(0, 700, 300, 800, 400));
    delayed_no_hand.push_back(det(2, 100, 300, 200, 400));
    delayed_no_hand.push_back(det(2, 300, 111, 400, 211));
    // 无关 D 的第一张无手直接框必须先等待下一帧，故意让 #1 的遮挡
    // proof 跨过一次 deferred settlement。
    delayed_no_hand.push_back(det(25, 420, 100, 520, 200));
    return delayed_no_hand;
}

void test_provisional_causal_occlusion_survives_delayed_no_hand_settlement() {
    fridge::SessionManager session;
    int frame = 1;
    const std::vector<fridge::Detection> delayed_no_hand =
        prepare_delayed_causal_occlusion_session(&session, &frame);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        delayed_no_hand, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 1));

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        delayed_no_hand, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 6);
    assert(has_event(second.settlement, fridge::EventKind::MOVED, 9));
    assert(has_event(second.settlement, fridge::EventKind::OCCLUDED, 1));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(second.settlement, fridge::EventKind::MOVED, 7));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 7));
    assert_event_before(second.settlement, fridge::EventKind::MOVED, 9,
                        fridge::EventKind::OCCLUDED, 1);
    const fridge::InventoryItem* hidden = session.inventory().find_by_item(1);
    assert(hidden != 0);
    assert(hidden->status == fridge::ItemStatus::OCCLUDED);
    assert(hidden->block_ids.size() == 1);
    assert(hidden->block_ids.count(9));
    assert(hidden->occlusion_proof.kind ==
           fridge::OcclusionProofKind::CAUSAL_FRONT_MISSING);
    assert(hidden->occlusion_proof.witness_blocker_ids.size() == 1);
    assert(hidden->occlusion_proof.witness_blocker_ids.count(9));
}

// 临时 causal proof 不是永久忽略目标。若下一张无手帧已重新得到 #1 的
// 合法直接框，proof 必须撤销；最终应保留 VISIBLE，而不是强行提交遮挡。
void test_provisional_causal_occlusion_clears_on_target_direct_observation() {
    fridge::SessionManager session;
    int frame = 1;
    const std::vector<fridge::Detection> delayed_no_hand =
        prepare_delayed_causal_occlusion_session(&session, &frame);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        delayed_no_hand, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());

    std::vector<fridge::Detection> revealed_no_hand;
    revealed_no_hand.push_back(det(0, 300, 100, 400, 200));  // target #1 directly seen
    revealed_no_hand.insert(revealed_no_hand.end(), delayed_no_hand.begin(),
                            delayed_no_hand.end());

    fridge::SettlementResult result;
    for (int attempt = 0; attempt < 3; ++attempt) {
        const int no_hand_frame = frame++;
        fridge::FrameProcessResult next = session.process_frame(
            revealed_no_hand, std::vector<fridge::BBox>(),
            no_hand_frame, no_hand_frame);
        assert(next.no_hand_frame_processed);
        result = next.settlement;
        if (result.committed) break;
    }

    assert(result.committed);
    assert(session.inventory().size() == 6);
    assert(has_event(result, fridge::EventKind::MOVED, 9));
    assert(!has_event(result, fridge::EventKind::OCCLUDED, 1));
    assert(!has_event(result, fridge::EventKind::OUT, 1));
    const fridge::InventoryItem* target = session.inventory().find_by_item(1);
    assert(target != 0);
    assert(target->status == fridge::ItemStatus::VISIBLE);
    // 直接观察会撤销完整遮挡 proof；前景关系本身仍可保留为 VISIBLE + block_ids。
    assert(target->block_ids.count(9));
    assert(target->occlusion_proof.kind == fridge::OcclusionProofKind::NONE);
}

// 因果遮挡不是严格矩形全覆盖，但也不能退化成“任意一点交集”。这里前景
// 只覆盖目标旧框的一半，目标在有手阶段有 HAND/POSSIBLE_MOVED 线索却没有
// 自己的唯一终点；前景完成既有自匹配确认后，应采用 causal proof 而不是 OUT。
void test_causal_front_missing_accepts_substantial_partial_cover() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // moved front
    initial.push_back(item(2, 2, 300, 100, 400, 200));  // hand-affected target
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(70, 70, 430, 230)),
               &frame);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(140, 70, 500, 230)),
               &frame);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 240, 100, 340, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(210, 70, 570, 230)),
               &frame);
    const fridge::Detection final_front = det(0, 350, 100, 450, 200);
    send_frame(&session, std::vector<fridge::Detection>(1, final_front),
               std::vector<fridge::BBox>(1, fridge::BBox(270, 70, 430, 240)),
               &frame);

    fridge::FrameProcessResult first = session.process_frame(
        std::vector<fridge::Detection>(1, final_front),
        std::vector<fridge::BBox>(), frame, frame);
    ++frame;
    assert(first.no_hand_frame_processed);
    assert(first.settlement.committed);
    assert(session.inventory().size() == 2);
    assert(has_event(first.settlement, fridge::EventKind::MOVED, 1));
    assert(has_event(first.settlement, fridge::EventKind::OCCLUDED, 2));
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 2));
    assert_event_before(first.settlement, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::OCCLUDED, 2);
    const fridge::InventoryItem* target = session.inventory().find_by_item(2);
    assert(target != 0);
    assert(target->status == fridge::ItemStatus::OCCLUDED);
    assert(target->occlusion_proof.kind ==
           fridge::OcclusionProofKind::CAUSAL_FRONT_MISSING);
    assert(target->occlusion_proof.witness_blocker_ids.size() == 1);
    assert(target->occlusion_proof.witness_blocker_ids.count(1));
}

// A 移到新位置时，A 在旧位置曾被 B 遮挡这一历史关系不能自动迁移到
// 新位置。二维重叠本身不提供 B 仍在 A 前方的深度证据。
void test_moved_target_drops_stale_historical_blocker() {
    std::map<int, fridge::InventoryItem> operation_start;
    std::map<int, fridge::InventoryItem> working;
    fridge::InventoryItem moved = item(1, 0, 100, 100, 200, 200);
    fridge::InventoryItem historical_front = item(2, 2, 100, 100, 200, 200);
    moved.block_ids.insert(historical_front.item_id);
    operation_start[moved.item_id] = moved;
    operation_start[historical_front.item_id] = historical_front;

    moved.box = fridge::BBox(500, 100, 600, 200);
    moved.base_box = moved.box;
    working[moved.item_id] = moved;
    working[historical_front.item_id] = historical_front;

    std::set<int> confirmed_front_ids;
    confirmed_front_ids.insert(moved.item_id);
    std::set<int> confirmed_moved_ids;
    confirmed_moved_ids.insert(moved.item_id);
    const fridge::session_internal::BlockerRelationGraph graph =
        fridge::session_internal::build_event_driven_blocker_graph(
            operation_start, working, confirmed_front_ids, confirmed_moved_ids,
            std::set<int>());

    const std::map<int, std::set<int> >::const_iterator effective =
        graph.effective_by_target.find(moved.item_id);
    const std::map<int, std::set<int> >::const_iterator removed =
        graph.removed_by_target.find(moved.item_id);
    assert(effective != graph.effective_by_target.end());
    assert(removed != graph.removed_by_target.end());
    assert(effective->second.empty());
    assert(removed->second.count(historical_front.item_id));
}

// 正式库存中 OCCLUDED 不是一个孤立状态：它必须给出当前 blocker 图中的
// witness。失败提交不能半更新已经持久化的库存；VISIBLE 则一律清空旧 proof。
void test_inventory_rejects_invalid_formal_occlusion_proof() {
    fridge::InventoryDB inventory;
    std::map<int, fridge::InventoryItem> baseline;
    baseline[1] = item(1, 0, 100, 100, 200, 200);
    assert(inventory.replace_all(baseline, 2));

    std::map<int, fridge::InventoryItem> candidate = baseline;
    candidate[2] = item(2, 2, 100, 100, 200, 200);
    candidate[1].status = fridge::ItemStatus::OCCLUDED;
    candidate[1].block_ids.insert(2);

    // No kind and no witness is malformed; the old atomic snapshot survives.
    assert(!inventory.replace_all(candidate, 3));
    assert(inventory.size() == 1);
    assert(inventory.find_by_item(1)->status == fridge::ItemStatus::VISIBLE);

    candidate[1].occlusion_proof.kind = fridge::OcclusionProofKind::STRICT_UNION;
    candidate[1].occlusion_proof.witness_blocker_ids.insert(99);
    assert(!inventory.replace_all(candidate, 3));
    assert(inventory.size() == 1);

    candidate[1].block_ids.insert(99);
    assert(!inventory.replace_all(candidate, 3));
    assert(inventory.size() == 1);
    candidate[1].block_ids.erase(99);

    candidate[1].occlusion_proof.witness_blocker_ids.clear();
    candidate[1].occlusion_proof.witness_blocker_ids.insert(2);
    // VISIBLE proof must be normalized away even if a caller accidentally
    // carries an old proof alongside an ordinary visible item.
    candidate[2].occlusion_proof.kind = fridge::OcclusionProofKind::STRICT_UNION;
    candidate[2].occlusion_proof.witness_blocker_ids.insert(1);
    assert(inventory.replace_all(candidate, 3));
    assert(inventory.find_by_item(1)->status == fridge::ItemStatus::OCCLUDED);
    assert(inventory.find_by_item(1)->occlusion_proof.witness_blocker_ids.count(2));
    assert(inventory.find_by_item(2)->occlusion_proof.kind ==
           fridge::OcclusionProofKind::NONE);
    assert(inventory.find_by_item(2)->occlusion_proof.witness_blocker_ids.empty());
}

// 多层 blocker 的 OUT 求值只能收缩候选集合。这里模拟第一轮中 B 已被
// 确认遮挡保护，下一轮中 C 又因 B 的保留关系进入 pending；无论候选集合
// 的遍历顺序如何，最终都只能留下仍具有普通 OUT 证据的 A。
void test_pending_out_candidates_converge_monotonically() {
    std::set<int> candidates;
    candidates.insert(1);  // A: ordinary OUT remains legal.
    candidates.insert(2);  // B: current confirmed occlusion protects it.
    candidates.insert(3);  // C: becomes pending after the first replan.

    std::map<int, fridge::session_internal::BlockerTransitionPlan> first_plan;
    first_plan[1].out = fridge::session_internal::OutDisposition::NORMAL_OUT_EVIDENCE;
    first_plan[2].out =
        fridge::session_internal::OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION;
    first_plan[3].out = fridge::session_internal::OutDisposition::NORMAL_OUT_EVIDENCE;
    const std::set<int> after_first =
        fridge::session_internal::retain_pending_out_candidates(candidates, first_plan);
    assert(after_first.size() == 2);
    assert(after_first.count(1));
    assert(after_first.count(3));

    std::map<int, fridge::session_internal::BlockerTransitionPlan> stable_plan;
    stable_plan[1].out = fridge::session_internal::OutDisposition::NORMAL_OUT_EVIDENCE;
    stable_plan[3].out =
        fridge::session_internal::OutDisposition::HOLD_FOR_PENDING_OCCLUSION;
    const std::set<int> after_second =
        fridge::session_internal::retain_pending_out_candidates(after_first, stable_plan);
    const std::set<int> after_stable =
        fridge::session_internal::retain_pending_out_candidates(after_second, stable_plan);
    assert(after_second.size() == 1);
    assert(after_second.count(1));
    assert(after_stable == after_second);
}

void test_moved_front_item_occludes_then_reveals() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // A: apple
    initial.push_back(item(2, 2, 300, 100, 400, 200));  // B: orange
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // A 移到 B 前面，当前无手帧只看得到 A。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 340, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    fridge::SettlementResult first = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)), &frame);
    assert(first.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::OCCLUDED);
    assert(session.inventory().find_by_item(2)->block_ids.count(1));
    assert_event_before(first, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::OCCLUDED, 2);

    // 下一次操作中 A 仍停在 B 前方，且本轮没有新的 MOVED / IN。历史前景关系
    // 不能因为当前帧没有新的 confirmed_front_ids 而被清空，也不能重复产生事件。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    fridge::SettlementResult still_blocked = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)), &frame);
    assert(still_blocked.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::OCCLUDED);
    assert(session.inventory().find_by_item(2)->block_ids.count(1));
    assert(!has_event(still_blocked, fridge::EventKind::OCCLUDED, 2));
    assert(!has_event(still_blocked, fridge::EventKind::REVEALED, 2));

    // 再把 A 移开；A 和 B 同时可见，B 必须恢复 item#2 而不是新入库。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 230, 100, 330, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(210, 80, 310, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 160, 100, 260, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(140, 80, 240, 220)), &frame);
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 300, 100, 400, 200));
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    fridge::SettlementResult second = settle_after_hand(&session, stable, &frame);
    assert(second.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::VISIBLE);
    assert(has_event(second, fridge::EventKind::REVEALED, 2));
    assert_event_before(second, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::REVEALED, 2);
}

// A 11px top-edge residual is wider than the existing 8px edge-residual
// tolerance, but the confirmed moved front covers 89% of the target.  Two
// comparable no-hand frames must use the dedicated disappearance proof rather
// than allowing ordinary two-frame OUT to delete the target.
void test_disappearance_supported_moved_front_occludes_without_out() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // moved front apple
    initial.push_back(item(2, 2, 300, 100, 400, 200));  // hidden orange
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 240, 105, 340, 205)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 85, 320, 225)), &frame);
    const fridge::BBox final_front(300, 111, 400, 211);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, final_front.x1, final_front.y1,
                   final_front.x2, final_front.y2)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 91, 380, 231)), &frame);

    const std::vector<fridge::Detection> no_hand_front(1,
        det(0, final_front.x1, final_front.y1,
            final_front.x2, final_front.y2));
    fridge::SettlementResult result = settle_after_hand(
        &session, no_hand_front, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 2);
    const fridge::InventoryItem* target = session.inventory().find_by_item(2);
    assert(target != 0);
    assert(target->status == fridge::ItemStatus::OCCLUDED);
    assert(target->block_ids.size() == 1);
    assert(target->block_ids.count(1));
    assert(target->occlusion_proof.kind ==
           fridge::OcclusionProofKind::DISAPPEARANCE_SUPPORTED);
    assert(target->occlusion_proof.witness_blocker_ids.size() == 1);
    assert(target->occlusion_proof.witness_blocker_ids.count(1));
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(has_event(result, fridge::EventKind::OCCLUDED, 2));
    assert(!has_event(result, fridge::EventKind::OUT, 2));
    assert_event_before(result, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::OCCLUDED, 2);
}

// The target is HAND_FULL/POSSIBLE_MOVED during the same operation, but never
// gets a unique endpoint.  A confirmed moved front covers 89% of its old box;
// the 11px top residual intentionally fails the existing 8px edge proof.  The
// causal front-missing proof must still settle the target without OUT.
void test_causal_front_missing_releases_hand_target_without_out() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // moved front
    initial.push_back(item(2, 2, 300, 100, 400, 200));  // hand-affected target
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)),
               &frame);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)),
               &frame);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 240, 105, 340, 205)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 340, 240)),
               &frame);

    const fridge::BBox final_front(300, 111, 400, 211);
    // The hand fully covers the target's old box in the last HAND frame while
    // the front object's own endpoint remains directly observable.
    send_frame(&session,
               std::vector<fridge::Detection>(1,
                   det(0, final_front.x1, final_front.y1,
                       final_front.x2, final_front.y2)),
               std::vector<fridge::BBox>(1, fridge::BBox(270, 70, 430, 240)),
               &frame);

    const fridge::SettlementResult result = settle_after_hand(
        &session,
        std::vector<fridge::Detection>(1,
            det(0, final_front.x1, final_front.y1,
                final_front.x2, final_front.y2)),
        &frame);
    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(has_event(result, fridge::EventKind::OCCLUDED, 2));
    assert(!has_event(result, fridge::EventKind::OUT, 2));
    assert_event_before(result, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::OCCLUDED, 2);

    const fridge::InventoryItem* target = session.inventory().find_by_item(2);
    assert(target != 0);
    assert(target->status == fridge::ItemStatus::OCCLUDED);
    assert(target->block_ids.size() == 1);
    assert(target->block_ids.count(1));
    assert(target->occlusion_proof.kind ==
           fridge::OcclusionProofKind::CAUSAL_FRONT_MISSING);
    assert(target->occlusion_proof.witness_blocker_ids.size() == 1);
    assert(target->occlusion_proof.witness_blocker_ids.count(1));
    assert(target->base_box.x1 == 300.0f && target->base_box.y1 == 100.0f &&
           target->base_box.x2 == 400.0f && target->base_box.y2 == 200.0f);
}

// DISAPPEARANCE_SUPPORTED 需要连续无手帧。中间重新出现手时，前后两段
// 候选不能拼成两帧，否则一次短暂漏检会被错误提交为正式 OCCLUDED。
void test_disappearance_evidence_resets_when_hand_returns() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 300, 100, 400, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, 240, 105, 340, 205)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 85, 320, 225)), &frame);
    const fridge::BBox final_front(300, 111, 400, 211);
    send_frame(&session, std::vector<fridge::Detection>(1,
               det(0, final_front.x1, final_front.y1,
                   final_front.x2, final_front.y2)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 91, 380, 231)), &frame);

    const std::vector<fridge::Detection> no_hand_front(1,
        det(0, final_front.x1, final_front.y1,
            final_front.x2, final_front.y2));
    fridge::FrameProcessResult first = session.process_frame(
        no_hand_front, std::vector<fridge::BBox>(), frame, frame);
    ++frame;
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::VISIBLE);

    // This hand frame is intentionally stationary at the last hand position;
    // it only breaks the no-hand evidence segment.
    send_frame(&session, no_hand_front,
               std::vector<fridge::BBox>(1, fridge::BBox(280, 91, 380, 231)), &frame);
    fridge::FrameProcessResult after_hand = session.process_frame(
        no_hand_front, std::vector<fridge::BBox>(), frame, frame);
    ++frame;
    assert(after_hand.no_hand_frame_processed);
    assert(!after_hand.settlement.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::VISIBLE);

    fridge::SettlementResult result = settle_after_hand(
        &session, no_hand_front, &frame);
    assert(result.committed);
    const fridge::InventoryItem* target = session.inventory().find_by_item(2);
    assert(target != 0);
    assert(target->status == fridge::ItemStatus::OCCLUDED);
    assert(target->occlusion_proof.kind ==
           fridge::OcclusionProofKind::DISAPPEARANCE_SUPPORTED);
    assert(has_event(result, fridge::EventKind::OCCLUDED, 2));
    assert(!has_event(result, fridge::EventKind::OUT, 2));
}

// 第一事务中 A 的确认终点在 B 的左侧留下 6px 外边缘残余。当前 confirmed
// MOVED front 可以按既有 edge-residual 规则使 B 正式 OCCLUDED；下一事务
// A 离开后，历史 before 也必须用同一正式语义，否则 B 会残留 OCCLUDED。
void test_edge_residual_historical_blocker_reveals_after_move() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // A: apple
    initial.push_back(item(2, 2, 300, 100, 400, 200));  // B: orange
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // A 的最终框 (306,100)-(406,200) 在 B 的左侧留下 6px 残余，严格覆盖
    // 失败但仍满足 FLOW3_CONFIRMED_OCCLUSION_EDGE_RESIDUAL_PX。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 340, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 306, 100, 406, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(286, 80, 386, 220)), &frame);
    fridge::SettlementResult occluded = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 306, 100, 406, 200)), &frame);
    assert(occluded.committed);
    const fridge::InventoryItem* hidden = session.inventory().find_by_item(2);
    assert(hidden != 0);
    assert(hidden->status == fridge::ItemStatus::OCCLUDED);
    assert(hidden->block_ids.size() == 1);
    assert(hidden->block_ids.count(1));
    assert(has_event(occluded, fridge::EventKind::MOVED, 1));
    assert(has_event(occluded, fridge::EventKind::OCCLUDED, 2));
    assert_event_before(occluded, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::OCCLUDED, 2);

    // 下一事务 A 确认移到 B 外，B 同时有唯一且合法的直接框。旧实现的 before
    // 只做严格覆盖，会错误得到 false -> false，因而不允许 REVEALED。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 306, 100, 406, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(286, 80, 386, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 380, 100, 480, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(360, 80, 460, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 460, 100, 560, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(440, 80, 540, 220)), &frame);
    std::vector<fridge::Detection> final_frame;
    final_frame.push_back(det(0, 500, 100, 600, 200));
    final_frame.push_back(det(2, 300, 100, 400, 200));
    send_frame(&session, final_frame,
               std::vector<fridge::BBox>(1, fridge::BBox(480, 80, 580, 220)), &frame);
    fridge::SettlementResult revealed = settle_after_hand(&session, final_frame, &frame);
    assert(revealed.committed);
    const fridge::InventoryItem* visible = session.inventory().find_by_item(2);
    assert(visible != 0);
    assert(visible->status == fridge::ItemStatus::VISIBLE);
    assert(visible->block_ids.empty());
    assert(session.inventory().size() == 2);
    assert(has_event(revealed, fridge::EventKind::MOVED, 1));
    assert(has_event(revealed, fridge::EventKind::REVEALED, 2));
    assert(!has_event(revealed, fridge::EventKind::IN, 3));
    assert_event_before(revealed, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::REVEALED, 2);
}

// 现场密集拓扑回放：三个苹果与一个橙子同时存在。item#2 先以边缘残余
// 正式遮挡 item#4；item#1 与 item#3 始终可见，其中 item#3 靠近橙子但从未
// 成为确认前景。item#2 随后移走时，不能因同类苹果密度而丢失其 MOVED 身份，
// 也不能把最终三张苹果框中的任何一张误建成新的 D/IN。
void test_dense_three_apple_topology_reveals_edge_residual_target() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 550, 300, 650, 400));  // 静态 apple
    initial.push_back(item(2, 0, 100, 100, 200, 200));  // 历史 blocker apple
    initial.push_back(item(3, 0, 410, 100, 510, 200));  // 静态 apple，紧邻 orange
    initial.push_back(item(4, 2, 300, 100, 400, 200));  // 被遮挡 orange
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 550, 300, 650, 400));
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(0, 410, 100, 510, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // 第一事务：item#2 移至 item#4 前方。最终保留 6px 左边缘残余，因此
    // item#4 的正式遮挡只能由既有 edge-residual 语义建立；两张静态苹果框
    // 全程可见，避免把密集货架简化为单物品路径。
    std::vector<fridge::Detection> first_hand_frame;
    first_hand_frame.push_back(det(0, 550, 300, 650, 400));
    first_hand_frame.push_back(det(0, 100, 100, 200, 200));
    first_hand_frame.push_back(det(0, 410, 100, 510, 200));
    send_frame(&session, first_hand_frame,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);

    std::vector<fridge::Detection> first_middle;
    first_middle.push_back(det(0, 550, 300, 650, 400));
    first_middle.push_back(det(0, 170, 100, 270, 200));
    first_middle.push_back(det(0, 410, 100, 510, 200));
    send_frame(&session, first_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);

    std::vector<fridge::Detection> first_near_target;
    first_near_target.push_back(det(0, 550, 300, 650, 400));
    first_near_target.push_back(det(0, 240, 100, 340, 200));
    first_near_target.push_back(det(0, 410, 100, 510, 200));
    send_frame(&session, first_near_target,
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);

    std::vector<fridge::Detection> first_final;
    first_final.push_back(det(0, 550, 300, 650, 400));
    first_final.push_back(det(0, 306, 100, 406, 200));
    first_final.push_back(det(0, 410, 100, 510, 200));
    send_frame(&session, first_final,
               std::vector<fridge::BBox>(1, fridge::BBox(286, 80, 386, 220)), &frame);
    fridge::SettlementResult occluded = settle_after_hand(&session, first_final, &frame);
    assert(occluded.committed);
    assert(session.inventory().size() == 4);
    const fridge::InventoryItem* hidden = session.inventory().find_by_item(4);
    assert(hidden != 0);
    assert(hidden->status == fridge::ItemStatus::OCCLUDED);
    assert(hidden->block_ids.size() == 1);
    assert(hidden->block_ids.count(2));
    assert(has_event(occluded, fridge::EventKind::MOVED, 2));
    assert(has_event(occluded, fridge::EventKind::OCCLUDED, 4));
    assert_event_before(occluded, fridge::EventKind::MOVED, 2,
                        fridge::EventKind::OCCLUDED, 4);

    // 第二事务：blocker 离开 B。最终无手帧同时有三个 apple 和一个 orange，
    // 要求维持 item#2 的真实终点身份，并由已确认 blocker 移走这一因果关系
    // 使 item#4 REVEALED。
    send_frame(&session, first_final,
               std::vector<fridge::BBox>(1, fridge::BBox(286, 80, 386, 220)), &frame);

    std::vector<fridge::Detection> second_middle;
    second_middle.push_back(det(0, 550, 300, 650, 400));
    second_middle.push_back(det(0, 250, 160, 350, 260));
    second_middle.push_back(det(0, 410, 100, 510, 200));
    send_frame(&session, second_middle,
               std::vector<fridge::BBox>(1, fridge::BBox(230, 140, 370, 280)), &frame);

    std::vector<fridge::Detection> second_near_destination;
    second_near_destination.push_back(det(0, 550, 300, 650, 400));
    second_near_destination.push_back(det(0, 180, 230, 280, 330));
    second_near_destination.push_back(det(0, 410, 100, 510, 200));
    send_frame(&session, second_near_destination,
               std::vector<fridge::BBox>(1, fridge::BBox(160, 210, 300, 350)), &frame);

    std::vector<fridge::Detection> final_frame;
    final_frame.push_back(det(0, 550, 300, 650, 400));
    final_frame.push_back(det(0, 100, 300, 200, 400));
    final_frame.push_back(det(0, 410, 100, 510, 200));
    final_frame.push_back(det(2, 300, 100, 400, 200));
    send_frame(&session, final_frame,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 280, 220, 420)), &frame);
    fridge::SettlementResult revealed = settle_after_hand(&session, final_frame, &frame);
    assert(revealed.committed);
    assert(session.inventory().size() == 4);
    assert(revealed.events.size() == 2);

    const fridge::InventoryItem* static_first = session.inventory().find_by_item(1);
    const fridge::InventoryItem* moved = session.inventory().find_by_item(2);
    const fridge::InventoryItem* static_near_target = session.inventory().find_by_item(3);
    const fridge::InventoryItem* visible = session.inventory().find_by_item(4);
    assert(static_first != 0 && moved != 0 && static_near_target != 0 && visible != 0);
    assert(static_first->status == fridge::ItemStatus::VISIBLE);
    assert(moved->status == fridge::ItemStatus::VISIBLE);
    assert(static_near_target->status == fridge::ItemStatus::VISIBLE);
    assert(visible->status == fridge::ItemStatus::VISIBLE);
    assert(static_first->base_box.x1 == 550.0f && static_first->base_box.y1 == 300.0f);
    assert(static_first->base_box.x2 == 650.0f && static_first->base_box.y2 == 400.0f);
    assert(moved->base_box.x1 == 100.0f && moved->base_box.y1 == 300.0f);
    assert(moved->base_box.x2 == 200.0f && moved->base_box.y2 == 400.0f);
    assert(static_near_target->base_box.x1 == 410.0f &&
           static_near_target->base_box.y1 == 100.0f);
    assert(static_near_target->base_box.x2 == 510.0f &&
           static_near_target->base_box.y2 == 200.0f);
    assert(visible->base_box.x1 == 300.0f && visible->base_box.y1 == 100.0f);
    assert(visible->base_box.x2 == 400.0f && visible->base_box.y2 == 200.0f);
    assert(static_first->block_ids.empty());
    assert(moved->block_ids.empty());
    assert(static_near_target->block_ids.empty());
    assert(visible->block_ids.empty());
    assert(has_event(revealed, fridge::EventKind::MOVED, 2));
    assert(has_event(revealed, fridge::EventKind::REVEALED, 4));
    for (size_t event = 0; event < revealed.events.size(); ++event) {
        assert(revealed.events[event].kind != fridge::EventKind::IN);
    }
    assert_event_before(revealed, fridge::EventKind::MOVED, 2,
                        fridge::EventKind::REVEALED, 4);
}

// 目标 C 已经正式 OCCLUDED，但历史 blocker 在本轮没有任何确认移动或
// OUT。当前检测器再次给出 C 的直接框只能是临时 observation，不能凭它
// 直接把正式状态改成 VISIBLE 或生成 REVEALED。
void test_occluded_target_observation_without_blocker_change_stays_occluded() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 300, 100, 400, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // 第一轮建立 A -> C 的正式完整遮挡关系。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 340, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    fridge::SettlementResult occluded = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)), &frame);
    assert(occluded.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::OCCLUDED);
    assert(session.inventory().find_by_item(2)->block_ids.count(1));

    // 无关操作中 A 保持原位置，C 又出现一张直接框。没有 blocker 变化，
    // 因此本轮仍提交 OCCLUDED，保留历史关系且没有 REVEALED。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    std::vector<fridge::Detection> direct_c;
    direct_c.push_back(det(0, 300, 100, 400, 200));
    direct_c.push_back(det(2, 300, 100, 400, 200));
    fridge::SettlementResult stable = settle_after_hand(&session, direct_c, &frame);
    assert(stable.committed);
    const fridge::InventoryItem* c = session.inventory().find_by_item(2);
    assert(c != 0);
    assert(c->status == fridge::ItemStatus::OCCLUDED);
    assert(c->block_ids.count(1));
    assert(!has_event(stable, fridge::EventKind::REVEALED, 2));
}

// 历史 blocker A 只有正式 OUT 后才可解除 C 的完整遮挡；C 还必须在同一
// 无手事务中有合法直接框。事件必须保持 OUT(A) 在 REVEALED(C) 之前。
void test_out_blocker_reveals_observed_target_in_causal_order() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 300, 100, 400, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // 先用既有路径建立 A 完整遮挡 C 的正式关系。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 340, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    fridge::SettlementResult occluded = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)), &frame);
    assert(occluded.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::OCCLUDED);
    assert(session.inventory().find_by_item(2)->block_ids.count(1));

    // A 被取走，C 从第一张无手帧起重新可见。A 的普通 OUT 仍须满足既有
    // 连续缺失门槛；达到门槛的同一事务才能解除遮挡。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 420, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(420, 80, 560, 220)), &frame);
    const std::vector<fridge::Detection> direct_c(
        1, det(2, 300, 100, 400, 200));
    fridge::SettlementResult result = settle_after_hand(&session, direct_c, &frame);
    assert(result.committed);
    assert(session.inventory().find_by_item(1) == 0);
    const fridge::InventoryItem* c = session.inventory().find_by_item(2);
    assert(c != 0);
    assert(c->status == fridge::ItemStatus::VISIBLE);
    assert(c->block_ids.empty());
    assert_event_before(result, fridge::EventKind::OUT, 1,
                        fridge::EventKind::REVEALED, 2);
}

// 低分跨类别近同框只能降低 A 原位候选的身份权威。A 已有真实移动路径时，
// 真实终点必须继续认领 A；低分框不能把 A 静态结算，也不能制造新的 D/IN。
void test_cross_class_duplicate_original_does_not_create_d() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 100, 100, 200, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // 模拟上一事务已经确认的历史完整遮挡关系。它是测试正式遮挡生命周期
    // 的边界，不把后端初始化过程当作本轮 blocker 事件。
    fridge::InventoryItem* hidden_target = session.inventory().find_by_item(2);
    assert(hidden_target != 0);
    hidden_target->status = fridge::ItemStatus::OCCLUDED;
    hidden_target->block_ids.insert(1);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::OCCLUDED);
    assert(session.inventory().find_by_item(2)->block_ids.count(1));

    // A 从 blocker 原位移走。橙子高分框和一个几乎同框的低分苹果
    // 同时出现；低分苹果正好会命中 A 的 operation-start original_box。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(100, 80, 200, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 180, 100, 280, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(160, 80, 260, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 340, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);

    std::vector<fridge::Detection> final_frame;
    final_frame.push_back(det(0, 300, 100, 400, 200, 0.90f));
    final_frame.push_back(det(2, 100, 100, 200, 200, 0.70f));
    final_frame.push_back(det(0, 100, 100, 200, 200, 0.30f));
    fridge::SettlementResult result = settle_after_hand(&session, final_frame, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(has_event(result, fridge::EventKind::REVEALED, 2));
    assert(!has_event(result, fridge::EventKind::IN, 3));
    int moved_index = -1;
    int revealed_index = -1;
    for (size_t i = 0; i < result.events.size(); ++i) {
        if (result.events[i].kind == fridge::EventKind::MOVED &&
            result.events[i].item_id == 1) moved_index = static_cast<int>(i);
        if (result.events[i].kind == fridge::EventKind::REVEALED &&
            result.events[i].item_id == 2) revealed_index = static_cast<int>(i);
    }
    assert(moved_index >= 0 && revealed_index >= 0 && moved_index < revealed_index);
}

// 历史 blocker 正式 OUT 后，原先被完整遮挡的 C 不能立即 REVEALED，也不能
// 带着遮挡期间的缺失帧直接 OUT。C 从 blocker 失效后的第一张无手帧重新开始
// 计数，在既有阈值达到前持续阻止提交。
void test_historical_blocker_out_starts_new_missing_chain() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 300, 100, 400, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // 先用正式流程建立 item#1 -> item#2 的历史完整遮挡关系。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 170, 100, 270, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(150, 80, 250, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 340, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    fridge::SettlementResult occluded = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)), &frame);
    assert(occluded.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::OCCLUDED);
    assert(session.inventory().find_by_item(2)->block_ids.count(1));

    // 再取走前景 item#1。item#2 已是 OCCLUDED，因此不会被新的 HAND 轨迹
    // 误当作直接可见物品；它只能在 blocker 正式 OUT 后走新的缺失链。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 420, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(420, 80, 560, 220)), &frame);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());

    // item#1 在这一帧完成既有 OUT；item#2 才开始自己的“遮挡解释失效”第 1 帧。
    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(!second.settlement.committed);
    assert(session.operation_pending());
    assert(!has_event(second.settlement, fridge::EventKind::REVEALED, 2));

    const int third_no_hand = frame++;
    fridge::FrameProcessResult third = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        third_no_hand, third_no_hand);
    assert(third.no_hand_frame_processed);
    assert(third.settlement.committed);
    assert(session.inventory().find_by_item(1) == 0);
    assert(session.inventory().find_by_item(2) == 0);
    assert(has_event(third.settlement, fridge::EventKind::OUT, 1));
    assert(has_event(third.settlement, fridge::EventKind::OUT, 2));
    assert(!has_event(third.settlement, fridge::EventKind::REVEALED, 2));
}

// VISIBLE + block_ids 只代表局部前景关系，不是完整遮挡。当前景 A 随后 OUT、
// B 又恰好漏检时，B 不能走“历史完整遮挡失效”的专用 OUT 链，更不能被 A
// 的 OUT 连带删除；没有 B 自己的旧 C 证据时应保留正式库存。
void test_partial_front_relation_does_not_start_occlusion_loss_out_chain() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 260, 100, 360, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> initial_foods;
    initial_foods.push_back(det(0, 100, 100, 200, 200));
    initial_foods.push_back(det(2, 260, 100, 360, 200));
    initial_no_hand_frame(&session, initial_foods, &frame);

    // A 确认移动到 B 的左侧，只与 B 有部分重叠；B 仍被直接看见，因此
    // 提交后必须是 VISIBLE，但保存 A 的前景关系。
    send_frame(&session, initial_foods,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    std::vector<fridge::Detection> move_1;
    move_1.push_back(det(0, 130, 100, 230, 200));
    move_1.push_back(det(2, 260, 100, 360, 200));
    send_frame(&session, move_1,
               std::vector<fridge::BBox>(1, fridge::BBox(110, 80, 170, 220)), &frame);
    std::vector<fridge::Detection> move_2;
    move_2.push_back(det(0, 160, 100, 260, 200));
    move_2.push_back(det(2, 260, 100, 360, 200));
    send_frame(&session, move_2,
               std::vector<fridge::BBox>(1, fridge::BBox(140, 80, 200, 220)), &frame);
    std::vector<fridge::Detection> partial_front;
    partial_front.push_back(det(0, 180, 100, 280, 200));
    partial_front.push_back(det(2, 260, 100, 360, 200));
    send_frame(&session, partial_front,
               std::vector<fridge::BBox>(1, fridge::BBox(160, 80, 220, 220)), &frame);
    fridge::SettlementResult first = settle_after_hand(&session, partial_front, &frame);
    assert(first.committed);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::VISIBLE);
    assert(session.inventory().find_by_item(2)->block_ids.count(1));

    // 只操作 A，B 没有自己的 HAND/CONTACT 轨迹。A 消失并完成正常 OUT 后，
    // B 的短暂漏检不得被误当作“完整遮挡失效后的连续缺失”。
    send_frame(&session, partial_front,
               std::vector<fridge::BBox>(1, fridge::BBox(170, 80, 230, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 150, 100, 250, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(140, 80, 200, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 120, 100, 220, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(110, 80, 170, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 90, 100, 190, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 140, 220)), &frame);
    fridge::SettlementResult second = settle_after_hand(
        &session, std::vector<fridge::Detection>(), &frame);
    assert(second.committed);
    assert(session.inventory().find_by_item(1) == 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::VISIBLE);
    assert(!has_event(second, fridge::EventKind::OUT, 2));
}

// 多 blocker 的覆盖关系必须按最终位置一起重算。A 移开并不等于 C 露出：
// 只要历史 B 仍完整覆盖 C，C 必须继续 OCCLUDED，且关系中只移除 A。
void test_remaining_historical_blocker_keeps_c_occluded_after_other_moves() {
    fridge::SessionManager session;
    session.start_new_session();

    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 300, 100, 400, 200));
    initial.push_back(item(2, 0, 300, 100, 400, 200));
    initial.push_back(item(3, 1, 300, 100, 400, 200));
    session.init_from_backend(initial, true);

    // init_from_backend() 有意将后端快照标准化为 VISIBLE 并清空 blocker。
    // 这里模拟的是前一轮已经由本地事务提交的遮挡关系，因此在加载完成后
    // 恢复该持久状态，避免把“后端初始化边界”误测成 blocker 生命周期。
    fridge::InventoryItem* hidden = session.inventory().find_by_item(1);
    assert(hidden != 0);
    hidden->status = fridge::ItemStatus::OCCLUDED;
    hidden->block_ids.insert(2);
    hidden->block_ids.insert(3);
    int frame = 1;

    std::vector<fridge::Detection> original_fronts;
    original_fronts.push_back(det(0, 300, 100, 400, 200));
    original_fronts.push_back(det(1, 300, 100, 400, 200));
    send_frame(&session, original_fronts,
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);

    std::vector<fridge::Detection> move_1;
    move_1.push_back(det(0, 230, 100, 330, 200));
    move_1.push_back(det(1, 300, 100, 400, 200));
    send_frame(&session, move_1,
               std::vector<fridge::BBox>(1, fridge::BBox(210, 80, 310, 220)), &frame);
    std::vector<fridge::Detection> move_2;
    move_2.push_back(det(0, 160, 100, 260, 200));
    move_2.push_back(det(1, 300, 100, 400, 200));
    send_frame(&session, move_2,
               std::vector<fridge::BBox>(1, fridge::BBox(140, 80, 240, 220)), &frame);
    std::vector<fridge::Detection> final_fronts;
    final_fronts.push_back(det(0, 100, 100, 200, 200));
    final_fronts.push_back(det(1, 300, 100, 400, 200));
    send_frame(&session, final_fronts,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);

    fridge::SettlementResult result = settle_after_hand(&session, final_fronts, &frame);
    assert(result.committed);
    const fridge::InventoryItem* still_hidden = session.inventory().find_by_item(1);
    assert(still_hidden != 0);
    assert(still_hidden->status == fridge::ItemStatus::OCCLUDED);
    assert(!still_hidden->block_ids.count(2));
    assert(still_hidden->block_ids.count(3));
    assert(has_event(result, fridge::EventKind::MOVED, 2));
    assert(!has_event(result, fridge::EventKind::REVEALED, 1));
}

// POST_HAND_REVEAL_D 的第一张无手直接帧只能创建候选；第二张能连续匹配
// 到同一 D 后，才允许提交 IN。这里刻意逐张断言，防止恢复快照投票语义。
void test_post_hand_reveal_commits_on_second_direct_frame() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(260, 80, 360, 220)), &frame);

    const std::vector<fridge::Detection> revealed(
        1, det(0, 170, 100, 290, 220));
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        revealed, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().size() == 0);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        revealed, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(second.settlement, fridge::EventKind::IN, 1));
}

// C 在手离开后第一次重新出现，只能先成为 reappear_candidate。下一张直接
// 无手帧自匹配成功后，才可把它确认成 MOVED。
void test_c_reappear_commits_after_second_direct_frame() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    // C 被完整遮住后随手移动，但有手阶段没有可用的真实 B。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 360, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(340, 80, 480, 220)), &frame);

    const std::vector<fridge::Detection> reappeared(
        1, det(0, 240, 100, 340, 200));
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        reappeared, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(!has_event(first.settlement, fridge::EventKind::MOVED, 1));

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        reappeared, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(second.settlement, fridge::EventKind::MOVED, 1));
}

// 细节10 + 细节15：A 在有手阶段暂时看不见时，静止 B/C 的直接原位框现在
// 必须先被原位所有权计划保留，A 不应再把 C 写进 reappear_candidate。无手
// 阶段仍要正常找到 A 的真实终点，不能抢 B/C、误 OUT 或误建 D。
void test_stale_same_class_reappear_candidate_falls_back_to_moved_a() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 800, 100, 900, 200));  // A：右侧，被移动
    initial.push_back(item(2, 0, 300, 100, 400, 200));  // B：静止
    initial.push_back(item(3, 0, 500, 100, 600, 200));  // C：静止，但误入 A 候选
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 800, 100, 900, 200));
    stable.push_back(det(0, 300, 100, 400, 200));
    stable.push_back(det(0, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::Detection> b_and_c;
    b_and_c.push_back(det(0, 300, 100, 400, 200));
    b_and_c.push_back(det(0, 500, 100, 600, 200));

    // t0：手完全遮住 A。t1：手路过 C；细节15要求 C 的原位框不能再被
    // A 的宽松路径借走。
    send_frame(&session, b_and_c,
               std::vector<fridge::BBox>(1, fridge::BBox(780, 80, 920, 220)), &frame);
    send_frame(&session, b_and_c,
               std::vector<fridge::BBox>(1, fridge::BBox(480, 80, 620, 220)), &frame);
    // t2：手继续到 A 的真正放下位置；C 仍会被旧候选自匹配一次。
    send_frame(&session, b_and_c,
               std::vector<fridge::BBox>(1, fridge::BBox(180, 330, 320, 470)), &frame);

    const std::map<int, fridge::OperationTrack>& before_no_hand =
        session.operation_tracks();
    assert(before_no_hand.find(1) != before_no_hand.end());
    assert(!before_no_hand.find(1)->second.has_reappear_candidate_box);

    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(det(0, 200, 350, 300, 450));  // A 的真正新位置
    final_boxes.push_back(det(0, 300, 100, 400, 200));  // B 原位
    final_boxes.push_back(det(0, 500, 100, 600, 200));  // C 原位

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(!has_event(first.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(first.settlement, fridge::EventKind::IN, 4));
    const std::map<int, fridge::OperationTrack>& after_first =
        session.operation_tracks();
    assert(after_first.find(1) != after_first.end());
    assert(!after_first.find(1)->second.no_hand_candidate_ambiguous);
    assert(after_first.find(1)->second.has_reappear_candidate_box);
    assert(after_first.find(1)->second.reappear_candidate_box.x1 == 200.0f);
    assert(session.inventory().size() == 3);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 3);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(session.inventory().find_by_item(3) != 0);
    assert(has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(second.settlement, fridge::EventKind::MOVED, 2));
    assert(!has_event(second.settlement, fridge::EventKind::MOVED, 3));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 2));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 3));
    assert(!has_event(second.settlement, fridge::EventKind::IN, 4));
}

// 细节10-7.2 + 细节15：静止 B/C 已在有手帧直接保留，A 不应先保存错误
// 候选。若无手后仍完全没有 A，只能等待正常连续 OUT 证据，不能把 B/C
// 交给 A、单帧 OUT，或借剩余框创建同类 D。
void test_stale_same_class_reappear_candidate_without_a_waits_then_out() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 800, 100, 900, 200));  // A：被取出
    initial.push_back(item(2, 0, 300, 100, 400, 200));  // B：静止
    initial.push_back(item(3, 0, 500, 100, 600, 200));  // C：静止，误入 A 候选
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 800, 100, 900, 200));
    stable.push_back(det(0, 300, 100, 400, 200));
    stable.push_back(det(0, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::Detection> b_and_c;
    b_and_c.push_back(det(0, 300, 100, 400, 200));
    b_and_c.push_back(det(0, 500, 100, 600, 200));
    send_frame(&session, b_and_c,
               std::vector<fridge::BBox>(1, fridge::BBox(780, 80, 920, 220)), &frame);
    send_frame(&session, b_and_c,
               std::vector<fridge::BBox>(1, fridge::BBox(480, 80, 620, 220)), &frame);
    send_frame(&session, b_and_c,
               std::vector<fridge::BBox>(1, fridge::BBox(180, 330, 320, 470)), &frame);

    const std::map<int, fridge::OperationTrack>& before_no_hand =
        session.operation_tracks();
    assert(before_no_hand.find(1) != before_no_hand.end());
    assert(!before_no_hand.find(1)->second.has_reappear_candidate_box);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        b_and_c, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().size() == 3);
    assert(!has_event(first.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(first.settlement, fridge::EventKind::IN, 4));
    const std::map<int, fridge::OperationTrack>& after_first =
        session.operation_tracks();
    assert(after_first.find(1) != after_first.end());
    assert(!after_first.find(1)->second.no_hand_candidate_ambiguous);
    assert(!after_first.find(1)->second.has_reappear_candidate_box);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        b_and_c, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) == 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(session.inventory().find_by_item(3) != 0);
    assert(!has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(has_event(second.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 2));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 3));
    assert(!has_event(second.settlement, fridge::EventKind::IN, 4));
}

// OUT 同样使用连续的无手直接缺失证据：首张缺失保留工作库存，第二张才
// 可以结束这次操作并提交 OUT。
void test_out_requires_two_direct_missing_frames() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 360, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(340, 80, 480, 220)), &frame);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().find_by_item(1) != 0);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        second_no_hand, second_no_hand);
    assert(second.settlement.committed);
    assert(session.inventory().find_by_item(1) == 0);
    assert(has_event(second.settlement, fridge::EventKind::OUT, 1));
}

// 单个部分遮挡的新 D 不足以把缺失 C 标成 OCCLUDED；两个已确认 D 的矩形
// 覆盖并集完整盖住 C 时，才应建立遮挡关系。这覆盖矩形差集的两条边界。
void test_no_hand_occlusion_uses_cover_union() {
    {
        fridge::SessionManager session;
        session.start_new_session();
        std::vector<fridge::InventoryItem> initial;
        initial.push_back(item(1, 2, 100, 100, 200, 200));
        session.init_from_backend(initial, true);
        int frame = 1;
        initial_no_hand_frame(&session,
            std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

        const std::vector<fridge::Detection> left_d(
            1, det(0, 50, 100, 150, 200));
        const fridge::BBox left_hand(30, 145, 45, 160);
        send_frame(&session, left_d, std::vector<fridge::BBox>(1, left_hand), &frame);
        send_frame(&session, left_d, std::vector<fridge::BBox>(1, left_hand), &frame);

        const int first_no_hand = frame++;
        fridge::FrameProcessResult first = session.process_frame(
            left_d, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
        assert(!first.settlement.committed);
        const int second_no_hand = frame++;
        fridge::FrameProcessResult second = session.process_frame(
            left_d, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
        assert(second.settlement.committed);
        assert(session.inventory().find_by_item(1)->status == fridge::ItemStatus::VISIBLE);
        assert(!has_event(second.settlement, fridge::EventKind::OCCLUDED, 1));
    }

    {
        fridge::SessionManager session;
        session.start_new_session();
        std::vector<fridge::InventoryItem> initial;
        initial.push_back(item(1, 2, 100, 100, 200, 200));
        session.init_from_backend(initial, true);
        int frame = 1;
        initial_no_hand_frame(&session,
            std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

        const fridge::Detection left = det(0, 50, 100, 150, 200);
        const fridge::Detection right = det(1, 150, 100, 250, 200);
        const fridge::BBox left_hand(30, 145, 45, 160);
        const fridge::BBox right_hand(255, 145, 270, 160);
        send_frame(&session, std::vector<fridge::Detection>(1, left),
                   std::vector<fridge::BBox>(1, left_hand), &frame);
        send_frame(&session, std::vector<fridge::Detection>(1, left),
                   std::vector<fridge::BBox>(1, left_hand), &frame);

        std::vector<fridge::Detection> both;
        both.push_back(left);
        both.push_back(right);
        send_frame(&session, both, std::vector<fridge::BBox>(1, right_hand), &frame);
        send_frame(&session, both, std::vector<fridge::BBox>(1, right_hand), &frame);

        const int first_no_hand = frame++;
        fridge::FrameProcessResult first = session.process_frame(
            both, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
        assert(!first.settlement.committed);
        const int second_no_hand = frame++;
        fridge::FrameProcessResult second = session.process_frame(
            both, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
        assert(second.settlement.committed);
        const fridge::InventoryItem* hidden = session.inventory().find_by_item(1);
        assert(hidden != 0);
        assert(hidden->status == fridge::ItemStatus::OCCLUDED);
        assert(hidden->block_ids.size() == 2);
        assert(has_event(second.settlement, fridge::EventKind::OCCLUDED, 1));
        assert_event_before(second.settlement, fridge::EventKind::IN, 2,
                            fridge::EventKind::OCCLUDED, 1);
        assert_event_before(second.settlement, fridge::EventKind::IN, 3,
                            fridge::EventKind::OCCLUDED, 1);
    }
}

// A 在无手阶段已有真实移动路径，但同类 C-D alias 尚未完成连续性确认。
// 目标 C 已经连续两帧没有直接框时，不能在 A 的 alias/身份确认完成前先写
// OUT；只有 A 按既有门槛确认 MOVED 后，才允许建立 blocker/OCCLUDED。
void test_unconfirmed_moving_front_defers_out_until_occlusion() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 0, 700, 100, 800, 200));
    initial.push_back(item(3, 2, 430, 95, 550, 205));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 700, 100, 800, 200));
    stable.push_back(det(2, 430, 95, 550, 205));
    initial_no_hand_frame(&session, stable, &frame);

    // A 的同类完整终点与一个窄 decoy 同时出现，构成真实 C-D alias；
    // 在 alias 尚未完成两帧无手仲裁前，A 不能仅凭宽路径提前确认 MOVED。
    const fridge::BBox old_hand(80, 80, 220, 220);
    const fridge::BBox moved_hand(370, 80, 610, 220);
    const fridge::Detection decoy = det(0, 440, 100, 540, 200);
    const fridge::Detection complete_a = det(0, 430, 95, 550, 205);
    const fridge::Detection static_b = det(0, 700, 100, 800, 200);
    send_frame(&session,
               std::vector<fridge::Detection>{static_b, stable[2]},
               std::vector<fridge::BBox>(1, old_hand), &frame);
    std::vector<fridge::Detection> conflicting_hand_frame;
    conflicting_hand_frame.push_back(decoy);
    conflicting_hand_frame.push_back(complete_a);
    conflicting_hand_frame.push_back(static_b);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);

    const std::vector<fridge::Detection> first_candidate(1, complete_a);
    const std::vector<fridge::Detection> final_cover(1, complete_a);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        first_candidate, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);

    // 这里正好是旧代码 C 的第二张 direct-missing。A 的 alias 在本帧才
    // 达到既有连续确认门槛；A 的 MOVED 与 C 的 OCCLUDED 必须在同一提交
    // 中完成，不能让 C 先变成 OUT。
    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        final_cover, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    const fridge::InventoryItem* hidden = session.inventory().find_by_item(3);
    assert(hidden != 0);
    assert(hidden->status == fridge::ItemStatus::OCCLUDED);
    assert(hidden->block_ids.count(1));
    assert(has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(has_event(second.settlement, fridge::EventKind::OCCLUDED, 3));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 3));
    assert_event_before(second.settlement, fridge::EventKind::MOVED, 1,
                        fridge::EventKind::OCCLUDED, 3);
}

// 和上一回放只差最后 A 候选消失。保护必须在这一帧自动失效，使 C 从
// threshold-1 接回正常 direct-missing OUT，而不是留下 blocker 或永久 defer。
void test_unconfirmed_moving_front_failure_restores_out_chain() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 300, 100, 400, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 300, 100, 400, 200));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 360, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(340, 80, 480, 220)), &frame);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        std::vector<fridge::Detection>(1, det(0, 250, 100, 350, 200)),
        std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        std::vector<fridge::Detection>(1, det(0, 390, 180, 450, 260)),
        std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(!second.settlement.committed);

    const int third_no_hand = frame++;
    fridge::FrameProcessResult third = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        third_no_hand, third_no_hand);
    assert(third.no_hand_frame_processed);
    // C 已从 threshold-1 接回普通 OUT 链；A 自己也需要两张连续
    // direct-missing，故本帧只允许暂存 C 的 OUT，不能永久冻结事务。
    assert(!third.settlement.committed);
    assert(session.operation_pending());

    const int fourth_no_hand = frame++;
    fridge::FrameProcessResult fourth = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        fourth_no_hand, fourth_no_hand);
    assert(fourth.no_hand_frame_processed);
    assert(fourth.settlement.committed);
    assert(session.inventory().find_by_item(1) == 0);
    assert(session.inventory().find_by_item(2) == 0);
    assert(has_event(fourth.settlement, fridge::EventKind::OUT, 1));
    assert(has_event(fourth.settlement, fridge::EventKind::OUT, 2));
}

// 纯几何层只允许真正接触目标外边界的窄残余；内部细缝和中间空洞即使很窄
// 也必须保持严格失败。业务调用层另行保证 covers 都来自确认 blocker。
void test_confirmed_occlusion_edge_residual_geometry() {
    using namespace fridge::session_internal;
    const fridge::BBox target(100, 100, 200, 200);

    std::vector<fridge::BBox> left_edge;
    left_edge.push_back(fridge::BBox(106, 100, 200, 200));
    assert(!fully_covered_by(target, left_edge));
    assert(edge_residual_within_target_border(
        target, left_edge,
        fridge::FLOW3_CONFIRMED_OCCLUSION_EDGE_RESIDUAL_PX));

    std::vector<fridge::BBox> internal_gap;
    internal_gap.push_back(fridge::BBox(100, 100, 145, 200));
    internal_gap.push_back(fridge::BBox(155, 100, 200, 200));
    assert(!fully_covered_by(target, internal_gap));
    assert(!edge_residual_within_target_border(
        target, internal_gap,
        fridge::FLOW3_CONFIRMED_OCCLUSION_EDGE_RESIDUAL_PX));

    std::vector<fridge::BBox> deep_edge_gap;
    deep_edge_gap.push_back(fridge::BBox(112, 100, 200, 200));
    assert(!edge_residual_within_target_border(
        target, deep_edge_gap,
        fridge::FLOW3_CONFIRMED_OCCLUSION_EDGE_RESIDUAL_PX));
}

// 细节13 / r16：手边短暂出现同类低分假框时，r15 必须先保护 alias D。
// 这里的旧苏打罐 C 有自己的无手直接框，但其中心偏移约 14.4px：超过
// CONTACT 原位 12px、低于正式 MOVED 28px。C 需要先用两张尺度一致的
// 无手框完成静态 release，之后 r15 才能按既有两帧缺失规则回收 fake D。
// 同一轮操作中的苹果 A 仍必须正常提交 MOVED。
void test_static_gray_zone_old_c_settles_before_r15_stale_alias_cleanup() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 520, 100, 620, 200));      // moved apple A
    initial.push_back(item(2, 25, 163, 138, 401, 400));     // static soda C
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 520, 100, 620, 200));
    stable.push_back(det(25, 163, 138, 401, 400));
    initial_no_hand_frame(&session, stable, &frame);

    // 两张静止手帧只让 C 的 claim grace 自然结束，不能积累“没有拿起”的
    // 移动证据。之后的两张有效手移动帧让 A 获得正常整理路径，同时 C
    // 仍稳定在原位；第一个成熟手移动帧中额外出现一个同类 fake D。
    const fridge::BBox hand0(0, 0, 1000, 600);
    const fridge::BBox hand1(30, 0, 1030, 600);
    const fridge::BBox hand2(60, 0, 1060, 600);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);

    std::vector<fridge::Detection> first_moved_hand;
    first_moved_hand.push_back(det(0, 550, 100, 650, 200));
    first_moved_hand.push_back(det(25, 163, 138, 401, 400));
    first_moved_hand.push_back(det(25, 850, 450, 940, 550, 0.266f));  // fake D
    send_frame(&session, first_moved_hand,
               std::vector<fridge::BBox>(1, hand1), &frame);

    bool found_quarantined_fake_d = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (track.is_suspect_new && track.pending_d_quarantined_by_old_c &&
            track.conflicting_old_item_ids.count(2)) {
            found_quarantined_fake_d = true;
            assert(track.item_id <= 0);
            assert(!track.promoted_to_working_inventory);
        }
    }
    assert(found_quarantined_fake_d);

    std::vector<fridge::Detection> second_moved_hand;
    second_moved_hand.push_back(det(0, 580, 100, 680, 200));
    second_moved_hand.push_back(det(25, 163, 138, 401, 400));
    send_frame(&session, second_moved_hand,
               std::vector<fridge::BBox>(1, hand2), &frame);

    const std::map<int, fridge::OperationTrack>& after_hand =
        session.operation_tracks();
    assert(after_hand.find(2) != after_hand.end());
    assert(after_hand.find(2)->second.state == fridge::OperationTrackState::NORMAL);
    assert(after_hand.find(2)->second.contact_state == fridge::ContactState::NONE);
    assert(after_hand.find(2)->second.resolution ==
           fridge::ExistingItemResolution::STATIC_CONFIRMED);
    assert(after_hand.find(2)->second.needs_no_hand_settlement);
    assert(after_hand.find(2)->second.has_hand_estimate_anchor_box);

    // 真实 C 的无手框相对 original 的中心偏移为约 14.4px，严格位于
    // 现有 12px CONTACT 门槛与 28px 正式 MOVED 门槛之间。
    std::vector<fridge::Detection> no_hand;
    no_hand.push_back(det(0, 580, 100, 680, 200));
    no_hand.push_back(det(25, 145, 141, 392, 407));

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        no_hand, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    const std::map<int, fridge::OperationTrack>& after_first =
        session.operation_tracks();
    assert(after_first.find(2) != after_first.end());
    assert(after_first.find(2)->second.needs_no_hand_settlement);
    assert(after_first.find(2)->second.stable_near_original_no_hand_count == 1);
    bool fake_d_waits_for_c = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             after_first.begin(); it != after_first.end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (track.is_suspect_new && track.pending_d_quarantined_by_old_c &&
            track.conflicting_old_item_ids.count(2)) {
            fake_d_waits_for_c = track.alias_no_hand_missing_count == 0;
        }
    }
    assert(fake_d_waits_for_c);
    assert(!has_event(first.settlement, fridge::EventKind::IN, 3));
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 2));

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        no_hand, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(!second.settlement.committed);
    assert(session.operation_pending());
    const std::map<int, fridge::OperationTrack>& after_second =
        session.operation_tracks();
    assert(after_second.find(2) != after_second.end());
    assert(!after_second.find(2)->second.needs_no_hand_settlement);
    assert(after_second.find(2)->second.release_reason ==
           fridge::ReleaseReason::STABLE_NEAR_ORIGINAL_NO_HAND);
    assert(after_second.find(2)->second.stable_near_original_no_hand_count == 0);
    assert(!after_second.find(2)->second.has_stable_near_original_box);
    bool fake_d_started_stale_missing = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             after_second.begin(); it != after_second.end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (track.is_suspect_new && track.pending_d_quarantined_by_old_c &&
            track.conflicting_old_item_ids.count(2)) {
            fake_d_started_stale_missing = track.alias_no_hand_missing_count == 1;
        }
    }
    assert(fake_d_started_stale_missing);
    assert(!has_event(second.settlement, fridge::EventKind::IN, 3));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 2));

    const int third_no_hand = frame++;
    fridge::FrameProcessResult third = session.process_frame(
        no_hand, std::vector<fridge::BBox>(), third_no_hand, third_no_hand);
    assert(third.no_hand_frame_processed);
    assert(third.settlement.committed);
    assert(session.inventory().size() == 2);
    const fridge::InventoryItem* final_a = session.inventory().find_by_item(1);
    const fridge::InventoryItem* final_c = session.inventory().find_by_item(2);
    assert(final_a != 0);
    assert(final_c != 0);
    assert(final_c->cls_id == 25);
    assert(final_c->status == fridge::ItemStatus::VISIBLE);
    assert(has_event(third.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(third.settlement, fridge::EventKind::MOVED, 2));
    assert(!has_event(third.settlement, fridge::EventKind::IN, 3));
    assert(!has_event(third.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(third.settlement, fridge::EventKind::OUT, 2));
}

// r16 的两张近原位证据必须是连续的无手帧。手一旦重新进入，哪怕它没有
// 再次改变 C 的最终结论，也必须清除已有的 1/2，不能让手前后一张灰区框
// 拼成错误的静态 release。
void test_static_gray_zone_evidence_resets_when_hand_returns() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 25, 163, 138, 401, 400));
    session.init_from_backend(initial, true);
    int frame = 1;
    const std::vector<fridge::Detection> stable(
        1, det(25, 163, 138, 401, 400));
    initial_no_hand_frame(&session, stable, &frame);

    const fridge::BBox hand0(0, 0, 1000, 600);
    const fridge::BBox hand1(30, 0, 1030, 600);
    const fridge::BBox hand2(60, 0, 1060, 600);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand1), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand2), &frame);

    const std::vector<fridge::Detection> gray_zone(
        1, det(25, 145, 141, 392, 407));
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        gray_zone, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_tracks().find(1)->second
           .stable_near_original_no_hand_count == 1);

    // 复用最后一个手框，避免给 C 注入新的移动路径；这里只验证 hand return
    // 作为时间边界清除静态证据。
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand2), &frame);
    const fridge::OperationTrack& after_hand_return =
        session.operation_tracks().find(1)->second;
    assert(after_hand_return.needs_no_hand_settlement);
    assert(after_hand_return.stable_near_original_no_hand_count == 0);
    assert(!after_hand_return.has_stable_near_original_box);

    const int first_after_hand = frame++;
    fridge::FrameProcessResult after_hand_first = session.process_frame(
        gray_zone, std::vector<fridge::BBox>(), first_after_hand, first_after_hand);
    assert(after_hand_first.no_hand_frame_processed);
    assert(!after_hand_first.settlement.committed);
    assert(session.operation_pending());
}

// 细节16现场回放：item#8 苹果确实移动，邻近的 item#7 橙子没有主动整理，
// 但手离开后 YOLO 框从完整框向左扩张，中心变化约 21.7px。该变化位于
// CONTACT 的 12px 原位门槛和正式归一化 MOVED 门槛之间；稳定、唯一的灰区
// 候选必须完成静态恢复，不能让整轮操作永久 defer-commit。
void test_item7_gray_recovery_does_not_block_item8_move() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(7, 2, 638, 460, 785, 634));  // orange: recovery gray zone
    initial.push_back(item(8, 0, 500, 100, 600, 200));  // apple: real move
    session.init_from_backend(initial, true);
    int frame = 1;

    const fridge::Detection orange_original = det(2, 638, 460, 785, 634);
    const fridge::Detection apple_original = det(0, 500, 100, 600, 200);
    std::vector<fridge::Detection> stable;
    stable.push_back(orange_original);
    stable.push_back(apple_original);
    initial_no_hand_frame(&session, stable, &frame);

    // 先给两个旧 C 三张稳定的有手帧，让原有 claim-grace/静态保护逻辑
    // 自然完成；手框本身不依赖物品移动方向。
    const fridge::BBox hand0(0, 0, 1000, 720);
    const fridge::BBox hand1(30, 0, 1030, 720);
    const fridge::BBox hand2(60, 0, 1060, 720);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand0), &frame);

    // 苹果沿明显的新位置移动；橙子仍提供完整原框，避免把两件物品的
    // 身份证据混成一个路径。再给一张连续有手帧巩固苹果移动链路。
    std::vector<fridge::Detection> moved;
    moved.push_back(orange_original);
    moved.push_back(det(0, 550, 100, 650, 200));
    send_frame(&session, moved, std::vector<fridge::BBox>(1, hand1), &frame);
    moved[1] = det(0, 580, 100, 680, 200);
    send_frame(&session, moved, std::vector<fridge::BBox>(1, hand2), &frame);

    // 现场恢复框：(594,460)~(787,645)，与完整原框中心相差约 21.7px。
    // 连续无手帧允许边界/中心有轻微抖动，但这里使用稳定框以精确复现灰区。
    const fridge::Detection orange_recovered = det(2, 594, 460, 787, 645);
    const fridge::Detection apple_moved = det(0, 580, 100, 680, 200);
    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(orange_recovered);
    final_boxes.push_back(apple_moved);

    fridge::SettlementResult result;
    for (int i = 0; i < 4; ++i) {
        const int no_hand_frame = frame++;
        const fridge::FrameProcessResult processed = session.process_frame(
            final_boxes, std::vector<fridge::BBox>(), no_hand_frame, no_hand_frame);
        assert(processed.no_hand_frame_processed);
        result = processed.settlement;
        if (result.committed) break;
    }

    assert(result.committed);
    assert(session.inventory().size() == 2);
    const fridge::InventoryItem* orange = session.inventory().find_by_item(7);
    const fridge::InventoryItem* apple = session.inventory().find_by_item(8);
    assert(orange != 0);
    assert(apple != 0);
    assert(orange->status == fridge::ItemStatus::VISIBLE);
    assert(apple->status == fridge::ItemStatus::VISIBLE);
    assert(has_event(result, fridge::EventKind::MOVED, 8));
    assert(!has_event(result, fridge::EventKind::MOVED, 7));
    assert(!has_event(result, fridge::EventKind::IN, 7));
    assert(!has_event(result, fridge::EventKind::OUT, 7));
    assert(!has_event(result, fridge::EventKind::OCCLUDED, 7));
    assert(!has_event(result, fridge::EventKind::REVEALED, 7));
}

// 细节15-5.1：矩形差集过程中不能按单个残片面积提前丢弃。目标 C 的
// 三个已确认前景 D 留下两个 0.4 x 10 的缝隙；每个缝隙面积正好为 4，
// 但总面积为 8，仍然不能算作完全遮挡。这里刻意让 C 在最后一张有手帧
// 才进入 HAND_* 且没有移动证据：旧实现若误判完整遮挡会完成操作并写
// OCCLUDED；修复后应保留 C 的 VISIBLE 身份并继续等待直接无手证据。
void test_cover_union_keeps_multiple_tiny_residuals_until_total_area_check() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 120, 110));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(2, 100, 100, 120, 110)), &frame);

    // 三个 D 分别向目标 C 的左、中、右延伸。它们对 C 的覆盖依次为
    // [100,106]、[106.4,113]、[113.4,120]，仅留下两个面积各约为 4 的
    // 缝隙；单个残片都不超过 COVER_REMAINING_AREA_EPS，但总面积约为 8。
    const fridge::Detection c = det(2, 100, 100, 120, 110);
    const fridge::Detection left = det(0, 50, 100, 106, 110);
    const fridge::Detection middle = det(1, 106.4f, 50, 113, 110);
    const fridge::Detection right = det(3, 113.4f, 100, 120, 160);
    const fridge::BBox hand(95, 95, 125, 115);

    // 同一静止手框下连续观察三个 D。C 的完整框在前两张仍可见，第三张
    // 只漏掉 C；前两张让三个 D 达到提升门槛。第三张把手移到远处，
    // 让每个已提升 D 得到明确的脱手/放下证据；C 仍保留自己的 HAND_* runtime。
    std::vector<fridge::Detection> all;
    all.push_back(c);
    all.push_back(left);
    all.push_back(middle);
    all.push_back(right);
    send_frame(&session, all, std::vector<fridge::BBox>(1, hand), &frame);
    send_frame(&session, all, std::vector<fridge::BBox>(1, hand), &frame);

    std::vector<fridge::Detection> blockers_only;
    blockers_only.push_back(left);
    blockers_only.push_back(middle);
    blockers_only.push_back(right);
    const fridge::BBox hand_moved_away(300, 300, 330, 330);
    const fridge::BBox hand_moved_further(320, 320, 350, 350);
    // 让 D 先脱手，同时给 C 两张连续的“仍在原位”反向证据；这样 C
    // 会暂时释放为静态，不把这段手位移历史带进后面的遮挡专用轨迹。
    send_frame(&session, all,
               std::vector<fridge::BBox>(1, hand_moved_away), &frame);
    send_frame(&session, all,
               std::vector<fridge::BBox>(1, hand_moved_further), &frame);
    // 先插入一张无手帧，完成 C 的直接无手静态结算。若在这一步之前
    // 立刻把手移回 C，reopen_released_static_tracks_ 会合法地恢复旧 C，
    // 并把手位移历史带入测试，导致它走 OUT，而不是验证遮挡差集。
    const int settle_static_frame = frame++;
    const fridge::FrameProcessResult static_result = session.process_frame(
        all, std::vector<fridge::BBox>(), settle_static_frame, settle_static_frame);
    assert(static_result.no_hand_frame_processed);
    assert(!static_result.settlement.committed);

    // 远离 C 的静止手帧只清掉 released-hand candidate 保护标记，
    // 不重新建立 C 轨迹。
    send_frame(&session, blockers_only,
               std::vector<fridge::BBox>(1, hand_moved_further), &frame);
    // 手重新回到 C 原位，但 C 本帧故意漏检；这是没有历史移动证据的新
    // HAND_* runtime。已放下的三个 D 仍是当前操作的确认前景。
    send_frame(&session, blockers_only,
               std::vector<fridge::BBox>(1, hand), &frame);

    int promoted_d_count = 0;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        if (!it->second.is_suspect_new ||
            !it->second.promoted_to_working_inventory) {
            continue;
        }
        ++promoted_d_count;
        assert(it->second.drop_confirmed);
        assert(it->second.has_placed_box);
    }
    // 没有这三个已确认前景 D，后面的 no-hand 遮挡结算不会调用目标差集。
    assert(promoted_d_count == 3);

    for (int no_hand_index = 0; no_hand_index < 5; ++no_hand_index) {
        const int no_hand_frame = frame++;
        const fridge::FrameProcessResult result = session.process_frame(
            blockers_only, std::vector<fridge::BBox>(), no_hand_frame, no_hand_frame);
        assert(result.no_hand_frame_processed);
        assert(!result.settlement.committed);
    }
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(1)->status == fridge::ItemStatus::VISIBLE);
}

// 细节31：整盒放入。手在中间一大片活动过（拿透明盒），无手后 3 个互不重叠、
// 都落在手活动区域内的同类框稳定 3 帧 → 补出 3 个 IN。
void test_no_hand_correction_box_in_hand_region_gets_in() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

    std::vector<fridge::Detection> none;
    send_frame(&session, none, std::vector<fridge::BBox>(1, fridge::BBox(300, 420, 620, 720)), &frame);
    send_frame(&session, none, std::vector<fridge::BBox>(1, fridge::BBox(360, 410, 700, 720)), &frame);
    send_frame(&session, none, std::vector<fridge::BBox>(1, fridge::BBox(300, 420, 760, 720)), &frame);

    std::vector<fridge::Detection> eggs;
    eggs.push_back(det(18, 380, 470, 500, 570));
    eggs.push_back(det(18, 520, 470, 640, 570));
    eggs.push_back(det(18, 450, 590, 570, 690));
    fridge::SettlementResult result = settle_after_hand(&session, eggs, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 3);
    int in_events = 0;
    for (size_t i = 0; i < result.events.size(); ++i) {
        if (result.events[i].kind == fridge::EventKind::IN) ++in_events;
    }
    assert(in_events == 3);
}

// 细节31：整盒取走。初始 3 个鸡蛋在库，手在那片活动过，之后无手快照持续为空
// → 补出 3 个 OUT，库存清零。
void test_no_hand_correction_box_out_hand_region_gets_out() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> init;
    init.push_back(item(1, 18, 380, 470, 500, 570));
    init.push_back(item(2, 18, 520, 470, 640, 570));
    init.push_back(item(3, 18, 450, 590, 570, 690));
    session.init_from_backend(init, true);
    int frame = 1;
    std::vector<fridge::Detection> eggs;
    eggs.push_back(det(18, 380, 470, 500, 570));
    eggs.push_back(det(18, 520, 470, 640, 570));
    eggs.push_back(det(18, 450, 590, 570, 690));
    initial_no_hand_frame(&session, eggs, &frame);

    std::vector<fridge::Detection> none;
    send_frame(&session, none, std::vector<fridge::BBox>(1, fridge::BBox(360, 440, 660, 720)), &frame);
    send_frame(&session, none, std::vector<fridge::BBox>(1, fridge::BBox(360, 440, 660, 720)), &frame);
    send_frame(&session, none, std::vector<fridge::BBox>(1, fridge::BBox(360, 440, 660, 720)), &frame);
    fridge::SettlementResult result = settle_after_hand(&session, none, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 0);
    int out_events = 0;
    for (size_t i = 0; i < result.events.size(); ++i) {
        if (result.events[i].kind == fridge::EventKind::OUT) ++out_events;
    }
    assert(out_events == 3);
}

// 细节31 反向守护：手只在左侧碰旧苹果；右侧远处稳定出现一个高分框（手从没去过）
// → 绝不补 IN。这正是旧 test_unbound_no_hand_box_never_auto_in 想保护的红线，
// 现在由"手活动区域"这把更精确的钥匙来守护。
void test_no_hand_correction_box_outside_hand_region_not_in() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> init;
    init.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(init, true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 900, 100, 1000, 200));  // 远在右侧，手从没去过
    fridge::SettlementResult result = settle_after_hand(&session, stable, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(!has_event(result, fridge::EventKind::IN, 2));
    assert(session.inventory().find_by_item(2) == 0);
}
}  // namespace session3_replay
