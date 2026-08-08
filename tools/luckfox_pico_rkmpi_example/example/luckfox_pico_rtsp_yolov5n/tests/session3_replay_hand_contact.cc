// Host-side SessionManager replay scenarios: hand_contact.
#include "session3_replay_support.h"

#include <algorithm>
#include <assert.h>
#include <map>
#include <vector>

namespace session3_replay {


// HAND_* 的分类必须只使用手对旧物品完整参考框的覆盖率，不能根据本帧
// 是否恰好还有 YOLO 局部框来猜。这里特意不给物品检测框，验证三个区间。
void test_hand_cover_ratio_controls_hand_state() {
    {
        fridge::SessionManager session;
        session.start_new_session();
        std::vector<fridge::InventoryItem> initial;
        initial.push_back(item(1, 0, 100, 100, 200, 200));
        session.init_from_backend(initial, true);
        int frame = 1;
        initial_no_hand_frame(&session,
            std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

        // 覆盖 20%，低于 e2=0.30，不能建立 HAND_* 候选。
        send_frame(&session, std::vector<fridge::Detection>(),
                   std::vector<fridge::BBox>(1, fridge::BBox(100, 100, 120, 200)), &frame);
        assert(session.operation_tracks().find(1) == session.operation_tracks().end());
    }

    {
        fridge::SessionManager session;
        session.start_new_session();
        std::vector<fridge::InventoryItem> initial;
        initial.push_back(item(1, 0, 100, 100, 200, 200));
        session.init_from_backend(initial, true);
        int frame = 1;
        initial_no_hand_frame(&session,
            std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

        // 覆盖 50%，位于 e2 与 e1 之间；即使没有当前检测框也必须是 PARTIAL。
        send_frame(&session, std::vector<fridge::Detection>(),
                   std::vector<fridge::BBox>(1, fridge::BBox(100, 100, 150, 200)), &frame);
        const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
        assert(tracks.find(1) != tracks.end());
        assert(tracks.find(1)->second.state ==
               fridge::OperationTrackState::HAND_PARTIAL_BLOCKED);
    }

    {
        fridge::SessionManager session;
        session.start_new_session();
        std::vector<fridge::InventoryItem> initial;
        initial.push_back(item(1, 0, 100, 100, 200, 200));
        session.init_from_backend(initial, true);
        int frame = 1;
        initial_no_hand_frame(&session,
            std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

        // 覆盖 90%，高于 e1=0.88，才是 FULL。
        send_frame(&session, std::vector<fridge::Detection>(),
                   std::vector<fridge::BBox>(1, fridge::BBox(100, 100, 190, 200)), &frame);
        const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
        assert(tracks.find(1) != tracks.end());
        assert(tracks.find(1)->second.state ==
               fridge::OperationTrackState::HAND_FULL_BLOCKED);
    }
}

// D 放下后手会继续移开。此时物品的最近真实观测比“旧框 + 手累计位移”可靠；
// 若仍只使用估计框，D 会在无手直接帧中无法绑定而被错误丢弃。
void test_new_item_dropped_before_hand_moves_away_keeps_its_identity() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 160, 100, 230, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 370, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 320, 220)), &frame);
    // D 已停在 x=300 一带，手继续向右移开。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 370, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(420, 80, 520, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 270, 80, 420, 220)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(result, fridge::EventKind::IN, 1));
}

// 放下不能因单帧“B 看起来掉队”就确认：需要连续两帧都满足
// 手继续移动、B 基本停住、且 B 已与手脱离/更接近完整框的组合证据。
void test_drop_requires_continuous_evidence() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    // 新建 HAND_* 需先完整经过两张后续有效帧；随后成熟帧才可把连续
    // 本地 B 升级为正式候选。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 220, 100, 320, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 300, 220)), &frame);
    // 保护期结束后仍需一张真正移动的手帧才能累计第二条 HAND 证据；静止
    // 手帧只更新观测，不会把 hold_and_move 推到确认态。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(360, 80, 460, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(320, 80, 420, 220)), &frame);
    assert(session.operation_tracks().find(1)->second.hold_and_move);

    // B 已停在 x=340；第一次手继续离开只累计第 1 个 drop 证据。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(500, 80, 600, 220)), &frame);
    const fridge::OperationTrack& after_first_drop =
        session.operation_tracks().find(1)->second;
    assert(after_first_drop.state != fridge::OperationTrackState::PLACED);
    assert(after_first_drop.drop_evidence_count == 1);

    // 第二个连续证据才允许进入 PLACED。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(700, 80, 800, 220)), &frame);
    assert(session.operation_tracks().find(1)->second.state ==
           fridge::OperationTrackState::PLACED);

    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)), &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
}

// 同类相邻物品都可能解释手边的局部框时，不能因为 map 遍历顺序把框强塞给
// item#1 或 item#2，更不能把它新建为 D；本轮应保持两个旧身份不变。
void test_ambiguous_hand_partial_box_does_not_create_d() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 300, 300));
    initial.push_back(item(2, 0, 300, 100, 500, 300));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 300, 300));
    stable.push_back(det(0, 300, 100, 500, 300));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 420, 300)),
               std::vector<fridge::BBox>(1, fridge::BBox(210, 80, 350, 320)), &frame);
    fridge::SettlementResult result = settle_after_hand(&session, stable, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(!has_event(result, fridge::EventKind::IN, 3));
}

// 手指框很小、手腕完全不动时，物品仍应进入 CONTACT_*，而不是被登记成 D。
void test_low_coverage_contact_move_keeps_identity() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    const fridge::BBox finger(90, 145, 105, 160);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 120, 100, 220, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 140, 100, 240, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);
    // t0/t1/t2 是新建 CONTACT_* 的保护期；t3 才可以把连续 tentative B
    // 升级为正式 CONTACT_MOVING。
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 160, 100, 260, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    assert(tracks.find(1) != tracks.end());
    assert(tracks.find(1)->second.contact_state ==
           fridge::ContactState::CONTACT_MOVING);
    assert(tracks.find(1)->second.hold_and_move);
    // 保护期内的 B 是 tentative，不会写入正式 observed_track；转 HAND
    // 时仍会以最新 tentative B 作为估计锚点。
    assert(tracks.find(1)->second.observed_track.size() >= 1);

    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 140, 100, 240, 200)), &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

// 低覆盖率候选一直没有可靠 B 时，手离开后的缺失不能单凭一帧直接 OUT。
void test_low_coverage_contact_without_endpoint_stays_pending() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    const fridge::BBox finger(90, 145, 105, 160);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, finger), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(), &frame);
    // CONTACT_CANDIDATE 没有原位置重现、也没有可靠的实际物品终点时，
    // 无手逐帧收尾必须继续保持未决，不能按固定窗口强制提交。
    assert(!result.committed);
    assert(session.operation_pending());
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(!has_event(result, fridge::EventKind::OUT, 1));
}

// CONTACT_CANDIDATE 在原位置连续重新出现时应释放，不能因为手框接近就重复创建 D。
void test_low_coverage_contact_candidate_releases_at_original() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    const std::vector<fridge::Detection> stable(1, det(0, 100, 100, 200, 200));
    initial_no_hand_frame(&session, stable, &frame);

    const fridge::BBox finger(90, 145, 105, 160);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, finger), &frame);
    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    assert(tracks.find(1) == tracks.end() ||
           tracks.find(1)->second.contact_state == fridge::ContactState::NONE);

    fridge::SettlementResult result = settle_after_hand(&session, stable, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(!has_event(result, fridge::EventKind::OUT, 1));
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

// 先用手指推开、随后改为大面积握住时，HAND_* 的估计原点必须是最后一次
// 真实 B，而不是最初 A.box；否则小幅移动仍与旧框局部重叠时会被错误释放。
void test_contact_to_hand_transition_keeps_observed_anchor() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    const fridge::BBox finger(90, 145, 105, 160);
    const fridge::BBox covering_hand(140, 80, 240, 220);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 130, 100, 230, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 130, 100, 230, 200)),
               std::vector<fridge::BBox>(1, covering_hand), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    assert(tracks.find(1) != tracks.end());
    assert(tracks.find(1)->second.contact_state == fridge::ContactState::NONE);
    assert(tracks.find(1)->second.state ==
           fridge::OperationTrackState::HAND_FULL_BLOCKED);
    assert(tracks.find(1)->second.has_hand_estimate_anchor_box);
    assert(tracks.find(1)->second.hand_estimate_anchor_box.x1 == 130.0f);
    // 保护期内的 B 不写入正式 observed_track；不过它已被提升为 HAND
    // 的 estimate anchor，因此只需保留最初可靠观测即可。
    assert(tracks.find(1)->second.observed_track.size() >= 1);

    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 130, 100, 230, 200)), &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

// 细节1：HAND_* 尚未确认握持时，手框静止是模糊帧。即使当前检测框仍在
// 原位置，也不能用重复帧累计 not_hold，或把候选提前释放。
void test_stationary_hand_frame_does_not_change_hand_evidence() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    const std::vector<fridge::Detection> stable(1, det(0, 100, 100, 200, 200));
    initial_no_hand_frame(&session, stable, &frame);

    const fridge::BBox hand(100, 100, 150, 200);  // 仅覆盖 C 的 50%。
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand), &frame);
    const fridge::OperationTrack& first = session.operation_tracks().find(1)->second;
    assert(!first.hold_and_move);
    assert(first.hold_evidence_count == 0);
    assert(first.not_hold_evidence_count == 0);

    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand), &frame);
    send_frame(&session, stable, std::vector<fridge::BBox>(1, hand), &frame);
    const fridge::OperationTrack& after_static_frames =
        session.operation_tracks().find(1)->second;
    assert(!after_static_frames.hold_and_move);
    assert(after_static_frames.hold_evidence_count == 0);
    assert(after_static_frames.not_hold_evidence_count == 0);
    assert(after_static_frames.state ==
           fridge::OperationTrackState::HAND_PARTIAL_BLOCKED);
}

// 一条 HandTrack 可以同时服务多个物品；两个 HAND_* 都必须记录同一条
// hand_id 的位移，而不是依赖“当前只检测到一只手”的全局开关。
void test_one_hand_moves_two_items_with_one_hand_id() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 280, 100, 380, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 280, 100, 380, 200));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(70, 70, 410, 230)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(110, 70, 450, 230)), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    const fridge::OperationTrack& left = tracks.find(1)->second;
    const fridge::OperationTrack& right = tracks.find(2)->second;
    assert(left.carrier_hand_id >= 0);
    assert(left.carrier_hand_id == right.carrier_hand_id);
    assert(!left.carrier_hand_ambiguous);
    assert(!right.carrier_hand_ambiguous);
    assert(left.move_values.size() == 1);
    assert(right.move_values.size() == 1);
    assert(left.move_values.back().dx > 30.0f);
    assert(right.move_values.back().dx > 30.0f);
}

// 输入数组顺序不是手身份。两只手反向移动时，每件物品只能积累自己所关联
// hand_id 的 delta；把第二张的检测数组反转可防止实现悄悄依赖 YOLO 下标。
void test_two_hands_keep_opposite_item_deltas_when_detection_order_changes() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 500, 100, 600, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::BBox> first_hands;
    first_hands.push_back(fridge::BBox(80, 80, 220, 220));
    first_hands.push_back(fridge::BBox(480, 80, 620, 220));
    send_frame(&session, std::vector<fridge::Detection>(), first_hands, &frame);

    // H2 排在前面且向左，H1 排在后面且向右。
    std::vector<fridge::BBox> reversed_hands;
    reversed_hands.push_back(fridge::BBox(380, 80, 520, 220));
    reversed_hands.push_back(fridge::BBox(180, 80, 320, 220));
    send_frame(&session, std::vector<fridge::Detection>(), reversed_hands, &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    const fridge::OperationTrack& left = tracks.find(1)->second;
    const fridge::OperationTrack& right = tracks.find(2)->second;
    assert(left.carrier_hand_id >= 0);
    assert(right.carrier_hand_id >= 0);
    assert(left.carrier_hand_id != right.carrier_hand_id);
    assert(!left.carrier_hand_ambiguous);
    assert(!right.carrier_hand_ambiguous);
    assert(left.move_values.size() == 1);
    assert(right.move_values.size() == 1);
    assert(left.move_values.back().dx > 80.0f);
    assert(right.move_values.back().dx < -80.0f);
}

// 两条旧手轨迹合并成一个检测框时，不允许选择面积较大、列表靠前或距离更近
// 的一条手来伪造位移。两个已有物品都应保留此前路径并暂停本帧 delta。
void test_merged_hand_detection_suspends_item_deltas() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 500, 100, 600, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::BBox> separate_hands;
    separate_hands.push_back(fridge::BBox(80, 80, 220, 220));
    separate_hands.push_back(fridge::BBox(480, 80, 620, 220));
    send_frame(&session, std::vector<fridge::Detection>(), separate_hands, &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 620, 220)), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    const fridge::OperationTrack& left = tracks.find(1)->second;
    const fridge::OperationTrack& right = tracks.find(2)->second;
    assert(left.move_values.empty());
    assert(right.move_values.empty());
    assert(left.carrier_hand_ambiguous);
    assert(right.carrier_hand_ambiguous);
}

// 两只手先各自覆盖 C，下一帧被合并成一个手框。此时每个 C 都必须保留
// hand delta 已中断的历史，而不是借另一只手的位移。手离开后，两个不同
// 类别的终点均连续、唯一地直接可见时，应在同一事务内一起确认 MOVED。
void test_interrupted_two_hand_recovery_commits_both_moves() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 500, 100, 600, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::BBox> separate_hands;
    separate_hands.push_back(fridge::BBox(80, 80, 220, 220));
    separate_hands.push_back(fridge::BBox(480, 80, 620, 220));
    // 首帧仍保留 C 自己的直接框，建立 strict first_hand_block_box；随后
    // 才用合并手框制造 hand_id/delta 中断。
    send_frame(&session, stable, separate_hands, &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 620, 220)), &frame);

    const std::map<int, fridge::OperationTrack>& interrupted =
        session.operation_tracks();
    const fridge::OperationTrack& left = interrupted.find(1)->second;
    const fridge::OperationTrack& right = interrupted.find(2)->second;
    assert(left.hand_delta_interrupted);
    assert(right.hand_delta_interrupted);
    assert(left.has_first_hand_block_box);
    assert(right.has_first_hand_block_box);
    assert(left.move_values.empty());
    assert(right.move_values.empty());

    std::vector<fridge::Detection> endpoints;
    endpoints.push_back(det(0, 180, 100, 280, 200));
    endpoints.push_back(det(2, 420, 100, 520, 200));
    fridge::SettlementResult result = settle_after_hand(&session, endpoints, &frame);

    assert(result.committed);
    assert(!session.operation_pending());
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(has_event(result, fridge::EventKind::MOVED, 2));
}

// 有手阶段已唯一看到两个旧 C 各自贴手地离开原位，但手框随后合并，因而
// 两条 delta 都被保护性暂停。两个 C 消失后仍只能经过原有连续无手缺失
// 窗口，最后在一个原子事务里一起 OUT，不能因“同时不见”而单帧直接 OUT。
void test_interrupted_two_hand_exit_commits_both_outs_after_missing_window() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 500, 100, 600, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::BBox> separate_hands;
    separate_hands.push_back(fridge::BBox(80, 80, 220, 220));
    separate_hands.push_back(fridge::BBox(480, 80, 620, 220));
    send_frame(&session, stable, separate_hands, &frame);

    std::vector<fridge::Detection> moving_while_merged;
    moving_while_merged.push_back(det(0, 180, 100, 280, 200));
    moving_while_merged.push_back(det(2, 420, 100, 520, 200));
    send_frame(&session, moving_while_merged,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 620, 220)), &frame);

    const std::map<int, fridge::OperationTrack>& before_exit =
        session.operation_tracks();
    const fridge::OperationTrack& left = before_exit.find(1)->second;
    const fridge::OperationTrack& right = before_exit.find(2)->second;
    assert(left.hand_delta_interrupted);
    assert(right.hand_delta_interrupted);
    assert(left.has_direct_exit_evidence);
    assert(right.has_direct_exit_evidence);
    assert(left.move_values.empty());
    assert(right.move_values.empty());

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 2));

    fridge::SettlementResult result;
    for (int i = 0; i < 4; ++i) {
        const int no_hand_frame = frame++;
        fridge::FrameProcessResult next = session.process_frame(
            std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
            no_hand_frame, no_hand_frame);
        if (next.no_hand_frame_processed) result = next.settlement;
        if (result.committed) break;
    }

    assert(result.committed);
    assert(!session.operation_pending());
    assert(session.inventory().size() == 0);
    assert(has_event(result, fridge::EventKind::OUT, 1));
    assert(has_event(result, fridge::EventKind::OUT, 2));
}

// hand_id 中断本身绝不是移动证据。两个 C 先留下离开候选、再在手仍存在时
// 回到原位置，最后无手稳定看到原位时，必须沿已有静态收尾提交，不产生
// MOVED 或 OUT。
void test_interrupted_two_hand_return_to_original_stays_static() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 500, 100, 600, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::BBox> separate_hands;
    separate_hands.push_back(fridge::BBox(80, 80, 220, 220));
    separate_hands.push_back(fridge::BBox(480, 80, 620, 220));
    send_frame(&session, stable, separate_hands, &frame);

    std::vector<fridge::Detection> moved;
    moved.push_back(det(0, 180, 100, 280, 200));
    moved.push_back(det(2, 420, 100, 520, 200));
    send_frame(&session, moved,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 620, 220)), &frame);

    const std::map<int, fridge::OperationTrack>& interrupted =
        session.operation_tracks();
    const fridge::OperationTrack& left = interrupted.find(1)->second;
    const fridge::OperationTrack& right = interrupted.find(2)->second;
    assert(left.hand_delta_interrupted);
    assert(right.hand_delta_interrupted);
    assert(left.has_direct_exit_evidence);
    assert(right.has_direct_exit_evidence);

    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 620, 220)), &frame);

    fridge::SettlementResult result = settle_after_hand(&session, stable, &frame);
    assert(result.committed);
    assert(!session.operation_pending());
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(!has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::MOVED, 2));
    assert(!has_event(result, fridge::EventKind::OUT, 1));
    assert(!has_event(result, fridge::EventKind::OUT, 2));
}

// 两条旧手轨迹都能几何接上同一个当前框时，即使 H1 的代价显著更低，
// 当前框仍可能是合并框。不能让 H1 继续把这个框的位移写给左侧物品。
void test_skewed_merged_hand_detection_suspends_all_item_deltas() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 2, 350, 100, 450, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 350, 100, 450, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::BBox> separate_hands;
    separate_hands.push_back(fridge::BBox(0, 50, 300, 250));
    separate_hands.push_back(fridge::BBox(300, 50, 500, 250));
    send_frame(&session, std::vector<fridge::Detection>(), separate_hands, &frame);

    // H1 到合并框的代价低于 H2，但两个旧 hand_id 都仍是这个框的连续候选。
    // 这必须按“合并/歧义”处理，而不是把 H1 选作胜者。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(0, 50, 500, 250)), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    const fridge::OperationTrack& left = tracks.find(1)->second;
    const fridge::OperationTrack& right = tracks.find(2)->second;
    assert(left.move_values.empty());
    assert(right.move_values.empty());
    assert(left.carrier_hand_ambiguous);
    assert(right.carrier_hand_ambiguous);
}

// 本帧有两只可靠 hand_id，但第二只手也覆盖到已有物品时，先检查物品的
// 预测位置是否存在多手关联。若有，不能先写旧 carrier 的 delta、再标歧义。
void test_second_hand_overlap_suspends_existing_item_delta() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    std::vector<fridge::BBox> first_hands;
    first_hands.push_back(fridge::BBox(0, 50, 300, 250));
    first_hands.push_back(fridge::BBox(300, 50, 900, 250));
    send_frame(&session, std::vector<fridge::Detection>(), first_hands, &frame);

    // 两条 hand_id 的全局匹配仍唯一；但 H2 本帧已经覆盖到 item#1 的预测位置。
    std::vector<fridge::BBox> overlapping_hands;
    overlapping_hands.push_back(fridge::BBox(20, 50, 320, 250));
    overlapping_hands.push_back(fridge::BBox(100, 50, 700, 250));
    send_frame(&session, std::vector<fridge::Detection>(), overlapping_hands, &frame);

    const fridge::OperationTrack& track = session.operation_tracks().find(1)->second;
    assert(track.carrier_hand_ambiguous);
    assert(track.move_values.empty());
}

// 一只手在另一只手仍被识别时短暂漏检，恢复到同一 hand_id 的首帧只能恢复
// 身份，不能把漏检间隔两端的差值补写为可靠物品位移；下一张连续帧才可继续。
void test_temporarily_lost_hand_recovers_without_gap_delta() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    std::vector<fridge::BBox> first_hands;
    first_hands.push_back(fridge::BBox(80, 80, 220, 220));
    first_hands.push_back(fridge::BBox(700, 80, 840, 220));
    send_frame(&session, std::vector<fridge::Detection>(), first_hands, &frame);
    const int original_hand_id = session.operation_tracks().find(1)->second.carrier_hand_id;
    assert(original_hand_id >= 0);

    // H1 漏检，H2 仍在，因此该帧属于有手连续操作，而不是无手结算。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(700, 80, 840, 220)), &frame);
    assert(session.operation_tracks().find(1)->second.move_values.empty());

    // 让输出顺序反转，确保恢复依赖内部 hand_id 而非检测数组下标。
    std::vector<fridge::BBox> recovered_hands;
    recovered_hands.push_back(fridge::BBox(700, 80, 840, 220));
    recovered_hands.push_back(fridge::BBox(100, 80, 240, 220));
    send_frame(&session, std::vector<fridge::Detection>(), recovered_hands, &frame);
    const fridge::OperationTrack& recovered = session.operation_tracks().find(1)->second;
    assert(recovered.carrier_hand_id == original_hand_id);
    assert(!recovered.carrier_hand_ambiguous);
    assert(recovered.move_values.empty());

    std::vector<fridge::BBox> continuous_hands;
    continuous_hands.push_back(fridge::BBox(120, 80, 260, 220));
    continuous_hands.push_back(fridge::BBox(700, 80, 840, 220));
    send_frame(&session, std::vector<fridge::Detection>(), continuous_hands, &frame);
    const fridge::OperationTrack& continued = session.operation_tracks().find(1)->second;
    assert(continued.carrier_hand_id == original_hand_id);
    assert(continued.move_values.size() == 1);
    assert(continued.move_values.back().dx > 12.0f);
}

// 细节8-7.2 的自然时序：A 先连续两帧被直接看到仍在原位，触发暂时的
// STATIC_CONFIRMED；手尚未离开就再次移动，A 在旧位置消失。此时必须重新
// 激活 A 的轨迹，最终认回移动后的 A，而不是把它登记成新的同类 D。
void test_provisional_static_a_reopens_when_moved_later_in_same_operation() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // A
    initial.push_back(item(2, 0, 700, 100, 800, 200));  // B，远离 A 的移动路径
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 700, 100, 800, 200));
    initial_no_hand_frame(&session, stable, &frame);

    // t0/t1/t2：手移动，但 A 的独立完整框一直在原位。t2 后 A 应仅是
    // 有手阶段的暂时静态确认，仍须等待无手结算或后续重新激活。
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(100, 80, 240, 220)), &frame);
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(120, 80, 260, 220)), &frame);
    const fridge::OperationTrack& temporarily_static =
        session.operation_tracks().find(1)->second;
    assert(temporarily_static.state == fridge::OperationTrackState::NORMAL);
    assert(temporarily_static.resolution == fridge::ExistingItemResolution::STATIC_CONFIRMED);
    assert(temporarily_static.needs_no_hand_settlement);

    // t3：同一次手操作再次移动，A 暂时完全被挡住；B 仍有自己的原位框。
    std::vector<fridge::Detection> only_b;
    only_b.push_back(det(0, 700, 100, 800, 200));
    send_frame(&session, only_b,
               std::vector<fridge::BBox>(1, fridge::BBox(240, 80, 380, 220)), &frame);
    const fridge::OperationTrack& reopened = session.operation_tracks().find(1)->second;
    assert(reopened.state != fridge::OperationTrackState::NORMAL);
    assert(reopened.resolution == fridge::ExistingItemResolution::NONE);
    assert(reopened.needs_no_hand_settlement);
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        assert(!it->second.is_suspect_new);
    }

    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(det(0, 220, 100, 320, 200));
    final_boxes.push_back(det(0, 700, 100, 800, 200));
    fridge::SettlementResult result = settle_after_hand(&session, final_boxes, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::OUT, 1));
    assert(!has_event(result, fridge::EventKind::IN, 3));
}

// 细节15-4.4：有手阶段的 MOVED 只是临时工作状态。若同一操作尚未提交前
// 再次出现唯一属于自己的严格原位框，必须清掉 placed/drop/confirmed_moved
// 残留，最终不能提交一条假的 MOVED。
void test_direct_original_evidence_rolls_back_provisional_moved() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 220, 100, 320, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 300, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(360, 80, 460, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(320, 80, 420, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(500, 80, 600, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(700, 80, 800, 220)), &frame);

    const fridge::OperationTrack& moved = session.operation_tracks().find(1)->second;
    assert(moved.state == fridge::OperationTrackState::PLACED);
    assert(moved.resolution == fridge::ExistingItemResolution::MOVED_CONFIRMED);
    assert(moved.has_placed_box);
    assert(moved.drop_confirmed);

    // 同一只手仍在画面中，但当前唯一框已经严格回到 item#1 的原位置。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    const fridge::OperationTrack& rolled_back = session.operation_tracks().find(1)->second;
    assert(rolled_back.state == fridge::OperationTrackState::NORMAL);
    assert(rolled_back.resolution == fridge::ExistingItemResolution::STATIC_CONFIRMED);
    assert(!rolled_back.has_placed_box);
    assert(!rolled_back.drop_confirmed);

    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(1)->status == fridge::ItemStatus::VISIBLE);
    assert(!has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::IN, 2));
    assert(!has_event(result, fridge::EventKind::OUT, 1));
}

// D 在 HAND 期已经提升并获得放下证据时，D + 手覆盖 C 仍只是临时解释。
// 没有 D 的无手直接身份确认前，C 不能被 runtime 标成 OCCLUDED_CONFIRMED，
// 也不能失去自己的无手结算义务。
void test_hand_front_cover_remains_provisional_until_no_hand_confirmation() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    const fridge::Detection front = det(0, 100, 100, 200, 200);
    const fridge::BBox covering_hand(80, 80, 220, 220);
    send_frame(&session, std::vector<fridge::Detection>(1, front),
               std::vector<fridge::BBox>(1, covering_hand), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, front),
               std::vector<fridge::BBox>(1, covering_hand), &frame);

    // 两张持续脱手的有手帧让 D 走完既有提升/放下链，但故意不进入无手确认。
    const fridge::BBox hand_moved_once(300, 80, 440, 220);
    const fridge::BBox hand_moved_twice(500, 80, 640, 220);
    send_frame(&session, std::vector<fridge::Detection>(1, front),
               std::vector<fridge::BBox>(1, hand_moved_once), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, front),
               std::vector<fridge::BBox>(1, hand_moved_twice), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    const std::map<int, fridge::OperationTrack>::const_iterator target = tracks.find(1);
    assert(target != tracks.end());
    assert(target->second.resolution !=
           fridge::ExistingItemResolution::OCCLUDED_CONFIRMED);
    assert(target->second.needs_no_hand_settlement);

    bool saw_confirmed_front = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        if (it->second.is_suspect_new && it->second.promoted_to_working_inventory &&
            it->second.drop_confirmed) {
            saw_confirmed_front = true;
        }
    }
    assert(saw_confirmed_front);
    assert(session.inventory().find_by_item(1)->status == fridge::ItemStatus::VISIBLE);

    const int first_no_hand = frame++;
    const fridge::FrameProcessResult result = session.process_frame(
        std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
        first_no_hand, first_no_hand);
    assert(result.no_hand_frame_processed);
    assert(!result.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().find_by_item(1)->status == fridge::ItemStatus::VISIBLE);
}

// 细节27：完全不可见物品的手组见证只应在同一条内部 hand_id 连续到达
// 画面边界时成立；一旦 hand identity 失效或中途重绑，它就不能再作为 OUT
// 的入口。
void test_hand_group_exit_witness_requires_continuous_edge_hand() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 220, 220));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 220, 220)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 260, 260)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(260, 80, 440, 260)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(460, 80, 640, 260)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(660, 80, 840, 260)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(860, 80, 1040, 260)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(1080, 80, 1280, 260)), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    const std::map<int, fridge::OperationTrack>::const_iterator target = tracks.find(1);
    assert(target != tracks.end());
    assert(target->second.carrier_capture_context);
    assert(target->second.capture_was_fully_hidden);
    assert(target->second.hand_group_exit_witness);
    assert(!target->second.hand_group_identity_invalid);
    assert(target->second.hand_group_exit_frame > 0);
}
}  // namespace session3_replay
