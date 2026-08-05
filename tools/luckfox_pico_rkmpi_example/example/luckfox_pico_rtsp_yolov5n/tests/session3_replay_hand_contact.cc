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
}  // namespace session3_replay
