// Host-side SessionManager replay scenarios: no_hand.
#include "session3_replay_support.h"

#include <algorithm>
#include <assert.h>
#include <map>
#include <vector>

namespace session3_replay {


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
    }
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
}  // namespace session3_replay
