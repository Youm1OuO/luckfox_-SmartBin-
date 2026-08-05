// Host-side SessionManager replay scenarios: basic.
#include "session3_replay_support.h"

#include <algorithm>
#include <assert.h>
#include <map>
#include <vector>

namespace session3_replay {


void test_move_keeps_identity() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 160, 100, 260, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(140, 80, 240, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 230, 100, 330, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(210, 80, 310, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 380, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 300, 100, 400, 200)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
}

// 物品在手的候选路径中段已放下，手又继续移动离开时，最终稳定框不一定
// 靠近 hand 的最后位置。收尾必须回查整条 track，而不是把它误判 OUT。
void test_drop_at_middle_of_candidate_path_keeps_identity() {
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
    // 物品已停在 x=220，手继续移动时本帧看不到物品。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(320, 80, 420, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 220, 100, 320, 200)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
}

void test_out_removes_existing_identity() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    // 两张有效手帧都看到物品跟着手到旧位置之外，先建立 hold_and_move；
    // 随后连续无手直接帧完全找不到它，才允许 OUT。
    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 220, 100, 320, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 300, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(320, 80, 420, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 0);
    assert(has_event(result, fridge::EventKind::OUT, 1));
}

// hold_and_move=False 仍是“待确认”，不是“确认没移动”。如果手已经发生
// 有效移动，且后续无手直接帧在旧位置、候选路径都找不到它，就应当补确认 OUT。
void test_unconfirmed_hand_candidate_is_out_after_final_absence() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 300, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(320, 80, 420, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 0);
    assert(has_event(result, fridge::EventKind::OUT, 1));
}

// 反过来，即使手移动后还没有凑够 hold 证据，只要无手直接帧把物品重新
// 绑定回原位置，就必须保留原身份，不能因为动态参考框已偏移而 OUT。
void test_unconfirmed_hand_candidate_reappears_at_old_position() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 300, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(!has_event(result, fridge::EventKind::OUT, 1));
}

// 真实取出时，物品可能从手刚盖住开始就完全没有 YOLO 框。只要手随后
// 发生过有效移动、稳定收尾仍在旧位置/候选路径找不到它，就必须能 OUT。
void test_fully_hand_hidden_item_can_out() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 25, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(25, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(220, 80, 360, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 0);
    assert(has_event(result, fridge::EventKind::OUT, 1));
}
}  // namespace session3_replay
