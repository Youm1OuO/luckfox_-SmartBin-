// 3.0 SessionManager 的主机端最小回放测试。
// 不依赖摄像头或 RKNN，只喂入已经得到的 Detection / hand box。
#include <assert.h>

#include <vector>

#include "fridge_config.h"
#include "session.h"

// inventory.cc 的调试打印会引用该符号；回放测试不链接完整 postprocess。
char* coco_cls_to_name(int) {
    static char name[] = "test";
    return name;
}

namespace {

fridge::Detection det(int cls, float x1, float y1, float x2, float y2,
                      float score = 0.9f) {
    fridge::Detection d;
    d.cls_id = cls;
    d.score = score;
    d.box = fridge::BBox(x1, y1, x2, y2);
    return d;
}

fridge::InventoryItem item(int id, int cls, float x1, float y1,
                           float x2, float y2) {
    fridge::InventoryItem result;
    result.item_id = id;
    result.cls_id = cls;
    result.box = fridge::BBox(x1, y1, x2, y2);
    result.base_box = result.box;
    result.score = 0.9f;
    result.status = fridge::ItemStatus::VISIBLE;
    return result;
}

void initial_snapshot(fridge::SessionManager* session,
    const std::vector<fridge::Detection>& foods,
                      int* frame) {
    for (int i = 0; i < fridge::SNAPSHOT_N; ++i) {
        const int id = (*frame)++;
        session->process_frame(foods, std::vector<fridge::BBox>(), id, id);
    }
}

bool has_event(const fridge::SettlementResult& result, fridge::EventKind kind,
               int item_id) {
    for (size_t i = 0; i < result.events.size(); ++i) {
        if (result.events[i].kind == kind && result.events[i].item_id == item_id) return true;
    }
    return false;
}

void send_frame(fridge::SessionManager* session,
                const std::vector<fridge::Detection>& foods,
                const std::vector<fridge::BBox>& hands, int* frame) {
    const int id = (*frame)++;
    session->process_frame(foods, hands, id, id);
}

fridge::SettlementResult settle_after_hand(
        fridge::SessionManager* session,
        const std::vector<fridge::Detection>& stable_foods, int* frame) {
    fridge::SettlementResult result;
    for (int i = 0; i < fridge::SNAPSHOT_N; ++i) {
        const int id = (*frame)++;
        fridge::FrameProcessResult frame_result = session->process_frame(
            stable_foods, std::vector<fridge::BBox>(), id, id);
        if (frame_result.stable_snapshot_generated) result = frame_result.settlement;
    }
    return result;
}

void test_move_keeps_identity() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

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
    initial_snapshot(&session,
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
    initial_snapshot(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    // 两张有效手帧都看到物品跟着手到旧位置之外，先建立 hold_and_move；
    // 随后无手稳定帧完全找不到它，才允许 OUT。
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
// 有效移动，且最终稳定快照在旧位置、候选路径都找不到它，就应当补确认 OUT。
void test_unconfirmed_hand_candidate_is_out_after_final_absence() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

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

// 反过来，即使手移动后还没有凑够 hold 证据，只要稳定快照把物品重新
// 绑定回原位置，就必须保留原身份，不能因为动态参考框已偏移而 OUT。
void test_unconfirmed_hand_candidate_reappears_at_old_position() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

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
    initial_snapshot(&session,
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
        initial_snapshot(&session,
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
        initial_snapshot(&session,
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
        initial_snapshot(&session,
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

void test_new_item_requires_d_chain() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 182, 100, 262, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(100, 100, 180, 180)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 212, 100, 292, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(130, 100, 210, 180)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(2, 212, 100, 292, 180)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(result, fridge::EventKind::IN, 1));
}

// 新物品可能只在一张有手帧露出局部框，手离开后才第一次看到完整框。
// 这正是实际“放下后迅速抽手”的常见路径；不能因此丢失已建立的 D 链路。
void test_new_item_can_confirm_from_first_no_hand_frame() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 160, 100, 230, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 140, 80, 280, 220)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(result, fridge::EventKind::IN, 1));
}

// D 首帧只有被手露出的细条，后续完全看不见；它在手路径中段被放下，
// 手继续移开后才在无手帧看到完整框。不能只拿终点或初始细条做匹配。
void test_partially_seen_new_item_can_reappear_full_on_middle_path() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(), &frame);

    // 初次只看到宽 30 像素的 D 局部框。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 160, 100, 190, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(100, 80, 180, 220)), &frame);
    // D 已完全被手遮住；手分别向右移动 100、200 像素，路径中会保留 P1 和 P2。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 280, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(400, 80, 480, 220)), &frame);

    // 完整苹果位于路径中段 P1 附近，既不重叠初始细条，也不重叠最终 P2。
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 290, 50, 430, 270)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::IN, 1));
}

// D 可以从进入到放下始终完全被手遮住，整个有手阶段没有任何 D 检测框。
// 手离开后，只要完整 B 出现在本次公共手轨迹附近并连续稳定出现，也必须
// 建立 POST_HAND_REVEAL_D -> 工作库存 -> IN 链路。
void test_fully_hidden_new_item_can_enter_from_post_hand_reveal() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(), &frame);

    // 有手期间始终没有苹果检测；手在路径中段放下它后继续向右移开。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(260, 80, 360, 220)), &frame);

    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 170, 100, 290, 220)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::IN, 1));
}

// C 被手挡住后，手移开时在 C 原位置留下一个不贴手的、不同类别 D。
// D 不能因为“不贴手”被漏掉；它应以 C_POSITION_REPLACEMENT_D 预登记，
// 最终让 C 保持遮挡、D 正常 IN。
void test_new_item_replacing_c_old_position_is_registered_without_hand_contact() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));  // C: orange
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session,
        std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    // 先把 C 完全挡住。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    // 手已移到右边；苹果 D 留在 C 原位置，和手框不相贴。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 220, 220)),
               std::vector<fridge::BBox>(1, fridge::BBox(300, 80, 440, 220)), &frame);

    bool found_replacement_source = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        if (it->second.suspect_source ==
            fridge::SuspectSource::C_POSITION_REPLACEMENT_D) {
            found_replacement_source = true;
        }
    }
    assert(found_replacement_source);

    // 第二张有效有手帧让 D 完成一次自匹配；随后稳定无手快照正式结算。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 220, 220)),
               std::vector<fridge::BBox>(1, fridge::BBox(500, 80, 640, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 100, 100, 220, 220)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(1)->status == fridge::ItemStatus::OCCLUDED);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(result, fridge::EventKind::IN, 2));
}

// POST_HAND_REVEAL_D 的首帧完整 B 不是正式入库证据。若下一张有效无手帧
// 消失，则候选必须丢弃，之后也不能因为旧 B 偶然重现就补发 IN。
void test_post_hand_reveal_requires_continuous_no_hand_confirmation() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(), &frame);

    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(260, 80, 360, 220)), &frame);

    const int first_no_hand = frame++;
    session.process_frame(std::vector<fridge::Detection>(1, det(0, 170, 100, 290, 220)),
                          std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    const int second_no_hand = frame++;
    session.process_frame(std::vector<fridge::Detection>(), std::vector<fridge::BBox>(),
                          second_no_hand, second_no_hand);
    const int third_no_hand = frame++;
    fridge::FrameProcessResult final_frame = session.process_frame(
        std::vector<fridge::Detection>(1, det(0, 170, 100, 290, 220)),
        std::vector<fridge::BBox>(), third_no_hand, third_no_hand);

    assert(final_frame.stable_snapshot_generated);
    assert(session.inventory().size() == 0);
    assert(!has_event(final_frame.settlement, fridge::EventKind::IN, 1));
}

// 即使一个 B 在 D 的候选路径附近，只要它可以严格认领给已有库存，D 就不能
// 抢占它并制造重复 IN。
void test_d_reappearance_does_not_claim_strict_existing_inventory() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 290, 50, 430, 270));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session,
        std::vector<fridge::Detection>(1, det(0, 290, 50, 430, 270)), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 160, 100, 190, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(100, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 280, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(400, 80, 480, 220)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 290, 50, 430, 270)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

// D 放下后手会继续移开。此时物品的最近真实观测比“旧框 + 手累计位移”可靠；
// 若仍只使用估计框，D 会在无手稳定快照中无法绑定而被错误丢弃。
void test_new_item_dropped_before_hand_moves_away_keeps_its_identity() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(), &frame);

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

// 旧物品刚被手遮住时，YOLO 常给出与完整框只部分重叠的局部框。它必须优先
// 认领给已有 item，而不是被 scan_or_update_suspects_ 误建成新的 D。
void test_partial_existing_item_is_not_registered_as_new_d() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 300, 300));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 300, 300)), &frame);

    // 首帧局部框向右偏，IoM(完整框, 局部框) < 0.84，旧实现会把它错误建成 D。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 410, 300)),
               std::vector<fridge::BBox>(1, fridge::BBox(210, 80, 350, 320)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 470, 300)),
               std::vector<fridge::BBox>(1, fridge::BBox(270, 80, 410, 320)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 300, 100, 500, 300)),
               std::vector<fridge::BBox>(1, fridge::BBox(300, 80, 440, 320)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 300, 100, 500, 300)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

// 有效帧之间手框发生较大跳动时，C 的“旧框 + 手位移”终点可能错过实际 B。
// 同类贴手 B 仍应先成为 C 的 reappear_candidate，而不是新建 HAND_VISIBLE_D。
void test_fast_same_class_b_becomes_c_candidate_not_d() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session,
        std::vector<fridge::Detection>(1, det(2, 100, 100, 200, 200)), &frame);

    // C 先被完整遮住，随后手快速移到右侧。实际橙子 B 留在候选轨迹的
    // 中间位置，和当前终点预计框不匹配，但与手相贴。
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 180, 100, 280, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(300, 80, 440, 220)), &frame);

    const std::map<int, fridge::OperationTrack>& first_tracks = session.operation_tracks();
    assert(first_tracks.find(1) != first_tracks.end());
    assert(first_tracks.find(1)->second.has_reappear_candidate_box);
    for (std::map<int, fridge::OperationTrack>::const_iterator it = first_tracks.begin();
         it != first_tracks.end(); ++it) {
        assert(!it->second.is_suspect_new);
    }

    // 第二次看到同一个 B 后，候选才完成自匹配；整个过程中仍不应出现 D。
    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 180, 100, 280, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(460, 80, 600, 220)), &frame);
    const std::map<int, fridge::OperationTrack>& second_tracks = session.operation_tracks();
    assert(second_tracks.find(1) != second_tracks.end());
    assert(second_tracks.find(1)->second.reappear_candidate_match_count >=
           fridge::FLOW3_REAPPEAR_CANDIDATE_CONFIRM_FRAMES);
    for (std::map<int, fridge::OperationTrack>::const_iterator it = second_tracks.begin();
         it != second_tracks.end(); ++it) {
        assert(!it->second.is_suspect_new);
    }

    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(2, 180, 100, 280, 200)), &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::IN, 2));
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
    initial_snapshot(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    // 前两次跟手观察建立 hold_and_move。
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 180, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 220, 100, 320, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 300, 220)), &frame);
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
    initial_snapshot(&session, stable, &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 240, 100, 420, 300)),
               std::vector<fridge::BBox>(1, fridge::BBox(210, 80, 350, 320)), &frame);
    fridge::SettlementResult result = settle_after_hand(&session, stable, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(!has_event(result, fridge::EventKind::IN, 3));
}

// 旧苹果仍然有自己的严格检测框时，旁边贴手出现的第二个苹果不能被
// item#1 的宽松局部匹配吞掉；它必须走 D -> 工作库存 -> 正式 IN 链路。
void test_adjacent_same_class_new_item_gets_its_own_d_chain() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 800, 100, 1000, 300));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session,
        std::vector<fridge::Detection>(1, det(0, 800, 100, 1000, 300)), &frame);

    std::vector<fridge::Detection> hand_foods;
    hand_foods.push_back(det(0, 800, 100, 1000, 300));  // 已有苹果的完整框
    hand_foods.push_back(det(0, 740, 100, 940, 300));   // 新苹果，紧挨旧苹果
    // 手同时碰到两个苹果，模拟真实“把第二个苹果紧挨第一个放下”的情况。
    send_frame(&session, hand_foods,
               std::vector<fridge::BBox>(1, fridge::BBox(760, 80, 900, 320)), &frame);
    // 第二张有效帧仍同时碰到两者；逐帧应按各自的唯一最近框分别认领。
    send_frame(&session, hand_foods,
               std::vector<fridge::BBox>(1, fridge::BBox(760, 90, 900, 330)), &frame);
    fridge::SettlementResult result = settle_after_hand(&session, hand_foods, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(result, fridge::EventKind::IN, 2));
}

void test_unbound_stable_box_never_auto_in() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

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
    initial_snapshot(&session, initial_foods, &frame);

    // A 移到 B 前面，稳定快照只看得到 A。
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

// 手指框很小、手腕完全不动时，物品仍应进入 CONTACT_*，而不是被登记成 D。
void test_low_coverage_contact_move_keeps_identity() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_snapshot(&session,
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

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    assert(tracks.find(1) != tracks.end());
    assert(tracks.find(1)->second.contact_state ==
           fridge::ContactState::CONTACT_MOVING);
    assert(tracks.find(1)->second.hold_and_move);
    assert(tracks.find(1)->second.observed_track.size() >= 3);

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
    initial_snapshot(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    const fridge::BBox finger(90, 145, 105, 160);
    send_frame(&session,
               std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)),
               std::vector<fridge::BBox>(1, finger), &frame);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, finger), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(), &frame);
    assert(result.committed);
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
    initial_snapshot(&session, stable, &frame);

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
    initial_snapshot(&session,
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
    assert(tracks.find(1)->second.observed_track.size() >= 2);

    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(0, 130, 100, 230, 200)), &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

}  // namespace

int main() {
    test_move_keeps_identity();
    test_drop_at_middle_of_candidate_path_keeps_identity();
    test_out_removes_existing_identity();
    test_unconfirmed_hand_candidate_is_out_after_final_absence();
    test_unconfirmed_hand_candidate_reappears_at_old_position();
    test_fully_hand_hidden_item_can_out();
    test_hand_cover_ratio_controls_hand_state();
    test_new_item_requires_d_chain();
    test_new_item_can_confirm_from_first_no_hand_frame();
    test_partially_seen_new_item_can_reappear_full_on_middle_path();
    test_fully_hidden_new_item_can_enter_from_post_hand_reveal();
    test_new_item_replacing_c_old_position_is_registered_without_hand_contact();
    test_post_hand_reveal_requires_continuous_no_hand_confirmation();
    test_d_reappearance_does_not_claim_strict_existing_inventory();
    test_new_item_dropped_before_hand_moves_away_keeps_its_identity();
    test_partial_existing_item_is_not_registered_as_new_d();
    test_fast_same_class_b_becomes_c_candidate_not_d();
    test_drop_requires_continuous_evidence();
    test_ambiguous_hand_partial_box_does_not_create_d();
    test_adjacent_same_class_new_item_gets_its_own_d_chain();
    test_unbound_stable_box_never_auto_in();
    test_moved_front_item_occludes_then_reveals();
    test_low_coverage_contact_move_keeps_identity();
    test_low_coverage_contact_without_endpoint_stays_pending();
    test_low_coverage_contact_candidate_releases_at_original();
    test_contact_to_hand_transition_keeps_observed_anchor();
    return 0;
}
