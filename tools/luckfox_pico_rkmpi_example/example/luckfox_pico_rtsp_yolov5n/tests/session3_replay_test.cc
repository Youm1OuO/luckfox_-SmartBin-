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

void initial_no_hand_frame(fridge::SessionManager* session,
                           const std::vector<fridge::Detection>& foods,
                           int* frame) {
    const int id = (*frame)++;
    session->process_frame(foods, std::vector<fridge::BBox>(), id, id);
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
        const std::vector<fridge::Detection>& no_hand_foods, int* frame) {
    fridge::SettlementResult result;
    // 连续帧在这里仅让业务状态机获得所需的直接自匹配/缺失证据；测试
    // 不会构造投票快照或平均框。
    for (int i = 0; i < 8; ++i) {
        const int id = (*frame)++;
        fridge::FrameProcessResult frame_result = session->process_frame(
            no_hand_foods, std::vector<fridge::BBox>(), id, id);
        if (frame_result.no_hand_frame_processed) {
            result = frame_result.settlement;
            if (result.committed) break;
        }
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

void test_new_item_requires_d_chain() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 182, 100, 262, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(100, 100, 180, 180)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(2, 212, 100, 292, 180)),
               std::vector<fridge::BBox>(1, fridge::BBox(130, 100, 210, 180)), &frame);
    fridge::SettlementResult result = settle_after_hand(
        &session, std::vector<fridge::Detection>(1, det(2, 212, 100, 292, 180)), &frame);

    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(result, fridge::EventKind::IN, 1));
    const fridge::InventoryItem* inserted = session.inventory().find_by_item(1);
    assert(inserted != 0);
    assert(inserted->created_frame > 0);
    assert(inserted->updated_frame >= inserted->created_frame);
}

// 新物品可能只在一张有手帧露出局部框，手离开后才第一次看到完整框。
// 这正是实际“放下后迅速抽手”的常见路径；不能因此丢失已建立的 D 链路。
void test_new_item_can_confirm_from_first_no_hand_frame() {
    fridge::SessionManager session;
    session.start_new_session();
    session.init_from_backend(std::vector<fridge::InventoryItem>(), true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

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
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

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
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

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
    initial_no_hand_frame(&session,
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
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(), &frame);

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

    assert(final_frame.no_hand_frame_processed);
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
    initial_no_hand_frame(&session,
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

// 手边橙子区域短暂额外出现 cls=0 苹果候选时，它可以先成为 provisional D，
// 但不能在下一帧借远处静态旧苹果的严格框把 self_match_count 从 1 提到 2。
// 这里第二个小手框的中心位移故意把假 D 的预测框带到旧苹果附近，同时对
// 旧苹果的覆盖率仍低于 e2，因而旧苹果应被静态预约而非进入 HAND_*。
void test_hand_visible_d_does_not_confirm_from_strict_static_old_c() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // 静态旧苹果 C
    initial.push_back(item(2, 2, 500, 100, 600, 200));  // 橙子
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    std::vector<fridge::Detection> first_hand_foods = stable;
    // 和橙子框几乎完全重叠的跨类别假苹果候选。
    first_hand_foods.push_back(det(0, 500, 100, 600, 200, 0.56f));
    send_frame(&session, first_hand_foods,
               std::vector<fridge::BBox>(1, fridge::BBox(540, 140, 560, 160)),
               &frame);

    bool found_provisional_false_d = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        if (it->second.is_suspect_new && it->second.cls_id == 0) {
            found_provisional_false_d = true;
            assert(!it->second.promoted_to_working_inventory);
            assert(it->second.self_match_count == 1);
        }
    }
    assert(found_provisional_false_d);

    // 手中心左移 400px，使假 D 的宽松预测位置落到旧苹果；手框仅覆盖旧
    // 苹果 4%，因此该苹果仍是本帧可静态预约的旧 C。
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(140, 140, 160, 160)),
               &frame);

    bool false_d_still_unconfirmed = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        if (it->second.is_suspect_new && it->second.cls_id == 0) {
            false_d_still_unconfirmed = true;
            assert(!it->second.promoted_to_working_inventory);
            assert(it->second.self_match_count == 1);
        }
    }
    assert(false_d_still_unconfirmed);

    // 假 D 被让渡后，真实橙子仍可按原有 CONTACT 移动链路正常整理。
    std::vector<fridge::Detection> moved_orange;
    moved_orange.push_back(det(0, 100, 100, 200, 200));
    moved_orange.push_back(det(2, 520, 100, 620, 200));
    send_frame(&session, moved_orange,
               std::vector<fridge::BBox>(1, fridge::BBox(560, 140, 580, 160)),
               &frame);
    moved_orange[1] = det(2, 540, 100, 640, 200);
    send_frame(&session, moved_orange,
               std::vector<fridge::BBox>(1, fridge::BBox(580, 140, 600, 160)),
               &frame);
    moved_orange[1] = det(2, 560, 100, 660, 200);
    send_frame(&session, moved_orange,
               std::vector<fridge::BBox>(1, fridge::BBox(600, 140, 620, 160)),
               &frame);
    send_frame(&session, moved_orange,
               std::vector<fridge::BBox>(1, fridge::BBox(600, 140, 620, 160)),
               &frame);

    fridge::SettlementResult result = settle_after_hand(&session, moved_orange, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(1)->status ==
           fridge::ItemStatus::VISIBLE);
    assert(has_event(result, fridge::EventKind::MOVED, 2));
    assert(!has_event(result, fridge::EventKind::IN, 3));
    assert(!has_event(result, fridge::EventKind::OCCLUDED, 1));
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

// 旧物品刚被手遮住时，YOLO 常给出与完整框只部分重叠的局部框。它必须优先
// 认领给已有 item，而不是被 scan_or_update_suspects_ 误建成新的 D。
void test_partial_existing_item_is_not_registered_as_new_d() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 300, 300));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session, std::vector<fridge::Detection>(1, det(0, 100, 100, 300, 300)), &frame);

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
    initial_no_hand_frame(&session,
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

// 细节7：新建 C 在 t0/t1/t2 的保护期内仍会累计自己的 tentative B，
// 但不能将 B 排他认领、更不能把它建成同类 D；t3 成熟后才允许升级。
void test_new_track_claim_grace_defers_same_class_d() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    const fridge::BBox hand0(80, 80, 220, 220);
    const fridge::BBox hand1(200, 80, 340, 220);
    const std::vector<fridge::Detection> moved(1, det(0, 220, 100, 320, 200));
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, hand0), &frame);  // t0
    send_frame(&session, moved, std::vector<fridge::BBox>(1, hand1), &frame);  // t1
    const fridge::OperationTrack& after_t1 = session.operation_tracks().find(1)->second;
    assert(after_t1.claim_grace_remaining == 1);
    assert(after_t1.has_tentative_b_box);
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        assert(!it->second.is_suspect_new);
    }

    send_frame(&session, moved, std::vector<fridge::BBox>(1, hand1), &frame);  // t2
    const fridge::OperationTrack& after_t2 = session.operation_tracks().find(1)->second;
    assert(after_t2.claim_grace_remaining == 0);
    assert(after_t2.tentative_b_match_count >= 2);

    const fridge::BBox hand2(260, 80, 400, 220);
    const std::vector<fridge::Detection> moved_again(1, det(0, 280, 100, 380, 200));
    send_frame(&session, moved_again, std::vector<fridge::BBox>(1, hand2), &frame);  // t3
    const fridge::OperationTrack& after_t3 = session.operation_tracks().find(1)->second;
    assert(after_t3.hold_and_move);
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        assert(!it->second.is_suspect_new);
    }

    fridge::SettlementResult result = settle_after_hand(&session, moved_again, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 1);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::IN, 2));
}

// 有手阶段两个成熟同类 C 都能解释同一个 B 时，仍不得按 map 遍历顺序
// 抢身份或退化成新 D。手离开后若持续只看见一个框，细节9改为按一框
// 一物品结算：稳定保留一个，另一个 OUT。
void test_mature_same_class_tracks_keep_shared_b_ambiguous_until_no_hand_count_settlement() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 0, 220, 100, 320, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 220, 100, 320, 200));
    initial_no_hand_frame(&session, stable, &frame);

    const fridge::BBox hand(80, 80, 340, 220);
    const std::vector<fridge::Detection> shared(1, det(0, 160, 100, 260, 200));
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, hand), &frame);  // t0
    send_frame(&session, shared, std::vector<fridge::BBox>(1, hand), &frame);  // t1
    send_frame(&session, shared, std::vector<fridge::BBox>(1, hand), &frame);  // t2
    send_frame(&session, shared, std::vector<fridge::BBox>(1, hand), &frame);  // t3

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    assert(tracks.find(1) != tracks.end());
    assert(tracks.find(2) != tracks.end());
    assert(tracks.find(1)->second.claim_grace_remaining == 0);
    assert(tracks.find(2)->second.claim_grace_remaining == 0);
    assert(!tracks.find(1)->second.hold_and_move);
    assert(!tracks.find(2)->second.hold_and_move);
    for (std::map<int, fridge::OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        assert(!it->second.is_suspect_new);
    }

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        shared, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().size() == 2);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        shared, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.settlement.committed);
    // 两个候选成本完全相同，细节9规定以稳定 item_id 决胜保留 #1。
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) == 0);
    assert(!has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(second.settlement, fridge::EventKind::MOVED, 2));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 1));
    assert(has_event(second.settlement, fridge::EventKind::OUT, 2));
    assert(!has_event(second.settlement, fridge::EventKind::IN, 3));
}

// 手离开后的无手帧也要推进保护期。第一张无手 B 只能成为 fresh C 的
// tentative，不能被 POST_HAND_REVEAL_D 抢走；成熟后才恢复旧 C 的身份。
void test_no_hand_frame_respects_new_track_claim_grace() {
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
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 340, 220)), &frame);

    const std::vector<fridge::Detection> moved(1, det(0, 220, 100, 320, 200));
    const int first_no_hand = frame++;
    session.process_frame(moved, std::vector<fridge::BBox>(), first_no_hand,
                          first_no_hand);
    const std::map<int, fridge::OperationTrack>& after_first =
        session.operation_tracks();
    assert(after_first.find(1) != after_first.end());
    assert(after_first.find(1)->second.claim_grace_remaining == 0);
    assert(after_first.find(1)->second.has_tentative_b_box);
    for (std::map<int, fridge::OperationTrack>::const_iterator it = after_first.begin();
         it != after_first.end(); ++it) {
        assert(!it->second.is_suspect_new);
    }

    fridge::SettlementResult result = settle_after_hand(&session, moved, &frame);
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

// 旧苹果仍然有自己的严格检测框时，旁边贴手出现的第二个苹果不能被
// item#1 的宽松局部匹配吞掉；它必须走 D -> 工作库存 -> 正式 IN 链路。
void test_adjacent_same_class_new_item_gets_its_own_d_chain() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 800, 100, 1000, 300));
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
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

// 细节8-7.1：A 被移动到同类 B 旁边时，B 的独立原位框必须仍属于 B，
// A 则沿自己的 HAND 路径在无手帧完成 MOVED；不能增加同类 D，也不能 OUT A。
void test_active_a_moves_next_to_static_same_class_b() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // A
    initial.push_back(item(2, 0, 420, 100, 520, 200));  // B
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 420, 100, 520, 200));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    std::vector<fridge::Detection> moved_once;
    moved_once.push_back(det(0, 200, 100, 300, 200));
    moved_once.push_back(det(0, 420, 100, 520, 200));
    send_frame(&session, moved_once,
               std::vector<fridge::BBox>(1, fridge::BBox(180, 80, 320, 220)), &frame);
    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(det(0, 300, 100, 400, 200));
    final_boxes.push_back(det(0, 420, 100, 520, 200));
    send_frame(&session, final_boxes,
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 420, 220)), &frame);

    fridge::SettlementResult result = settle_after_hand(&session, final_boxes, &frame);
    assert(result.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::OUT, 1));
    assert(!has_event(result, fridge::EventKind::IN, 3));
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

// 细节9：A 向同类 B 靠近后，YOLO 只剩一个严格属于 B、又落在 A 路径
// 附近的框。该框必须一对一保留给 B，并成为“不是 A”的排除证据；A 在
// 连续两张无手帧后 OUT，而不是永久未决。
void test_same_class_static_neighbor_box_allows_active_a_out() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // A
    initial.push_back(item(2, 0, 340, 100, 440, 200));  // B
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 340, 100, 440, 200));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 340, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 420, 220)), &frame);

    const std::vector<fridge::Detection> merged_or_b_only(
        1, det(0, 340, 100, 440, 200));
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        merged_or_b_only, std::vector<fridge::BBox>(),
        first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().size() == 2);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        merged_or_b_only, std::vector<fridge::BBox>(),
        second_no_hand, second_no_hand);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) == 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(!has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(has_event(second.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(second.settlement, fridge::EventKind::IN, 3));
}

// 细节9-7.1：复现实机日志中的相邻橙子框。#24 的独立原位框要稳定保留，
// 被手取走的 #23 只在第二张直接无手帧 OUT，不能等到 #24 也被拿走。
void test_adjacent_same_class_real_boxes_out_only_removed_item() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(23, 2, 238, 252, 430, 443));  // A: orange
    initial.push_back(item(24, 2, 409, 245, 610, 443));  // B: orange
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(2, 238, 252, 430, 443));
    stable.push_back(det(2, 409, 245, 610, 443));
    initial_no_hand_frame(&session, stable, &frame);

    const std::vector<fridge::Detection> only_b(
        1, det(2, 409, 245, 610, 443));
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(220, 230, 440, 460)),
               &frame);
    send_frame(&session, only_b,
               std::vector<fridge::BBox>(1, fridge::BBox(300, 230, 520, 460)),
               &frame);
    send_frame(&session, only_b,
               std::vector<fridge::BBox>(1, fridge::BBox(360, 230, 580, 460)),
               &frame);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        only_b, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().size() == 2);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        only_b, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(23) == 0);
    assert(session.inventory().find_by_item(24) != 0);
    assert(has_event(second.settlement, fridge::EventKind::OUT, 23));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 24));
    assert(!has_event(second.settlement, fridge::EventKind::IN, 25));
}

// 细节9-7.6：一张无手帧短暂漏掉相邻同类 A 后立刻恢复两个独立框时，
// 数量不足候选必须撤销，不能在后续帧补发 OUT。
void test_adjacent_same_class_single_frame_deficit_recovers_without_out() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 0, 340, 100, 440, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 340, 100, 440, 200));
    initial_no_hand_frame(&session, stable, &frame);

    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 340, 220)), &frame);
    send_frame(&session, std::vector<fridge::Detection>(1, det(0, 340, 100, 440, 200)),
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 420, 220)), &frame);

    const std::vector<fridge::Detection> only_b(
        1, det(0, 340, 100, 440, 200));
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        only_b, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(!first.settlement.committed);
    assert(session.operation_pending());

    const int recovered_no_hand = frame++;
    fridge::FrameProcessResult recovered = session.process_frame(
        stable, std::vector<fridge::BBox>(), recovered_no_hand, recovered_no_hand);
    if (!recovered.settlement.committed) {
        recovered.settlement = settle_after_hand(&session, stable, &frame);
    }
    assert(recovered.settlement.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(!has_event(recovered.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(recovered.settlement, fridge::EventKind::OUT, 2));
}

// 细节9-5.5：上一张无手帧留下的 B 若突然跳到明显不连续的位置，不能继续
// 复用上一帧的 A 缺失计数；新的稳定位置必须从第一张直接无手帧重新计数。
void test_adjacent_same_class_deficit_requires_continuous_survivor_box() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));
    initial.push_back(item(2, 0, 340, 100, 440, 200));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 340, 100, 440, 200));
    initial_no_hand_frame(&session, stable, &frame);

    const std::vector<fridge::Detection> only_b(
        1, det(0, 340, 100, 440, 200));
    send_frame(&session, only_b,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 220, 220)), &frame);
    send_frame(&session, only_b,
               std::vector<fridge::BBox>(1, fridge::BBox(200, 80, 340, 220)), &frame);
    send_frame(&session, only_b,
               std::vector<fridge::BBox>(1, fridge::BBox(280, 80, 420, 220)), &frame);

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        only_b, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(!first.settlement.committed);
    assert(session.operation_pending());

    const std::vector<fridge::Detection> jumped_b(
        1, det(0, 700, 100, 800, 200));
    const int discontinuous_no_hand = frame++;
    fridge::FrameProcessResult discontinuous = session.process_frame(
        jumped_b, std::vector<fridge::BBox>(),
        discontinuous_no_hand, discontinuous_no_hand);
    assert(!discontinuous.settlement.committed);
    assert(session.operation_pending());
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);

    // 同一新位置再连续出现一帧，才从新的计数序列确认 #1 OUT。
    const int second_continuous_no_hand = frame++;
    fridge::FrameProcessResult second_continuous = session.process_frame(
        jumped_b, std::vector<fridge::BBox>(),
        second_continuous_no_hand, second_continuous_no_hand);
    assert(second_continuous.settlement.committed);
    assert(session.inventory().size() == 1);
    assert(session.inventory().find_by_item(1) == 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(second_continuous.settlement, fridge::EventKind::OUT, 1));
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

// 细节11-8.2：同一个被移动的旧 C 在有手帧中可能同时被 YOLO 给出一个
// 局部/过期候选和一个完整新位置框。后者不能被直接 promote 成工作库存
// D；它应保留为 C-D alias runtime track，并在两张直接无手帧后归回旧 C。
void test_quarantined_same_class_duplicate_merges_back_to_old_c() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 2, 100, 100, 200, 200));  // old orange C
    initial.push_back(item(2, 2, 700, 100, 800, 200));  // static same-class B
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(2, 100, 100, 200, 200));
    stable.push_back(det(2, 700, 100, 800, 200));
    initial_no_hand_frame(&session, stable, &frame);

    // t0: C is fully covered. t1/t2: the hand displacement predicts the
    // narrow decoy box; the complete box is the same physical C at its new
    // location. The old claim grace intentionally keeps existing behavior.
    const fridge::BBox old_hand(80, 80, 220, 220);
    const fridge::BBox moved_hand(370, 80, 610, 220);
    const fridge::Detection decoy = det(2, 440, 100, 540, 200);
    const fridge::Detection complete_c = det(2, 430, 95, 550, 205);
    std::vector<fridge::Detection> only_b;
    only_b.push_back(det(2, 700, 100, 800, 200));
    send_frame(&session, only_b, std::vector<fridge::BBox>(1, old_hand), &frame);

    std::vector<fridge::Detection> conflicting_hand_frame;
    conflicting_hand_frame.push_back(decoy);
    conflicting_hand_frame.push_back(complete_c);
    conflicting_hand_frame.push_back(det(2, 700, 100, 800, 200));
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);

    const std::map<int, fridge::OperationTrack>& hand_tracks =
        session.operation_tracks();
    assert(hand_tracks.find(1) != hand_tracks.end());
    assert(hand_tracks.find(1)->second.live_state ==
           fridge::LiveObservationState::C_D_ALIAS);
    bool found_quarantined_duplicate = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             hand_tracks.begin(); it != hand_tracks.end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (!track.is_suspect_new || !track.pending_d_quarantined_by_old_c ||
            !track.conflicting_old_item_ids.count(1)) {
            continue;
        }
        found_quarantined_duplicate = true;
        assert(track.item_id <= 0);
        assert(!track.promoted_to_working_inventory);
        assert(track.live_state == fridge::LiveObservationState::C_D_ALIAS);
        assert(track.live_state_provisional);
        assert(track.has_last_seen_box);
    }
    assert(found_quarantined_duplicate);
    assert(session.inventory().size() == 2);

    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(complete_c);
    final_boxes.push_back(det(2, 700, 100, 800, 200));
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(!has_event(first.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(first.settlement, fridge::EventKind::IN, 3));

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(second.settlement, fridge::EventKind::IN, 3));
}

// 细节12：有手期间 YOLO 可能给移动旧 C 额外带来一个同类假框。若手离开后
// C 已独立确认 MOVED、静止邻居也仍在，而该 D 既没有独立框也没有共享框，
// 第 1 张无手帧只能继续等待；第 2 张连续无手帧必须回收 runtime-only D，
// 不能让 defer-commit 无限循环，更不能产生 IN/OUT。
void test_quarantined_same_class_stale_alias_disappears_after_old_c_settles() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // moved old apple C
    initial.push_back(item(2, 0, 700, 100, 800, 200));  // static same-class B
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 700, 100, 800, 200));
    initial_no_hand_frame(&session, stable, &frame);

    const fridge::BBox old_hand(80, 80, 220, 220);
    const fridge::BBox moved_hand(370, 80, 610, 220);
    const fridge::Detection complete_c = det(0, 430, 95, 550, 205);
    const fridge::Detection false_d = det(0, 560, 100, 660, 200);
    const fridge::Detection static_b = det(0, 700, 100, 800, 200);

    std::vector<fridge::Detection> only_b;
    only_b.push_back(static_b);
    send_frame(&session, only_b, std::vector<fridge::BBox>(1, old_hand), &frame);

    std::vector<fridge::Detection> conflicting_hand_frame;
    conflicting_hand_frame.push_back(complete_c);
    conflicting_hand_frame.push_back(false_d);
    conflicting_hand_frame.push_back(static_b);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);
    send_frame(&session, conflicting_hand_frame,
               std::vector<fridge::BBox>(1, moved_hand), &frame);

    bool found_quarantined_false_d = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (!track.is_suspect_new || !track.pending_d_quarantined_by_old_c ||
            !track.conflicting_old_item_ids.count(1)) {
            continue;
        }
        found_quarantined_false_d = true;
        assert(track.item_id <= 0);
        assert(!track.promoted_to_working_inventory);
    }
    assert(found_quarantined_false_d);

    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(complete_c);
    final_boxes.push_back(static_b);
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());
    assert(!has_event(first.settlement, fridge::EventKind::IN, 3));
    assert(!has_event(first.settlement, fridge::EventKind::OUT, 1));

    bool saw_missing_count_one = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (track.is_suspect_new && track.pending_d_quarantined_by_old_c &&
            track.conflicting_old_item_ids.count(1)) {
            saw_missing_count_one = track.alias_no_hand_missing_count == 1;
        }
    }
    assert(saw_missing_count_one);

    // 手重新出现必须切断这段“D 消失”证据；不能把手前后两张无手帧
    // 拼成连续 2 帧并错误清理 alias。
    send_frame(&session, final_boxes, std::vector<fridge::BBox>(1, moved_hand), &frame);
    bool saw_missing_count_reset = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (track.is_suspect_new && track.pending_d_quarantined_by_old_c &&
            track.conflicting_old_item_ids.count(1)) {
            saw_missing_count_reset = track.alias_no_hand_missing_count == 0;
        }
    }
    assert(saw_missing_count_reset);

    const int first_after_hand = frame++;
    fridge::FrameProcessResult after_hand_first = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), first_after_hand, first_after_hand);
    assert(after_hand_first.no_hand_frame_processed);
    assert(!after_hand_first.settlement.committed);
    assert(session.operation_pending());

    const int second_after_hand = frame++;
    fridge::FrameProcessResult after_hand_second = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), second_after_hand, second_after_hand);
    assert(after_hand_second.no_hand_frame_processed);
    assert(!after_hand_second.settlement.committed);
    assert(session.operation_pending());

    const int third_after_hand = frame++;
    fridge::FrameProcessResult after_hand_third = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), third_after_hand, third_after_hand);
    assert(after_hand_third.no_hand_frame_processed);
    assert(after_hand_third.settlement.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(after_hand_third.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(after_hand_third.settlement, fridge::EventKind::IN, 3));
    assert(!has_event(after_hand_third.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(after_hand_third.settlement, fridge::EventKind::OUT, 2));
}

// 细节11：C 已在无手帧独立 release、随后又被手接触时，旧 runtime 会被
// 重建。仍被隔离的 D 必须继续与新 C 保持双向 alias；同一无手框应作为
// shared detection，而不是先把 D 误记为 stale missing。
void test_retouch_old_c_preserves_alias_links_and_shared_resolution() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // old apple C
    initial.push_back(item(2, 0, 700, 100, 800, 200));  // static same-class B
    session.init_from_backend(initial, true);
    int frame = 1;

    const fridge::Detection original_c = det(0, 100, 100, 200, 200);
    const fridge::Detection static_b = det(0, 700, 100, 800, 200);
    std::vector<fridge::Detection> stable;
    stable.push_back(original_c);
    stable.push_back(static_b);
    initial_no_hand_frame(&session, stable, &frame);

    const fridge::BBox original_hand(80, 80, 220, 220);
    const fridge::BBox moved_hand(370, 80, 610, 220);
    const fridge::Detection complete_c = det(0, 430, 95, 550, 205);
    const fridge::Detection alias_d = det(0, 560, 100, 660, 200);
    send_frame(&session, std::vector<fridge::Detection>(1, static_b),
               std::vector<fridge::BBox>(1, original_hand), &frame);

    std::vector<fridge::Detection> conflicting_hand_frame;
    conflicting_hand_frame.push_back(complete_c);
    conflicting_hand_frame.push_back(alias_d);
    conflicting_hand_frame.push_back(static_b);
    for (int i = 0; i < 3; ++i) {
        send_frame(&session, conflicting_hand_frame,
                   std::vector<fridge::BBox>(1, moved_hand), &frame);
    }

    int alias_key = 0;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        if (it->second.is_suspect_new &&
            it->second.pending_d_quarantined_by_old_c &&
            it->second.conflicting_old_item_ids.count(1)) {
            alias_key = it->first;
            break;
        }
    }
    assert(alias_key < 0);

    // C 在原位被独立 release，但 D 仍是 quarantined runtime。
    const int release_frame = frame++;
    fridge::FrameProcessResult released = session.process_frame(
        stable, std::vector<fridge::BBox>(), release_frame, release_frame);
    assert(released.no_hand_frame_processed);
    assert(session.operation_tracks().find(1) != session.operation_tracks().end());
    assert(session.operation_tracks().find(alias_key) != session.operation_tracks().end());
    assert(session.operation_tracks().find(1)->second.conflicting_suspect_keys.count(
               alias_key));

    // 重接触创建新的 C runtime。D 的 D -> C 关系必须恢复新 C 的 C -> D。
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, original_hand), &frame);
    const std::map<int, fridge::OperationTrack>& after_retouch =
        session.operation_tracks();
    assert(after_retouch.find(1) != after_retouch.end());
    assert(after_retouch.find(alias_key) != after_retouch.end());
    assert(after_retouch.find(1)->second.conflicting_suspect_keys.count(alias_key));
    assert(after_retouch.find(alias_key)->second.conflicting_old_item_ids.count(1));

    // C 现在移动到 D 上次出现的位置。无手阶段只给出同一个框，必须按
    // shared detection 连续两帧回收重复 D，而不是走 D missing 链路。
    std::vector<fridge::Detection> moved_hand_frame;
    moved_hand_frame.push_back(alias_d);
    moved_hand_frame.push_back(static_b);
    for (int i = 0; i < 3; ++i) {
        send_frame(&session, moved_hand_frame,
                   std::vector<fridge::BBox>(1, moved_hand), &frame);
    }

    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        moved_hand_frame, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_tracks().find(alias_key) != session.operation_tracks().end());
    const fridge::OperationTrack& d_after_first =
        session.operation_tracks().find(alias_key)->second;
    assert(d_after_first.alias_no_hand_match_count == 1);
    assert(d_after_first.alias_no_hand_missing_count == 0);
    assert(d_after_first.alias_no_hand_matched_this_frame);

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        moved_hand_frame, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    assert(session.operation_tracks().find(alias_key) == session.operation_tracks().end());
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(!has_event(second.settlement, fridge::EventKind::IN, 3));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 1));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 2));
}

// 细节11-8.3：防止假 D 不能退化为吞掉真实 D。C 的完整新框和一个真实
// 同类 D 都在有手阶段形成 provisional alias；无手两帧分别得到不同框后，
// C 保留 old item_id 并 MOVED，D 才正式 IN。
void test_quarantined_same_class_real_d_confirms_after_distinct_no_hand_boxes() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // old apple C
    session.init_from_backend(initial, true);
    int frame = 1;
    initial_no_hand_frame(&session,
        std::vector<fridge::Detection>(1, det(0, 100, 100, 200, 200)), &frame);

    const fridge::BBox old_hand(80, 80, 220, 220);
    const fridge::BBox c_hand(370, 80, 610, 220);
    const fridge::BBox c_and_d_hand(180, 80, 800, 220);
    const fridge::Detection decoy = det(0, 440, 100, 540, 200);
    const fridge::Detection complete_c = det(0, 430, 95, 550, 205);
    const fridge::Detection real_d = det(0, 650, 100, 750, 200);
    send_frame(&session, std::vector<fridge::Detection>(),
               std::vector<fridge::BBox>(1, old_hand), &frame);

    std::vector<fridge::Detection> c_conflict;
    c_conflict.push_back(decoy);
    c_conflict.push_back(complete_c);
    send_frame(&session, c_conflict, std::vector<fridge::BBox>(1, c_hand), &frame);
    send_frame(&session, c_conflict, std::vector<fridge::BBox>(1, c_hand), &frame);

    std::vector<fridge::Detection> c_and_d_conflict;
    c_and_d_conflict.push_back(decoy);
    c_and_d_conflict.push_back(complete_c);
    c_and_d_conflict.push_back(real_d);
    send_frame(&session, c_and_d_conflict,
               std::vector<fridge::BBox>(1, c_and_d_hand), &frame);

    bool found_c_alias = false;
    bool found_real_d_alias = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        const fridge::OperationTrack& track = it->second;
        if (!track.is_suspect_new || !track.pending_d_quarantined_by_old_c ||
            !track.conflicting_old_item_ids.count(1)) {
            continue;
        }
        assert(!track.promoted_to_working_inventory);
        if (track.last_seen_box.x1 < 600.0f) {
            found_c_alias = true;
        } else {
            found_real_d_alias = true;
        }
    }
    assert(found_c_alias);
    assert(found_real_d_alias);

    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(complete_c);
    final_boxes.push_back(real_d);
    const int first_no_hand = frame++;
    fridge::FrameProcessResult first = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), first_no_hand, first_no_hand);
    assert(first.no_hand_frame_processed);
    assert(!first.settlement.committed);
    assert(session.operation_pending());

    const int second_no_hand = frame++;
    fridge::FrameProcessResult second = session.process_frame(
        final_boxes, std::vector<fridge::BBox>(), second_no_hand, second_no_hand);
    assert(second.no_hand_frame_processed);
    assert(second.settlement.committed);
    assert(session.inventory().size() == 2);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(has_event(second.settlement, fridge::EventKind::MOVED, 1));
    assert(has_event(second.settlement, fridge::EventKind::IN, 2));
    assert(!has_event(second.settlement, fridge::EventKind::OUT, 1));
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

// 细节15-5.2：C1 已被手影响时，不得先以更低几何成本参加静态预约竞争，
// 否则会挡住真正静态的 C2，令既有 HAND_VISIBLE_D 借 C2 的框自匹配升级。
void test_hand_affected_old_c_does_not_block_static_owner_reservation() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 200, 200));  // C1，第二帧被手影响
    initial.push_back(item(2, 0, 130, 100, 230, 200));  // C2，第二帧静态
    initial.push_back(item(3, 2, 500, 100, 600, 200));  // 仅用于制造假 D 的橙子
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 130, 100, 230, 200));
    stable.push_back(det(2, 500, 100, 600, 200));
    initial_no_hand_frame(&session, stable, &frame);

    // t0 在橙子处建立一个 cls=0 的 HAND_VISIBLE_D。手中心为 512.5，下一帧
    // 移到 117.5 时，它的预测位置恰好落到 C2 的近原位框。
    std::vector<fridge::Detection> first = stable;
    first.push_back(det(0, 500, 100, 600, 200, 0.56f));
    send_frame(&session, first,
               std::vector<fridge::BBox>(1, fridge::BBox(502.5f, 140, 522.5f, 160)),
               &frame);

    bool found_false_d = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        if (it->second.is_suspect_new && it->second.cls_id == 0) {
            found_false_d = true;
            assert(it->second.self_match_count == 1);
            assert(!it->second.promoted_to_working_inventory);
        }
    }
    assert(found_false_d);

    // 这个框对 C1 的 strict cost 更低，但手只覆盖 C1 的 35%，只覆盖 C2
    // 的 5%。因此 C1 必须在候选写入前被排除，C2 才能唯一预约该检测框。
    std::vector<fridge::Detection> second;
    second.push_back(det(0, 105, 100, 205, 200));
    second.push_back(det(2, 500, 100, 600, 200));
    send_frame(&session, second,
               std::vector<fridge::BBox>(1, fridge::BBox(100, 100, 135, 200)),
               &frame);

    bool false_d_still_unconfirmed = false;
    for (std::map<int, fridge::OperationTrack>::const_iterator it =
             session.operation_tracks().begin();
         it != session.operation_tracks().end(); ++it) {
        if (it->second.is_suspect_new && it->second.cls_id == 0) {
            false_d_still_unconfirmed = true;
            assert(it->second.self_match_count == 1);
            assert(!it->second.promoted_to_working_inventory);
        }
    }
    assert(false_d_still_unconfirmed);
}

// 同一个回放同时覆盖细节15-4.1（B 完整原位框）、4.2（B 只露局部框）和
// 检测数组顺序独立性。真正移动的只有 soda_can；A 暂时漏检，B 的原位观测
// 不能被 A 的宽松手路径借走。
void run_same_class_old_owner_replay(bool b_is_partial, bool reverse_detection_order) {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 25, 520, 100, 620, 200));  // soda_can，唯一真实 MOVED
    initial.push_back(item(2, 0, 100, 100, 200, 200));   // apple A，暂时漏检
    initial.push_back(item(3, 0, 260, 100, 360, 200));   // apple B，始终留在原位
    session.init_from_backend(initial, true);
    int frame = 1;

    std::vector<fridge::Detection> stable;
    stable.push_back(det(25, 520, 100, 620, 200));
    stable.push_back(det(0, 100, 100, 200, 200));
    stable.push_back(det(0, 260, 100, 360, 200));
    initial_no_hand_frame(&session, stable, &frame);

    const fridge::BBox hand0(50, 50, 680, 250);
    const fridge::BBox hand1(120, 50, 750, 250);
    const fridge::BBox hand2(190, 50, 820, 250);
    const fridge::BBox hand3(260, 50, 890, 250);
    const fridge::Detection b_full = det(0, 260, 100, 360, 200);
    const fridge::Detection b_partial = det(0, 270, 100, 330, 200);

    // A 在 t0 已被大手框完整盖住且漏检；B 与 soda 仍给出各自的真实框。
    std::vector<fridge::Detection> first;
    first.push_back(det(25, 520, 100, 620, 200));
    first.push_back(b_full);
    if (reverse_detection_order) std::reverse(first.begin(), first.end());
    send_frame(&session, first, std::vector<fridge::BBox>(1, hand0), &frame);

    const fridge::Detection b = b_is_partial ? b_partial : b_full;
    for (int i = 0; i < 3; ++i) {
        std::vector<fridge::Detection> moving;
        moving.push_back(det(25, 590.0f + 70.0f * i, 100,
                             690.0f + 70.0f * i, 200));
        moving.push_back(b);
        if (reverse_detection_order) std::reverse(moving.begin(), moving.end());
        const fridge::BBox hand = i == 0 ? hand1 : (i == 1 ? hand2 : hand3);
        send_frame(&session, moving, std::vector<fridge::BBox>(1, hand), &frame);
    }

    std::vector<fridge::Detection> final_boxes;
    final_boxes.push_back(det(25, 730, 100, 830, 200));
    final_boxes.push_back(det(0, 100, 100, 200, 200));
    final_boxes.push_back(b_full);
    if (reverse_detection_order) std::reverse(final_boxes.begin(), final_boxes.end());
    fridge::SettlementResult result = settle_after_hand(&session, final_boxes, &frame);

    assert(result.committed);
    assert(session.inventory().size() == 3);
    assert(session.inventory().find_by_item(1) != 0);
    assert(session.inventory().find_by_item(2) != 0);
    assert(session.inventory().find_by_item(3) != 0);
    assert(session.inventory().find_by_item(2)->status == fridge::ItemStatus::VISIBLE);
    assert(session.inventory().find_by_item(3)->status == fridge::ItemStatus::VISIBLE);
    assert(result.events.size() == 1);
    assert(has_event(result, fridge::EventKind::MOVED, 1));
    assert(!has_event(result, fridge::EventKind::MOVED, 2));
    assert(!has_event(result, fridge::EventKind::MOVED, 3));
    assert(!has_event(result, fridge::EventKind::OCCLUDED, 2));
    assert(!has_event(result, fridge::EventKind::OCCLUDED, 3));
    assert(!has_event(result, fridge::EventKind::IN, 4));
    assert(!has_event(result, fridge::EventKind::OUT, 2));
    assert(!has_event(result, fridge::EventKind::OUT, 3));
}

void test_same_class_old_owner_blocks_path_match_before_identity_swap() {
    run_same_class_old_owner_replay(false, false);
    run_same_class_old_owner_replay(false, true);
    run_same_class_old_owner_replay(true, false);
    run_same_class_old_owner_replay(true, true);
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

// 细节15-2.3：两个旧 C 对同一个局部框都只有同级的连续匹配时，局部框
// 更靠近其中一个 C 的中心也不能决定 owner。必须保留为原位置歧义，避免
// 把遮挡导致的框中心偏移误写成某一个 C 的 tentative B / MOVED 证据。
void test_same_grade_local_old_owners_remain_ambiguous_despite_center_distance() {
    fridge::SessionManager session;
    session.start_new_session();
    std::vector<fridge::InventoryItem> initial;
    initial.push_back(item(1, 0, 100, 100, 300, 300));
    initial.push_back(item(2, 0, 400, 100, 600, 300));
    session.init_from_backend(initial, true);
    int frame = 1;
    std::vector<fridge::Detection> stable;
    stable.push_back(det(0, 100, 100, 300, 300));
    stable.push_back(det(0, 400, 100, 600, 300));
    initial_no_hand_frame(&session, stable, &frame);

    // t0 为两个旧 C 分别建立自己的 last_hand_block_box。
    send_frame(&session, stable,
               std::vector<fridge::BBox>(1, fridge::BBox(80, 80, 620, 320)), &frame);

    // 该局部框与两个 last_hand_block_box 都只满足 TRACK 连续性；它更靠近
    // item#2，但对 item#1 也仍在相同的连续性等级内。它不满足任一完整
    // 原位置的 strict / hand_partial，因此不能被 old-position 分支掩盖。
    const std::vector<fridge::Detection> shared_local(
        1, det(0, 259, 100, 459, 300));
    send_frame(&session, shared_local,
               std::vector<fridge::BBox>(1, fridge::BBox(100, 80, 640, 320)), &frame);

    const std::map<int, fridge::OperationTrack>& tracks = session.operation_tracks();
    assert(tracks.find(1) != tracks.end());
    assert(tracks.find(2) != tracks.end());
    assert(!tracks.find(1)->second.has_tentative_b_box);
    assert(!tracks.find(2)->second.has_tentative_b_box);
    assert(!tracks.find(1)->second.hold_and_move);
    assert(!tracks.find(2)->second.hold_and_move);
    for (std::map<int, fridge::OperationTrack>::const_iterator it = tracks.begin();
         it != tracks.end(); ++it) {
        assert(!it->second.is_suspect_new);
    }
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
    test_hand_visible_d_does_not_confirm_from_strict_static_old_c();
    test_new_item_dropped_before_hand_moves_away_keeps_its_identity();
    test_partial_existing_item_is_not_registered_as_new_d();
    test_fast_same_class_b_becomes_c_candidate_not_d();
    test_new_track_claim_grace_defers_same_class_d();
    test_mature_same_class_tracks_keep_shared_b_ambiguous_until_no_hand_count_settlement();
    test_no_hand_frame_respects_new_track_claim_grace();
    test_drop_requires_continuous_evidence();
    test_ambiguous_hand_partial_box_does_not_create_d();
    test_adjacent_same_class_new_item_gets_its_own_d_chain();
    test_unbound_no_hand_box_never_auto_in();
    test_moved_front_item_occludes_then_reveals();
    test_low_coverage_contact_move_keeps_identity();
    test_low_coverage_contact_without_endpoint_stays_pending();
    test_low_coverage_contact_candidate_releases_at_original();
    test_contact_to_hand_transition_keeps_observed_anchor();
    test_stationary_hand_frame_does_not_change_hand_evidence();
    test_post_hand_reveal_commits_on_second_direct_frame();
    test_c_reappear_commits_after_second_direct_frame();
    test_active_a_moves_next_to_static_same_class_b();
    test_stale_same_class_reappear_candidate_falls_back_to_moved_a();
    test_stale_same_class_reappear_candidate_without_a_waits_then_out();
    test_provisional_static_a_reopens_when_moved_later_in_same_operation();
    test_same_class_static_neighbor_box_allows_active_a_out();
    test_adjacent_same_class_real_boxes_out_only_removed_item();
    test_adjacent_same_class_single_frame_deficit_recovers_without_out();
    test_adjacent_same_class_deficit_requires_continuous_survivor_box();
    test_out_requires_two_direct_missing_frames();
    test_no_hand_occlusion_uses_cover_union();
    test_static_gray_zone_old_c_settles_before_r15_stale_alias_cleanup();
    test_static_gray_zone_evidence_resets_when_hand_returns();
    test_quarantined_same_class_duplicate_merges_back_to_old_c();
    test_quarantined_same_class_stale_alias_disappears_after_old_c_settles();
    test_retouch_old_c_preserves_alias_links_and_shared_resolution();
    test_quarantined_same_class_real_d_confirms_after_distinct_no_hand_boxes();
    test_cover_union_keeps_multiple_tiny_residuals_until_total_area_check();
    test_hand_affected_old_c_does_not_block_static_owner_reservation();
    test_same_class_old_owner_blocks_path_match_before_identity_swap();
    test_direct_original_evidence_rolls_back_provisional_moved();
    test_same_grade_local_old_owners_remain_ambiguous_despite_center_distance();
    return 0;
}
