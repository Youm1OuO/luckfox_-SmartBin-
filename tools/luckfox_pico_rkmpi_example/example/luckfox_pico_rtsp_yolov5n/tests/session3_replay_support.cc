// Shared host-side replay fixture and assertions.
#include "session3_replay_support.h"

// inventory.cc debug output references this symbol; replay does not link postprocess.
char* coco_cls_to_name(int) {
    static char name[] = "test";
    return name;
}

namespace session3_replay {

fridge::Detection det(int cls, float x1, float y1, float x2, float y2,
                      float score) {
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
}  // namespace session3_replay
