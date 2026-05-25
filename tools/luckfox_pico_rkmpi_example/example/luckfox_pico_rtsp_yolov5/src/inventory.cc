// ============================================================================
//  inventory.cc
//  库存数据库实现（内存版）
// ============================================================================
#include "inventory.h"
#include "fridge_config.h"

#include <cstdio>
#include <limits>

// 用 coco_cls_to_name() 把 cls_id 转成可读名字。
// 注意：这里要 include "yolov5.h" 而不是直接 include "postprocess.h"，
// 因为 postprocess.h 里用到了 yolov5.h 中定义的 rknn_app_context_t 类型。
#include "yolov5.h"

namespace fridge {

int InventoryDB::find_closest(int cls_id, const BBox& query) const {
    int best = -1;
    float best_dist = std::numeric_limits<float>::max();
    for (size_t i = 0; i < items_.size(); ++i) {
        if (items_[i].cls_id != cls_id) continue;
        float d = center_distance(items_[i].pos, query);
        if (d < best_dist) {
            best_dist = d;
            best = (int)i;
        }
    }
    return best;
}

int InventoryDB::handle_in(const FinalEvent& e) {
    InventoryItem it;
    it.item_id = next_item_id_++;
    it.cls_id = e.cls_id;
    it.pos = e.to_pos;
    it.created_frame = e.frame_end;
    it.updated_frame = e.frame_end;
    items_.push_back(it);
    return it.item_id;
}

int InventoryDB::handle_out(const FinalEvent& e) {
    // 找位置最近的同类项删除
    int idx = find_closest(e.cls_id, e.from_pos);
    if (idx < 0) return -1;        // 库存里本来就没有，可能是误识别
    int id = items_[idx].item_id;
    items_.erase(items_.begin() + idx);
    return id;
}

int InventoryDB::handle_relocate(const FinalEvent& e) {
    int idx = find_closest(e.cls_id, e.from_pos);
    if (idx < 0) {
        // 库存里没有这个物品的记录，但又发生了"整理"
        // 退化策略：把它当作"新出现"，按 IN 处理
        return handle_in(e);
    }
    items_[idx].pos = e.to_pos;
    items_[idx].updated_frame = e.frame_end;
    return items_[idx].item_id;
}

std::vector<int> InventoryDB::apply_events(const std::vector<FinalEvent>& events) {
    std::vector<int> affected;
    for (const auto& e : events) {
        int id = -1;
        switch (e.type) {
            case FinalEventType::IN:       id = handle_in(e);       break;
            case FinalEventType::OUT:      id = handle_out(e);      break;
            case FinalEventType::RELOCATE: id = handle_relocate(e); break;
        }
        if (id >= 0) affected.push_back(id);

        // 这里也直接打日志（方便调试）
        printf("[EVENT] %s cls=%d(%s) "
               "from=(%.0f,%.0f)~(%.0f,%.0f) to=(%.0f,%.0f)~(%.0f,%.0f) "
               "frame=%d~%d hand=#%d -> item_id=%d\n",
               final_event_type_to_str(e.type),
               e.cls_id,
               coco_cls_to_name(e.cls_id),
               e.from_pos.x1, e.from_pos.y1, e.from_pos.x2, e.from_pos.y2,
               e.to_pos.x1, e.to_pos.y1, e.to_pos.x2, e.to_pos.y2,
               e.frame_begin, e.frame_end,
               e.hand_track_id,
               id);
    }
    return affected;
}

void InventoryDB::print(const char* prefix) const {
    printf("%s[Inventory] %zu items:\n", prefix, items_.size());
    for (const auto& it : items_) {
        printf("  - id=%d cls=%d(%s) pos=(%.0f,%.0f)~(%.0f,%.0f) "
               "created@%d updated@%d\n",
               it.item_id, it.cls_id,
               coco_cls_to_name(it.cls_id),
               it.pos.x1, it.pos.y1, it.pos.x2, it.pos.y2,
               it.created_frame, it.updated_frame);
    }
}

}  // namespace fridge
