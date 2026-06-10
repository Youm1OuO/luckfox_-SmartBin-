// ============================================================================
//  inventory.cc
//  本地工作库存实现 — 新业务流程6：三态模型
// ============================================================================
#include "inventory.h"
#include "fridge_config.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include "yolov5.h"

namespace fridge {

const char* item_status_to_str(ItemStatus s) {
    switch (s) {
        case ItemStatus::VISIBLE:  return "可见";
        case ItemStatus::OCCLUDED: return "遮挡";
        case ItemStatus::OUT:      return "出库";
    }
    return "?";
}

int InventoryDB::add_item(int track_id, int cls_id, const BBox& box,
                          float score, int frame_id, long long time_ms) {
    InventoryItem it;
    it.item_id = next_item_id_++;
    it.track_id = track_id;
    it.cls_id = cls_id;
    it.box = box;
    it.score = score;
    it.status = ItemStatus::VISIBLE;  // 新入库默认"可见"
    it.created_frame = frame_id;
    it.updated_frame = frame_id;
    it.created_time_ms = time_ms;
    it.out_time_ms = 0;
    it.out_order_id = 0;
    items_[it.item_id] = it;
    return it.item_id;
}

InventoryItem* InventoryDB::find_by_item(int item_id) {
    auto it = items_.find(item_id);
    return it == items_.end() ? nullptr : &it->second;
}

const InventoryItem* InventoryDB::find_by_item(int item_id) const {
    auto it = items_.find(item_id);
    return it == items_.end() ? nullptr : &it->second;
}

bool InventoryDB::set_status(int item_id, ItemStatus new_status, long long time_ms) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return false;

    ItemStatus old_status = it->second.status;
    if (old_status == new_status) {
        return false;
    }

    it->second.status = new_status;
    if (new_status == ItemStatus::OUT) {
        it->second.out_time_ms = time_ms;
        it->second.out_order_id = next_out_order_id_++;
    } else {
        it->second.out_time_ms = 0;
        it->second.out_order_id = 0;
    }
    return true;
}

void InventoryDB::update_item(int item_id, int track_id, const BBox& box,
                               float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.track_id = track_id;
    it->second.box = box;
    it->second.score = score;
    it->second.updated_frame = frame_id;
}

void InventoryDB::remove_item(int item_id) {
    items_.erase(item_id);
}

size_t InventoryDB::count_by_status(ItemStatus s) const {
    size_t n = 0;
    for (const auto& kv : items_) {
        if (kv.second.status == s) n++;
    }
    return n;
}

bool InventoryDB::cleanup_expired(long long now_ms) {
    std::vector<int> expired;
    for (const auto& kv : items_) {
        if (kv.second.status == ItemStatus::OUT && kv.second.out_time_ms > 0) {
            if (now_ms - kv.second.out_time_ms > OUT_ITEM_EXPIRE_MS) {
                expired.push_back(kv.first);
            }
        }
    }
    for (int id : expired) {
        printf("[INVENTORY] item#%d 出库超时，清除记录\n", id);
        items_.erase(id);
    }
    return !expired.empty();
}

void InventoryDB::print(const char* prefix) const {
    // 先打印可见物品
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        if (it.status != ItemStatus::VISIBLE) continue;
        printf("%s  - item#%d cls=%d(%s) [可见] "
               "pos=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f tid=%d\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               it.box.x1, it.box.y1, it.box.x2, it.box.y2,
               it.score, it.track_id);
    }
    // 再打印遮挡物品
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        if (it.status != ItemStatus::OCCLUDED) continue;
        printf("%s  - item#%d cls=%d(%s) [遮挡] "
               "pos=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f tid=%d\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               it.box.x1, it.box.y1, it.box.x2, it.box.y2,
               it.score, it.track_id);
    }
    // 再打印出库物品
    bool has_out = false;
    for (const auto& kv : items_) {
        if (kv.second.status == ItemStatus::OUT) { has_out = true; break; }
    }
    if (has_out) {
        printf("%s  --- 出库（临时记录）---\n", prefix);
        for (const auto& kv : items_) {
            const auto& it = kv.second;
            if (it.status != ItemStatus::OUT) continue;
            printf("%s  - out#%lld item#%d cls=%d(%s) [出库] "
                   "pos=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f tid=%d\n",
                   prefix,
                   it.out_order_id, it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
                   it.box.x1, it.box.y1, it.box.x2, it.box.y2,
                   it.score, it.track_id);
        }
    }
}

std::string InventoryDB::to_json(const char* device_id, long long timestamp_ms,
                                 const char* session_id) const {
    std::string s;
    char buf[768];

    snprintf(buf, sizeof(buf),
             "{\"device_id\":\"%s\",\"timestamp\":%lld,"
             "\"session_id\":\"%s\","
             "\"event_type\":\"DOOR_CLOSE\",\"inventory\":[",
             device_id, timestamp_ms, session_id ? session_id : "");
    s += buf;

    bool first = true;
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        // 只上报 VISIBLE + OCCLUDED（在冰箱中的物品）
        if (it.status == ItemStatus::OUT) continue;
        float bw = it.box.x2 - it.box.x1;
        float bh = it.box.y2 - it.box.y1;
        snprintf(buf, sizeof(buf),
                 "%s{\"item_id\":%d,\"category\":\"%s\",\"fine_class\":\"%s\","
                 "\"status\":\"%s\",\"bbox\":[%.0f,%.0f,%.0f,%.0f]}",
                 first ? "" : ",",
                 it.item_id, coarse_category(it.cls_id), coco_cls_to_name(it.cls_id),
                 item_status_to_str(it.status),
                 it.box.x1, it.box.y1, bw, bh);
        s += buf;
        first = false;
    }
    s += "]}";
    return s;
}

}  // namespace fridge
