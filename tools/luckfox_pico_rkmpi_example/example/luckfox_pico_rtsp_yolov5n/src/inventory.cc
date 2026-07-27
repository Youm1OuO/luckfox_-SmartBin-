// ============================================================================
//  inventory.cc
//  本地工作库存实现 — 新业务流程6：三态模型
// ============================================================================
#include "inventory.h"
#include "fridge_config.h"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "yolov5n.h"

namespace fridge {

const char* item_status_to_str(ItemStatus s) {
    switch (s) {
        case ItemStatus::VISIBLE:  return "可见";
        case ItemStatus::OCCLUDED: return "遮挡";
        case ItemStatus::OUT:      return "出库";
    }
    return "?";
}

int InventoryDB::add_item(int cls_id, const BBox& box,
                          float score, int frame_id, long long time_ms) {
    InventoryItem it;
    it.item_id = next_item_id_++;
    it.cls_id = cls_id;
    it.anchor_box = box;
    it.anchor_valid = true;
    it.last_seen_box = box;
    it.score = score;
    it.status = ItemStatus::VISIBLE;  // 新入库默认"可见"
    it.no_occluder_count = 0;
    it.duplicate_count = 0;
    it.created_frame = frame_id;
    it.updated_frame = frame_id;
    it.created_time_ms = time_ms;
    it.out_time_ms = 0;
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

void InventoryDB::set_status(int item_id, ItemStatus new_status, long long time_ms) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.status = new_status;
    if (new_status == ItemStatus::OUT) {
        it->second.out_time_ms = time_ms;
    } else {
        it->second.out_time_ms = 0;
    }
}

void InventoryDB::update_seen_item(int item_id, const BBox& box,
                                    float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.last_seen_box = box;
    it->second.score = score;
    it->second.updated_frame = frame_id;
}

void InventoryDB::update_anchor_item(int item_id, const BBox& box,
                                     float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.anchor_box = box;
    it->second.anchor_valid = true;
    it->second.last_seen_box = box;
    it->second.score = score;
    it->second.updated_frame = frame_id;
}

void InventoryDB::replace_all(const std::map<int, InventoryItem>& items,
                              int next_item_id) {
    items_ = items;
    next_item_id_ = std::max(1, next_item_id);
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

void InventoryDB::cleanup_expired(long long now_ms) {
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
}

void InventoryDB::print(const char* prefix) const {
    // 先打印可见物品
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        if (it.status != ItemStatus::VISIBLE) continue;
        printf("%s  - item#%d cls=%d(%s) [可见] "
               "anchor=(%.0f,%.0f)~(%.0f,%.0f) last_seen=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               it.anchor_box.x1, it.anchor_box.y1, it.anchor_box.x2, it.anchor_box.y2,
               it.last_seen_box.x1, it.last_seen_box.y1, it.last_seen_box.x2, it.last_seen_box.y2,
               it.score);
    }
    // 再打印遮挡物品
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        if (it.status != ItemStatus::OCCLUDED) continue;
        printf("%s  - item#%d cls=%d(%s) [遮挡] "
               "anchor=(%.0f,%.0f)~(%.0f,%.0f) last_seen=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               it.anchor_box.x1, it.anchor_box.y1, it.anchor_box.x2, it.anchor_box.y2,
               it.last_seen_box.x1, it.last_seen_box.y1, it.last_seen_box.x2, it.last_seen_box.y2,
               it.score);
    }
    // OUT 状态仅作为内部恢复匹配的临时记录，不在库存列表里展示。
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
        snprintf(buf, sizeof(buf),
                 "%s{\"item_id\":%d,\"category\":\"%s\","
                 "\"status\":\"%s\",\"bbox\":[%.0f,%.0f,%.0f,%.0f]}",
                 first ? "" : ",",
                 it.item_id, cls_id_to_chinese(it.cls_id),
                 item_status_to_str(it.status),
                 it.anchor_box.x1, it.anchor_box.y1,
                 it.anchor_box.x2, it.anchor_box.y2);
        s += buf;
        first = false;
    }
    s += "]}";
    return s;
}

}  // namespace fridge
