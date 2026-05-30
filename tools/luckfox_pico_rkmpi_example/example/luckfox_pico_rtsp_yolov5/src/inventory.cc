// ============================================================================
//  inventory.cc
//  本地工作库存实现 — item_id 身份版
// ============================================================================
#include "inventory.h"
#include "fridge_config.h"

#include <cstdio>
#include <cstring>
#include "yolov5.h"   // coco_cls_to_name()

namespace fridge {

const char* item_status_to_str(ItemStatus s) {
    switch (s) {
        case ItemStatus::VISIBLE:  return "VISIBLE";
        case ItemStatus::OCCLUDED: return "OCCLUDED";
        case ItemStatus::GONE:     return "GONE";
    }
    return "?";
}

int InventoryDB::add_item(int track_id, int cls_id, const BBox& box,
                          float score, int frame_id) {
    InventoryItem it;
    it.item_id = next_item_id_++;
    it.track_id = track_id;
    it.cls_id = cls_id;
    it.box = box;
    it.score = score;
    it.status = ItemStatus::VISIBLE;
    it.created_frame = frame_id;
    it.updated_frame = frame_id;
    it.last_seen_frame = frame_id;
    items_[it.item_id] = it;
    return it.item_id;
}

InventoryItem* InventoryDB::find_by_track(int track_id) {
    for (auto& kv : items_) {
        if (kv.second.track_id == track_id) return &kv.second;
    }
    return nullptr;
}

const InventoryItem* InventoryDB::find_by_track(int track_id) const {
    for (const auto& kv : items_) {
        if (kv.second.track_id == track_id) return &kv.second;
    }
    return nullptr;
}

InventoryItem* InventoryDB::find_by_item(int item_id) {
    auto it = items_.find(item_id);
    return it == items_.end() ? nullptr : &it->second;
}

void InventoryDB::update_seen(int item_id, int track_id, const BBox& box,
                              float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.track_id = track_id;
    it->second.box = box;
    it->second.score = score;
    it->second.status = ItemStatus::VISIBLE;
    it->second.updated_frame = frame_id;
    it->second.last_seen_frame = frame_id;
}

void InventoryDB::mark_occluded(int item_id) {
    auto it = items_.find(item_id);
    if (it != items_.end() && it->second.status == ItemStatus::VISIBLE) {
        it->second.status = ItemStatus::OCCLUDED;
    }
}

void InventoryDB::relocate_item(int item_id, int new_track_id, const BBox& new_box,
                                float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.track_id = new_track_id;   // 重新绑定 track（身份 item_id 不变！）
    it->second.box = new_box;
    it->second.score = score;
    it->second.status = ItemStatus::VISIBLE;
    it->second.updated_frame = frame_id;
    it->second.last_seen_frame = frame_id;
}

void InventoryDB::remove_item(int item_id) {
    items_.erase(item_id);
}

size_t InventoryDB::count_visible() const {
    size_t n = 0;
    for (const auto& kv : items_) {
        if (kv.second.status == ItemStatus::VISIBLE) n++;
    }
    return n;
}

size_t InventoryDB::count_occluded() const {
    size_t n = 0;
    for (const auto& kv : items_) {
        if (kv.second.status == ItemStatus::OCCLUDED) n++;
    }
    return n;
}

void InventoryDB::print(const char* prefix) const {
    printf("%s[Inventory] %zu items (visible=%zu, occluded=%zu)\n",
           prefix, items_.size(), count_visible(), count_occluded());
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        printf("%s  - item#%d cls=%d(%s) status=%s "
               "pos=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f tid=%d seen@%d\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               item_status_to_str(it.status),
               it.box.x1, it.box.y1, it.box.x2, it.box.y2,
               it.score, it.track_id, it.last_seen_frame);
    }
}

std::string InventoryDB::to_json(const char* device_id, long long timestamp_ms) const {
    // 手写 JSON（避免引入第三方库）。格式：
    // {"device_id":"...","timestamp":...,"event_type":"DOOR_CLOSE","inventory":[...]}
    std::string s;
    char buf[512];

    snprintf(buf, sizeof(buf),
             "{\"device_id\":\"%s\",\"timestamp\":%lld,"
             "\"event_type\":\"DOOR_CLOSE\",\"inventory\":[",
             device_id, timestamp_ms);
    s += buf;

    bool first = true;
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        // 只上报还在库存里的（VISIBLE / OCCLUDED 都算"在冰箱里"）
        snprintf(buf, sizeof(buf),
                 "%s{\"item_id\":%d,\"category\":\"%s\",\"status\":\"%s\","
                 "\"bbox\":[%.0f,%.0f,%.0f,%.0f]}",
                 first ? "" : ",",
                 it.item_id, coco_cls_to_name(it.cls_id),
                 item_status_to_str(it.status),
                 it.box.x1, it.box.y1, it.box.x2, it.box.y2);
        s += buf;
        first = false;
    }
    s += "]}";
    return s;
}

}  // namespace fridge
