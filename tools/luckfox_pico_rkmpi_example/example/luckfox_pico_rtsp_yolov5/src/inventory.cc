// ============================================================================
//  inventory.cc
//  本地工作库存实现 — v4 简化版（无【被遮挡】状态）
// ============================================================================
#include "inventory.h"
#include "fridge_config.h"

#include <cstdio>
#include <cstring>
#include "yolov5.h"

namespace fridge {

const char* item_status_to_str(ItemStatus s) {
    switch (s) {
        case ItemStatus::INVENTORY: return "在库";
        case ItemStatus::TAKEN:     return "被拿走";
        case ItemStatus::HELD:      return "遮挡";
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
    it.status = ItemStatus::INVENTORY;  // 新入库默认"在库"
    it.created_frame = frame_id;
    it.updated_frame = frame_id;
    it.last_seen_frame = frame_id;
    it.stable_frames = 0;
    it.last_bbox = box;
    it.created_time_ms = time_ms;
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

const InventoryItem* InventoryDB::find_by_item(int item_id) const {
    auto it = items_.find(item_id);
    return it == items_.end() ? nullptr : &it->second;
}

void InventoryDB::update_seen(int item_id, int track_id, const BBox& box,
                              float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.last_bbox = it->second.box;
    it->second.track_id = track_id;
    it->second.box = box;
    it->second.score = score;
    it->second.updated_frame = frame_id;
    it->second.last_seen_frame = frame_id;
    it->second.stable_frames++;
}

void InventoryDB::set_status(int item_id, ItemStatus new_status) {
    auto it = items_.find(item_id);
    if (it != items_.end()) {
        it->second.status = new_status;
    }
}

void InventoryDB::increment_stable_frames(int item_id) {
    auto it = items_.find(item_id);
    if (it != items_.end()) {
        it->second.stable_frames++;
    }
}

void InventoryDB::relocate_item(int item_id, int new_track_id, const BBox& new_box,
                                float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.last_bbox = it->second.box;
    it->second.track_id = new_track_id;
    it->second.box = new_box;
    it->second.score = score;
    it->second.updated_frame = frame_id;
    it->second.last_seen_frame = frame_id;
    it->second.stable_frames++;
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

void InventoryDB::print(const char* prefix) const {
    // 先打印在库物品（非 TAKEN/HELD）
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        if (it.status == ItemStatus::TAKEN || it.status == ItemStatus::HELD) continue;
        printf("%s  - item#%d cls=%d(%s) [在库] "
               "pos=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f tid=%d seen@%d\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               it.box.x1, it.box.y1, it.box.x2, it.box.y2,
               it.score, it.track_id, it.last_seen_frame);
    }
    // 再打印 TAKEN 的物品（用空行隔开）
    bool has_taken = false;
    for (const auto& kv : items_) {
        if (kv.second.status == ItemStatus::TAKEN) { has_taken = true; break; }
    }
    if (has_taken) {
        printf("%s  --- 被拿走（临时记录，不上传后台）---\n", prefix);
        for (const auto& kv : items_) {
            const auto& it = kv.second;
            if (it.status != ItemStatus::TAKEN) continue;
            printf("%s  - item#%d cls=%d(%s) [被拿走] "
                   "pos=(%.0f,%.0f)~(%.0f,%.0f) score=%.2f tid=%d seen@%d\n",
                   prefix,
                   it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
                   it.box.x1, it.box.y1, it.box.x2, it.box.y2,
                   it.score, it.track_id, it.last_seen_frame);
        }
    }
}

std::string InventoryDB::to_json(const char* device_id, long long timestamp_ms) const {
    std::string s;
    char buf[768];

    snprintf(buf, sizeof(buf),
             "{\"device_id\":\"%s\",\"timestamp\":%lld,"
             "\"event_type\":\"DOOR_CLOSE\",\"inventory\":[",
             device_id, timestamp_ms);
    s += buf;

    bool first = true;
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        // 排除 TAKEN/HELD，只上报真正"在库"的物品
        if (it.status == ItemStatus::TAKEN || it.status == ItemStatus::HELD) continue;
        float bw = it.box.x2 - it.box.x1;
        float bh = it.box.y2 - it.box.y1;
        snprintf(buf, sizeof(buf),
                 "%s{\"local_track_id\":%d,\"category\":\"%s\",\"fine_class\":\"%s\","
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
