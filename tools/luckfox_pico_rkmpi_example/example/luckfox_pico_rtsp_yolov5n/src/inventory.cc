// ============================================================================
//  inventory.cc
//  本地持久库存实现（2.0）
// ============================================================================
#include "inventory.h"
#include "fridge_config.h"
#include <cstdio>
#include <cstring>
#include <algorithm>
#include "yolov5n.h"

namespace fridge {

namespace {

bool formal_occlusion_proof_is_valid(
        const std::map<int, InventoryItem>& items,
        const InventoryItem& item, const char** reason) {
    if (item.occlusion_proof.kind == OcclusionProofKind::NONE) {
        if (reason) *reason = "missing-proof-kind";
        return false;
    }
    if (item.occlusion_proof.witness_blocker_ids.empty()) {
        if (reason) *reason = "missing-proof-witness";
        return false;
    }
    for (std::set<int>::const_iterator witness =
             item.occlusion_proof.witness_blocker_ids.begin();
         witness != item.occlusion_proof.witness_blocker_ids.end(); ++witness) {
        if (*witness == item.item_id) {
            if (reason) *reason = "self-proof-witness";
            return false;
        }
        if (!item.block_ids.count(*witness)) {
            if (reason) *reason = "proof-witness-not-in-block-ids";
            return false;
        }
        if (!items.count(*witness)) {
            if (reason) *reason = "proof-witness-not-active";
            return false;
        }
    }
    return true;
}

}  // namespace

const char* item_status_to_str(ItemStatus s) {
    switch (s) {
        case ItemStatus::VISIBLE:  return "可见";
        case ItemStatus::OCCLUDED: return "遮挡";
    }
    return "?";
}

int InventoryDB::add_item(int cls_id, const BBox& box,
                          float score, int frame_id, long long time_ms) {
    InventoryItem it;
    it.item_id = next_item_id_++;
    it.cls_id = cls_id;
    it.box = box;
    it.base_box = box;
    it.score = score;
    it.status = ItemStatus::VISIBLE;
    it.block_ids.clear();
    it.created_frame = frame_id;
    it.updated_frame = frame_id;
    it.created_time_ms = time_ms;
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

void InventoryDB::set_status(int item_id, ItemStatus new_status) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.status = new_status;
    if (new_status == ItemStatus::VISIBLE) {
        it->second.occlusion_proof.clear();
    }
}

void InventoryDB::update_seen_item(int item_id, const BBox& box,
                                    float score, int frame_id) {
    auto it = items_.find(item_id);
    if (it == items_.end()) return;
    it->second.box = box;
    it->second.score = score;
    it->second.updated_frame = frame_id;
}

bool InventoryDB::replace_all(const std::map<int, InventoryItem>& items,
                              int next_item_id) {
    std::map<int, InventoryItem> normalized = items;
    // A visible committed item cannot retain a historical occlusion proof.
    // Keep the normalization local until every OCCLUDED record has passed
    // validation, so a rejected commit cannot partially mutate persistence.
    for (std::map<int, InventoryItem>::iterator it = normalized.begin();
         it != normalized.end(); ++it) {
        if (it->second.status == ItemStatus::VISIBLE) {
            it->second.occlusion_proof.clear();
        }
    }

    for (std::map<int, InventoryItem>::const_iterator it = normalized.begin();
         it != normalized.end(); ++it) {
        if (it->second.status != ItemStatus::OCCLUDED) continue;
        const char* reason = "unknown";
        if (formal_occlusion_proof_is_valid(normalized, it->second, &reason)) {
            continue;
        }
        // Do not silently downgrade legacy malformed OCCLUDED records to
        // VISIBLE.  The caller receives an explicit failure and retains the
        // previous atomic snapshot until a valid lifecycle proof is available.
        fprintf(stderr,
                "[InventoryDB] reject replace_all: item#%d OCCLUDED proof invalid (%s)\n",
                it->first, reason);
        return false;
    }

    items_.swap(normalized);
    next_item_id_ = std::max(1, next_item_id);
    return true;
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
    // 保持原来的显示顺序：先可见，再遮挡。
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        if (it.status != ItemStatus::VISIBLE) continue;
        printf("%s  - item#%d cls=%d(%s) [可见] "
               "box=(%.0f,%.0f)~(%.0f,%.0f) base=(%.0f,%.0f)~(%.0f,%.0f) blockers=%zu score=%.2f\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               it.box.x1, it.box.y1, it.box.x2, it.box.y2,
               it.base_box.x1, it.base_box.y1, it.base_box.x2, it.base_box.y2,
               it.block_ids.size(),
               it.score);
    }
    for (const auto& kv : items_) {
        const auto& it = kv.second;
        if (it.status != ItemStatus::OCCLUDED) continue;
        printf("%s  - item#%d cls=%d(%s) [遮挡] "
               "last_visible_box=(%.0f,%.0f)~(%.0f,%.0f) base=(%.0f,%.0f)~(%.0f,%.0f) blockers=%zu score=%.2f\n",
               prefix,
               it.item_id, it.cls_id, coco_cls_to_name(it.cls_id),
               it.box.x1, it.box.y1, it.box.x2, it.box.y2,
               it.base_box.x1, it.base_box.y1, it.base_box.x2, it.base_box.y2,
               it.block_ids.size(),
               it.score);
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
        snprintf(buf, sizeof(buf),
                 "%s{\"item_id\":%d,\"category\":\"%s\","
                 "\"status\":\"%s\",\"bbox\":[%.0f,%.0f,%.0f,%.0f]}",
                 first ? "" : ",",
                 it.item_id, cls_id_to_chinese(it.cls_id),
                 item_status_to_str(it.status),
                 it.box.x1, it.box.y1,
                 it.box.x2, it.box.y2);
        s += buf;
        first = false;
    }
    s += "]}";
    return s;
}

}  // namespace fridge
