// ============================================================================
//  inventory.h
//  本地工作库存 — 以 item_id 为稳定身份
//
//  设计要点:
//    - 每个物品有一个永不变的 item_id（物理身份证号），上传后台用这个
//    - track_id 是当前绑定的 ByteTrack id，整理时会重新绑定（身份不变）
//    - 抗抖动 / 抗遮挡：物品消失不立刻删，由 session 控制删除时机
// ============================================================================
#ifndef __FRIDGE_INVENTORY_H
#define __FRIDGE_INVENTORY_H

#include <map>
#include <vector>
#include <string>
#include "snapshot.h"
#include "geometry.h"

namespace fridge {

enum class ItemStatus {
    VISIBLE,    // 当前能稳定看到
    OCCLUDED,   // 看不到但还在（被遮挡/手拿着/暂时丢失）
    GONE,       // 已离场（通常升级到 GONE 时直接移除）
};

struct InventoryItem {
    int item_id;           // ★ 稳定身份，永不变
    int track_id;          // 当前绑定的 ByteTrack id（整理时会更新；-1 表示未绑定）
    int cls_id;
    BBox box;              // 最近位置
    float score;
    ItemStatus status;
    int created_frame;     // 入库帧号
    int updated_frame;     // 最近更新帧号
    int last_seen_frame;   // 最近一次被 YOLO 看到的帧号（抗抖动/超时用）
};

class InventoryDB {
public:
    InventoryDB() : next_item_id_(1) {}

    // 新增一个物品，返回分配的 item_id
    int add_item(int track_id, int cls_id, const BBox& box, float score, int frame_id);

    // 按 track_id / item_id 查找（返回指针，找不到返回 nullptr）
    InventoryItem* find_by_track(int track_id);
    InventoryItem* find_by_item(int item_id);
    const InventoryItem* find_by_track(int track_id) const;

    // 更新某个物品（按 item_id）的位置和状态为 VISIBLE
    void update_seen(int item_id, int track_id, const BBox& box, float score, int frame_id);

    // 标记某个物品为 OCCLUDED（按 item_id）
    void mark_occluded(int item_id);

    // 整理：把某个 item 重新绑定到新 track_id 并更新位置（身份 item_id 不变）
    void relocate_item(int item_id, int new_track_id, const BBox& new_box,
                       float score, int frame_id);

    // 删除某个物品（按 item_id）
    void remove_item(int item_id);

    const std::map<int, InventoryItem>& items() const { return items_; }
    size_t size() const { return items_.size(); }
    size_t count_visible() const;
    size_t count_occluded() const;

    // 调试打印
    void print(const char* prefix = "") const;

    // 关门时上传后台用：把库存导出成 JSON 字符串
    std::string to_json(const char* device_id, long long timestamp_ms) const;

private:
    std::map<int, InventoryItem> items_;   // item_id -> item
    int next_item_id_;
};

const char* item_status_to_str(ItemStatus s);

}  // namespace fridge

#endif  // __FRIDGE_INVENTORY_H
