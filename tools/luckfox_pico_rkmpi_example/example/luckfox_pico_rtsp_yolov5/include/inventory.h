// ============================================================================
//  inventory.h
//  本地工作库存 — 以 item_id 为稳定身份（新业务流程4-1）
//
//  设计要点:
//    - 物品状态：VISIBLE（可见，在冰箱中）、OCCLUDED（被其他物品遮挡）、TAKEN（被拿走，临时）
//    - VISIBLE 和 OCCLUDED 都被认为是在冰箱库存中
//    - 手遮挡通过 HELD 机制处理（held_items_），不改变物品的 VISIBLE/OCCLUDED 状态
//    - 关门时上传库存：所有物品（排除 TAKEN）
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
    VISIBLE,    // 【可见】— 物品在冰箱中，可见（含新放入的、遮挡恢复的）
    OCCLUDED,   // 【遮挡】— 被其他物品物理遮挡（在库但不可见）
    TAKEN,      // 【拿走】— 确认被取出（临时保留，关门时不上传后台）
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
    int last_seen_frame;   // 最近一次被 YOLO 看到的帧号
    int stable_frames;     // 连续被看到/存在的帧数
    BBox last_bbox;        // 上一帧位置（用于移动检测）
    long long created_time_ms;  // 入库时的毫秒时间戳
};

class InventoryDB {
public:
    InventoryDB() : next_item_id_(1) {}

    // 新增一个物品，返回分配的 item_id（默认状态：在库中）
    int add_item(int track_id, int cls_id, const BBox& box, float score,
                 int frame_id, long long time_ms = 0);

    // 按 track_id / item_id 查找（返回指针，找不到返回 nullptr）
    InventoryItem* find_by_track(int track_id);
    InventoryItem* find_by_item(int item_id);
    const InventoryItem* find_by_track(int track_id) const;
    const InventoryItem* find_by_item(int item_id) const;

    // 更新某个物品（按 item_id）的位置
    void update_seen(int item_id, int track_id, const BBox& box, float score, int frame_id);

    // 标记某个物品为指定状态（按 item_id）
    void set_status(int item_id, ItemStatus new_status);

    // 增加某个物品的 stable_frames 计数（按 item_id）
    void increment_stable_frames(int item_id);

    // 整理：把某个 item 重新绑定到新 track_id 并更新位置（身份 item_id 不变）
    void relocate_item(int item_id, int new_track_id, const BBox& new_box,
                       float score, int frame_id);

    // 删除某个物品（按 item_id）
    void remove_item(int item_id);

    const std::map<int, InventoryItem>& items() const { return items_; }
    size_t size() const { return items_.size(); }
    size_t count_by_status(ItemStatus s) const;

    // 调试打印
    void print(const char* prefix = "") const;

    // 关门时上传后台用：把库存导出成 JSON 字符串（排除 TAKEN）
    std::string to_json(const char* device_id, long long timestamp_ms) const;

private:
    std::map<int, InventoryItem> items_;
    int next_item_id_;
};

const char* item_status_to_str(ItemStatus s);

}  // namespace fridge

#endif  // __FRIDGE_INVENTORY_H
