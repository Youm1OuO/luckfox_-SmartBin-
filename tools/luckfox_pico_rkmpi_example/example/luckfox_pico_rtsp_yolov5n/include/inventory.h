// ============================================================================
//  inventory.h
//  本地工作库存 — 单框快照裁决
//
//  设计要点:
//    - 物品状态只有：VISIBLE（可见）、OCCLUDED（遮挡）
//    - 出库只存在于本轮 planned 结果；提交时直接从活跃库存删除
//    - item_id 只属于库存；Track 只作为本轮整理辅助证据
//    - last_box 是最近一次确认看见的框，允许是完整框或局部框
//    - 库存只在稳定快照比较结束后整体提交
// ============================================================================
#ifndef __FRIDGE_INVENTORY_H
#define __FRIDGE_INVENTORY_H

#include <map>
#include <vector>
#include <string>
#include "geometry.h"

namespace fridge {

enum class ItemStatus {
    VISIBLE,    // 【可见】— YOLO能识别到，在冰箱中
    OCCLUDED,   // 【遮挡】— 被其他物品/手遮挡，仍在冰箱中
};

struct InventoryItem {
    int item_id = -1;         // 稳定身份ID，永不变
    int cls_id = -1;          // 物品类别
    BBox last_box;            // 最近一次确认看见的框，允许是完整框或局部框
    float score = 0.0f;       // 最新检测分数
    ItemStatus status = ItemStatus::VISIBLE;
    int blocker_id = -1;      // 上一次确认挡住该物品的库存 item_id
    bool new_item_pending = false; // 新入库的首个框尚未完成一次后续确认
    int created_frame = 0;    // 入库帧号
    int updated_frame = 0;    // 最近更新帧号
    long long created_time_ms = 0; // 入库时间戳
};

class InventoryDB {
public:
    InventoryDB() : next_item_id_(1) {}

    // 新增一个物品，返回分配的 item_id。
    int add_item(int cls_id, const BBox& box, float score,
                 int frame_id, long long time_ms = 0,
                 bool new_item_pending = true);

    // 按 item_id 查找（返回指针，找不到返回 nullptr）
    InventoryItem* find_by_item(int item_id);
    const InventoryItem* find_by_item(int item_id) const;

    // 标记某个物品为指定状态。
    void set_status(int item_id, ItemStatus new_status);

    // 更新某个物品的当前确认框。
    void update_seen_item(int item_id, const BBox& box, float score, int frame_id);

    // 删除某个物品（按 item_id）
    void remove_item(int item_id);

    const std::map<int, InventoryItem>& items() const { return items_; }
    size_t size() const { return items_.size(); }
    size_t count_by_status(ItemStatus s) const;
    int next_item_id() const { return next_item_id_; }

    // SessionManager 在 planned_changes 全部算完后一次性替换库存，避免半提交。
    void replace_all(const std::map<int, InventoryItem>& items, int next_item_id);

    // 调试打印
    void print(const char* prefix = "") const;

    // 关门时上传后台用：导出全部活跃库存。
    std::string to_json(const char* device_id, long long timestamp_ms,
                        const char* session_id = nullptr) const;

private:
    std::map<int, InventoryItem> items_;
    int next_item_id_;
};

const char* item_status_to_str(ItemStatus s);

}  // namespace fridge

#endif  // __FRIDGE_INVENTORY_H
