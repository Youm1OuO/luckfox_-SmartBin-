// ============================================================================
//  inventory.h
//  本地工作库存 — 单稳定快照直接裁决
//
//  设计要点:
//    - 物品状态：VISIBLE（可见）、OCCLUDED（遮挡）、OUT（出库）
//    - 【可见】与【遮挡】都算"在冰箱中"
//    - 【出库】是临时保留状态，用于后续恢复匹配
//    - item_id 只属于库存；临时 Track 不写入库存、更不上传后台
//    - anchor_box 是可靠的身份/完整框；last_seen_box 是最近一次确认看到的框
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
    OUT,        // 【出库】— 不在冰箱中（临时保留，用于恢复匹配）
};

struct InventoryItem {
    int item_id = -1;         // 稳定身份ID，永不变
    int cls_id = -1;          // 物品类别
    BBox anchor_box;          // 可靠的完整参考框；整理确认后才更新
    bool anchor_valid = false;// false 表示尚未取得可靠完整框
    BBox last_seen_box;       // 最近一次稳定快照确认看到的 YOLO 框，允许是局部框
    float score = 0.0f;       // 最新检测分数
    ItemStatus status = ItemStatus::VISIBLE;
    int no_occluder_count = 0;// 遮挡物连续没有遮挡依据的快照次数
    int duplicate_count = 0;  // 同一稳定快照里疑似重复出现的次数
    int created_frame = 0;    // 入库帧号
    int updated_frame = 0;    // 最近更新帧号
    long long created_time_ms = 0; // 入库时间戳
    long long out_time_ms = 0;     // 出库时间戳（仅OUT状态有效，用于过期清理）
};

class InventoryDB {
public:
    InventoryDB() : next_item_id_(1) {}

    // 新增一个物品，返回分配的 item_id（默认状态：VISIBLE、anchor 有效）
    int add_item(int cls_id, const BBox& box, float score,
                 int frame_id, long long time_ms = 0);

    // 按 item_id 查找（返回指针，找不到返回 nullptr）
    InventoryItem* find_by_item(int item_id);
    const InventoryItem* find_by_item(int item_id) const;

    // 标记某个物品为指定状态
    void set_status(int item_id, ItemStatus new_status, long long time_ms = 0);

    // 更新某个物品的当前确认框，不修改 anchor_box
    void update_seen_item(int item_id, const BBox& box, float score, int frame_id);

    // 整理确认或稳定快照得到可靠完整框时调用
    void update_anchor_item(int item_id, const BBox& box,
                            float score, int frame_id);

    // 删除某个物品（按 item_id）
    void remove_item(int item_id);

    const std::map<int, InventoryItem>& items() const { return items_; }
    size_t size() const { return items_.size(); }
    size_t count_by_status(ItemStatus s) const;
    int next_item_id() const { return next_item_id_; }

    // SessionManager 在 planned_changes 全部算完后一次性替换库存，避免半提交。
    void replace_all(const std::map<int, InventoryItem>& items, int next_item_id);

    // 清理过期的 OUT 物品
    void cleanup_expired(long long now_ms);

    // 调试打印
    void print(const char* prefix = "") const;

    // 关门时上传后台用：把库存导出成 JSON 字符串（只含 VISIBLE + OCCLUDED）
    std::string to_json(const char* device_id, long long timestamp_ms,
                        const char* session_id = nullptr) const;

private:
    std::map<int, InventoryItem> items_;
    int next_item_id_;
};

const char* item_status_to_str(ItemStatus s);

}  // namespace fridge

#endif  // __FRIDGE_INVENTORY_H
