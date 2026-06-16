// ============================================================================
//  inventory.h
//  本地工作库存 — 新业务流程6：三态模型
//
//  设计要点:
//    - 物品状态：VISIBLE（可见）、OCCLUDED（遮挡）、OUT（出库）
//    - 【可见】与【遮挡】都算"在冰箱中"
//    - 【出库】是临时保留状态，用于后续恢复匹配
//    - 所有身份匹配都基于库存清单进行
//    - 库存清单实时更新
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
    int item_id;              // 稳定身份ID，永不变
    int track_id;             // 当前绑定的ByteTrack ID（-1表示未绑定）
    int cls_id;               // 物品类别
    BBox box;                 // 当前位置（被动bbox，会因为遮挡等原因被动变化）
    BBox active_box;          // 主动bbox（物品刚拿进来或整理后的bbox，只有主动移动时才更新）
    float score;              // 最新检测分数
    ItemStatus status;        // 当前状态
    int created_frame;        // 入库帧号
    int updated_frame;        // 最近更新帧号
    long long created_time_ms;// 入库时间戳
    long long out_time_ms;    // 出库时间戳（仅OUT状态有效，用于过期清理）
    int stable_frames;        // 连续未被YOLO识别到的快照周期数（仅VISIBLE有效，>=9时删除）
};

class InventoryDB {
public:
    InventoryDB() : next_item_id_(1) {}

    // 新增一个物品，返回分配的 item_id（默认状态：VISIBLE）
    int add_item(int track_id, int cls_id, const BBox& box, float score,
                 int frame_id, long long time_ms = 0);

    // 按 item_id 查找（返回指针，找不到返回 nullptr）
    InventoryItem* find_by_item(int item_id);
    const InventoryItem* find_by_item(int item_id) const;

    // 标记某个物品为指定状态
    void set_status(int item_id, ItemStatus new_status, long long time_ms = 0);

    // 更新某个物品的位置和信息
    void update_item(int item_id, int track_id, const BBox& box, float score, int frame_id);

    // 删除某个物品（按 item_id）
    void remove_item(int item_id);

    const std::map<int, InventoryItem>& items() const { return items_; }
    std::map<int, InventoryItem>& items() { return items_; }
    size_t size() const { return items_.size(); }
    size_t count_by_status(ItemStatus s) const;

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
