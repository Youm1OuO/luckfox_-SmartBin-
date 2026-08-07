// ============================================================================
//  inventory.h
//  本地持久库存（2.0）
//
//  设计要点:
//    - 物品状态只有：VISIBLE（可见）、OCCLUDED（遮挡）
//    - 出库只存在于本轮 working_inventory；成功结算后才删除正式物品
//    - item_id 只属于库存；Track 只提供整理/出库的必要证据
//    - box 是最近一次确认看见的框；base_box 是该物品可靠的完整位置
//    - block_ids 是已确认位于该物品前方、且仍与其 base_box 相交的物品 ID；
//      是否完全遮挡由这些前景框的覆盖并集单独判断
//    - InventoryDB 用 std::map 保存物品，按 item_id 查找且对象地址稳定
// ============================================================================
#ifndef __FRIDGE_INVENTORY_H
#define __FRIDGE_INVENTORY_H

#include <map>
#include <set>
#include <vector>
#include <string>
#include "geometry.h"

namespace fridge {

enum class ItemStatus {
    VISIBLE,    // 【可见】— YOLO能识别到，在冰箱中
    OCCLUDED,   // 【遮挡】— 被其他物品/手遮挡，仍在冰箱中
};

// A committed explanation for a formal OCCLUDED status.  This is deliberately
// separate from block_ids: block_ids describes the complete live relation graph,
// while witnesses records the blockers that actually proved full coverage.
enum class OcclusionProofKind {
    NONE,
    STRICT_UNION,
    EDGE_RESIDUAL_UNION,
    DISAPPEARANCE_SUPPORTED,
};

struct OcclusionProof {
    OcclusionProofKind kind = OcclusionProofKind::NONE;
    std::set<int> witness_blocker_ids;

    void clear() {
        kind = OcclusionProofKind::NONE;
        witness_blocker_ids.clear();
    }
};

struct InventoryItem {
    int item_id = -1;         // 稳定身份ID，永不变
    int cls_id = -1;          // 物品类别
    BBox box;                 // 最近一次确认看见的框，允许是局部框
    BBox base_box;            // 初次入库或整理后确认的完整位置
    float score = 0.0f;       // 最新检测分数
    ItemStatus status = ItemStatus::VISIBLE;
    std::set<int> block_ids;  // 当前被确认位于其前方的多个库存物品
    OcclusionProof occlusion_proof; // formal proof when status == OCCLUDED
    int created_frame = 0;    // 入库帧号
    int updated_frame = 0;    // 最近更新帧号
    long long created_time_ms = 0; // 入库时间戳
};

class InventoryDB {
public:
    InventoryDB() : next_item_id_(1) {}

    // 新增一个物品，返回分配的 item_id。
    int add_item(int cls_id, const BBox& box, float score,
                 int frame_id, long long time_ms = 0);

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
    std::map<int, InventoryItem>& mutable_items() { return items_; }
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
