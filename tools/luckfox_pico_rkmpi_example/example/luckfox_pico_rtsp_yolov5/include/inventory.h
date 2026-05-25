// ============================================================================
//  inventory.h
//  库存数据库 - 极简内存版本
//
//  当前实现：std::vector<InventoryItem>，按 cls_id+位置 去匹配。
//  后续要持久化到 SQLite/文件，或者通过 HTTP 上报后台，
//  都只需要改 InventoryDB 的实现，不影响事件层的接口。
// ============================================================================
#ifndef __FRIDGE_INVENTORY_H
#define __FRIDGE_INVENTORY_H

#include <vector>
#include <string>
#include "hand_state.h"
#include "geometry.h"

namespace fridge {

// 一条库存项
struct InventoryItem {
    int item_id;            // 库存内部 id（自增）
    int cls_id;             // 类别 id
    BBox pos;               // 最近位置
    int created_frame;      // 入库帧号
    int updated_frame;      // 上次更新帧号
};

class InventoryDB {
public:
    InventoryDB() : next_item_id_(1) {}

    // 应用一批 final events
    // 返回：本次操作影响的库存项 id 列表（用于上报后台）
    std::vector<int> apply_events(const std::vector<FinalEvent>& events);

    const std::vector<InventoryItem>& items() const { return items_; }

    // 调试用：打印当前库存
    void print(const char* prefix = "") const;

private:
    // IN：新增库存项
    int handle_in(const FinalEvent& e);
    // OUT：找到位置最匹配的同类项删除
    int handle_out(const FinalEvent& e);
    // RELOCATE：找到位置最匹配的同类项更新位置
    int handle_relocate(const FinalEvent& e);

    // 在 items_ 中找一个 cls_id 相同、位置离 query 最近的库存项的下标
    // 找不到返回 -1
    int find_closest(int cls_id, const BBox& query) const;

private:
    std::vector<InventoryItem> items_;
    int next_item_id_;
};

}  // namespace fridge

#endif  // __FRIDGE_INVENTORY_H
