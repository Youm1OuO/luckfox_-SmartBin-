// ============================================================================
//  snapshot.h
//  稳态快照 + diff
//
//  设计要点:
//    - Snapshot 只记录"稳态那一刻"画面里所有 food 类 track 的状态
//    - 不记录手 track（手不可能出现在稳态里）
//    - 主键是 track_id（ByteTrack 给的全局唯一 id），不是位置/类别
//    - diff 完全按 track_id 求差集，不做位置匹配也不做类别匹配
//      原因：ByteTrack 已经在时序上保证了"同一物理物品 → 同一 track_id"
// ============================================================================
#ifndef __FRIDGE_SNAPSHOT_H
#define __FRIDGE_SNAPSHOT_H

#include <map>
#include <vector>
#include "tracker.h"
#include "geometry.h"

namespace fridge {

// 稳态那一刻一个 food track 的"档案"
// 字段精简到只保留事件层 / 库存层需要的东西
struct SnapshotItem {
    int track_id;
    int cls_id;
    BBox box;
    float score;
};

// 一份完整稳态快照
// 用 map 而不是 vector 是因为 diff 时按 track_id 查找特别频繁
struct Snapshot {
    std::map<int, SnapshotItem> items;   // track_id -> SnapshotItem
    int frame_id = -1;                   // 截取时的帧号（调试用）
    bool valid = false;                  // 是否是一份有效快照
};

// 业务事件类型 — 稳态切片 diff 的输出
enum class ChangeType {
    IN,         // 放入：track_id 在 after 但不在 before
    OUT,        // 拿走：track_id 在 before 但不在 after
    RELOCATE    // 整理：两边都有，位置变化超过阈值
};

// 一条业务事件
struct ChangeEvent {
    ChangeType type;
    int track_id;            // diff 维度：用谁的 track_id
                             //   IN 用 after 的 track_id
                             //   OUT 用 before 的 track_id
                             //   RELOCATE 两边一致
    int cls_id;
    BBox from_box;           // OUT/RELOCATE 用：操作前的位置；IN 时 = to_box
    BBox to_box;             // IN/RELOCATE 用：操作后的位置；OUT 时 = from_box
    float score;             // 取 after（IN/RELOCATE）或 before（OUT）的分数
    int frame_before;        // before 快照帧号
    int frame_after;         // after 快照帧号
};

// 把当前 tracker 输出的 tracks 凝固成一份快照
// 只收 food 类、score >= SNAPSHOT_MIN_SCORE 的 track
Snapshot make_snapshot(const std::vector<Track>& tracks, int frame_id);

// 比对 before 和 after，按 track_id 求差集，得到一组事件
// 注意：纯几何 diff，不做"再认领"和"部分取出"等高级逻辑（未来在 reconcile 阶段补）
std::vector<ChangeEvent> diff_snapshots(const Snapshot& before, const Snapshot& after);

// 调试辅助
const char* change_type_to_str(ChangeType t);

}  // namespace fridge

#endif  // __FRIDGE_SNAPSHOT_H
