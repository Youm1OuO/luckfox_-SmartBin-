// ============================================================================
//  snapshot.h
//  多帧投票快照 — 新业务流程6
//
//  设计要点:
//    - 不使用单帧作为快照，而是用 N 帧的综合结果（投票过滤）
//    - N 帧内同一个物品必须出现 >= N*s 次，才被写入快照
//    - SnapshotBuffer 负责收集 N 帧的检测结果，满了就生成快照
//    - 快照包含 has_hand 标记：有手的快照不用于对比
// ============================================================================
#ifndef __FRIDGE_SNAPSHOT_H
#define __FRIDGE_SNAPSHOT_H

#include <vector>
#include "tracker.h"
#include "geometry.h"

namespace fridge {

// 投票缓冲区中的一个候选物品
struct VotingItem {
    int cls_id;
    BBox box;           // 平均位置（多帧加权平均）
    float best_score;   // 最高分数
    int count;          // 出现帧数
};

// 一份快照（投票过滤后的最终结果）
struct Snapshot {
    std::vector<VotingItem> items;  // 通过投票阈值的物品
    int frame_id;                   // 快照帧号（最后一帧的帧号）
    bool valid;                     // 是否是一份有效快照
    bool has_hand;                  // 是否包含手（包含手的快照不用于对比）
};

// 多帧投票缓冲区
class SnapshotBuffer {
public:
    SnapshotBuffer(int N = 3, float s = 0.6f);

    // 每帧调用：推入一帧的检测结果（只含食物检测，不含手）
    // has_hand: 当前帧是否检测到手
    void push(const std::vector<Detection>& detections, int frame_id, bool has_hand);

    // 缓冲区是否满（攒够 N 帧）
    bool full() const;

    // 取出快照并重置缓冲区
    Snapshot take_snapshot();

    // 外部门状态切换时调用，避免跨会话残留帧混进第一份快照
    void reset();

private:
    int N_;
    float s_;
    std::vector<std::vector<Detection>> frames_;
    std::vector<int> frame_ids_;
    std::vector<bool> hand_flags_;
};

}  // namespace fridge

#endif  // __FRIDGE_SNAPSHOT_H
