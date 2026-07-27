// ============================================================================
//  snapshot.h
//  多帧投票快照 — 连续无手稳定帧
//
//  设计要点:
//    - 不使用单帧作为快照，而是用 N 帧的综合结果（投票过滤）
//    - N 帧内同一个物品必须出现 >= N*s 次，才被写入快照
//    - SessionManager 在看到手时会先 reset()；所以这里仅收集连续无手帧
//    - SnapshotBuffer 负责收集 N 帧的检测结果，满了就生成快照
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
};

// 多帧投票缓冲区
class SnapshotBuffer {
public:
    SnapshotBuffer(int N = 3, float s = 0.6f);

    // 每帧调用：推入一帧已确认无手的物品检测结果。
    void push(const std::vector<Detection>& detections, int frame_id);

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
};

}  // namespace fridge

#endif  // __FRIDGE_SNAPSHOT_H
