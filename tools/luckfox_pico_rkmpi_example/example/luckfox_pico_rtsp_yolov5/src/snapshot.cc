// ============================================================================
//  snapshot.cc
//  多帧投票快照实现 — 新业务流程6
// ============================================================================
#include "snapshot.h"
#include "fridge_config.h"

#include <cmath>
#include <algorithm>

namespace fridge {

// ============================================================================
//  SnapshotBuffer
// ============================================================================

SnapshotBuffer::SnapshotBuffer(int N, float s) : N_(N), s_(s) {
    frames_.reserve(N);
    frame_ids_.reserve(N);
    hand_flags_.reserve(N);
}

void SnapshotBuffer::push(const std::vector<Detection>& detections,
                           int frame_id, bool has_hand) {
    frames_.push_back(detections);
    frame_ids_.push_back(frame_id);
    hand_flags_.push_back(has_hand);
}

bool SnapshotBuffer::full() const {
    return (int)frames_.size() >= N_;
}

void SnapshotBuffer::reset() {
    frames_.clear();
    frame_ids_.clear();
    hand_flags_.clear();
}

Snapshot SnapshotBuffer::take_snapshot() {
    Snapshot snap;
    snap.valid = false;
    snap.has_hand = false;
    snap.frame_id = 0;

    if (frames_.empty()) return snap;

    // 检查是否有足够稳定的阻塞手。单帧疑似手不直接废掉整个快照。
    int hand_count = 0;
    for (bool h : hand_flags_) {
        if (h) hand_count++;
    }
    snap.has_hand = hand_count >= SNAPSHOT_HAND_BLOCK_MIN_COUNT;

    // 最后一帧的帧号
    snap.frame_id = frame_ids_.back();

    // ================================================================
    //  多帧投票算法
    //  对每一帧的每个 detection，与投票表中的候选物品做严格身份匹配：
    //    匹配上 → count++，位置取加权平均
    //    没匹配上 → 新增到投票表
    //  N帧结束后，保留 count >= N*s 的物品
    // ================================================================
    std::vector<VotingItem> voting_table;
    int min_count = (int)std::ceil(N_ * s_);

    for (int fi = 0; fi < (int)frames_.size(); fi++) {
        const auto& dets = frames_[fi];
        for (const auto& det : dets) {
            if (det.score < SNAPSHOT_MIN_SCORE) continue;

            // 在投票表中查找匹配（严格身份匹配）
            bool matched = false;
            for (auto& vi : voting_table) {
                // 1. 类别相同
                if (vi.cls_id != det.cls_id) continue;
                // 2. 中心距离近
                if (center_distance(vi.box, det.box) >= IDENTITY_CENTER_DIST) continue;
                // 3. 面积比差异小
                if (area_ratio_diff(vi.box, det.box) >= IDENTITY_AREA_RATIO) continue;
                // 4. IoU 高
                if (iou(vi.box, det.box) < IDENTITY_IOU_THRESH) continue;
                // 5. 像素颜色差异（这里没有frame，跳过颜色检查，用前4个条件）
                //    投票阶段连续帧间颜色几乎不变，前4个条件已经足够

                // 匹配上 → 更新
                vi.count++;
                // 位置取加权平均（累积平均）
                float n = (float)vi.count;
                vi.box.x1 = vi.box.x1 * (n-1)/n + det.box.x1 / n;
                vi.box.y1 = vi.box.y1 * (n-1)/n + det.box.y1 / n;
                vi.box.x2 = vi.box.x2 * (n-1)/n + det.box.x2 / n;
                vi.box.y2 = vi.box.y2 * (n-1)/n + det.box.y2 / n;
                if (det.score > vi.best_score) vi.best_score = det.score;
                matched = true;
                break;
            }

            if (!matched) {
                // 新增到投票表
                VotingItem vi;
                vi.cls_id = det.cls_id;
                vi.box = det.box;
                vi.best_score = det.score;
                vi.count = 1;
                voting_table.push_back(vi);
            }
        }
    }

    // 过滤：只保留 count >= N*s 的物品
    for (const auto& vi : voting_table) {
        if (vi.count >= min_count) {
            snap.items.push_back(vi);
        }
    }

    snap.valid = true;

    // 重置缓冲区
    reset();

    return snap;
}

}  // namespace fridge
