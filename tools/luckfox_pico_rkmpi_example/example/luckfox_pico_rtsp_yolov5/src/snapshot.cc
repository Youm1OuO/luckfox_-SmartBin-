// ============================================================================
//  snapshot.cc
//  多帧投票快照实现 — 新业务流程6
// ============================================================================
#include "snapshot.h"
#include "fridge_config.h"

#include <cmath>
#include <algorithm>

namespace fridge {

namespace {

void merge_appearance(AppearanceFeature& dst,
                      const AppearanceFeature& src,
                      int old_count,
                      int new_count) {
    if (!src.valid) return;
    if (!dst.valid || old_count <= 0) {
        dst = src;
        return;
    }

    float old_w = (float)old_count / (float)new_count;
    float new_w = 1.0f / (float)new_count;
    for (int c = 0; c < 3; ++c) {
        dst.mean_bgr[c] = dst.mean_bgr[c] * old_w + src.mean_bgr[c] * new_w;
    }
    for (int p = 0; p < 9; ++p) {
        for (int c = 0; c < 3; ++c) {
            dst.patch_bgr[p][c] = dst.patch_bgr[p][c] * old_w + src.patch_bgr[p][c] * new_w;
        }
    }
    for (int h = 0; h < 8; ++h) {
        dst.hue_hist[h] = dst.hue_hist[h] * old_w + src.hue_hist[h] * new_w;
    }
    dst.valid = true;
}

}  // namespace

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
    //  对每一帧的每个 detection，与投票表中的候选物品做快照聚合匹配：
    //    匹配上 → count++，位置取加权平均
    //    没匹配上 → 新增到投票表
    //  N帧结束后，保留 count >= N*s 的物品
    // ================================================================
    std::vector<VotingItem> voting_table;
    std::vector<int> voting_last_frame_index;
    int min_count = (int)std::ceil(N_ * s_);

    for (int fi = 0; fi < (int)frames_.size(); fi++) {
        const auto& dets = frames_[fi];
        for (const auto& det : dets) {
            if (det.score < SNAPSHOT_MIN_SCORE) continue;

            // 在投票表中查找最佳匹配。
            // 同一个 voting item 在同一帧最多匹配一次，避免近距离同类物品
            // 或同帧重复框被合并成一个候选并虚增 count。
            int best_idx = -1;
            float best_match_score = -1.0f;
            for (int vi_idx = 0; vi_idx < (int)voting_table.size(); ++vi_idx) {
                auto& vi = voting_table[vi_idx];
                if (voting_last_frame_index[vi_idx] == fi) continue;
                // 1. 类别相同
                if (vi.cls_id != det.cls_id) continue;
                // 2. 中心距离近
                float dist = center_distance(vi.box, det.box);
                if (dist >= SNAPSHOT_CLUSTER_CENTER_DIST) continue;
                // 3. 面积比差异小
                float area_diff = area_ratio_diff(vi.box, det.box);
                if (area_diff >= SNAPSHOT_CLUSTER_AREA_RATIO) continue;
                // 4. IoU 高
                float overlap = iou(vi.box, det.box);
                if (overlap < SNAPSHOT_CLUSTER_IOU_THRESH) continue;
                // 5. 像素颜色差异（这里没有frame，跳过颜色检查，用前4个条件）
                //    投票阶段连续帧间颜色几乎不变，前4个条件已经足够

                float dist_score = SNAPSHOT_CLUSTER_CENTER_DIST <= 0.0f
                                 ? 0.0f
                                 : 1.0f - dist / SNAPSHOT_CLUSTER_CENTER_DIST;
                float match_score = 0.70f * overlap
                                  + 0.20f * dist_score
                                  + 0.10f * (1.0f - area_diff);
                if (match_score > best_match_score) {
                    best_match_score = match_score;
                    best_idx = vi_idx;
                }
            }

            if (best_idx >= 0) {
                VotingItem& vi = voting_table[best_idx];
                int old_count = vi.count;
                vi.count++;
                float n = (float)vi.count;
                vi.box.x1 = vi.box.x1 * (n-1)/n + det.box.x1 / n;
                vi.box.y1 = vi.box.y1 * (n-1)/n + det.box.y1 / n;
                vi.box.x2 = vi.box.x2 * (n-1)/n + det.box.x2 / n;
                vi.box.y2 = vi.box.y2 * (n-1)/n + det.box.y2 / n;
                merge_appearance(vi.appearance, det.appearance, old_count, vi.count);
                if (det.score > vi.best_score) vi.best_score = det.score;
                voting_last_frame_index[best_idx] = fi;
                continue;
            }

            // 新增到投票表
            VotingItem vi;
            vi.cls_id = det.cls_id;
            vi.box = det.box;
            vi.best_score = det.score;
            vi.count = 1;
            vi.appearance = det.appearance;
            voting_table.push_back(vi);
            voting_last_frame_index.push_back(fi);
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
