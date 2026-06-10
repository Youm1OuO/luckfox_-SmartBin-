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

constexpr float CLUSTER_AREA_DIFF_MAX = 0.50f;       // 面积比例约 0.5 ~ 2.0
constexpr float CLUSTER_CENTER_NORM_MAX = 1.05f;     // 明显宽于 strict_origin_match
constexpr float CLUSTER_IOU_ENOUGH = 0.20f;
constexpr float CLUSTER_OVERLAP_ENOUGH = 0.25f;
constexpr float CLUSTER_SCORE_MIN = 0.22f;
constexpr float CLUSTER_MOTION_NORM_FOR_LAST_BOX = 0.35f;

struct SnapshotCluster {
    VotingItem item;
    BBox first_box;
    BBox last_box;
    int last_frame_index = -1;
};

float snapshot_cluster_score(const SnapshotCluster& cluster,
                             const Detection& det) {
    if (cluster.item.cls_id != det.cls_id) return -1.0f;

    float area_diff = area_ratio_diff(cluster.last_box, det.box);
    if (area_diff > CLUSTER_AREA_DIFF_MAX) return -1.0f;

    float center_norm = normalized_nearby_distance(cluster.last_box, det.box);
    float box_iou = iou(cluster.last_box, det.box);
    float box_overlap = overlap_ratio_of_smaller(cluster.last_box, det.box);

    bool close_enough = center_norm <= CLUSTER_CENTER_NORM_MAX;
    bool overlaps_enough = box_iou >= CLUSTER_IOU_ENOUGH ||
                           box_overlap >= CLUSTER_OVERLAP_ENOUGH;
    if (!close_enough && !overlaps_enough) return -1.0f;

    float distance_score = 1.0f - std::min(center_norm / CLUSTER_CENTER_NORM_MAX, 1.0f);
    float iou_score = std::min(box_iou / CLUSTER_IOU_ENOUGH, 1.0f);
    float overlap_score = std::min(box_overlap / CLUSTER_OVERLAP_ENOUGH, 1.0f);
    float area_score = 1.0f - std::min(area_diff / CLUSTER_AREA_DIFF_MAX, 1.0f);

    return 0.45f * distance_score +
           0.25f * iou_score +
           0.15f * overlap_score +
           0.15f * area_score;
}

void update_cluster(SnapshotCluster& cluster,
                    const Detection& det,
                    int frame_index) {
    cluster.item.count++;

    float n = (float)cluster.item.count;
    cluster.item.box.x1 = cluster.item.box.x1 * (n - 1.0f) / n + det.box.x1 / n;
    cluster.item.box.y1 = cluster.item.box.y1 * (n - 1.0f) / n + det.box.y1 / n;
    cluster.item.box.x2 = cluster.item.box.x2 * (n - 1.0f) / n + det.box.x2 / n;
    cluster.item.box.y2 = cluster.item.box.y2 * (n - 1.0f) / n + det.box.y2 / n;

    if (det.score > cluster.item.best_score) {
        cluster.item.best_score = det.score;
    }
    cluster.last_box = det.box;
    cluster.last_frame_index = frame_index;
}

SnapshotCluster make_cluster(const Detection& det, int frame_index) {
    SnapshotCluster cluster;
    cluster.item.cls_id = det.cls_id;
    cluster.item.box = det.box;
    cluster.item.best_score = det.score;
    cluster.item.count = 1;
    cluster.first_box = det.box;
    cluster.last_box = det.box;
    cluster.last_frame_index = frame_index;
    return cluster;
}

VotingItem finalize_cluster(const SnapshotCluster& cluster) {
    VotingItem item = cluster.item;
    float motion_norm = normalized_nearby_distance(cluster.first_box, cluster.last_box);
    if (motion_norm >= CLUSTER_MOTION_NORM_FOR_LAST_BOX) {
        item.box = cluster.last_box;
    }
    return item;
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
    //  对每一帧的每个 detection，与投票表中的候选物品做宽松聚合匹配：
    //    匹配上 → count++，位置取加权平均
    //    没匹配上 → 新增到投票表
    //  同一帧内每个候选物品最多接收一个 detection，避免相邻同类物品被合并。
    //  N帧结束后，保留 count >= N*s 的物品
    // ================================================================
    std::vector<SnapshotCluster> clusters;
    int min_count = (int)std::ceil(N_ * s_);

    for (int fi = 0; fi < (int)frames_.size(); fi++) {
        const auto& dets = frames_[fi];
        std::vector<bool> used_clusters(clusters.size(), false);
        for (const auto& det : dets) {
            if (det.score < SNAPSHOT_MIN_SCORE) continue;

            int best_idx = -1;
            float best_score = -1.0f;
            for (int ci = 0; ci < (int)clusters.size(); ++ci) {
                if (used_clusters[ci]) continue;
                float score = snapshot_cluster_score(clusters[ci], det);
                if (score > best_score) {
                    best_score = score;
                    best_idx = ci;
                }
            }

            if (best_idx >= 0 && best_score >= CLUSTER_SCORE_MIN) {
                update_cluster(clusters[best_idx], det, fi);
                used_clusters[best_idx] = true;
            } else {
                clusters.push_back(make_cluster(det, fi));
                used_clusters.push_back(true);
            }
        }
    }

    // 过滤：只保留 count >= N*s 的物品
    for (const auto& cluster : clusters) {
        if (cluster.item.count >= min_count) {
            snap.items.push_back(finalize_cluster(cluster));
        }
    }

    snap.valid = true;

    // 重置缓冲区
    reset();

    return snap;
}

}  // namespace fridge
