// ============================================================================
//  snapshot.cc
//  多帧投票快照实现 — 新业务流程7
// ============================================================================
#include "snapshot.h"
#include "fridge_config.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <map>

namespace fridge {

namespace {

constexpr float CLUSTER_AREA_DIFF_MAX = 0.50f;       // 面积比例约 0.5 ~ 2.0
constexpr float CLUSTER_CENTER_NORM_MAX = 1.05f;     // 明显宽于 strict_origin_match
constexpr float CLUSTER_IOU_ENOUGH = 0.20f;
constexpr float CLUSTER_OVERLAP_ENOUGH = 0.25f;
constexpr float CLUSTER_SCORE_MIN = 0.22f;
constexpr float CLUSTER_MOTION_NORM_FOR_LAST_BOX = 0.35f;

struct ClassVote {
    int count = 0;
    float best_score = 0.0f;
};

struct SnapshotCluster {
    VotingItem item;
    BBox first_box;
    BBox last_box;
    int last_frame_index = -1;
    std::map<int, ClassVote> class_votes;
};

void update_class_vote(SnapshotCluster& cluster, const Detection& det) {
    ClassVote& vote = cluster.class_votes[det.cls_id];
    vote.count++;
    if (det.score > vote.best_score) {
        vote.best_score = det.score;
    }
}

float same_class_cluster_score(const SnapshotCluster& cluster,
                               const Detection& det) {
    if (cluster.class_votes.find(det.cls_id) == cluster.class_votes.end()) {
        return -1.0f;
    }

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

float class_conflict_cluster_score(const SnapshotCluster& cluster,
                                   const Detection& det) {
    if (cluster.class_votes.find(det.cls_id) != cluster.class_votes.end()) {
        return -1.0f;
    }

    float area_diff = area_ratio_diff(cluster.last_box, det.box);
    if (area_diff > SNAPSHOT_CLASS_CONFLICT_AREA_RATIO) return -1.0f;

    float center_norm = normalized_nearby_distance(cluster.last_box, det.box);
    if (center_norm > SNAPSHOT_CLASS_CONFLICT_CENTER_RATIO) return -1.0f;

    float box_iou = iou(cluster.last_box, det.box);
    if (box_iou < SNAPSHOT_CLASS_CONFLICT_IOU) return -1.0f;

    float motion_norm = normalized_nearby_distance(cluster.first_box, det.box);
    if (motion_norm > SNAPSHOT_CLASS_CONFLICT_MOTION_RATIO) return -1.0f;

    float distance_score = 1.0f - std::min(center_norm / SNAPSHOT_CLASS_CONFLICT_CENTER_RATIO, 1.0f);
    float area_score = 1.0f - std::min(area_diff / SNAPSHOT_CLASS_CONFLICT_AREA_RATIO, 1.0f);

    return 0.50f * box_iou +
           0.30f * distance_score +
           0.20f * area_score;
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
    update_class_vote(cluster, det);
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
    update_class_vote(cluster, det);
    return cluster;
}

bool finalize_cluster(const SnapshotCluster& cluster, VotingItem* out_item) {
    if (!out_item || cluster.class_votes.empty()) return false;

    int best_cls = -1;
    int best_count = -1;
    float best_score = 0.0f;
    int second_cls = -1;
    int second_count = 0;

    for (std::map<int, ClassVote>::const_iterator it = cluster.class_votes.begin();
         it != cluster.class_votes.end(); ++it) {
        int cls = it->first;
        int count = it->second.count;
        float score = it->second.best_score;

        if (count > best_count ||
            (count == best_count && score > best_score)) {
            second_cls = best_cls;
            second_count = best_count < 0 ? 0 : best_count;
            best_cls = cls;
            best_count = count;
            best_score = score;
        } else if (count > second_count) {
            second_cls = cls;
            second_count = count;
        }
    }

    bool has_class_conflict = cluster.class_votes.size() > 1;
    if (has_class_conflict) {
        float class_ratio = (float)best_count / (float)std::max(1, cluster.item.count);
        int class_margin = best_count - second_count;
        if (class_ratio < SNAPSHOT_CLASS_STABLE_RATIO ||
            class_margin < SNAPSHOT_CLASS_COUNT_MARGIN) {
            std::printf("[SNAPSHOT] 类别不稳定，丢弃候选: best_cls=%d count=%d second_cls=%d count=%d total=%d ratio=%.2f\n",
                        best_cls, best_count, second_cls, second_count,
                        cluster.item.count, class_ratio);
            return false;
        }

        std::printf("[SNAPSHOT] 类别抖动已投票稳定: cls=%d count=%d second_cls=%d count=%d total=%d\n",
                    best_cls, best_count, second_cls, second_count,
                    cluster.item.count);
    }

    VotingItem item = cluster.item;
    item.cls_id = best_cls;
    item.best_score = best_score;

    float motion_norm = normalized_nearby_distance(cluster.first_box, cluster.last_box);
    if (motion_norm >= CLUSTER_MOTION_NORM_FOR_LAST_BOX) {
        item.box = cluster.last_box;
    }
    *out_item = item;
    return true;
}

}  // namespace

// ============================================================================
//  SnapshotBuffer
// ============================================================================

SnapshotBuffer::SnapshotBuffer(int N, float object_stable_ratio)
    : N_(N), object_stable_ratio_(object_stable_ratio) {
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
    //    同类别匹配上 → count++，位置取加权平均
    //    同类别没匹配上 → 严格判断是否为同一位置同一大小的跨类别抖动
    //    没匹配上 → 新增到投票表
    //  同一帧内每个候选物品最多接收一个 detection，避免相邻同类物品被合并。
    //  N帧结束后，先保留达到物体稳定阈值的候选，再做类别稳定投票。
    // ================================================================
    std::vector<SnapshotCluster> clusters;
    int min_count = (int)std::ceil(N_ * object_stable_ratio_);

    for (int fi = 0; fi < (int)frames_.size(); fi++) {
        const auto& dets = frames_[fi];
        std::vector<bool> used_clusters(clusters.size(), false);
        for (const auto& det : dets) {
            if (det.score < SNAPSHOT_MIN_SCORE) continue;

            int best_idx = -1;
            float best_score = -1.0f;
            for (int ci = 0; ci < (int)clusters.size(); ++ci) {
                if (used_clusters[ci]) continue;
                float score = same_class_cluster_score(clusters[ci], det);
                if (score > best_score) {
                    best_score = score;
                    best_idx = ci;
                }
            }

            if (best_idx >= 0 && best_score >= CLUSTER_SCORE_MIN) {
                update_cluster(clusters[best_idx], det, fi);
                used_clusters[best_idx] = true;
            } else {
                int conflict_idx = -1;
                float conflict_score = -1.0f;
                for (int ci = 0; ci < (int)clusters.size(); ++ci) {
                    if (used_clusters[ci]) continue;
                    float score = class_conflict_cluster_score(clusters[ci], det);
                    if (score > conflict_score) {
                        conflict_score = score;
                        conflict_idx = ci;
                    }
                }

                if (conflict_idx >= 0) {
                    update_cluster(clusters[conflict_idx], det, fi);
                    used_clusters[conflict_idx] = true;
                } else {
                    clusters.push_back(make_cluster(det, fi));
                    used_clusters.push_back(true);
                }
            }
        }
    }

    // 过滤：物体出现要稳定；如果同一 spatial cluster 内出现过类别冲突，
    // 还必须通过类别投票，避免把短暂误识别写入库存。
    for (const auto& cluster : clusters) {
        if (cluster.item.count >= min_count) {
            VotingItem item;
            if (finalize_cluster(cluster, &item)) {
                snap.items.push_back(item);
            }
        }
    }

    snap.valid = true;

    // 重置缓冲区
    reset();

    return snap;
}

}  // namespace fridge
