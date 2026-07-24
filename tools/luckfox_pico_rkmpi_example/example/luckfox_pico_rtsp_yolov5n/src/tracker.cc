// ============================================================================
//  tracker.cc
//  ByteTrack-Lite 实现
//
//  实现思路（每次 update 干这几件事）：
//    Step 0  分类 detection：score 高的进 high_dets，score 较低的进 low_dets。
//            <LOW_SCORE_THRESH 的太低，直接丢弃（噪声）。
//    Step 1  把已有 track 分类：TRACKED+LOST 一起进入第一轮匹配池。
//            （新出生的 NEW track 单独留着进第三轮）
//    Step 2  第一轮匹配：high_dets vs (TRACKED+LOST)，IoU 阈值较严。
//    Step 3  第二轮匹配：low_dets vs 第一轮没匹配上的 TRACKED tracks，IoU 阈值放宽。
//            这是 ByteTrack 的关键创新——遮挡时检测分会掉，但 IoU 还能救回来。
//    Step 4  第三轮：剩余 high_dets vs 第一轮没匹配上的 NEW tracks。
//    Step 5  没匹配上的 TRACKED → 转 LOST；没匹配上的 LOST → 计数++，超时 → REMOVED。
//    Step 6  没匹配上的 NEW（出生第一帧后立刻消失）→ 直接 REMOVED。
//    Step 7  没匹配上的高分 detection → 创建新 track。
//    Step 8  清理 REMOVED 的 track，输出当前活跃 track 给调用方。
// ============================================================================
#include "tracker.h"
#include "fridge_config.h"

#include <algorithm>
#include <vector>

namespace fridge {

ByteTrackLite::ByteTrackLite() : next_id_(1) {
    tracks_.reserve(64);
    output_tracks_.reserve(64);
}

int ByteTrackLite::next_track_id() {
    return next_id_++;
}

// 把一条 detection 的位置和分数更新到一条 track 上
void ByteTrackLite::update_track_with_detection(Track& trk,
                                                const Detection& det,
                                                int frame_id) {
    trk.box = det.box;
    trk.score = det.score;
    // 注意：cls_id 不每帧更新。出生时是什么类别就是什么，避免 YOLO 偶发误分类导致 ID 飘
    // 如果你后续需要类别投票，可以在外层做
    trk.frames_since_update = 0;
    trk.hit_count++;
    trk.last_frame_id = frame_id;

    // 根据 hit_count 升级状态
    if (trk.state == TrackState::NEW) {
        // NEW → TRACKED 需要至少匹配上 2 次（出生+本次匹配 = hit_count>=2）
        // 这是为了防止"YOLO 偶尔抖出一个假 detection"立刻成 track
        if (trk.hit_count >= 2) {
            trk.state = TrackState::TRACKED;
        }
    } else {
        // LOST → TRACKED：只要匹配上就回到 TRACKED
        // TRACKED → TRACKED：保持
        trk.state = TrackState::TRACKED;
    }
}

// 创建一条新 track
void ByteTrackLite::create_new_track(const Detection& det, int frame_id) {
    Track t;
    t.track_id = next_track_id();
    t.cls_id = det.cls_id;
    t.box = det.box;
    t.score = det.score;
    t.state = TrackState::NEW;
    t.frames_since_update = 0;
    t.hit_count = 1;          // 出生算第一次"hit"
    t.last_frame_id = frame_id;
    t.start_frame = frame_id;
    tracks_.push_back(t);
}

// 一轮匹配：基于 IoU 矩阵 + 贪心算法
//
// 贪心算法步骤（单轮匹配）：
//   1. 计算所有 (det, trk) 对的 IoU
//   2. 把 IoU >= 阈值的 (det, trk, iou) 三元组收集起来
//   3. 按 iou 降序排序
//   4. 从大到小取，每次拿最大的；若该 det 和 trk 都没被占用，则配对成功
//   5. 已配对的 det/trk 标记占用，跳过后续含它们的三元组
void ByteTrackLite::match_round(const std::vector<int>& det_indices,
                                const std::vector<int>& track_indices,
                                const std::vector<Detection>& detections,
                                float iou_threshold,
                                std::vector<int>& matched_dets,
                                std::vector<int>& matched_trks,
                                std::vector<std::pair<int,int>>& match_pairs) {
    matched_dets.clear();
    matched_trks.clear();
    match_pairs.clear();

    // 三元组 (iou, det_idx, trk_idx)
    struct Candidate { float iou; int di; int ti; };
    std::vector<Candidate> candidates;
    candidates.reserve(det_indices.size() * track_indices.size());

    // 计算 IoU 矩阵（其实是个三元组列表）
    for (int di : det_indices) {
        const BBox& db = detections[di].box;
        int dcls = detections[di].cls_id;
        for (int ti : track_indices) {
            const BBox& tb = tracks_[ti].box;
            // 类别约束：不同类别的不允许配对
            // (例如手 track 不能匹配到牛奶 detection；冰箱场景这个约束很重要)
            if (tracks_[ti].cls_id != dcls) continue;
            float v = iou(db, tb);
            if (v >= iou_threshold) {
                candidates.push_back({v, di, ti});
            }
        }
    }

    // 按 IoU 降序排序
    std::sort(candidates.begin(), candidates.end(),
              [](const Candidate& a, const Candidate& b){ return a.iou > b.iou; });

    // 贪心配对
    for (const auto& c : candidates) {
        // 检查 det 和 trk 是否都未被占用
        bool det_used = std::find(matched_dets.begin(), matched_dets.end(), c.di)
                        != matched_dets.end();
        if (det_used) continue;
        bool trk_used = std::find(matched_trks.begin(), matched_trks.end(), c.ti)
                        != matched_trks.end();
        if (trk_used) continue;
        // 配对成功
        matched_dets.push_back(c.di);
        matched_trks.push_back(c.ti);
        match_pairs.push_back({c.di, c.ti});
    }
}

const std::vector<Track>& ByteTrackLite::update(
        const std::vector<Detection>& detections,
        int frame_id) {

    // ===== Step 0：分流 detection（按分数高低） =====
    std::vector<int> high_det_indices;   // score >= HIGH
    std::vector<int> low_det_indices;    // LOW <= score < HIGH
    high_det_indices.reserve(detections.size());
    low_det_indices.reserve(detections.size());
    for (size_t i = 0; i < detections.size(); ++i) {
        float s = detections[i].score;
        if (s >= HIGH_SCORE_THRESH)      high_det_indices.push_back(i);
        else if (s >= LOW_SCORE_THRESH)  low_det_indices.push_back(i);
        // 更低分的直接忽略
    }

    // ===== Step 1：分流已有 track =====
    std::vector<int> tracked_or_lost_indices;  // TRACKED + LOST → 进第一轮
    std::vector<int> new_indices;              // NEW → 进第三轮
    tracked_or_lost_indices.reserve(tracks_.size());
    new_indices.reserve(tracks_.size());
    for (size_t i = 0; i < tracks_.size(); ++i) {
        TrackState s = tracks_[i].state;
        if (s == TrackState::TRACKED || s == TrackState::LOST) {
            tracked_or_lost_indices.push_back(i);
        } else if (s == TrackState::NEW) {
            new_indices.push_back(i);
        }
        // REMOVED 不应该出现在 tracks_ 里（每帧末尾会清掉）
    }

    // ===== Step 2：第一轮匹配 high_dets vs (TRACKED+LOST) =====
    std::vector<int> matched_high_dets, matched_first_trks;
    std::vector<std::pair<int,int>> first_pairs;
    match_round(high_det_indices, tracked_or_lost_indices, detections,
                MATCH_IOU_THRESH,
                matched_high_dets, matched_first_trks, first_pairs);

    // 应用第一轮配对：用 detection 更新 track
    for (auto& p : first_pairs) {
        update_track_with_detection(tracks_[p.second], detections[p.first], frame_id);
    }

    // 第一轮没匹配上的 high_det 和 track（只看 TRACKED 状态的，LOST 不参与第二轮）
    std::vector<int> remain_high_dets;
    for (int di : high_det_indices) {
        if (std::find(matched_high_dets.begin(), matched_high_dets.end(), di)
            == matched_high_dets.end()) {
            remain_high_dets.push_back(di);
        }
    }
    std::vector<int> remain_tracked_for_second;
    for (int ti : tracked_or_lost_indices) {
        if (std::find(matched_first_trks.begin(), matched_first_trks.end(), ti)
            == matched_first_trks.end()) {
            // 第二轮只让 TRACKED 状态的参与（不让 LOST 参与，避免错配）
            if (tracks_[ti].state == TrackState::TRACKED) {
                remain_tracked_for_second.push_back(ti);
            }
        }
    }

    // ===== Step 3：第二轮匹配 low_dets vs 剩余 TRACKED tracks =====
    std::vector<int> matched_low_dets, matched_second_trks;
    std::vector<std::pair<int,int>> second_pairs;
    match_round(low_det_indices, remain_tracked_for_second, detections,
                MATCH_IOU_THRESH_LOW,
                matched_low_dets, matched_second_trks, second_pairs);

    for (auto& p : second_pairs) {
        update_track_with_detection(tracks_[p.second], detections[p.first], frame_id);
    }

    // ===== Step 4：第三轮：剩余 high_dets vs NEW tracks =====
    // (ByteTrack 论文中处理 unconfirmed 的步骤)
    std::vector<int> matched_third_dets, matched_third_trks;
    std::vector<std::pair<int,int>> third_pairs;
    match_round(remain_high_dets, new_indices, detections,
                MATCH_IOU_THRESH_UNCFM,
                matched_third_dets, matched_third_trks, third_pairs);

    for (auto& p : third_pairs) {
        update_track_with_detection(tracks_[p.second], detections[p.first], frame_id);
    }

    // ===== Step 5：处理本帧没匹配上的 track =====
    // 5a) 第一轮没配上的 TRACKED 转 LOST，frames_since_update 累加
    // 5b) 第一轮没配上的 LOST 继续 LOST，frames_since_update 累加；超时则 REMOVED
    for (int ti : tracked_or_lost_indices) {
        bool matched = (std::find(matched_first_trks.begin(),
                                  matched_first_trks.end(), ti)
                        != matched_first_trks.end())
                    || (std::find(matched_second_trks.begin(),
                                  matched_second_trks.end(), ti)
                        != matched_second_trks.end());
        if (matched) continue;

        Track& t = tracks_[ti];
        t.frames_since_update++;
        if (t.state == TrackState::TRACKED) {
            t.state = TrackState::LOST;
        }
        // 超过 buffer 帧没回来 → 销毁
        if (t.frames_since_update > TRACK_BUFFER_FRAMES) {
            t.state = TrackState::REMOVED;
        }
    }

    // ===== Step 6：处理 NEW 状态没配上的 track =====
    // NEW 出生当帧后立刻没匹配上 = YOLO 偶发误检，直接 REMOVED
    for (int ti : new_indices) {
        bool matched = std::find(matched_third_trks.begin(),
                                 matched_third_trks.end(), ti)
                       != matched_third_trks.end();
        if (!matched) {
            tracks_[ti].state = TrackState::REMOVED;
        }
    }

    // ===== Step 7：剩余 high_dets 创建新 track =====
    for (int di : remain_high_dets) {
        bool used = std::find(matched_third_dets.begin(),
                              matched_third_dets.end(), di)
                    != matched_third_dets.end();
        if (used) continue;
        // 新 track 出生分数门槛
        if (detections[di].score < NEW_TRACK_SCORE_THRESH) continue;
        create_new_track(detections[di], frame_id);
    }

    // ===== Step 8：清理 REMOVED 并准备输出 =====
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const Track& t){ return t.state == TrackState::REMOVED; }),
        tracks_.end());

    // 输出：只返回 TRACKED 状态的 track（NEW 还未确认，LOST 当前帧没出现）
    output_tracks_.clear();
    for (const auto& t : tracks_) {
        if (t.state == TrackState::TRACKED) {
            output_tracks_.push_back(t);
        }
    }
    return output_tracks_;
}

}  // namespace fridge
