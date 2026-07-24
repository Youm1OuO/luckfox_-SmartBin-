// ============================================================================
//  tracker.h
//  ByteTrack-Lite 多目标跟踪器
//
//  核心思想（来自 ByteTrack 论文，ECCV 2022）：
//    1. 高分检测（score >= HIGH_SCORE_THRESH）先和现有 track 做匹配；
//    2. 第一轮没匹配上的 track，再去尝试和"低分检测"匹配
//       (因为遮挡时检测分数会掉，但目标其实还在)；
//    3. 还没匹配上的 track，进入 "Lost" 状态保留若干帧；
//    4. 还没匹配上的高分检测，作为新 track 出生。
//
//  本实现的简化点：
//    - 不使用 Kalman Filter 做运动预测，直接用 track 上一帧的 bbox 做 IoU。
//      理由：冰箱场景物品多数静止，KF 预测无意义；10 FPS 下帧间位移很小。
//    - 用贪心匹配代替匈牙利算法。
//      理由：单帧目标数 < 20 时贪心结果与最优匈牙利基本一致。
//    - 保留 ByteTrack 的核心创新：双轮（high+low）检测匹配。
// ============================================================================
#ifndef __FRIDGE_TRACKER_H
#define __FRIDGE_TRACKER_H

#include <vector>
#include "geometry.h"

namespace fridge {

// 一条 detection（输入到 tracker 的原始检测结果）
struct Detection {
    BBox box;       // 像素坐标，原图分辨率
    float score;    // 置信度 [0, 1]
    int cls_id;     // YOLO 类别 id
};

// Track 的生命状态
enum class TrackState {
    NEW,        // 刚创建（第一次出现，还未确认）
    TRACKED,    // 正常跟踪中（已确认，本帧也匹配上了）
    LOST,       // 暂时丢失（最近若干帧没匹配上，但还没超时）
    REMOVED     // 已超时销毁（仅作为状态标记，不会出现在输出中）
};

// 一条 track（被跟踪器维护的、已分配 id 的目标）
struct Track {
    int track_id;           // 全局唯一 ID（每条 track 一个）
    int cls_id;             // 类别 id（取出生时的，跟踪过程中通常不变）
    BBox box;               // 最近一次匹配上的 bbox 位置
    float score;            // 最近一次匹配上的检测分数
    TrackState state;       // 当前状态

    int frames_since_update;    // 自上次成功匹配以来过了多少帧（用于判超时）
    int hit_count;              // 累计成功匹配过多少帧（用于从 NEW 升级到 TRACKED）
    int last_frame_id;          // 最后一次匹配上的全局帧号

    // 出生帧号
    int start_frame;
};

// 跟踪器主类
class ByteTrackLite {
public:
    ByteTrackLite();
    ~ByteTrackLite() = default;

    // 输入：当前帧 detection 列表 + 当前帧编号
    // 输出：当前帧所有"活跃"的 track（state=TRACKED 或刚被匹配上的）
    //       注意：返回的是引用，外部不要保存指针；下次 update 后内容会变
    const std::vector<Track>& update(const std::vector<Detection>& detections,
                                     int frame_id);

    // 调试用：取所有 track（包括 LOST 状态的）
    const std::vector<Track>& all_tracks() const { return tracks_; }

private:
    // 内部：分配下一个 track id（自增）
    int next_track_id();

    // 第一轮：高分 detection vs (TRACKED + LOST) tracks
    // 第二轮：低分 detection vs 第一轮没配上的 TRACKED tracks
    // 第三轮：剩余高分 detection vs 第一轮没配上的 NEW tracks
    // 用 IoU 矩阵 + 贪心配对
    //
    // 参数：
    //   det_indices    - 参与本轮的 detection 在 detections 数组中的下标
    //   track_indices  - 参与本轮的 track 在 tracks_ 数组中的下标
    //   detections     - 本帧所有 detection
    //   iou_threshold  - IoU 低于此值不允许配对
    //   matched_dets   - 输出：被配上的 detection 下标集合
    //   matched_trks   - 输出：被配上的 track 下标集合
    //   match_pairs    - 输出：(det_idx, trk_idx) 配对列表
    void match_round(const std::vector<int>& det_indices,
                     const std::vector<int>& track_indices,
                     const std::vector<Detection>& detections,
                     float iou_threshold,
                     std::vector<int>& matched_dets,
                     std::vector<int>& matched_trks,
                     std::vector<std::pair<int,int>>& match_pairs);

    // 把一条 detection 更新到一条已有 track 上
    void update_track_with_detection(Track& trk, const Detection& det, int frame_id);

    // 创建新 track
    void create_new_track(const Detection& det, int frame_id);

private:
    std::vector<Track> tracks_;             // 全部活跃 + LOST 的 track 池
    std::vector<Track> output_tracks_;      // update() 返回的"对外"track 列表
    int next_id_;                           // 下一个分配的 track id
};

}  // namespace fridge

#endif  // __FRIDGE_TRACKER_H
