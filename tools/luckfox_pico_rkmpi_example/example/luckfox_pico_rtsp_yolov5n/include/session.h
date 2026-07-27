// ============================================================================
//  session.h
//  会话业务层：库存表 + 单份稳定快照 + 轻量 OperationTrack
//
//  正式库存只由稳定快照提交；OperationTrack 只在一轮开门期间保存，唯一
//  用途是证明“旧位置的物品和新位置的物品是同一次整理”。
// ============================================================================
#ifndef __FRIDGE_SESSION_H
#define __FRIDGE_SESSION_H

#include <map>
#include <vector>

#include "inventory.h"
#include "snapshot.h"

namespace fridge {

// IN / OUT / MOVED 会上报云端；OCCLUDED / REVEALED 只用于本地终端和库存表，
// 因为它们表示冰箱内的可见状态变化，不是实际出入库。
enum class EventKind { IN, OUT, MOVED, OCCLUDED, REVEALED };

struct InventoryEvent {
    EventKind kind;
    int item_id;
    int cls_id;
    BBox box;          // IN / OUT 的位置；MOVED 时等于 after_box
    BBox before_box;   // 仅 MOVED 使用
    BBox after_box;    // 仅 MOVED 使用
    float score;
};

struct SettlementResult {
    bool happened = false;
    std::vector<InventoryEvent> events;
};

// process_frame 的返回值：主循环据此知道这一帧是否刚形成稳定快照。
struct FrameProcessResult {
    bool stable_snapshot_generated = false;
    SettlementResult settlement;
};

enum class InitState {
    WAIT_BACKEND,
    WAIT_FIRST_STABLE_SNAPSHOT,
    READY,
};

enum class BackendStatus {
    UNKNOWN,
    TRUSTED,
    NO_TRUSTED_BACKEND,
};

// 这不是 tracker.h 的 ByteTrack Track。它没有全局身份，也不会上传后台。
enum class OperationTrackState {
    TRACKING_VISIBLE,       // YOLO 还能看到物品（完整或局部框）
    FULL_HAND_OCCLUDED,     // YOLO 看不到物品，proxy_box 跟随手移动
    PLACED,                 // 已确认放下，release_box 可以作为整理证据
};

// Candidate 只记录“手可能正在碰这个物品”，本身不能作为整理证据。
// 只有确认物品框或手代理真正移动后，才会升级成 OperationTrack。
enum class OperationCandidateState {
    VISIBLE_CANDIDATE,          // YOLO 仍能看到物品
    FULL_HAND_OCCLUDED_CANDIDATE, // 手覆盖原位置，YOLO 暂时看不到物品
};

struct OperationCandidate {
    int bound_item_id = -1;
    int cls_id = -1;

    BBox source_box;            // Candidate 建立时库存中的可靠位置
    BBox last_yolo_box;         // 最近一次真实 YOLO 框（可为局部框）
    bool has_last_yolo_box = false;
    BBox start_hand_box;        // 完全遮挡时，手刚覆盖物品的位置
    BBox last_hand_box;
    bool has_last_hand_box = false;
    OperationCandidateState state = OperationCandidateState::VISIBLE_CANDIDATE;
};

struct OperationTrack {
    int track_id = -1;          // 仅本次开门会话内自增
    int bound_item_id = -1;     // 引用 InventoryItem.item_id
    int cls_id = -1;

    BBox start_box;             // 从 anchor_box（或 last_seen_box）复制的起点
    BBox proxy_box;             // 当前完整物品位置的代理框
    BBox last_yolo_box;         // 上一次真实 YOLO 物品框，允许是局部框
    bool has_last_yolo_box = false;
    BBox last_hand_box;
    bool has_last_hand_box = false;

    std::vector<BBox> path;     // 每次 Track 更新后的 proxy_box，全程保留
    BBox release_box;           // 确认放下位置
    bool has_release_box = false;
    OperationTrackState state = OperationTrackState::TRACKING_VISIBLE;
};

class SessionManager {
public:
    SessionManager();

    void start_new_session(long long time_ms = 0);
    void init_from_backend(const std::vector<InventoryItem>& items,
                           bool authoritative_empty = false);
    void mark_backend_unavailable();
    void finish_session(long long time_ms = 0);

    // 每一帧唯一的业务入口。
    // food_detections 不含手；hand_boxes 是已经通过业务阈值的手框。
    FrameProcessResult process_frame(const std::vector<Detection>& food_detections,
                                     const std::vector<BBox>& hand_boxes,
                                     int frame_id, long long time_ms);

    bool has_backend() const { return backend_status_ == BackendStatus::TRUSTED; }
    bool ready() const { return init_state_ == InitState::READY; }
    bool hand_present() const { return hand_present_; }
    int no_hand_streak() const { return no_hand_streak_; }
    const InventoryDB& inventory() const { return inventory_; }
    InventoryDB& inventory() { return inventory_; }
    const std::vector<OperationTrack>& operation_tracks() const { return tracks_; }

    // 终端调试用：用紧凑表格展示当前正式库存（不显示临时 OUT 记录）。
    void print_inventory() const;

private:
    void update_tracks_while_hand_present(const std::vector<Detection>& food_detections,
                                          const std::vector<BBox>& hand_boxes,
                                          int frame_id);

    void initialize_from_snapshot(const Snapshot& snapshot);
    SettlementResult settle_snapshot(const Snapshot& snapshot);
    void refresh_visible_items_without_operation(const Snapshot& snapshot);

    int find_unique_inventory_binding(const Detection& detection,
                                      const std::vector<BBox>& hand_boxes) const;
    bool item_matches_snapshot(const InventoryItem& item,
                               const VotingItem& observed) const;
    bool item_is_bound_to_operation(int item_id) const;

    InventoryDB inventory_;
    SnapshotBuffer no_hand_buffer_;
    std::vector<OperationTrack> tracks_;
    std::vector<OperationCandidate> candidates_;

    int next_operation_track_id_;
    bool operation_pending_;
    bool hand_present_;
    int no_hand_streak_;
    long long current_time_ms_;
    long long session_start_time_ms_;
    InitState init_state_;
    BackendStatus backend_status_;
};

}  // namespace fridge

#endif  // __FRIDGE_SESSION_H
