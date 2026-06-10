// ============================================================================
//  session.h
//  会话管理器 — 新业务流程6：统一快照裁决 + OperationContext 辅助证据
//
//  核心职责：
//    1. baseline_snapshot -> stable_snapshot 的单次差异裁决
//    2. OperationContext 记录手、HELD代理轨迹、ByteTrack移动证据
//    3. relocation_match 优先于普通出库/入库，避免整理被拆成出库+入库
//    4. low_confidence_relocation 进入跨快照 PendingRelocation
//    5. 手不直接产生库存事件，只作为统一裁决的辅助证据
//
//  所有操作实时更新进本地库存清单，身份匹配也基于库存清单进行
// ============================================================================
#ifndef __FRIDGE_SESSION_H
#define __FRIDGE_SESSION_H

#include <vector>
#include <map>
#include <set>
#include <opencv2/core.hpp>
#include "tracker.h"
#include "snapshot.h"
#include "inventory.h"

namespace fridge {

// 出入库事件类型
enum class EventKind { IN, OUT, MOVED };

// 一条出入库事件
struct InventoryEvent {
    EventKind kind;
    int       item_id;
    int       cls_id;
    BBox      box;
    float     score;
};

struct SettlementResult {
    bool happened = false;
    bool inventory_changed = false;
    bool reject_snapshot = false;
    std::vector<InventoryEvent> events;
};

// 快照状态机阶段
enum class SnapState {
    IDLE,        // 初始化阶段（等待第一份有效快照）
    COMPARE,     // 正常对比阶段
};

// 开门初始化状态：后台请求和第一份无手稳定快照是两个独立阶段
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

enum class RelocationDecision {
    NO_RELOCATION,
    LOW_CONFIDENCE,
    CONFIRMED,
};

struct TrackEvidence {
    int track_id = -1;
    int cls_id = -1;
    int associated_item_id = -1;
    BBox start_box;
    BBox end_box;
    int start_frame_id = 0;
    int end_frame_id = 0;
    int sample_count = 0;
    bool occluded_by_hand = false;
    float confidence = 0.0f;
};

struct HeldProxyEvidence {
    int item_id = -1;
    int original_object_track_id = -1;
    int held_by_hand_track_id = -1;
    BBox last_visible_box;
    BBox hand_bbox_at_hold_start;
    std::vector<BBox> proxy_boxes;
    int held_start_frame_id = 0;
    int held_end_frame_id = 0;
    float confidence = 0.0f;
};

struct OperationContext {
    int context_id = 0;
    int baseline_snapshot_id = 0;
    bool hand_seen = false;
    int hand_frame_count = 0;
    bool hand_long_present = false;
    std::set<int> hand_track_ids;
    std::map<int, int> candidate_held_items;
    std::set<int> confirmed_held_items;
    std::vector<HeldProxyEvidence> held_proxy_evidences;
    std::map<int, TrackEvidence> active_track_evidences;
    std::set<int> moving_tracks;
    std::set<int> moved_item_candidates;
    int unstable_frame_count = 0;
    int created_frame_id = 0;
    int updated_frame_id = 0;

    void reset(int new_context_id, int new_baseline_snapshot_id, int frame_id);
};

struct PendingRelocation {
    int pending_id = 0;
    int item_id_A = -1;
    int class_id = -1;
    BBox old_bbox;
    BBox candidate_new_bbox;
    VotingItem snapshot_object_B;
    float reid_score = 0.0f;
    float evidence_score = 0.0f;
    int created_snapshot_id = 0;
    int last_checked_snapshot_id = 0;
    int stable_count = 0;
    int expire_after_stable_count = 2;
};

class SessionManager {
public:
    SessionManager();

    // ===== 主接口 =====

    // 开门：开始一轮新会话，等待后台库存或首个无手稳定快照
    void start_new_session(long long time_ms = 0);

    // 初始化（开门时从后台获取库存）
    // authoritative_empty=true 表示后台明确知道库存为空；默认空列表不可信。
    void init_from_backend(const std::vector<InventoryItem>& items,
                           bool authoritative_empty = false);

    // 后台失败、超时、未接入或返回非权威空列表时调用
    void mark_backend_unavailable();

    // 关门：冻结本轮库存，清理未确认的跨快照证据
    void finish_session(long long time_ms = 0);

    // 每帧调用：更新 OperationContext
    // hand_boxes: 当前帧所有手的bbox
    // tracks: 当前帧ByteTrack输出的所有track
    // frame_id: 当前帧号
    // time_ms: 当前时间戳（毫秒）
    // 返回：保留兼容主循环；当前不会直接产生库存事件
    SettlementResult update_hand(const std::vector<BBox>& hand_boxes,
                                 const std::vector<Track>& tracks,
                                 int frame_id, long long time_ms);

    // 推入一份快照（由SnapshotBuffer生成）
    // frame: 当前帧图像（用于颜色比较）
    // 返回：事件结果
    SettlementResult push_snapshot(const Snapshot& snap, const cv::Mat& frame);

    // ===== 查询接口 =====
    bool has_backend() const { return backend_status_ == BackendStatus::TRUSTED; }
    bool ready() const { return init_state_ == InitState::READY; }
    const InventoryDB& inventory() const { return inventory_; }
    InventoryDB& inventory() { return inventory_; }

    // ===== 手状态信息 =====
    bool hand_present() const { return hand_present_; }
    int  no_hand_streak() const { return no_hand_streak_; }

private:
    // ---- 快照状态机（4变量：snap1/contrast/current + has_snap1） ----
    SnapState snap_state_;
    Snapshot snap1_;          // 基准快照（用于对比的"快照1"）
    Snapshot contrast_;       // 稳定性检测基准
    Snapshot current_;        // 当前快照
    bool has_snap1_;          // snap1_ 是否有效

    // ---- 手部显示状态（仅用于 UI / OSD，不直接裁决库存） ----
    bool hand_present_;
    int no_hand_streak_;

    // ---- 操作上下文与待确认整理 ----
    OperationContext operation_context_;
    int next_context_id_;
    std::vector<PendingRelocation> pending_relocations_;
    int next_pending_id_;

    // ---- 库存 ----
    InventoryDB inventory_;
    BackendStatus backend_status_;
    InitState init_state_;
    bool inventory_initialized_;                  // 库存是否已初始化（防止重复初始化）
    std::vector<InventoryItem> backend_items_;

    // ---- 时间 ----
    long long current_time_ms_;
    long long session_start_time_ms_;
    bool first_empty_grace_logged_;

    // ---- 辅助方法 ----

    // 初始化时拍第一份快照（匹配后台库存或直接用快照初始化）
    void init_snapshot(const Snapshot& snap, const cv::Mat& frame);

    // 判断两份快照是否"差不多"（大部分物品都匹配上了）
    bool snapshots_similar(const Snapshot& a, const Snapshot& b);

    // baseline vs stable 的统一差异裁决（更新库存）
    SettlementResult compare_snapshots(const Snapshot& snap2, const cv::Mat& frame);

    // 重置快照状态机（门状态/外部需要重新建立 baseline 时调用）
    void reset_snap_state();

    // 每轮 stable diff 完成后重建 OperationContext
    void reset_operation_context(int baseline_snapshot_id, int frame_id);

    // 每帧收集手、HELD代理、ByteTrack移动证据
    void update_operation_context(const std::vector<BBox>& hand_boxes,
                                  const std::vector<Track>& tracks,
                                  int frame_id);

    int find_inventory_item_strict(const BBox& box, int cls_id,
                                   bool include_out) const;
    int find_inventory_item_relaxed(const BBox& box, int cls_id,
                                    bool include_out) const;
    int find_inventory_item_for_track(const Track& track) const;
    int find_best_hand_track_id(const BBox& hand_box,
                                const std::vector<Track>& tracks) const;
    int find_original_object_track_id(int item_id) const;

    float reid_score(const VotingItem& a, const VotingItem& b) const;
    float relocation_evidence_score(int item_id,
                                    const BBox& from,
                                    const BBox& to) const;
    RelocationDecision relocation_match(int item_id,
                                        const VotingItem& a,
                                        const VotingItem& b,
                                        float reid,
                                        float second_reid,
                                        float* evidence_score) const;
    void process_pending_relocations(const Snapshot& snap2,
                                     SettlementResult& result,
                                     std::set<int>& reserved_snap2_indices);

    // 严格身份匹配（5条件，含颜色，frame为空时跳过颜色检查）
    bool match_strict(const BBox& a, int cls_a, const BBox& b, int cls_b,
                      const cv::Mat& frame);

    // 颜色差异（16x16 resize + 平均RGB差）
    float color_diff(const cv::Mat& frame, const BBox& a, const BBox& b);

    // 打印库存
    void print_inventory();
};

}  // namespace fridge

#endif  // __FRIDGE_SESSION_H
