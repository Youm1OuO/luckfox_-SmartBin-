// ============================================================================
//  hand_state.h
//  手部状态机 + 事件配对
//
//  职责：
//    - 给每只手 (track_id) 维护一个 4 状态机：EMPTY / SUS_HOLD / HOLD / SUS_RELEASE
//    - 状态变化时产生原始事件（PICKUP / PUTDOWN）
//    - 当手离开画面（track_id 销毁）时，把这只手期间产生的所有原始事件
//      做配对处理，得到最终业务事件（IN / OUT / RELOCATE）
//
//  状态转移示意：
//                 IoU突然>=HIGH
//      ┌─────┐ ──────────────▶ ┌──────────┐  连续 N 帧 IoU 仍高
//      │EMPTY│                  │SUS_HOLD  │ ───────────────▶ ┌─────┐
//      └─────┘ ◀────────────── └──────────┘                  │ HOLD │
//                IoU 掉回去                                    └─────┘
//                                                                │
//                                              IoU < LOW          │
//                                                ▼                │
//                              ┌──────────────────┐               │
//                              │ SUS_RELEASE      │ ◀─────────────┘
//                              └──────────────────┘
//                                          │ 连续 N 帧 IoU 仍低
//                                          ▼
//                                       回到 EMPTY，触发"放下"事件
// ============================================================================
#ifndef __FRIDGE_HAND_STATE_H
#define __FRIDGE_HAND_STATE_H

#include <vector>
#include <map>
#include <string>
#include "tracker.h"
#include "geometry.h"

namespace fridge {

// 手的 4 个状态
enum class HandState {
    EMPTY,          // 空手
    SUS_HOLD,       // 疑似拿起（候选物品 IoU 跃升，等待连续帧确认）
    HOLD,           // 已确认持物
    SUS_RELEASE     // 疑似放下（IoU 掉了，等待连续帧确认是否真的松开）
};

// 原始事件类型（手 ID 期间累积的、未经配对的事件）
enum class RawEventType {
    PICKUP,         // 拿起
    PUTDOWN         // 放下
};

// 一条原始事件
struct RawEvent {
    RawEventType type;
    int cls_id;             // 物品类别
    BBox pos;               // 位置（手框中心 ≈ 操作位置；这里直接存物品的 bbox）
    int frame_id;           // 发生帧号
};

// 最终业务事件类型（配对完成后的）
enum class FinalEventType {
    IN,             // 放入冰箱（库存 +1）
    OUT,            // 从冰箱取出（库存 -1）
    RELOCATE        // 整理（库存数量不变，仅位置变化）
};

struct FinalEvent {
    FinalEventType type;
    int cls_id;
    BBox from_pos;          // RELOCATE 用：原位置；OUT 用：取走前的位置
    BBox to_pos;            // RELOCATE 用：新位置；IN 用：放下的位置
    int frame_begin;
    int frame_end;
    int hand_track_id;      // 由哪只手完成的（追溯用）
};

// 一只手的"操作日志"：从这只手 track 出生到销毁期间的所有信息
struct HandLog {
    int hand_track_id;
    HandState state;
    int candidate_cls_id;        // SUS_HOLD/SUS_RELEASE 期间锁定的物品类别
    int candidate_obj_track_id;  // 锁定的物品 track id
    BBox candidate_box;          // 候选物品的最新位置
    int sus_counter;             // 疑似状态的连续帧计数
    int held_cls_id;             // 当前 HOLD 状态下持有的物品类别
    int held_obj_track_id;       // 当前 HOLD 状态下持有的物品 track id
    bool entered_holding;        // 这只手出现在画面里的第一帧是不是已经持物了

    std::vector<RawEvent> raw_events;   // 这只手期间累积的所有原始事件
    int last_seen_frame;
};

// 手部状态机管理器
class HandStateManager {
public:
    HandStateManager() = default;

    // 每帧调用一次。
    // 输入：本帧所有 track（手 + 物品）
    // 输出：本次 update 完成结算的"最终事件"（多数帧为空；
    //       仅当某只手刚消失时才会有事件被产出）
    std::vector<FinalEvent> update(const std::vector<Track>& tracks, int frame_id);

private:
    // 把 tracks 拆成手 + 物品两个列表
    void split_tracks(const std::vector<Track>& tracks,
                      std::vector<const Track*>& hands,
                      std::vector<const Track*>& objects);

    // 找一只手当前 IoU 最高的物品（>= IoU 阈值才返回，否则返回 nullptr）
    const Track* find_top_overlapping_object(const Track& hand,
                                             const std::vector<const Track*>& objects,
                                             float iou_thresh);

    // 处理一只手的本帧逻辑：状态转移 + 产生原始事件
    void process_hand(HandLog& log,
                      const Track& hand,
                      const std::vector<const Track*>& objects,
                      int frame_id);

    // 一只手离开画面（track 不再存在）：把它的 raw_events 配对成最终事件
    std::vector<FinalEvent> finalize_hand(HandLog& log);

private:
    // hand_track_id -> HandLog
    std::map<int, HandLog> hand_logs_;
};

// =========================================================================
//  调试辅助：枚举转字符串
// =========================================================================
const char* hand_state_to_str(HandState s);
const char* final_event_type_to_str(FinalEventType t);

}  // namespace fridge

#endif  // __FRIDGE_HAND_STATE_H
