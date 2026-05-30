// ============================================================================
//  stability.h
//  稳态监控状态机
//
//  系统层面只有两个状态：
//      STABLE      画面"安静"，没人动东西，可以拍快照
//      DISTURBED   有手 / 有物品在动，所有库存更新都暂停
//
//  本类只负责"判断当前是哪个状态" + "本帧是否发生了状态切换"，
//  不负责拍快照、不负责 diff、不负责更新库存（那是 SessionManager 的事）。
//
//  STABLE 判定条件 (全部满足才算稳)：
//    1) 画面里没有 hand 类的活跃 track
//    2) 所有 food track 在最近 STABLE_MOTION_WINDOW 帧内中心点位移都
//       <= STABLE_MOVE_PIX
//
//  在不稳的时候，我们要求"立刻"进 DISTURBED；但要从 DISTURBED 回到 STABLE，
//  必须连续 STABLE_CONFIRM_FRAMES 帧都满足上面两个条件。
//  这样设计的目的：
//    - 漏报扰动 → 库存被错改（代价大）
//    - 多扰动一次 → 多等 0.6 秒（代价小）
//  所以"宁愿多扰动，不漏扰动"。
// ============================================================================
#ifndef __FRIDGE_STABILITY_H
#define __FRIDGE_STABILITY_H

#include <map>
#include <vector>
#include <deque>
#include "tracker.h"
#include "geometry.h"

namespace fridge {

enum class SystemState {
    STABLE,     // 安静，可以做事件结算
    DISTURBED,  // 有手 / 有物品在动，所有库存更新挂起
};

// 本帧是否发生了状态切换（用于让外面知道"现在该拍 before/after 了"）
enum class StateTransition {
    NONE,                     // 状态没变
    STABLE_TO_DISTURBED,      // 进入扰动 → 上层应该冻结 before 快照
    DISTURBED_TO_STABLE,      // 进入稳态 → 上层应该取 after 快照并 diff
};

class StabilityMonitor {
public:
    StabilityMonitor();

    // 每帧调用。返回本帧的状态切换信号（多数帧是 NONE）
    StateTransition update(const std::vector<Track>& tracks, int frame_id);

    SystemState state() const { return state_; }

    // 调试用：当前为什么不稳
    const char* last_unstable_reason() const { return last_unstable_reason_; }

private:
    // 判断本帧是不是"看起来稳"（瞬时判断，不带状态）
    bool is_frame_stable(const std::vector<Track>& tracks, int frame_id);

    // 维护每个 food track 的最近 N 帧中心点轨迹，用于"位移判断"
    void update_motion_history(const std::vector<Track>& tracks, int frame_id);

    // 清掉超过 STABLE_MOTION_WINDOW 帧没更新的 track 历史，避免内存膨胀
    void gc_motion_history(int frame_id);

private:
    SystemState state_;

    // 进入新状态需要的连续达标帧数
    int stable_streak_;       // 连续 N 帧"看起来稳"
    int disturb_streak_;      // 连续 N 帧"看起来不稳"

    // food track_id -> 最近若干帧的中心点 (frame_id, cx, cy)
    struct MotionPoint { int frame_id; float cx; float cy; };
    std::map<int, std::deque<MotionPoint>> motion_history_;

    // 上一次判定不稳的原因（调试日志用，免得每帧打印）
    const char* last_unstable_reason_;
};

// 调试辅助
const char* system_state_to_str(SystemState s);
const char* state_transition_to_str(StateTransition t);

}  // namespace fridge

#endif  // __FRIDGE_STABILITY_H
