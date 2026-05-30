// ============================================================================
//  stability.cc
//  稳态监控状态机实现
// ============================================================================
#include "stability.h"
#include "fridge_config.h"

namespace fridge {

const char* system_state_to_str(SystemState s) {
    switch (s) {
        case SystemState::STABLE:    return "STABLE";
        case SystemState::DISTURBED: return "DISTURBED";
    }
    return "?";
}

const char* state_transition_to_str(StateTransition t) {
    switch (t) {
        case StateTransition::NONE:                return "NONE";
        case StateTransition::STABLE_TO_DISTURBED: return "STABLE→DISTURBED";
        case StateTransition::DISTURBED_TO_STABLE: return "DISTURBED→STABLE";
    }
    return "?";
}

StabilityMonitor::StabilityMonitor()
    : state_(SystemState::STABLE),       // 系统冷启动假设是稳态
      stable_streak_(STABLE_CONFIRM_FRAMES),  // 一上来就视为已稳
      disturb_streak_(0),
      last_unstable_reason_("init") {}

void StabilityMonitor::update_motion_history(
        const std::vector<Track>& tracks, int frame_id) {
    for (const auto& t : tracks) {
        if (!is_food(t.cls_id)) continue;
        auto& dq = motion_history_[t.track_id];
        MotionPoint p;
        p.frame_id = frame_id;
        p.cx = t.box.cx();
        p.cy = t.box.cy();
        dq.push_back(p);
        // 保留最近 STABLE_MOTION_WINDOW 个点
        while ((int)dq.size() > STABLE_MOTION_WINDOW) {
            dq.pop_front();
        }
    }
}

void StabilityMonitor::gc_motion_history(int frame_id) {
    // 超过 STABLE_MOTION_WINDOW * 4 帧没更新的 track 直接清掉
    // (track 销毁后历史记录就成了僵尸数据)
    const int stale = STABLE_MOTION_WINDOW * 4;
    for (auto it = motion_history_.begin(); it != motion_history_.end(); ) {
        if (it->second.empty()) {
            it = motion_history_.erase(it);
            continue;
        }
        int last = it->second.back().frame_id;
        if (frame_id - last > stale) {
            it = motion_history_.erase(it);
        } else {
            ++it;
        }
    }
}

bool StabilityMonitor::is_frame_stable(const std::vector<Track>& tracks,
                                       int frame_id) {
    // 条件 1：没有手
    for (const auto& t : tracks) {
        if (is_hand(t.cls_id)) {
            last_unstable_reason_ = "hand_present";
            return false;
        }
    }

    // 条件 2：所有 food track 在最近 STABLE_MOTION_WINDOW 帧位移 <= 阈值
    // 注意：刚出现的 track 还没攒够历史，不能立刻就判稳
    //       但如果是稳态下"突然冒出来一个 track"，那本身就该让它扰动一下（视为运动）
    //       这里保守处理：历史不足 STABLE_MOTION_WINDOW 个点 → 视为不稳
    for (const auto& t : tracks) {
        if (!is_food(t.cls_id)) continue;
        auto it = motion_history_.find(t.track_id);
        if (it == motion_history_.end()) {
            last_unstable_reason_ = "track_no_history";
            return false;
        }
        const auto& dq = it->second;
        if ((int)dq.size() < STABLE_MOTION_WINDOW) {
            last_unstable_reason_ = "track_warming_up";
            return false;
        }
        // 检查窗口内最大位移
        // 对窗口内每相邻两点的位移做累加上限不太合适，更直观的做法是
        // 取窗口内"最远点对"的距离 — 但 N=5~10 时窗口很短，
        // 直接比较"窗口首点 vs 窗口末点"的距离即可
        float dx = dq.back().cx - dq.front().cx;
        float dy = dq.back().cy - dq.front().cy;
        float d = std::sqrt(dx * dx + dy * dy);
        if (d > STABLE_MOVE_PIX) {
            last_unstable_reason_ = "track_moving";
            return false;
        }
    }

    last_unstable_reason_ = "stable";
    return true;
}

StateTransition StabilityMonitor::update(const std::vector<Track>& tracks,
                                         int frame_id) {
    // 先把本帧 food 的中心点写进历史（不论稳不稳都要记，下一帧才有得看）
    update_motion_history(tracks, frame_id);
    gc_motion_history(frame_id);

    bool frame_stable = is_frame_stable(tracks, frame_id);

    StateTransition transition = StateTransition::NONE;

    if (frame_stable) {
        stable_streak_++;
        disturb_streak_ = 0;
    } else {
        disturb_streak_++;
        stable_streak_ = 0;
    }

    if (state_ == SystemState::STABLE) {
        // 当前是稳态，如果连续 DISTURBED_CONFIRM_FRAMES 帧不稳 → 进入扰动
        if (disturb_streak_ >= DISTURBED_CONFIRM_FRAMES) {
            state_ = SystemState::DISTURBED;
            transition = StateTransition::STABLE_TO_DISTURBED;
        }
    } else {
        // 当前是扰动态，连续 STABLE_CONFIRM_FRAMES 帧稳 → 回到稳态
        if (stable_streak_ >= STABLE_CONFIRM_FRAMES) {
            state_ = SystemState::STABLE;
            transition = StateTransition::DISTURBED_TO_STABLE;
        }
    }

    return transition;
}

}  // namespace fridge
