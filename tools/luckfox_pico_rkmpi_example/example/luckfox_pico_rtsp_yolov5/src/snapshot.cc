// ============================================================================
//  snapshot.cc
//  快照构造 + diff 实现
// ============================================================================
#include "snapshot.h"
#include "fridge_config.h"

namespace fridge {

const char* change_type_to_str(ChangeType t) {
    switch (t) {
        case ChangeType::IN:       return "IN";
        case ChangeType::OUT:      return "OUT";
        case ChangeType::RELOCATE: return "RELOCATE";
    }
    return "?";
}

Snapshot make_snapshot(const std::vector<Track>& tracks, int frame_id) {
    Snapshot s;
    s.frame_id = frame_id;
    s.valid = true;
    for (const auto& t : tracks) {
        // 只收食材，手不进快照
        if (!is_food(t.cls_id)) continue;
        // 分数太低的 track（多半是 LOST 状态被强行带出来）也不收
        if (t.score < SNAPSHOT_MIN_SCORE) continue;

        SnapshotItem it;
        it.track_id = t.track_id;
        it.cls_id = t.cls_id;
        it.box = t.box;
        it.score = t.score;
        s.items[t.track_id] = it;
    }
    return s;
}

std::vector<ChangeEvent> diff_snapshots(const Snapshot& before, const Snapshot& after) {
    std::vector<ChangeEvent> events;

    // 没有 before（没有"操作前"基线）→ 不能产生事件
    // 这种情况发生在系统刚启动还没拿到第一份稳态时
    if (!before.valid || !after.valid) return events;

    // 1) 遍历 before：track_id 不在 after 中 → OUT
    //    在 after 中 → 比较位置，决定 RELOCATE 或忽略
    for (const auto& kv : before.items) {
        int tid = kv.first;
        const SnapshotItem& bef = kv.second;

        auto it = after.items.find(tid);
        if (it == after.items.end()) {
            // 拿走
            ChangeEvent e;
            e.type = ChangeType::OUT;
            e.track_id = tid;
            e.cls_id = bef.cls_id;
            e.from_box = bef.box;
            e.to_box = bef.box;
            e.score = bef.score;
            e.frame_before = before.frame_id;
            e.frame_after = after.frame_id;
            events.push_back(e);
        } else {
            const SnapshotItem& aft = it->second;
            float d = center_distance(bef.box, aft.box);
            if (d > RELOCATE_MOVE_PIX) {
                // 整理（位置变化超过阈值）
                ChangeEvent e;
                e.type = ChangeType::RELOCATE;
                e.track_id = tid;
                // 类别按"after"的，因为 ByteTrack 出生时定的 cls_id 不每帧更新，
                // 但 after 这一刻 YOLO 给出的判断更新；不过这套实现里 cls_id
                // 一旦绑定就不变（tracker.cc 里有注释），所以 bef/aft.cls_id
                // 两者其实相同
                e.cls_id = aft.cls_id;
                e.from_box = bef.box;
                e.to_box = aft.box;
                e.score = aft.score;
                e.frame_before = before.frame_id;
                e.frame_after = after.frame_id;
                events.push_back(e);
            }
            // 距离 <= 阈值 → 没动，跳过（这是绝大多数 track 的情况）
        }
    }

    // 2) 遍历 after：track_id 不在 before 中 → IN
    for (const auto& kv : after.items) {
        int tid = kv.first;
        if (before.items.find(tid) != before.items.end()) continue;

        const SnapshotItem& aft = kv.second;
        ChangeEvent e;
        e.type = ChangeType::IN;
        e.track_id = tid;
        e.cls_id = aft.cls_id;
        e.from_box = aft.box;
        e.to_box = aft.box;
        e.score = aft.score;
        e.frame_before = before.frame_id;
        e.frame_after = after.frame_id;
        events.push_back(e);
    }

    return events;
}

}  // namespace fridge
