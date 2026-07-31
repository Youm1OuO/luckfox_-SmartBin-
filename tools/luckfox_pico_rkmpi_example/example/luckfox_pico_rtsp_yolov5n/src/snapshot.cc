// ============================================================================
//  snapshot.cc
//  多帧投票快照（2.0）
//
//  每一帧先对 Detection 和既有临时候选做全局一对一匹配：优先匹配数量，
//  数量相同再比较匹配分数。这样相邻的同类物品不会在同一帧被并入同一票。
// ============================================================================
#include "snapshot.h"
#include "fridge_config.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

namespace fridge {
namespace {

float ratio_difference(float a, float b) {
    const float larger = std::max(std::fabs(a), std::fabs(b));
    return larger > 0.001f ? std::fabs(a - b) / larger : 0.0f;
}

struct SnapshotCandidate {
    VotingItem item;
    int last_vote_frame = -1;
};

bool snapshot_item_matches(const SnapshotCandidate& candidate,
                           const Detection& detection) {
    return candidate.item.cls_id == detection.cls_id &&
           candidate.item.box.area() > 0.0f && detection.box.area() > 0.0f &&
           normalized_nearby_distance(candidate.item.box, detection.box)
               <= SNAPSHOT_VOTE_CENTER_NORM &&
           ratio_difference(candidate.item.box.w(), detection.box.w())
               <= SNAPSHOT_VOTE_WIDTH_RATIO &&
           ratio_difference(candidate.item.box.h(), detection.box.h())
               <= SNAPSHOT_VOTE_HEIGHT_RATIO;
}

float snapshot_match_score(const SnapshotCandidate& candidate,
                           const Detection& detection) {
    const float center = normalized_nearby_distance(candidate.item.box, detection.box);
    const float width = ratio_difference(candidate.item.box.w(), detection.box.w());
    const float height = ratio_difference(candidate.item.box.h(), detection.box.h());
    // 仅在 snapshot_item_matches 已经成立时调用。分数越大越好。
    return 1.0f - 0.60f * (center / SNAPSHOT_VOTE_CENTER_NORM)
                 - 0.20f * (width / SNAPSHOT_VOTE_WIDTH_RATIO)
                 - 0.20f * (height / SNAPSHOT_VOTE_HEIGHT_RATIO);
}

struct FlowEdge {
    int to;
    int rev;
    int cap;
    int cost;
};

void add_flow_edge(std::vector<std::vector<FlowEdge> >& graph,
                   int from, int to, int cap, int cost) {
    FlowEdge forward;
    forward.to = to;
    forward.rev = static_cast<int>(graph[to].size());
    forward.cap = cap;
    forward.cost = cost;
    FlowEdge backward;
    backward.to = from;
    backward.rev = static_cast<int>(graph[from].size());
    backward.cap = 0;
    backward.cost = -cost;
    graph[from].push_back(forward);
    graph[to].push_back(backward);
}

// 返回这一帧 Detection 与候选的全局一对一最佳关联。
// 边的奖励中放入很大的“匹配数量优先”常数，因此最小费用流先最大化匹配数，
// 然后才在同样数量下最大化 score。所有循环按索引固定，平分时也可复现。
std::vector<std::pair<int, int> > select_best_one_to_one_pairs(
        const std::vector<Detection>& detections,
        const std::vector<SnapshotCandidate>& candidates,
        int frame_index) {
    const int detection_count = static_cast<int>(detections.size());
    const int candidate_count = static_cast<int>(candidates.size());
    std::vector<std::pair<int, int> > result;
    if (detection_count == 0 || candidate_count == 0) return result;

    const int source = 0;
    const int detection_begin = 1;
    const int candidate_begin = detection_begin + detection_count;
    const int sink = candidate_begin + candidate_count;
    std::vector<std::vector<FlowEdge> > graph(sink + 1);

    for (int di = 0; di < detection_count; ++di) {
        add_flow_edge(graph, source, detection_begin + di, 1, 0);
    }
    for (int ci = 0; ci < candidate_count; ++ci) {
        add_flow_edge(graph, candidate_begin + ci, sink, 1, 0);
    }

    const int kCardinalityBonus = 1000000;
    for (int di = 0; di < detection_count; ++di) {
        for (int ci = 0; ci < candidate_count; ++ci) {
            if (candidates[ci].last_vote_frame == frame_index ||
                !snapshot_item_matches(candidates[ci], detections[di])) {
                continue;
            }
            const int score = static_cast<int>(
                std::max(0.0f, snapshot_match_score(candidates[ci], detections[di])) * 1000.0f);
            add_flow_edge(graph, detection_begin + di, candidate_begin + ci,
                          1, -(kCardinalityBonus + score));
        }
    }

    const int node_count = sink + 1;
    while (true) {
        const int kInf = std::numeric_limits<int>::max() / 4;
        std::vector<int> dist(node_count, kInf);
        std::vector<int> prev_node(node_count, -1);
        std::vector<int> prev_edge(node_count, -1);
        std::vector<bool> in_queue(node_count, false);
        std::queue<int> q;
        dist[source] = 0;
        q.push(source);
        in_queue[source] = true;

        while (!q.empty()) {
            const int u = q.front();
            q.pop();
            in_queue[u] = false;
            for (int ei = 0; ei < static_cast<int>(graph[u].size()); ++ei) {
                const FlowEdge& edge = graph[u][ei];
                if (edge.cap <= 0 || dist[u] == kInf) continue;
                if (dist[edge.to] < dist[u] + edge.cost) continue;
                // 相同费用时保持先遇到的前驱，作为稳定的索引顺序 tie-break。
                if (dist[edge.to] == dist[u] + edge.cost) continue;
                dist[edge.to] = dist[u] + edge.cost;
                prev_node[edge.to] = u;
                prev_edge[edge.to] = ei;
                if (!in_queue[edge.to]) {
                    q.push(edge.to);
                    in_queue[edge.to] = true;
                }
            }
        }

        if (prev_node[sink] < 0 || dist[sink] >= 0) break;
        for (int v = sink; v != source; v = prev_node[v]) {
            const int u = prev_node[v];
            const int ei = prev_edge[v];
            FlowEdge& edge = graph[u][ei];
            edge.cap -= 1;
            graph[v][edge.rev].cap += 1;
        }
    }

    for (int di = 0; di < detection_count; ++di) {
        const std::vector<FlowEdge>& edges = graph[detection_begin + di];
        for (size_t ei = 0; ei < edges.size(); ++ei) {
            const FlowEdge& edge = edges[ei];
            if (edge.to < candidate_begin || edge.to >= sink || edge.cap != 0) continue;
            result.push_back(std::make_pair(di, edge.to - candidate_begin));
            break;
        }
    }
    return result;
}

void update_candidate(SnapshotCandidate& candidate,
                      const Detection& detection, int frame_index) {
    const int old_count = candidate.item.count;
    const float old_weight = static_cast<float>(old_count);
    const float new_weight = old_weight + 1.0f;
    candidate.item.box.x1 = (candidate.item.box.x1 * old_weight + detection.box.x1) / new_weight;
    candidate.item.box.y1 = (candidate.item.box.y1 * old_weight + detection.box.y1) / new_weight;
    candidate.item.box.x2 = (candidate.item.box.x2 * old_weight + detection.box.x2) / new_weight;
    candidate.item.box.y2 = (candidate.item.box.y2 * old_weight + detection.box.y2) / new_weight;
    candidate.item.best_score = std::max(candidate.item.best_score, detection.score);
    candidate.item.count++;
    candidate.last_vote_frame = frame_index;
}

SnapshotCandidate make_candidate(const Detection& detection, int frame_index) {
    SnapshotCandidate candidate;
    candidate.item.cls_id = detection.cls_id;
    candidate.item.box = detection.box;
    candidate.item.best_score = detection.score;
    candidate.item.count = 1;
    candidate.item.item_id = -1;
    candidate.last_vote_frame = frame_index;
    return candidate;
}

}  // namespace

SnapshotBuffer::SnapshotBuffer(int N, float s) : N_(N), s_(s) {
    frames_.reserve(N);
    frame_ids_.reserve(N);
}

void SnapshotBuffer::push(const std::vector<Detection>& detections, int frame_id) {
    frames_.push_back(detections);
    frame_ids_.push_back(frame_id);
}

bool SnapshotBuffer::full() const {
    return static_cast<int>(frames_.size()) >= N_;
}

void SnapshotBuffer::reset() {
    frames_.clear();
    frame_ids_.clear();
}

Snapshot SnapshotBuffer::take_snapshot() {
    Snapshot snapshot;
    if (frames_.empty()) return snapshot;

    snapshot.frame_id = frame_ids_.back();
    std::vector<SnapshotCandidate> candidates;

    for (int fi = 0; fi < static_cast<int>(frames_.size()); ++fi) {
        const std::vector<Detection>& frame_items = frames_[fi];
        std::vector<Detection> eligible;
        eligible.reserve(frame_items.size());
        for (size_t di = 0; di < frame_items.size(); ++di) {
            if (frame_items[di].score >= SNAPSHOT_MIN_SCORE) {
                eligible.push_back(frame_items[di]);
            }
        }

        const std::vector<std::pair<int, int> > pairs =
            select_best_one_to_one_pairs(eligible, candidates, fi);
        std::vector<bool> matched_detection(eligible.size(), false);
        for (size_t pi = 0; pi < pairs.size(); ++pi) {
            const int di = pairs[pi].first;
            const int ci = pairs[pi].second;
            update_candidate(candidates[ci], eligible[di], fi);
            matched_detection[di] = true;
        }
        for (size_t di = 0; di < eligible.size(); ++di) {
            if (!matched_detection[di]) {
                candidates.push_back(make_candidate(eligible[di], fi));
            }
        }
    }

    const float actual_frame_count = static_cast<float>(frames_.size());
    for (size_t ci = 0; ci < candidates.size(); ++ci) {
        const VotingItem& item = candidates[ci].item;
        if (static_cast<float>(item.count) <= actual_frame_count * s_) continue;
        if (item.box.w() < SNAPSHOT_MIN_BOX_WIDTH ||
            item.box.h() < SNAPSHOT_MIN_BOX_HEIGHT) {
            continue;
        }
        snapshot.items.push_back(item);
    }

    std::stable_sort(snapshot.items.begin(), snapshot.items.end(),
        [](const VotingItem& a, const VotingItem& b) {
            return a.box.area() > b.box.area();
        });
    for (size_t i = 0; i < snapshot.items.size(); ++i) {
        snapshot.items[i].temporary_id = -1 - static_cast<int>(i);
        snapshot.items[i].item_id = -1;
    }

    snapshot.valid = true;
    reset();
    return snapshot;
}

}  // namespace fridge
