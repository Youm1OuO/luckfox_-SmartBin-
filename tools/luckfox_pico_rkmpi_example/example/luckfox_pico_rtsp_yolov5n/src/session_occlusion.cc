// ============================================================================
//  session_occlusion.cc
//  Pure blocker-graph, coverage facts, and visibility lifecycle decisions.
//
//  This module deliberately has no access to SessionManager runtime state.  It
//  receives already-confirmed identity/event facts and returns a plan.  The
//  settlement module remains the only place that applies the plan and commits
//  InventoryDB.
// ============================================================================
#include "session_internal.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <vector>

namespace fridge {
namespace session_internal {

namespace {

BBox reliable_box(const InventoryItem& item) {
    return item.base_box.area() > 0.0f ? item.base_box : item.box;
}

std::vector<BBox> remaining_after_covers(
        const BBox& target, const std::vector<BBox>& covers) {
    std::vector<BBox> remaining;
    if (target.area() <= 0.0f) return remaining;
    remaining.push_back(target);
    for (size_t ci = 0; ci < covers.size(); ++ci) {
        std::vector<BBox> next;
        for (size_t ri = 0; ri < remaining.size(); ++ri) {
            subtract_cover(remaining[ri], covers[ci], &next);
        }
        remaining.swap(next);
        if (remaining.empty()) break;
    }
    return remaining;
}

float remaining_area(const std::vector<BBox>& pieces) {
    float area = 0.0f;
    for (size_t i = 0; i < pieces.size(); ++i) area += pieces[i].area();
    return area;
}

bool residual_touches_outer_boundary(const BBox& target,
                                     const std::vector<BBox>& residual) {
    if (target.area() <= 0.0f) return false;
    const float epsilon = 0.001f;
    bool has_residual = false;
    for (size_t i = 0; i < residual.size(); ++i) {
        const BBox& piece = residual[i];
        if (piece.area() <= 0.0f) continue;
        has_residual = true;
        const bool touches_left = piece.x1 <= target.x1 + epsilon;
        const bool touches_right = piece.x2 >= target.x2 - epsilon;
        const bool touches_top = piece.y1 <= target.y1 + epsilon;
        const bool touches_bottom = piece.y2 >= target.y2 - epsilon;
        // A residual that bridges two opposite outer edges is an internal
        // split/gap through the object, not a detector-truncated boundary.
        // Adjacent-edge corner residuals remain valid.
        if ((!touches_left && !touches_right &&
             !touches_top && !touches_bottom) ||
            (touches_left && touches_right &&
             !touches_top && !touches_bottom) ||
            (touches_top && touches_bottom &&
             !touches_left && !touches_right)) {
            return false;
        }
    }
    return has_residual;
}

void erase_blocker_from_all(std::map<int, std::set<int> >* relations,
                            int blocker_id) {
    if (!relations) return;
    for (std::map<int, std::set<int> >::iterator target = relations->begin();
         target != relations->end(); ++target) {
        target->second.erase(blocker_id);
    }
}

OcclusionProof make_proof(OcclusionProofKind kind,
                          const std::set<int>& witnesses) {
    OcclusionProof proof;
    proof.kind = kind;
    proof.witness_blocker_ids = witnesses;
    return proof;
}

}  // namespace

CoverageEvaluation evaluate_coverage_facts(
        const BBox& target_box, const std::vector<BBox>& strict_cover_boxes,
        const std::vector<BBox>& edge_cover_boxes, bool allow_edge_residual) {
    CoverageEvaluation evaluation;
    evaluation.cover_box_count = strict_cover_boxes.size();
    if (target_box.area() <= 0.0f) return evaluation;

    const std::vector<BBox> residual =
        remaining_after_covers(target_box, strict_cover_boxes);
    const float target_area = target_box.area();
    const float uncovered = std::max(0.0f, remaining_area(residual));
    evaluation.uncovered_ratio = std::min(1.0f, uncovered / target_area);
    evaluation.covered_ratio = std::max(0.0f,
        std::min(1.0f, 1.0f - evaluation.uncovered_ratio));
    evaluation.strict_full = uncovered <= COVER_REMAINING_AREA_EPS;
    evaluation.residual_is_outer_boundary_only =
        !evaluation.strict_full &&
        residual_touches_outer_boundary(target_box, residual);
    evaluation.edge_residual_full = !evaluation.strict_full &&
        allow_edge_residual && edge_residual_within_target_border(
            target_box, edge_cover_boxes,
            FLOW3_CONFIRMED_OCCLUSION_EDGE_RESIDUAL_PX);
    evaluation.full = evaluation.strict_full || evaluation.edge_residual_full;
    return evaluation;
}

BlockerRelationGraph build_event_driven_blocker_graph(
        const std::map<int, InventoryItem>& operation_start,
        const std::map<int, InventoryItem>& working,
        const std::set<int>& confirmed_front_ids,
        const std::set<int>& confirmed_moved_ids,
        const std::set<int>& confirmed_out_ids) {
    BlockerRelationGraph graph;

    // Start with the last committed relation graph.  A missing current
    // observation is not an exit event and therefore cannot erase an edge.
    for (std::map<int, InventoryItem>::const_iterator target = working.begin();
         target != working.end(); ++target) {
        const std::map<int, InventoryItem>::const_iterator old =
            operation_start.find(target->first);
        if (old != operation_start.end()) {
            graph.effective_by_target[target->first] = old->second.block_ids;
        } else {
            // A new D is not a target of inferred historical occlusion.  It can
            // still act as a confirmed front for an old target below.
            graph.effective_by_target[target->first] = std::set<int>();
        }
        graph.effective_by_target[target->first].erase(target->first);
    }

    // A confirmed MOVE is an atomic remove-old/add-new operation.  Remove all
    // old edges first so a target intersecting both positions cannot lose the
    // newly-created edge due to iteration order.
    for (std::set<int>::const_iterator moved = confirmed_moved_ids.begin();
         moved != confirmed_moved_ids.end(); ++moved) {
        erase_blocker_from_all(&graph.effective_by_target, *moved);
        // A 移动后，A 自己在旧位置被谁遮住的历史关系同样失效。二维重叠
        // 不能推断静态 B 仍位于 A 新位置之前；后续仅允许本轮已确认的前景
        // 事件按新位置重新加入关系。
        std::map<int, std::set<int> >::iterator moved_target =
            graph.effective_by_target.find(*moved);
        if (moved_target != graph.effective_by_target.end()) {
            moved_target->second.clear();
        }
    }
    for (std::set<int>::const_iterator out = confirmed_out_ids.begin();
         out != confirmed_out_ids.end(); ++out) {
        erase_blocker_from_all(&graph.effective_by_target, *out);
    }

    // Only confirmed IN/MOVED fronts create new edges.  The front is never
    // inferred as being blocked by the target it is covering, and a newly
    // promoted D is not itself assigned historical blockers in this pass.
    for (std::set<int>::const_iterator front_id = confirmed_front_ids.begin();
         front_id != confirmed_front_ids.end(); ++front_id) {
        if (confirmed_out_ids.count(*front_id)) continue;
        const std::map<int, InventoryItem>::const_iterator front =
            working.find(*front_id);
        if (front == working.end()) continue;
        const BBox front_box = reliable_box(front->second);
        if (front_box.area() <= 0.0f) continue;
        for (std::map<int, InventoryItem>::const_iterator target = working.begin();
             target != working.end(); ++target) {
            if (target->first == *front_id ||
                !operation_start.count(target->first) ||
                confirmed_out_ids.count(target->first)) {
                continue;
            }
            const BBox target_box = reliable_box(target->second);
            if (target_box.area() <= 0.0f ||
                intersection_area(target_box, front_box) <= BLOCK_OVERLAP_AREA_EPS) {
                continue;
            }
            graph.effective_by_target[target->first].insert(*front_id);
        }
    }

    for (std::map<int, std::set<int> >::const_iterator target =
             graph.effective_by_target.begin();
         target != graph.effective_by_target.end(); ++target) {
        const std::map<int, InventoryItem>::const_iterator old =
            operation_start.find(target->first);
        const std::set<int> empty;
        const std::set<int>& historical = old == operation_start.end()
            ? empty : old->second.block_ids;
        for (std::set<int>::const_iterator id = target->second.begin();
             id != target->second.end(); ++id) {
            if (!historical.count(*id)) graph.added_by_target[target->first].insert(*id);
        }
        for (std::set<int>::const_iterator id = historical.begin();
             id != historical.end(); ++id) {
            if (!target->second.count(*id)) {
                graph.removed_by_target[target->first].insert(*id);
            }
        }
        for (std::set<int>::const_iterator moved = confirmed_moved_ids.begin();
             moved != confirmed_moved_ids.end(); ++moved) {
            if (historical.count(*moved) || target->second.count(*moved)) {
                graph.moved_by_target[target->first].insert(*moved);
            }
        }
    }
    return graph;
}

std::set<int> retain_pending_out_candidates(
        const std::set<int>& pending_out_ids,
        const std::map<int, BlockerTransitionPlan>& transition_plans) {
    std::set<int> retained;
    for (std::set<int>::const_iterator candidate = pending_out_ids.begin();
         candidate != pending_out_ids.end(); ++candidate) {
        const std::map<int, BlockerTransitionPlan>::const_iterator plan =
            transition_plans.find(*candidate);
        if (plan == transition_plans.end()) {
            retained.insert(*candidate);
            continue;
        }
        const OutDisposition disposition = plan->second.out;
        if (disposition == OutDisposition::HOLD_FOR_PENDING_OCCLUSION ||
            disposition == OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION ||
            disposition == OutDisposition::NOT_APPLICABLE) {
            continue;
        }
        retained.insert(*candidate);
    }
    return retained;
}

bool occlusion_proof_witnesses_valid(
        const OcclusionProof& proof,
        const std::set<int>& effective_blocker_ids) {
    if (proof.kind == OcclusionProofKind::NONE ||
        proof.witness_blocker_ids.empty()) {
        return false;
    }
    for (std::set<int>::const_iterator id = proof.witness_blocker_ids.begin();
         id != proof.witness_blocker_ids.end(); ++id) {
        if (!effective_blocker_ids.count(*id)) return false;
    }
    return true;
}

bool front_witness_boxes_are_continuous(
        const std::map<int, InventoryItem>& items,
        const std::map<int, BBox>& previous_boxes,
        const std::map<int, BBox>& current_boxes) {
    if (previous_boxes.empty() ||
        previous_boxes.size() != current_boxes.size()) {
        return false;
    }
    for (std::map<int, BBox>::const_iterator current = current_boxes.begin();
         current != current_boxes.end(); ++current) {
        const std::map<int, BBox>::const_iterator previous =
            previous_boxes.find(current->first);
        const std::map<int, InventoryItem>::const_iterator item =
            items.find(current->first);
        if (previous == previous_boxes.end() || item == items.end() ||
            !track_match_box(item->second.cls_id, previous->second,
                             item->second.cls_id, current->second)) {
            return false;
        }
    }
    return true;
}

OcclusionDecisionResult decide_occlusion_lifecycle(
        const OcclusionDecisionInput& input) {
    OcclusionDecisionResult result;
    const bool after_strict = input.after_geometry.strict_full;
    const bool after_edge = input.after_geometry.edge_residual_full;
    const bool after_full = input.after_geometry.full;
    const bool new_front_cover = input.relation_changed_by_confirmed_front &&
        input.current_confirmed_front;

    // A target with a legal direct observation is never promoted to a new
    // occlusion in this transaction.  Existing OCCLUDED status is handled
    // below because a direct frame may prove a reveal after a blocker exits.
    if (input.previous_status == ItemStatus::VISIBLE) {
        result.proposed_proof.clear();
        if (input.valid_direct_observation) {
            result.visibility = VisibilityDecision::KEEP_VISIBLE;
            result.out = OutDisposition::NOT_APPLICABLE;
            return result;
        }
        // A confirmed front item with a strict/edge-residual full-coverage
        // proof has direct causal evidence for OCCLUDED.  An identity
        // reservation on a same-class candidate only says that this target
        // cannot claim that candidate; it does not invalidate this already
        // established geometric proof.
        if (new_front_cover && after_strict) {
            result.visibility = VisibilityDecision::ENTER_OCCLUDED;
            result.out = OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION;
            result.allow_occluded_transition = true;
            result.proposed_proof = make_proof(
                OcclusionProofKind::STRICT_UNION,
                input.after_witness_blocker_ids);
            return result;
        }
        if (new_front_cover && after_edge) {
            result.visibility = VisibilityDecision::ENTER_OCCLUDED;
            result.out = OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION;
            result.allow_occluded_transition = true;
            result.proposed_proof = make_proof(
                OcclusionProofKind::EDGE_RESIDUAL_UNION,
                input.after_witness_blocker_ids);
            return result;
        }

        if (input.observation_conflict) {
            // Ambiguity is neither disappearance nor direct observation.
            // It must stop both disappearance and ordinary OUT evidence so
            // a later clear frame cannot be stitched onto this frame.  The
            // formal full-coverage cases above are the intentional exception.
            result.visibility = VisibilityDecision::PENDING_OCCLUSION_EVIDENCE;
            result.out = OutDisposition::HOLD_FOR_PENDING_OCCLUSION;
            return result;
        }

        // This is intentionally narrower than ordinary strict/edge coverage:
        // the front relation must be a newly confirmed event, the target must
        // be unobserved without ambiguity, and the residual may only be an
        // outer-boundary detection truncation.
        const bool disappearance_geometry =
            new_front_cover && !after_full &&
            input.after_geometry.covered_ratio >=
                FLOW3_CONFIRMED_OCCLUSION_DISAPPEARANCE_MIN_COVER_RATIO &&
            input.after_geometry.residual_is_outer_boundary_only &&
            !input.target_has_independent_exit_evidence;
        if (disappearance_geometry) {
            result.disappearance_candidate = true;
            result.matching_missing_frames =
                input.prior_disappearance_missing_frames + 1;
            if (result.matching_missing_frames >= FLOW3_NO_HAND_OUT_MISSING_FRAMES) {
                result.visibility = VisibilityDecision::ENTER_OCCLUDED;
                result.out = OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION;
                result.allow_occluded_transition = true;
                result.proposed_proof = make_proof(
                    OcclusionProofKind::DISAPPEARANCE_SUPPORTED,
                    input.after_witness_blocker_ids);
            } else {
                result.visibility = VisibilityDecision::PENDING_OCCLUSION_EVIDENCE;
                result.out = OutDisposition::HOLD_FOR_PENDING_OCCLUSION;
            }
            return result;
        }
        result.visibility = VisibilityDecision::KEEP_VISIBLE;
        result.out = OutDisposition::NORMAL_OUT_EVIDENCE;
        return result;
    }

    // An item with its own confirmed move/exit must be settled by that fact;
    // another object's overlap cannot turn its own destination into a reveal.
    if (input.target_has_independent_exit_evidence &&
        input.valid_direct_observation && !after_full) {
        result.visibility = VisibilityDecision::KEEP_VISIBLE;
        result.out = OutDisposition::NOT_APPLICABLE;
        result.proposed_proof.clear();
        return result;
    }

    if (after_strict) {
        result.visibility = VisibilityDecision::KEEP_OCCLUDED;
        result.out = OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION;
        result.proposed_proof = make_proof(
            OcclusionProofKind::STRICT_UNION,
            input.after_witness_blocker_ids);
        return result;
    }
    if (after_edge) {
        result.visibility = VisibilityDecision::KEEP_OCCLUDED;
        result.out = OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION;
        result.proposed_proof = make_proof(
            OcclusionProofKind::EDGE_RESIDUAL_UNION,
            input.after_witness_blocker_ids);
        return result;
    }
    if (input.previous_proof_unchanged && input.previous_proof_valid) {
        result.visibility = VisibilityDecision::KEEP_OCCLUDED;
        result.out = OutDisposition::BLOCKED_BY_CONFIRMED_OCCLUSION;
        result.proposed_proof = input.previous_proof;
        return result;
    }
    if (input.previous_proof_valid && input.valid_direct_observation &&
        !input.observation_conflict && !after_full) {
        result.visibility = VisibilityDecision::REVEAL_VISIBLE;
        result.out = OutDisposition::NOT_APPLICABLE;
        result.allow_revealed_transition = true;
        result.proposed_proof.clear();
        return result;
    }
    if (input.valid_direct_observation || input.observation_conflict) {
        // If a committed witness moved/left and the only current observation
        // is ambiguous, neither the old proof nor a REVEALED event is valid.
        // Keep the target pending until identity becomes clear or ordinary
        // post-occlusion-loss OUT evidence is available.
        if (input.observation_conflict && !input.previous_proof_unchanged) {
            result.visibility = VisibilityDecision::PENDING_REVEAL_OR_OUT;
            result.out = OutDisposition::HOLD_FOR_PENDING_OCCLUSION;
            result.proposed_proof.clear();
            return result;
        }
        // A legacy/ambiguous historical OCCLUDED record is conservative: a
        // single frame cannot fabricate a REVEALED event.
        result.visibility = VisibilityDecision::KEEP_OCCLUDED;
        result.out = input.observation_conflict
            ? OutDisposition::HOLD_FOR_PENDING_OCCLUSION
            : OutDisposition::NOT_APPLICABLE;
        result.proposed_proof = input.previous_proof_valid &&
            input.previous_proof_unchanged
            ? input.previous_proof : OcclusionProof();
        return result;
    }

    result.visibility = VisibilityDecision::PENDING_REVEAL_OR_OUT;
    result.out = OutDisposition::START_AFTER_OCCLUSION_LOSS;
    result.proposed_proof.clear();
    return result;
}

}  // namespace session_internal
}  // namespace fridge
