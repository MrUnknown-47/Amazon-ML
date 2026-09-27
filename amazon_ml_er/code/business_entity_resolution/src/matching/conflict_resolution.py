"""
Target Conflict Resolution and 1:1 Target Constraint Enforcement.
Evaluates whether enforcing the empirical training invariant (each target entity ID
belongs to at most one Source 1 entity) improves macro F_0.5 on validation.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict


def resolve_target_conflicts_greedy(
    predictions_by_s1: Dict[str, Set[str]],
    scored_candidates_by_s1: Dict[str, List[Tuple[str, float, Optional[Dict[str, Any]]]]],
    min_confidence: float = 0.0
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """
    Greedily resolves duplicate target assignments across S1 entities.
    For each target entity ID claimed by multiple S1 entities, assigns the target
    exclusively to the S1 entity that generated the highest classifier score.

    Returns:
        resolved_predictions, diagnostic_stats
    """
    # 1. Map target_id -> list of (s1_id, score)
    target_claims = defaultdict(list)
    score_lookup = defaultdict(dict)

    for s1_id, cands in scored_candidates_by_s1.items():
        for tid, sc, _ in cands:
            score_lookup[s1_id][tid] = sc

    for s1_id, pred_set in predictions_by_s1.items():
        for tid in pred_set:
            sc = score_lookup[s1_id].get(tid, 0.0)
            target_claims[tid].append((s1_id, sc))

    # 2. Resolve conflicts
    resolved_preds = {s1_id: set() for s1_id in predictions_by_s1}
    conflict_targets_count = 0
    total_duplicate_claims = 0

    for tid, claims in target_claims.items():
        if len(claims) == 1:
            s1_id, sc = claims[0]
            if sc >= min_confidence:
                resolved_preds[s1_id].add(tid)
        else:
            conflict_targets_count += 1
            total_duplicate_claims += (len(claims) - 1)
            # Sort claims descending by score
            best_s1_id, best_score = max(claims, key=lambda x: x[1])
            if best_score >= min_confidence:
                resolved_preds[best_s1_id].add(tid)

    stats = {
        "total_targets_claimed": len(target_claims),
        "conflicted_targets_count": conflict_targets_count,
        "duplicate_claims_removed": total_duplicate_claims,
        "conflict_ratio": float(round(conflict_targets_count / max(1, len(target_claims)), 5))
    }

    return resolved_preds, stats
