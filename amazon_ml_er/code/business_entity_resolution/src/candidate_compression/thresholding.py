"""
Candidate compression and adaptive thresholding policies.
Implements:
1. Fixed K budget
2. Adaptive K based on score confidence (score >= tau)
3. Adaptive K based on score gap (score >= alpha * top_score)
4. Adaptive K based on blocking channel evidence (multi-channel retention)
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np


def apply_candidate_budget(
    ranked_candidates: List[Tuple[str, float]],
    policy: str = "fixed_k",
    k: int = 40,
    threshold: float = 0.30,
    score_gap_ratio: float = 0.50,
    min_candidates: int = 1,
    max_candidates: int = 100,
    provenance_by_tid: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[str]:
    """
    Filter ranked candidates according to a compression policy.
    Returns:
        List of selected target entity IDs.
    """
    if not ranked_candidates:
        return []

    if policy == "fixed_k":
        return [tid for tid, _ in ranked_candidates[:k]]

    elif policy == "score_threshold":
        # Keep candidates with score >= threshold, subject to min and max constraints
        selected = []
        for tid, score in ranked_candidates:
            if score >= threshold:
                selected.append(tid)
            elif len(selected) < min_candidates:
                selected.append(tid)
            if len(selected) >= max_candidates:
                break
        if not selected and ranked_candidates:
            selected.append(ranked_candidates[0][0])
        return selected

    elif policy == "score_gap":
        # Keep candidates within score_gap_ratio of top candidate
        top_score = ranked_candidates[0][1]
        cutoff = top_score * score_gap_ratio
        selected = []
        for tid, score in ranked_candidates:
            if score >= cutoff or len(selected) < min_candidates:
                selected.append(tid)
            if len(selected) >= max_candidates:
                break
        return selected

    elif policy == "channel_evidence":
        # Always retain candidates retrieved by >= 2 channels, fill remainder up to k
        prov_map = provenance_by_tid or {}
        selected_set = set()
        selected = []

        # 1. Multi-channel hits first
        for tid, score in ranked_candidates:
            prov = prov_map.get(tid, {})
            ch_count = len(prov.get("channels", []))
            if ch_count >= 2:
                selected.append(tid)
                selected_set.add(tid)
                if len(selected) >= max_candidates:
                    break

        # 2. Fill with top ranked single-channel hits up to k
        for tid, score in ranked_candidates:
            if tid not in selected_set:
                selected.append(tid)
                selected_set.add(tid)
            if len(selected) >= k:
                break

        return selected

    else:
        raise ValueError(f"Unknown compression policy: {policy}")


def evaluate_compression_policy(
    ranked_candidates_by_s1: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    policy: str = "fixed_k",
    k: int = 40,
    threshold: float = 0.30,
    score_gap_ratio: float = 0.50,
    min_candidates: int = 1,
    max_candidates: int = 100,
    provenance_map: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None,
    total_target_records: int = 10320219
) -> Dict[str, Any]:
    """
    Empirically evaluate candidate retention and size metrics under a candidate compression policy.
    """
    prov_full = provenance_map or {}
    compressed_cands = {}
    total_s1 = len(ranked_candidates_by_s1)

    for s1_id, ranked in ranked_candidates_by_s1.items():
        prov_by_tid = prov_full.get(s1_id, {})
        selected = apply_candidate_budget(
            ranked,
            policy=policy,
            k=k,
            threshold=threshold,
            score_gap_ratio=score_gap_ratio,
            min_candidates=min_candidates,
            max_candidates=max_candidates,
            provenance_by_tid=prov_by_tid
        )
        compressed_cands[s1_id] = set(selected)

    # Compute recall
    total_true = sum(len(tids) for tids in ground_truth.values())
    captured_true = 0
    for s1_id, tids in ground_truth.items():
        cands = compressed_cands.get(s1_id, set())
        captured_true += len(tids & cands)

    cand_lengths = [len(cands) for cands in compressed_cands.values()]
    arr = np.array(cand_lengths) if cand_lengths else np.array([0])
    total_cands = int(arr.sum())

    recall = float(round(captured_true / max(1, total_true), 5))
    reduction_ratio = float(1.0 - (total_cands / max(1, total_s1 * total_target_records)))

    return {
        "policy": policy,
        "k": k,
        "threshold": threshold,
        "score_gap_ratio": score_gap_ratio,
        "candidate_recall": recall,
        "captured_true_matches": captured_true,
        "total_true_matches": total_true,
        "missed_matches": total_true - captured_true,
        "mean_candidates_per_S1": float(round(arr.mean(), 2)),
        "median_candidates_per_S1": float(np.median(arr)),
        "P95_candidates_per_S1": float(np.percentile(arr, 95)),
        "P99_candidates_per_S1": float(np.percentile(arr, 99)),
        "max_candidates_per_S1": int(arr.max()),
        "total_candidates": total_cands,
        "reduction_ratio": reduction_ratio
    }
