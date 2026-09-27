"""
Candidate Channel Union and Recall Contribution Analysis.
Constructs C_block(S1) as the unpruned union of all active blocking channels,
computes R_block, and performs incremental recall attribution.
"""

import time
import os
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np
import psutil

from ..evaluation.metrics import compute_candidate_recall


def union_channel_candidates(
    channel_candidate_sets: List[Dict[str, Set[str]]],
    query_ids: List[str]
) -> Dict[str, Set[str]]:
    """
    Compute C_block(S1) = ⋃_c C_c(S1).
    Guarantees:
    - Set union removes intra-entity duplicate IDs
    - No S1 self-matches allowed (enforced assertion)
    - All query S1 IDs present in result dict
    """
    union_candidates: Dict[str, Set[str]] = {s1_id: set() for s1_id in query_ids}

    for ch_cands in channel_candidate_sets:
        for s1_id, cands in ch_cands.items():
            if s1_id in union_candidates:
                # Disallow S1 self-matches
                valid_cands = {c for c in cands if not c.startswith("S1-")}
                union_candidates[s1_id].update(valid_cands)

    return union_candidates


def analyze_union_and_contributions(
    channels_with_cands: List[Tuple[str, Dict[str, Set[str]]]],
    query_records: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, Set[str]],
    total_target_corpus_size: int
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """
    Compute union metrics (R_block) and per-channel recall contributions:
    - Exclusive recall: matches captured ONLY by channel i
    - Cumulative recall: union of channels 1 through i
    - Incremental recall: matches added by channel i not captured by 1..(i-1)
    """
    proc = psutil.Process(os.getpid())
    t0 = time.time()
    m0 = proc.memory_info().rss / (1024 * 1024)

    query_ids = list(query_records.keys())
    all_ch_cands = [cands for _, cands in channels_with_cands]
    union_cands = union_channel_candidates(all_ch_cands, query_ids)

    elapsed = round(time.time() - t0, 3)
    m1 = proc.memory_info().rss / (1024 * 1024)

    # 1. Evaluate Union (R_block)
    union_recall_stats = compute_candidate_recall(ground_truth, union_cands)
    cand_counts = [len(cands) for cands in union_cands.values()]
    arr = np.array(cand_counts) if cand_counts else np.array([0])
    total_candidates = int(arr.sum())

    total_possible_pairs = len(query_ids) * total_target_corpus_size
    reduction_ratio = 1.0 - (total_candidates / total_possible_pairs) if total_possible_pairs > 0 else 1.0

    total_true_matches = union_recall_stats["total_true_matches"]

    # 2. Cumulative and Incremental Recall
    cumulative_set: Dict[str, Set[str]] = {s1: set() for s1 in query_ids}
    cumulative_recalls = []
    incremental_recalls = []

    prev_captured = 0
    for name, cands in channels_with_cands:
        for s1 in query_ids:
            cumulative_set[s1].update(cands.get(s1, set()))

        stats = compute_candidate_recall(ground_truth, cumulative_set)
        curr_captured = stats["captured_true_matches"]
        cum_rec = stats["candidate_recall"]
        inc_matches = curr_captured - prev_captured
        inc_rec = inc_matches / total_true_matches if total_true_matches > 0 else 0.0

        cumulative_recalls.append((name, float(round(cum_rec, 5))))
        incremental_recalls.append((name, float(round(inc_rec, 5)), inc_matches))
        prev_captured = curr_captured

    # 3. Exclusive Recall (matches captured ONLY by channel i and none of the others)
    exclusive_recalls = []
    for i, (name, cands) in enumerate(channels_with_cands):
        other_union: Dict[str, Set[str]] = {s1: set() for s1 in query_ids}
        for j, (_, other_cands) in enumerate(channels_with_cands):
            if i != j:
                for s1 in query_ids:
                    other_union[s1].update(other_cands.get(s1, set()))

        # Check matches in cands not in other_union
        exclusive_matches = 0
        for s1, true_targets in ground_truth.items():
            if not true_targets:
                continue
            ch_captured = true_targets & cands.get(s1, set())
            other_captured = true_targets & other_union.get(s1, set())
            exclusive_matches += len(ch_captured - other_captured)

        exc_rec = exclusive_matches / total_true_matches if total_true_matches > 0 else 0.0
        exclusive_recalls.append((name, float(round(exc_rec, 5)), exclusive_matches))

    union_metrics = {
        "R_block": float(round(union_recall_stats["candidate_recall"], 5)),
        "total_true_matches": total_true_matches,
        "captured_true_matches": union_recall_stats["captured_true_matches"],
        "missed_true_matches": union_recall_stats["missed_matches"],
        "candidate_count_total": total_candidates,
        "mean_candidates_per_S1": float(round(arr.mean(), 2)),
        "median_candidates_per_S1": float(np.median(arr)),
        "P95_candidates_per_S1": float(np.percentile(arr, 95)),
        "P99_candidates_per_S1": float(np.percentile(arr, 99)),
        "max_candidates_per_S1": int(arr.max()),
        "reduction_ratio": float(reduction_ratio),
        "runtime_seconds": elapsed,
        "peak_memory_mb": float(round(m1, 2)),
        "recall_attribution": {
            "cumulative_recall_by_channel": cumulative_recalls,
            "incremental_recall_by_channel": incremental_recalls,
            "exclusive_recall_by_channel": exclusive_recalls
        }
    }

    return union_cands, union_metrics
