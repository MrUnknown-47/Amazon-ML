"""
Lightweight Candidate Ranking and Pruning Engine.
Evaluates candidate budget constraints (K = 10, 15, 25, 40, 60, 100)
and computes post-pruning recall R_pruned without using ground-truth labels.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from ..preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    extract_numeric_tokens
)
from ..evaluation.metrics import compute_candidate_recall


def compute_fast_pair_heuristic(
    s1_features: Dict[str, Any],
    target_features: Dict[str, Any],
    channel_hits: int
) -> float:
    """
    Compute an ultra-fast heuristic score for candidate ranking:
    - Exact normalized name match (+3.0)
    - Name token Jaccard similarity (+2.0 * jaccard)
    - Street number / numeric token agreement (+2.0)
    - Address token Jaccard similarity (+1.5 * jaccard)
    - Multi-channel hit support (+1.0 * min(hits, 4))
    """
    score = 0.0

    # 1. Exact normalized name match
    if s1_features["norm_name"] and s1_features["norm_name"] == target_features["norm_name"]:
        score += 3.0

    # 2. Name token Jaccard
    s1_nt = s1_features["name_tokens"]
    tgt_nt = target_features["name_tokens"]
    if s1_nt and tgt_nt:
        inter = len(s1_nt & tgt_nt)
        union = len(s1_nt | tgt_nt)
        if union > 0:
            score += 2.0 * (inter / union)

    # 3. Numeric address token agreement
    s1_nums = s1_features["addr_nums"]
    tgt_nums = target_features["addr_nums"]
    if s1_nums and tgt_nums:
        if s1_nums & tgt_nums:
            score += 2.0

    # 4. Address token Jaccard
    s1_at = s1_features["addr_tokens"]
    tgt_at = target_features["addr_tokens"]
    if s1_at and tgt_at:
        inter = len(s1_at & tgt_at)
        union = len(s1_at | tgt_at)
        if union > 0:
            score += 1.5 * (inter / union)

    # 5. Channel consensus
    score += 1.0 * min(channel_hits, 4)

    return score


def extract_compact_record_features(record: Dict[str, str]) -> Dict[str, Any]:
    """Precompute lightweight feature sets for rapid candidate ranking."""
    bname = record.get("business_name", "")
    baddr = record.get("business_address", "")

    norm_name = normalize_business_name(bname)
    norm_addr = normalize_business_address(baddr)

    return {
        "norm_name": norm_name.alphanumeric,
        "name_tokens": set(norm_name.significant_tokens) if norm_name.significant_tokens else set(norm_name.tokens),
        "addr_nums": set(extract_numeric_tokens(baddr)),
        "addr_tokens": set(norm_addr.tokens)
    }


def prune_candidate_set(
    union_candidates: Dict[str, Set[str]],
    query_records: Dict[str, Dict[str, str]],
    target_records: Dict[str, Dict[str, str]],
    channel_hit_counts: Optional[Dict[Tuple[str, str], int]] = None,
    K: int = 25
) -> Dict[str, Set[str]]:
    """
    Rank each S1's candidates using the lightweight heuristic and prune to top-K.
    """
    pruned: Dict[str, Set[str]] = {}

    # Precompute query features
    query_feats = {
        s1_id: extract_compact_record_features(rec)
        for s1_id, rec in query_records.items()
    }

    # Cache target features on the fly
    target_feat_cache: Dict[str, Dict[str, Any]] = {}

    for s1_id, cand_set in union_candidates.items():
        if len(cand_set) <= K:
            pruned[s1_id] = set(cand_set)
            continue

        q_feat = query_feats[s1_id]
        scored_cands = []

        for tid in cand_set:
            if tid not in target_feat_cache:
                t_rec = target_records.get(tid, {})
                target_feat_cache[tid] = extract_compact_record_features(t_rec)

            t_feat = target_feat_cache[tid]
            hits = channel_hit_counts.get((s1_id, tid), 1) if channel_hit_counts else 1
            s = compute_fast_pair_heuristic(q_feat, t_feat, hits)
            scored_cands.append((tid, s))

        # Sort descending by score and select top-K
        scored_cands.sort(key=lambda x: x[1], reverse=True)
        top_k_ids = {tid for tid, _ in scored_cands[:K]}
        pruned[s1_id] = top_k_ids

    return pruned


def evaluate_candidate_budgets(
    union_candidates: Dict[str, Set[str]],
    query_records: Dict[str, Dict[str, str]],
    target_records: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, Set[str]],
    channel_hit_counts: Optional[Dict[Tuple[str, str], int]] = None,
    budgets: List[int] = [10, 15, 25, 40, 60, 100],
    total_target_corpus_size: int = 0
) -> Dict[str, Any]:
    """
    Benchmark multiple candidate budgets K in {10, 15, 25, 40, 60, 100}.
    Computes R_pruned, candidate distributions, and counts of truncated true matches.
    """
    # 1. Baseline R_block stats
    block_stats = compute_candidate_recall(ground_truth, union_candidates)
    r_block = block_stats["candidate_recall"]
    total_true_matches = block_stats["total_true_matches"]

    results_by_k = {}

    # Precompute query features
    query_feats = {
        s1_id: extract_compact_record_features(rec)
        for s1_id, rec in query_records.items()
    }
    target_feat_cache: Dict[str, Dict[str, Any]] = {}

    # Pre-rank all candidates once per S1
    ranked_candidates: Dict[str, List[str]] = {}
    for s1_id, cand_set in union_candidates.items():
        if len(cand_set) <= min(budgets):
            ranked_candidates[s1_id] = list(cand_set)
            continue

        q_feat = query_feats[s1_id]
        scored_cands = []
        for tid in cand_set:
            if tid not in target_feat_cache:
                t_rec = target_records.get(tid, {})
                target_feat_cache[tid] = extract_compact_record_features(t_rec)

            t_feat = target_feat_cache[tid]
            hits = channel_hit_counts.get((s1_id, tid), 1) if channel_hit_counts else 1
            s = compute_fast_pair_heuristic(q_feat, t_feat, hits)
            scored_cands.append((tid, s))

        scored_cands.sort(key=lambda x: x[1], reverse=True)
        ranked_candidates[s1_id] = [tid for tid, _ in scored_cands]

    # Evaluate each budget K
    total_possible_pairs = len(query_records) * total_target_corpus_size

    for K in budgets:
        pruned_k: Dict[str, Set[str]] = {}
        cand_counts = []
        truncated_entities_count = 0
        true_matches_lost = 0

        for s1_id, c_list in ranked_candidates.items():
            top_k = set(c_list[:K])
            pruned_k[s1_id] = top_k
            cand_counts.append(len(top_k))

            # Check if any true match was in unpruned union but lost in top-K
            true_set = ground_truth.get(s1_id, set())
            if true_set:
                union_captured = true_set & union_candidates.get(s1_id, set())
                pruned_captured = true_set & top_k
                lost = len(union_captured - pruned_captured)
                if lost > 0:
                    truncated_entities_count += 1
                    true_matches_lost += lost

        recall_stats = compute_candidate_recall(ground_truth, pruned_k)
        arr = np.array(cand_counts) if cand_counts else np.array([0])
        total_cands = int(arr.sum())

        red_ratio = 1.0 - (total_cands / total_possible_pairs) if total_possible_pairs > 0 else 1.0

        results_by_k[str(K)] = {
            "K": K,
            "R_pruned": float(round(recall_stats["candidate_recall"], 5)),
            "captured_true_matches": recall_stats["captured_true_matches"],
            "missed_true_matches": recall_stats["missed_matches"],
            "true_matches_lost_to_pruning": true_matches_lost,
            "entities_with_truncated_matches": truncated_entities_count,
            "candidate_count_total": total_cands,
            "mean_candidates_per_S1": float(round(arr.mean(), 2)),
            "median_candidates_per_S1": float(np.median(arr)),
            "P95_candidates_per_S1": float(np.percentile(arr, 95)),
            "P99_candidates_per_S1": float(np.percentile(arr, 99)),
            "max_candidates_per_S1": int(arr.max()),
            "reduction_ratio": float(red_ratio)
        }

    return {
        "R_block": float(round(r_block, 5)),
        "total_true_matches": total_true_matches,
        "captured_true_matches_block": block_stats["captured_true_matches"],
        "budgets_evaluated": results_by_k
    }
