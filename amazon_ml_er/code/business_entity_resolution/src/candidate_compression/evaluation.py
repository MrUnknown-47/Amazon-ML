"""
Comprehensive evaluation metrics for candidate ranking and compression.
Computes Recall@K, PR-AUC, unbiased category recall, baseline failure recovery,
and transliteration feature distributions.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score

from .features import EntityProfile, extract_pair_features, FEATURE_NAMES


def compute_recall_at_k(
    ranked_candidates_by_s1: Dict[str, List[Tuple[str, float]]],
    ground_truth: Dict[str, Set[str]],
    k_values: List[int] = [10, 15, 25, 40, 60, 80, 100, 150, 200]
) -> Dict[str, Any]:
    """
    Compute Recall@K across multiple candidate budgets K.
    """
    total_true = sum(len(tids) for tids in ground_truth.values())
    results = {}

    for k in k_values:
        captured = 0
        cand_counts = []
        for s1_id, true_tids in ground_truth.items():
            ranked = ranked_candidates_by_s1.get(s1_id, [])
            top_k_tids = {tid for tid, _ in ranked[:k]}
            captured += len(true_tids & top_k_tids)
            cand_counts.append(len(top_k_tids))

        arr = np.array(cand_counts) if cand_counts else np.array([0])
        recall = float(round(captured / max(1, total_true), 5))
        results[f"Recall@{k}"] = {
            "k": k,
            "recall": recall,
            "captured_true_matches": captured,
            "total_true_matches": total_true,
            "missed_matches": total_true - captured,
            "mean_candidates": float(round(arr.mean(), 2)),
            "median_candidates": float(np.median(arr)),
            "p95_candidates": float(np.percentile(arr, 95)),
            "p99_candidates": float(np.percentile(arr, 99)),
            "total_candidates": int(arr.sum())
        }

    return results


def compute_pr_auc_and_roc(
    y_true: List[int],
    y_scores: List[float]
) -> Dict[str, float]:
    """
    Compute Area Under Precision-Recall Curve (PR-AUC) and ROC-AUC.
    """
    y_t = np.asarray(y_true, dtype=np.int32)
    y_s = np.asarray(y_scores, dtype=np.float32)

    if len(y_t) == 0 or len(np.unique(y_t)) < 2:
        return {"pr_auc": 0.0, "roc_auc": 0.0}

    precision, recall, _ = precision_recall_curve(y_t, y_s)
    pr_auc = float(round(auc(recall, precision), 5))
    roc_auc = float(round(roc_auc_score(y_t, y_s), 5))

    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc
    }


def categorize_unbiased_pair(s1: EntityProfile, t: EntityProfile) -> str:
    """
    Categorize a candidate pair strictly using observable pair attributes
    WITHOUT using any retrieval outcome.
    """
    s1_alpha = s1.alphanumeric_name
    t_alpha = t.alphanumeric_name

    if s1.is_non_latin != t.is_non_latin:
        return "transliteration"
    elif not t.has_address:
        return "missing_target_address"
    elif s1_alpha and s1_alpha == t_alpha:
        return "exact_name_match"
    else:
        inter_n = len(s1.sig_name_tokens & t.sig_name_tokens)
        union_n = len(s1.sig_name_tokens | t.sig_name_tokens)
        n_jacc = inter_n / max(1, union_n)

        inter_a = len(s1.addr_tokens & t.addr_tokens)
        union_a = len(s1.addr_tokens | t.addr_tokens)
        a_jacc = inter_a / max(1, union_a)

        num_match = bool(s1.addr_street_num and s1.addr_street_num == t.addr_street_num)

        if n_jacc >= 0.50:
            return "high_name_similarity"
        elif a_jacc >= 0.35 or num_match:
            return "low_name_sim_same_address"
        elif s1.addr_numeric_tokens and not t.addr_numeric_tokens:
            return "missing_street_number"
        else:
            return "other_low_similarity"


def evaluate_unbiased_categories(
    s1_profiles: Dict[str, EntityProfile],
    target_profiles: Dict[str, EntityProfile],
    ground_truth_pairs: List[Tuple[str, str]],
    raw_candidates_by_s1: Dict[str, Set[str]],
    compressed_candidates_by_s1: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Evaluate candidate recall by unbiased observable categories.
    """
    category_pairs = defaultdict(list)
    for s1_id, tid in ground_truth_pairs:
        s1 = s1_profiles.get(s1_id)
        t = target_profiles.get(tid)
        if not s1 or not t:
            continue
        cat = categorize_unbiased_pair(s1, t)
        category_pairs[cat].append((s1_id, tid))

    results = {}
    for cat, pairs in category_pairs.items():
        tot = len(pairs)
        raw_captured = sum(1 for s1_id, tid in pairs if tid in raw_candidates_by_s1.get(s1_id, set()))
        comp_captured = sum(1 for s1_id, tid in pairs if tid in compressed_candidates_by_s1.get(s1_id, set()))

        results[cat] = {
            "total_true_pairs": tot,
            "percentage_of_all_pairs": float(round(tot / max(1, len(ground_truth_pairs)) * 100, 2)),
            "raw_blocking_captured": raw_captured,
            "raw_blocking_recall": float(round(raw_captured / max(1, tot), 5)),
            "compressed_captured": comp_captured,
            "compressed_recall": float(round(comp_captured / max(1, tot), 5))
        }

    return results


def evaluate_transliteration_features(
    s1_profiles: Dict[str, EntityProfile],
    target_profiles: Dict[str, EntityProfile],
    transliteration_pairs: List[Tuple[str, str]],
    raw_candidates_by_s1: Dict[str, Set[str]],
    compressed_candidates_by_s1: Dict[str, Set[str]],
    hard_negatives: Optional[List[Tuple[str, str]]] = None
) -> Dict[str, Any]:
    """
    Compare feature distributions for transliteration true positives,
    false negatives (unretrieved/uncompressed), and hard negatives.
    """
    tp_features = []
    fn_features = []
    hn_features = []

    for s1_id, tid in transliteration_pairs:
        s1 = s1_profiles.get(s1_id)
        t = target_profiles.get(tid)
        if not s1 or not t:
            continue
        feat = extract_pair_features(s1, t)
        is_comp = tid in compressed_candidates_by_s1.get(s1_id, set())
        if is_comp:
            tp_features.append(feat)
        else:
            fn_features.append(feat)

    if hard_negatives:
        for s1_id, tid in hard_negatives:
            s1 = s1_profiles.get(s1_id)
            t = target_profiles.get(tid)
            if s1 and t:
                hn_features.append(extract_pair_features(s1, t))

    def summarize_matrix(mat: List[np.ndarray]) -> Dict[str, float]:
        if not mat:
            return {name: 0.0 for name in FEATURE_NAMES}
        arr = np.array(mat)
        means = arr.mean(axis=0)
        return {name: float(round(means[i], 4)) for i, name in enumerate(FEATURE_NAMES)}

    return {
        "transliteration_summary": {
            "total_transliteration_pairs": len(transliteration_pairs),
            "retained_true_positives": len(tp_features),
            "missed_false_negatives": len(fn_features),
            "retention_rate": float(round(len(tp_features) / max(1, len(transliteration_pairs)), 5))
        },
        "feature_means_true_positives": summarize_matrix(tp_features),
        "feature_means_false_negatives": summarize_matrix(fn_features),
        "feature_means_hard_negatives": summarize_matrix(hn_features)
    }
