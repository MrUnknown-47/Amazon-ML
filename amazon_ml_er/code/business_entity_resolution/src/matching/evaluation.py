"""
Evaluation and Detailed Diagnostic Module for Final Matcher.
Wraps official Macro F_0.5 evaluation and computes fine-grained breakdowns:
- Cardinality retention (0, 1, 2, 3, 4, 5, 6, 7+)
- Source retention (Source 2 vs Source 3)
- Country breakdown
- Predicted match count distributions
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from ..evaluation.metrics import compute_entity_f05, compute_detailed_evaluation


def evaluate_matcher_predictions(
    y_true: Dict[str, Set[str]],
    y_pred: Dict[str, Set[str]],
    entity_metadata: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Any]:
    """
    Comprehensive evaluation matching the official Amazon ML Challenge guidelines.
    Returns:
    - Official macro F_0.5, macro precision, macro recall
    - Singleton accuracy
    - Predicted matches distribution (mean, median, P95, max)
    - Cardinality breakdown (0, 1, 2, 3, 4, 5, 6, 7+)
    - Source breakdown (S2 vs S3)
    - Country breakdown
    """
    base_eval = compute_detailed_evaluation(y_true, y_pred)
    meta = entity_metadata or {}

    pred_lens = [len(y_pred.get(eid, set())) for eid in y_true]
    arr_pred = np.array(pred_lens)

    # 1. Cardinality Breakdown
    card_groups = defaultdict(lambda: {"total_s1": 0, "f05_sum": 0.0, "true_matches": 0, "captured_matches": 0, "all_correct_count": 0, "at_least_one_count": 0})
    for eid, t_set in y_true.items():
        p_set = y_pred.get(eid, set())
        card = len(t_set)
        card_label = "0" if card == 0 else "1" if card == 1 else "2" if card == 2 else "3" if card == 3 else "4" if card == 4 else "5" if card == 5 else "6" if card == 6 else "7+"

        _, _, f05 = compute_entity_f05(t_set, p_set)
        cg = card_groups[card_label]
        cg["total_s1"] += 1
        cg["f05_sum"] += f05
        cg["true_matches"] += card
        inter = len(t_set & p_set)
        cg["captured_matches"] += inter
        if inter > 0 or card == 0 and len(p_set) == 0:
            cg["at_least_one_count"] += 1
        if (t_set == p_set):
            cg["all_correct_count"] += 1

    cardinality_results = {}
    for c_label, cg in sorted(card_groups.items(), key=lambda x: str(x[0])):
        n = cg["total_s1"]
        cardinality_results[c_label] = {
            "s1_entity_count": n,
            "macro_f05": float(round(cg["f05_sum"] / max(1, n), 5)),
            "true_matches": cg["true_matches"],
            "captured_matches": cg["captured_matches"],
            "pairwise_recall": float(round(cg["captured_matches"] / max(1, cg["true_matches"]), 5)) if cg["true_matches"] > 0 else 1.0,
            "all_matches_correct_ratio": float(round(cg["all_correct_count"] / max(1, n), 5)),
            "at_least_one_match_ratio": float(round(cg["at_least_one_count"] / max(1, n), 5))
        }

    # 2. Source Breakdown
    source_stats = defaultdict(lambda: {"true_matches": 0, "captured_matches": 0, "false_positives": 0})
    for eid, t_set in y_true.items():
        p_set = y_pred.get(eid, set())
        for tid in t_set:
            src = "source3" if ("source3" in tid or tid.startswith("S3")) else "source2"
            source_stats[src]["true_matches"] += 1
            if tid in p_set:
                source_stats[src]["captured_matches"] += 1
        for tid in (p_set - t_set):
            src = "source3" if ("source3" in tid or tid.startswith("S3")) else "source2"
            source_stats[src]["false_positives"] += 1

    source_results = {}
    for src, st in source_stats.items():
        tp = st["captured_matches"]
        fp = st["false_positives"]
        fn = st["true_matches"] - tp
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        denom = 0.25 * p + r
        f05 = (1.25 * p * r / denom) if denom > 0 else 0.0
        source_results[src] = {
            "true_matches": st["true_matches"],
            "captured_matches": tp,
            "false_positives": fp,
            "pairwise_precision": float(round(p, 5)),
            "pairwise_recall": float(round(r, 5)),
            "pairwise_f05": float(round(f05, 5))
        }

    # 3. Country Breakdown
    country_groups = defaultdict(lambda: {"total_s1": 0, "f05_sum": 0.0, "true_matches": 0, "captured_matches": 0})
    for eid, t_set in y_true.items():
        p_set = y_pred.get(eid, set())
        country = meta.get(eid, {}).get("country", "Unknown")
        _, _, f05 = compute_entity_f05(t_set, p_set)
        country_groups[country]["total_s1"] += 1
        country_groups[country]["f05_sum"] += f05
        country_groups[country]["true_matches"] += len(t_set)
        country_groups[country]["captured_matches"] += len(t_set & p_set)

    country_results = {}
    for ctry, cg in sorted(country_groups.items()):
        n = cg["total_s1"]
        country_results[ctry] = {
            "s1_entity_count": n,
            "macro_f05": float(round(cg["f05_sum"] / max(1, n), 5)),
            "true_matches": cg["true_matches"],
            "captured_matches": cg["captured_matches"],
            "recall": float(round(cg["captured_matches"] / max(1, cg["true_matches"]), 5)) if cg["true_matches"] > 0 else 0.0
        }

    return {
        "macro_f05": base_eval["macro_f05"],
        "macro_precision": base_eval["macro_precision"],
        "macro_recall": base_eval["macro_recall"],
        "singleton_accuracy": base_eval["singletons"]["singleton_accuracy"],
        "total_true_positives": base_eval["target_level_aggregates"]["true_positives"],
        "total_false_positives": base_eval["target_level_aggregates"]["false_positives_false_merges"],
        "total_false_negatives": base_eval["target_level_aggregates"]["false_negatives_missed_matches"],
        "predicted_matches_distribution": {
            "mean": float(round(arr_pred.mean(), 2)),
            "median": float(np.median(arr_pred)),
            "p95": float(np.percentile(arr_pred, 95)),
            "max": int(arr_pred.max())
        },
        "cardinality_breakdown": cardinality_results,
        "source_breakdown": source_results,
        "country_breakdown": country_results
    }
