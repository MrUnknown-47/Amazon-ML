"""
Official Evaluation Metrics for Business Entity Resolution.
Implements the macro-averaged F_0.5 metric with precise singleton handling
and diagnostic performance indicators.
"""

from typing import Dict, Set, List, Any, Optional, Union
import numpy as np


def compute_entity_f05(true_set: Set[str], pred_set: Set[str]) -> Tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_0.5 for a single Source 1 entity.
    Returns:
        (precision, recall, f05)
    Special cases:
        - True singleton (empty true_set) and empty pred_set -> (1.0, 1.0, 1.0)
        - True singleton (empty true_set) and non-empty pred_set -> (0.0, 0.0, 0.0)
        - Non-singleton (non-empty true_set) and empty pred_set -> (0.0, 0.0, 0.0)
        - Non-singleton and TP == 0 -> (0.0, 0.0, 0.0)
    """
    if len(true_set) == 0:
        if len(pred_set) == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 0.0

    if len(pred_set) == 0:
        return 0.0, 0.0, 0.0

    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if tp == 0:
        return 0.0, 0.0, 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    # Beta = 0.5: F_0.5 = (1.25 * P * R) / (0.25 * P + R)
    denominator = 0.25 * precision + recall
    if denominator == 0.0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denominator

    return precision, recall, f05


# Type alias for Tuple import
from typing import Tuple


def compute_macro_f05(
    y_true: Dict[str, Set[str]],
    y_pred: Dict[str, Set[str]]
) -> float:
    """
    Calculate the official macro-averaged F_0.5 across all Source 1 entities.
    """
    if not y_true:
        return 0.0

    total_f05 = 0.0
    for s1_id, true_set in y_true.items():
        pred_set = y_pred.get(s1_id, set())
        _, _, f05 = compute_entity_f05(true_set, pred_set)
        total_f05 += f05

    return total_f05 / len(y_true)


def compute_detailed_evaluation(
    y_true: Dict[str, Set[str]],
    y_pred: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Comprehensive evaluation returning:
    - macro F_0.5
    - macro precision
    - macro recall
    - singleton exact accuracy
    - total true positives, false positives (false merges), false negatives (missed matches)
    - per-entity F_0.5 score distribution
    """
    n_entities = len(y_true)
    if n_entities == 0:
        return {"error": "Empty true set"}

    precisions = []
    recalls = []
    f05_scores = []

    total_tp = 0
    total_fp = 0
    total_fn = 0

    total_singletons = 0
    correct_singletons = 0
    false_merge_singletons = 0

    for s1_id, true_set in y_true.items():
        pred_set = y_pred.get(s1_id, set())
        p, r, f05 = compute_entity_f05(true_set, pred_set)

        precisions.append(p)
        recalls.append(r)
        f05_scores.append(f05)

        if len(true_set) == 0:
            total_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
            else:
                false_merge_singletons += 1
                total_fp += len(pred_set)
        else:
            tp = len(true_set & pred_set)
            fp = len(pred_set - true_set)
            fn = len(true_set - pred_set)
            total_tp += tp
            total_fp += fp
            total_fn += fn

    arr_f05 = np.array(f05_scores)
    arr_p = np.array(precisions)
    arr_r = np.array(recalls)

    return {
        "macro_f05": float(arr_f05.mean()),
        "macro_precision": float(arr_p.mean()),
        "macro_recall": float(arr_r.mean()),
        "total_entities_evaluated": n_entities,
        "singletons": {
            "total_true_singletons": total_singletons,
            "correct_predicted_singletons": correct_singletons,
            "false_merge_singletons": false_merge_singletons,
            "singleton_accuracy": float(correct_singletons / total_singletons) if total_singletons else 0.0
        },
        "target_level_aggregates": {
            "true_positives": total_tp,
            "false_positives_false_merges": total_fp,
            "false_negatives_missed_matches": total_fn
        },
        "f05_distribution": {
            "min": float(arr_f05.min()),
            "p25": float(np.percentile(arr_f05, 25)),
            "median": float(np.median(arr_f05)),
            "mean": float(arr_f05.mean()),
            "p75": float(np.percentile(arr_f05, 75)),
            "max": float(arr_f05.max())
        }
    }


def compute_candidate_recall(
    y_true: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]]
) -> Dict[str, Any]:
    """
    Evaluate candidate generation / blocking quality:
    - Candidate recall (recall ceiling): proportion of true matches captured in candidates
    - Entity coverage: proportion of entities whose full true match set is captured
    - Candidate set size statistics (mean, median, p95, max)
    """
    total_true_matches = 0
    captured_matches = 0

    full_coverage_entities = 0
    partial_coverage_entities = 0
    zero_coverage_entities = 0

    cand_counts = []

    for s1_id, true_set in y_true.items():
        cand_set = candidates.get(s1_id, set())
        cand_counts.append(len(cand_set))

        if len(true_set) == 0:
            continue

        total_true_matches += len(true_set)
        captured = len(true_set & cand_set)
        captured_matches += captured

        if captured == len(true_set):
            full_coverage_entities += 1
        elif captured > 0:
            partial_coverage_entities += 1
        else:
            zero_coverage_entities += 1

    cand_arr = np.array(cand_counts) if cand_counts else np.array([0])

    return {
        "candidate_recall": float(captured_matches / total_true_matches) if total_true_matches else 1.0,
        "total_true_matches": total_true_matches,
        "captured_true_matches": captured_matches,
        "missed_matches": total_true_matches - captured_matches,
        "entity_coverage": {
            "full_coverage_count": full_coverage_entities,
            "partial_coverage_count": partial_coverage_entities,
            "zero_coverage_count": zero_coverage_entities,
        },
        "candidate_size_stats": {
            "mean": float(round(cand_arr.mean(), 2)),
            "median": float(np.median(cand_arr)),
            "p90": float(np.percentile(cand_arr, 90)),
            "p95": float(np.percentile(cand_arr, 95)),
            "p99": float(np.percentile(cand_arr, 99)),
            "max": int(cand_arr.max())
        }
    }
