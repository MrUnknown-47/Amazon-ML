"""
Threshold Search and Hyperparameter Tuning on the TUNE Split.
Performs fine grid searches over classification thresholds, margin constraints,
and entity-decision strategies to maximize downstream official macro F_0.5.
Strictly isolated: Must only be executed on the tuning split (never on the 50K holdout).
"""

from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from .entity_decision import EntityDecisionEngine
from ..evaluation.metrics import compute_macro_f05, compute_detailed_evaluation


def search_optimal_threshold(
    tune_gt: Dict[str, Set[str]],
    scored_candidates_by_s1: Dict[str, List[Tuple[str, float, Optional[Dict[str, Any]]]]],
    threshold_grid: Optional[List[float]] = None,
    margin_grid: Optional[List[float]] = None,
    strategies: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Search over threshold, margin, and strategy combinations on the tuning set.
    Selects the parameter tuple that maximizes official Macro F_0.5.
    """
    th_grid = threshold_grid or [
        0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
        0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90
    ]
    m_grid = margin_grid or [0.10, 0.15, 0.20, 0.25, 0.30]
    strat_list = strategies or ["global_threshold", "threshold_and_margin", "score_drop_elbow", "adaptive_relative"]

    best_score = -1.0
    best_params = {}
    search_history = []

    # 1. First pass: Evaluate global threshold
    for th in th_grid:
        engine = EntityDecisionEngine(score_threshold=th, strategy="global_threshold")
        preds = engine.decide_batch(scored_candidates_by_s1)
        eval_res = compute_detailed_evaluation(tune_gt, preds)
        f05 = eval_res["macro_f05"]

        entry = {
            "strategy": "global_threshold",
            "score_threshold": th,
            "score_margin": 0.0,
            "macro_f05": float(round(f05, 5)),
            "macro_precision": float(round(eval_res["macro_precision"], 5)),
            "macro_recall": float(round(eval_res["macro_recall"], 5)),
            "singleton_accuracy": float(round(eval_res["singletons"]["singleton_accuracy"], 5))
        }
        search_history.append(entry)

        if f05 > best_score:
            best_score = f05
            best_params = entry

    # 2. Second pass: Evaluate threshold + margin strategies around top thresholds
    promising_th = sorted(search_history, key=lambda x: x["macro_f05"], reverse=True)[:5]
    focal_thresholds = [x["score_threshold"] for x in promising_th]

    for strat in ["threshold_and_margin", "score_drop_elbow", "adaptive_relative"]:
        for th in focal_thresholds:
            for margin in m_grid:
                engine = EntityDecisionEngine(score_threshold=th, score_margin=margin, strategy=strat)
                preds = engine.decide_batch(scored_candidates_by_s1)
                eval_res = compute_detailed_evaluation(tune_gt, preds)
                f05 = eval_res["macro_f05"]

                entry = {
                    "strategy": strat,
                    "score_threshold": th,
                    "score_margin": margin,
                    "macro_f05": float(round(f05, 5)),
                    "macro_precision": float(round(eval_res["macro_precision"], 5)),
                    "macro_recall": float(round(eval_res["macro_recall"], 5)),
                    "singleton_accuracy": float(round(eval_res["singletons"]["singleton_accuracy"], 5))
                }
                search_history.append(entry)

                if f05 > best_score:
                    best_score = f05
                    best_params = entry

    return {
        "best_params": best_params,
        "best_macro_f05": float(round(best_score, 5)),
        "total_configurations_searched": len(search_history),
        "search_history": sorted(search_history, key=lambda x: x["macro_f05"], reverse=True)
    }
