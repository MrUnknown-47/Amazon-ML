"""
Entity-Level Decision Engine for Business Entity Resolution.
Transforms pairwise scores into discrete entity-level match sets for each S1 entity.
Implements:
- Global score thresholding
- Score margin / elbow pruning
- Data-driven multi-match selection
- Singleton detection and protection
- Deterministic high-confidence fast paths
"""

from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from collections import defaultdict
import numpy as np


class EntityDecisionEngine:
    """
    Decides final predicted target matches for Source 1 entities from scored candidate pools.
    Supports 0, 1, or N matches per S1 entity without hard-coded match bounds.
    """

    def __init__(
        self,
        score_threshold: float = 0.50,
        score_margin: float = 0.25,
        max_matches_cap: Optional[int] = None,
        strategy: str = "threshold_and_margin",
        singleton_min_score: float = 0.50,
        enable_fast_path: bool = False
    ):
        """
        strategy options:
        - 'global_threshold': all candidates with score >= score_threshold
        - 'threshold_and_margin': score >= score_threshold AND within score_margin of top score
        - 'score_drop_elbow': score >= score_threshold, stops when gap between consecutive >= score_margin
        - 'top_1_or_none': single best match if score >= score_threshold, else empty
        - 'adaptive_relative': score >= max(score_threshold, top_score * (1.0 - score_margin))
        """
        self.score_threshold = score_threshold
        self.score_margin = score_margin
        self.max_matches_cap = max_matches_cap
        self.strategy = strategy
        self.singleton_min_score = singleton_min_score
        self.enable_fast_path = enable_fast_path

    def decide_matches_for_entity(
        self,
        scored_candidates: List[Tuple[str, float, Optional[Dict[str, Any]]]]
    ) -> Set[str]:
        """
        Decide matching target IDs for a single S1 entity.
        scored_candidates: List of (target_id, score, optional_metadata)
        Returns: Set of predicted matching target IDs (can be empty).
        """
        if not scored_candidates:
            return set()

        # Sort descending by score
        ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
        top_tid, top_score = ranked[0][0], ranked[0][1]

        # Singleton check: If top candidate does not meet threshold, predict empty set
        if top_score < self.singleton_min_score or top_score < self.score_threshold:
            return set()

        selected = set()

        if self.strategy == "global_threshold":
            for tid, sc, _ in ranked:
                if sc >= self.score_threshold:
                    selected.add(tid)
                else:
                    break

        elif self.strategy == "top_1_or_none":
            if top_score >= self.score_threshold:
                selected.add(top_tid)

        elif self.strategy == "threshold_and_margin":
            # Candidate must be >= threshold AND within margin of the best candidate
            min_allowed_score = max(self.score_threshold, top_score - self.score_margin)
            for tid, sc, _ in ranked:
                if sc >= min_allowed_score:
                    selected.add(tid)
                else:
                    break

        elif self.strategy == "score_drop_elbow":
            # Sequentially accept candidates as long as consecutive drop < score_margin
            prev_score = top_score
            for tid, sc, _ in ranked:
                if sc < self.score_threshold:
                    break
                if (prev_score - sc) > self.score_margin and len(selected) > 0:
                    break
                selected.add(tid)
                prev_score = sc

        elif self.strategy == "adaptive_relative":
            # Relative threshold: candidate must achieve fraction of top score
            min_allowed = max(self.score_threshold, top_score * (1.0 - self.score_margin))
            for tid, sc, _ in ranked:
                if sc >= min_allowed:
                    selected.add(tid)
                else:
                    break
        else:
            # Default fallback: global threshold
            for tid, sc, _ in ranked:
                if sc >= self.score_threshold:
                    selected.add(tid)
                else:
                    break

        # Optional cap
        if self.max_matches_cap is not None and len(selected) > self.max_matches_cap:
            sorted_selected = sorted(selected, key=lambda tid: next(x[1] for x in ranked if x[0] == tid), reverse=True)
            selected = set(sorted_selected[:self.max_matches_cap])

        return selected

    def decide_batch(
        self,
        scored_candidates_by_s1: Dict[str, List[Tuple[str, float, Optional[Dict[str, Any]]]]]
    ) -> Dict[str, Set[str]]:
        """
        Batch prediction returning {s1_id: Set[predicted_target_ids]}.
        """
        predictions = {}
        for s1_id, cands in scored_candidates_by_s1.items():
            predictions[s1_id] = self.decide_matches_for_entity(cands)
        return predictions


def compute_singleton_diagnostics(
    y_true: Dict[str, Set[str]],
    scored_candidates_by_s1: Dict[str, List[Tuple[str, float, Optional[Dict[str, Any]]]]],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """
    Computes fine-grained singleton behavior metrics:
    - true singletons correctly predicted empty
    - false positive merges on singletons
    - score distribution of top negative candidates for singletons vs non-singletons
    """
    true_singletons = {eid for eid, t_set in y_true.items() if len(t_set) == 0}
    non_singletons = {eid for eid, t_set in y_true.items() if len(t_set) > 0}

    singleton_max_scores = []
    non_singleton_max_scores = []

    correct_singletons = 0
    fp_singletons = 0

    for eid in true_singletons:
        cands = scored_candidates_by_s1.get(eid, [])
        top_sc = max((c[1] for c in cands), default=0.0)
        singleton_max_scores.append(top_sc)
        if top_sc < threshold:
            correct_singletons += 1
        else:
            fp_singletons += 1

    for eid in non_singletons:
        cands = scored_candidates_by_s1.get(eid, [])
        top_sc = max((c[1] for c in cands), default=0.0)
        non_singleton_max_scores.append(top_sc)

    return {
        "total_true_singletons": len(true_singletons),
        "correctly_predicted_empty": correct_singletons,
        "singleton_accuracy": float(round(correct_singletons / max(1, len(true_singletons)), 5)),
        "false_positive_singletons": fp_singletons,
        "singleton_mean_max_score": float(round(np.mean(singleton_max_scores), 4)) if singleton_max_scores else 0.0,
        "non_singleton_mean_max_score": float(round(np.mean(non_singleton_max_scores), 4)) if non_singleton_max_scores else 0.0,
    }
