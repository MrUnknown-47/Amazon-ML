"""
Leak-safe dataset builder for candidate ranker training.
Extracts training pairs exclusively from the raw blocking union of training S1 entities,
applies hard-negative sampling, and preserves strict separation from validation entities.
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from .features import EntityProfile, extract_batch_features, NUM_FEATURES
from .sampler import HardNegativeSampler


def build_candidate_ranker_dataset(
    s1_profiles: Dict[str, EntityProfile],
    target_profiles: Dict[str, EntityProfile],
    candidates_by_s1: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    ground_truth: Dict[str, Set[str]],
    sampler: Optional[HardNegativeSampler] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build training feature matrix X, binary labels y, and group array for learning-to-rank / classification.
    Only includes candidates from queries in s1_profiles.
    """
    neg_sampler = sampler or HardNegativeSampler()

    all_features = []
    all_labels = []
    all_groups = []

    total_positives = 0
    total_hard_negatives = 0
    total_other_negatives = 0
    sampled_hard_total = 0
    sampled_other_total = 0

    group_id = 0
    for s1_id, s1 in s1_profiles.items():
        cand_list = candidates_by_s1.get(s1_id, [])
        if not cand_list:
            continue

        true_tids = ground_truth.get(s1_id, set())

        # Resolve target profiles and provenance
        t_profs = []
        prov_list = []
        for tid, prov in cand_list:
            t_prof = target_profiles.get(tid)
            if t_prof:
                t_profs.append(t_prof)
                prov_list.append(prov)

        if not t_profs:
            continue

        # Sample candidates for this query
        s_targets, s_prov, s_labels, stats = neg_sampler.sample_candidates_for_query(
            s1, t_profs, prov_list, true_tids
        )

        if not s_targets:
            continue

        # Extract features
        feat_matrix = extract_batch_features(s1, s_targets, s_prov)

        all_features.append(feat_matrix)
        all_labels.extend(s_labels)
        all_groups.extend([group_id] * len(s_labels))
        group_id += 1

        total_positives += stats["total_positives"]
        total_hard_negatives += stats["total_hard_negatives"]
        total_other_negatives += stats["total_other_negatives"]
        sampled_hard_total += stats["sampled_hard"]
        sampled_other_total += stats["sampled_other"]

    if all_features:
        X = np.vstack(all_features)
        y = np.array(all_labels, dtype=np.int32)
        groups = np.array(all_groups, dtype=np.int32)
    else:
        X = np.empty((0, NUM_FEATURES), dtype=np.float32)
        y = np.empty(0, dtype=np.int32)
        groups = np.empty(0, dtype=np.int32)

    total_sampled_neg = sampled_hard_total + sampled_other_total
    pos_rate = float(round(total_positives / max(1, total_positives + total_sampled_neg), 5))

    sampling_summary = {
        "total_queries_with_positives": group_id,
        "total_positive_candidates": total_positives,
        "total_hard_negatives_available": total_hard_negatives,
        "total_other_negatives_available": total_other_negatives,
        "sampled_hard_negatives": sampled_hard_total,
        "sampled_other_negatives": sampled_other_total,
        "total_sampled_negatives": total_sampled_neg,
        "total_training_examples": len(y),
        "positive_rate": pos_rate,
        "hard_negative_percentage": float(round(sampled_hard_total / max(1, total_sampled_neg) * 100, 2))
    }

    return X, y, groups, sampling_summary
