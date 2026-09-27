"""
Leak-Safe Dataset Construction and Hard Negative Mining for Final Matcher Training.
Extracts positive examples exclusively from true matches retrieved by the authoritative blocker,
and systematically oversamples hard negative candidates from the physical candidate set.
Strictly isolated from tuning and validation holdouts.
"""

import random
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from .features import (
    MatchingEntityProfile,
    extract_matching_features,
    NUM_MATCHING_FEATURES
)


class MatcherHardNegativeSampler:
    """
    Controlled negative sampler tailored for the final matching classifier.
    Categorizes negatives into distinct hard failure modes:
    - High name similarity with discordant address
    - High address similarity with discordant name
    - Same street number / postal code collision
    - Multi-channel blocking hits that are false matches
    - Random candidate negatives for global regularization
    """

    def __init__(
        self,
        negatives_per_positive: int = 10,
        min_hard_ratio: float = 0.7,
        random_seed: int = 42
    ):
        self.negatives_per_positive = negatives_per_positive
        self.min_hard_ratio = min_hard_ratio
        self.rng = random.Random(random_seed)

    def sample_candidates_for_query(
        self,
        s1: MatchingEntityProfile,
        target_profiles: List[MatchingEntityProfile],
        provenances: List[Dict[str, Any]],
        ranker_scores: List[float],
        true_target_ids: Set[str]
    ) -> Tuple[List[MatchingEntityProfile], List[Dict[str, Any]], List[float], List[int], List[int], Dict[str, Any]]:
        """
        Sample training pairs for a single S1 query.
        Returns:
            sampled_targets, sampled_prov, sampled_scores, sampled_ranks, labels, stats
        """
        positives = []
        hard_negatives = []
        other_negatives = []

        for rank_idx, (t_prof, prov, score) in enumerate(zip(target_profiles, provenances, ranker_scores), start=1):
            tid = t_prof.eid
            item = (t_prof, prov, score, rank_idx)
            if tid in true_target_ids:
                positives.append(item)
            else:
                # Classify negative difficulty
                s1_sig = s1.sig_name_tokens
                t_sig = t_prof.sig_name_tokens
                name_jaccard = len(s1_sig & t_sig) / max(1, len(s1_sig | t_sig))

                s1_addr = s1.addr_tokens
                t_addr = t_prof.addr_tokens
                addr_jaccard = len(s1_addr & t_addr) / max(1, len(s1_addr | t_addr))

                is_hard = False
                channels = prov.get("channels", set())

                # Condition 1: High name, low address (name collision / different location)
                if name_jaccard >= 0.40 and addr_jaccard < 0.25:
                    is_hard = True
                # Condition 2: High address, low name (co-location / different business)
                elif addr_jaccard >= 0.40 and name_jaccard < 0.25:
                    is_hard = True
                # Condition 3: Street number match with distinct name
                elif s1.addr_street_num and s1.addr_street_num == t_prof.addr_street_num and name_jaccard < 0.30:
                    is_hard = True
                # Condition 4: Multi-channel hit non-match
                elif len(channels) >= 2:
                    is_hard = True
                # Condition 5: High candidate ranker score non-match
                elif score >= 0.30:
                    is_hard = True

                if is_hard:
                    hard_negatives.append(item)
                else:
                    other_negatives.append(item)

        n_pos = len(positives)
        target_total_neg = max(self.negatives_per_positive, n_pos * self.negatives_per_positive)
        target_hard_neg = int(target_total_neg * self.min_hard_ratio)
        target_other_neg = target_total_neg - target_hard_neg

        # Sample hard negatives
        if len(hard_negatives) > target_hard_neg:
            sampled_hard = self.rng.sample(hard_negatives, target_hard_neg)
        else:
            sampled_hard = list(hard_negatives)

        # Remaining negative quota filled by other negatives or additional hard negatives
        remaining_needed = target_total_neg - len(sampled_hard)
        if len(other_negatives) > remaining_needed:
            sampled_other = self.rng.sample(other_negatives, remaining_needed)
        else:
            sampled_other = list(other_negatives)

        # If still short, sample remaining available negatives
        sampled_items = positives + sampled_hard + sampled_other

        targets_out = [it[0] for it in sampled_items]
        prov_out = [it[1] for it in sampled_items]
        scores_out = [it[2] for it in sampled_items]
        ranks_out = [it[3] for it in sampled_items]
        labels_out = [1] * len(positives) + [0] * (len(sampled_hard) + len(sampled_other))

        stats = {
            "positives": len(positives),
            "available_hard_negatives": len(hard_negatives),
            "sampled_hard_negatives": len(sampled_hard),
            "available_other_negatives": len(other_negatives),
            "sampled_other_negatives": len(sampled_other)
        }

        return targets_out, prov_out, scores_out, ranks_out, labels_out, stats


def build_matching_dataset(
    s1_profiles: Dict[str, MatchingEntityProfile],
    target_profiles: Dict[str, MatchingEntityProfile],
    candidates_by_s1: Dict[str, List[Tuple[str, Dict[str, Any]]]],
    ground_truth: Dict[str, Set[str]],
    ranker_scores_by_s1: Optional[Dict[str, Dict[str, float]]] = None,
    sampler: Optional[MatcherHardNegativeSampler] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Build training/tuning feature matrix X, labels y, and query groups.
    Leak-safe: Only includes S1 entities in s1_profiles.
    """
    neg_sampler = sampler or MatcherHardNegativeSampler()
    scores_dict = ranker_scores_by_s1 or defaultdict(dict)

    all_features = []
    all_labels = []
    all_groups = []

    total_pos = 0
    total_hard_neg = 0
    total_other_neg = 0
    group_id = 0

    for s1_id, s1 in s1_profiles.items():
        cand_list = candidates_by_s1.get(s1_id, [])
        if not cand_list:
            continue

        true_tids = ground_truth.get(s1_id, set())

        # Resolve profiles and candidate ranker scores
        t_profs = []
        provs = []
        scores = []
        for tid, prov in cand_list:
            t_prof = target_profiles.get(tid)
            if t_prof:
                t_profs.append(t_prof)
                provs.append(prov)
                scores.append(scores_dict[s1_id].get(tid, 0.0))

        if not t_profs:
            continue

        s_targs, s_prov, s_scores, s_ranks, s_labels, stats = neg_sampler.sample_candidates_for_query(
            s1, t_profs, provs, scores, true_tids
        )

        if not s_targs:
            continue

        # Extract features for sampled pairs
        for i in range(len(s_targs)):
            feat = extract_matching_features(
                s1, s_targs[i], s_prov[i], s_scores[i], s_ranks[i]
            )
            all_features.append(feat)
            all_labels.append(s_labels[i])
            all_groups.append(group_id)

        total_pos += stats["positives"]
        total_hard_neg += stats["sampled_hard_negatives"]
        total_other_neg += stats["sampled_other_negatives"]
        group_id += 1

    X = np.array(all_features, dtype=np.float32) if all_features else np.zeros((0, NUM_MATCHING_FEATURES), dtype=np.float32)
    y = np.array(all_labels, dtype=np.int32) if all_labels else np.zeros((0,), dtype=np.int32)
    groups = np.array(all_groups, dtype=np.int32) if all_groups else np.zeros((0,), dtype=np.int32)

    dataset_stats = {
        "queries_processed": group_id,
        "total_pairs": len(y),
        "positive_pairs": int(np.sum(y == 1)),
        "negative_pairs": int(np.sum(y == 0)),
        "hard_negatives": total_hard_neg,
        "other_negatives": total_other_neg,
        "positive_ratio": float(np.mean(y == 1)) if len(y) > 0 else 0.0,
        "feature_count": NUM_MATCHING_FEATURES
    }

    return X, y, groups, dataset_stats
