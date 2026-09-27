"""
Hard negative sampling and class imbalance management for candidate ranker training.
Draws positive examples exclusively from true matches retrieved by raw blocking,
and oversamples hard negative candidates (high name overlap with different address,
same address with different name, multi-channel hits).
"""

import random
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
import numpy as np

from .features import EntityProfile


class HardNegativeSampler:
    """
    Controlled negative sampler that prioritizes difficult negative candidates
    from the raw candidate pool to train discriminative candidate rankers.
    """

    def __init__(
        self,
        negatives_per_positive: int = 8,
        min_hard_ratio: float = 0.6,
        random_seed: int = 42
    ):
        self.negatives_per_positive = negatives_per_positive
        self.min_hard_ratio = min_hard_ratio
        self.rng = random.Random(random_seed)

    def sample_candidates_for_query(
        self,
        s1_profile: EntityProfile,
        candidate_target_profiles: List[EntityProfile],
        candidate_provenance: List[Dict[str, Any]],
        true_target_ids: Set[str]
    ) -> Tuple[List[EntityProfile], List[Dict[str, Any]], List[int], Dict[str, Any]]:
        """
        Sample training pairs for a single S1 query.
        Returns:
            sampled_targets, sampled_provenance, labels (1 for match, 0 for negative), stats
        """
        positives = []
        pos_prov = []

        hard_negatives = []
        hard_neg_prov = []

        other_negatives = []
        other_neg_prov = []

        s1_sig = s1_profile.sig_name_tokens
        s1_atoks = s1_profile.addr_tokens
        s1_num = s1_profile.addr_street_num

        for t_prof, prov in zip(candidate_target_profiles, candidate_provenance):
            tid = t_prof.eid
            if tid in true_target_ids:
                # True positive conditional on raw blocking retrieval
                positives.append(t_prof)
                pos_prov.append(prov)
            else:
                # Determine hardness
                t_sig = t_prof.sig_name_tokens
                t_atoks = t_prof.addr_tokens
                t_num = t_prof.addr_street_num

                name_jacc = len(s1_sig & t_sig) / max(1, len(s1_sig | t_sig))
                addr_jacc = len(s1_atoks & t_atoks) / max(1, len(s1_atoks | t_atoks))
                num_match = bool(s1_num and s1_num == t_num)
                ch_count = len(prov.get("channels", []))

                is_hard = (
                    (name_jacc >= 0.40 and addr_jacc < 0.20) or  # Same/similar name, different address
                    (addr_jacc >= 0.35 and name_jacc < 0.20) or  # Same address, different name (DBA/neighbor)
                    (num_match and name_jacc >= 0.25) or          # Same building number + partial name
                    (ch_count >= 2)                              # Retrieved by multiple blocking channels
                )

                if is_hard:
                    hard_negatives.append(t_prof)
                    hard_neg_prov.append(prov)
                else:
                    other_negatives.append(t_prof)
                    other_neg_prov.append(prov)

        # If no positives retrieved, nothing to learn conditional on retrieval for this query
        if not positives:
            return [], [], [], {
                "total_positives": 0,
                "total_hard_negatives": len(hard_negatives),
                "total_other_negatives": len(other_negatives),
                "sampled_hard": 0,
                "sampled_other": 0
            }

        target_neg_count = len(positives) * self.negatives_per_positive
        target_hard_count = int(round(target_neg_count * self.min_hard_ratio))

        # Sample hard negatives
        sampled_hard = []
        sampled_hard_prov = []
        if len(hard_negatives) <= target_hard_count:
            sampled_hard = list(hard_negatives)
            sampled_hard_prov = list(hard_neg_prov)
        else:
            indices = self.rng.sample(range(len(hard_negatives)), target_hard_count)
            for idx in indices:
                sampled_hard.append(hard_negatives[idx])
                sampled_hard_prov.append(hard_neg_prov[idx])

        # Fill remaining negative budget with other negatives (or more hard if available)
        remaining_budget = target_neg_count - len(sampled_hard)
        sampled_other = []
        sampled_other_prov = []

        pool = other_negatives
        pool_prov = other_neg_prov
        if len(pool) < remaining_budget and len(hard_negatives) > len(sampled_hard):
            # Borrow unused hard negatives if other negatives are scarce
            unused_hard_idx = set(range(len(hard_negatives))) - set(indices if len(hard_negatives) > target_hard_count else range(len(hard_negatives)))
            pool = list(pool) + [hard_negatives[i] for i in unused_hard_idx]
            pool_prov = list(pool_prov) + [hard_neg_prov[i] for i in unused_hard_idx]

        if len(pool) <= remaining_budget:
            sampled_other = list(pool)
            sampled_other_prov = list(pool_prov)
        else:
            idx_list = self.rng.sample(range(len(pool)), remaining_budget)
            for idx in idx_list:
                sampled_other.append(pool[idx])
                sampled_other_prov.append(pool_prov[idx])

        # Assemble final query sample
        final_targets = positives + sampled_hard + sampled_other
        final_prov = pos_prov + sampled_hard_prov + sampled_other_prov
        final_labels = [1] * len(positives) + [0] * (len(sampled_hard) + len(sampled_other))

        stats = {
            "total_positives": len(positives),
            "total_hard_negatives": len(hard_negatives),
            "total_other_negatives": len(other_negatives),
            "sampled_hard": len(sampled_hard),
            "sampled_other": len(sampled_other)
        }

        return final_targets, final_prov, final_labels, stats
