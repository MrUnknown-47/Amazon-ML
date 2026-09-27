"""
Candidate ranking engine.
Scores raw candidate pools for S1 entities and produces sorted, ranked candidate lists
with deterministic tie-breaking and candidate budget support.
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from .features import EntityProfile, extract_batch_features
from .model import CandidateRanker


def rank_candidates_for_query(
    s1_profile: EntityProfile,
    target_profiles: List[EntityProfile],
    provenance_list: List[Dict[str, Any]],
    ranker: CandidateRanker
) -> List[Tuple[str, float]]:
    """
    Score and rank a list of target candidates for a single S1 query.
    Returns:
        List of (target_entity_id, ranking_score) sorted descending by score.
        Ties are broken deterministically by target_entity_id ascending.
    """
    if not target_profiles:
        return []

    # Extract batch features
    feat_matrix = extract_batch_features(s1_profile, target_profiles, provenance_list)

    # Score candidates
    scores = ranker.score(feat_matrix)

    # Build pairs (target_id, score)
    ranked_pairs = []
    for i, t_prof in enumerate(target_profiles):
        ranked_pairs.append((t_prof.eid, float(scores[i])))

    # Deterministic sort: score descending, target_id ascending
    ranked_pairs.sort(key=lambda item: (-item[1], item[0]))
    return ranked_pairs


def rank_candidates_batch(
    s1_profiles: List[EntityProfile],
    candidates_by_s1: Dict[str, List[Tuple[EntityProfile, Dict[str, Any]]]],
    ranker: CandidateRanker
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Batch-rank candidates for multiple S1 queries in a streaming-compatible manner.
    """
    results = {}
    for s1 in s1_profiles:
        cands = candidates_by_s1.get(s1.eid, [])
        if not cands:
            results[s1.eid] = []
            continue

        target_profs = [c[0] for c in cands]
        prov_list = [c[1] for c in cands]
        ranked = rank_candidates_for_query(s1, target_profs, prov_list, ranker)
        results[s1.eid] = ranked

    return results
