"""
Candidate Compression and Ranking Package for Business Entity Resolution.
Implements leak-safe candidate ranker, hard-negative sampling, feature extraction,
ranking evaluation (Recall@K), and adaptive thresholding policies.
"""

from .features import EntityProfile, extract_pair_features, FEATURE_NAMES
from .sampler import HardNegativeSampler
from .model import CandidateRanker
from .ranker import rank_candidates_for_query
from .thresholding import apply_candidate_budget, evaluate_compression_policy
from .evaluation import compute_recall_at_k, evaluate_unbiased_categories

__all__ = [
    "EntityProfile",
    "extract_pair_features",
    "FEATURE_NAMES",
    "HardNegativeSampler",
    "CandidateRanker",
    "rank_candidates_for_query",
    "apply_candidate_budget",
    "evaluate_compression_policy",
    "compute_recall_at_k",
    "evaluate_unbiased_categories",
]
