"""
Phase 2C Automated Test Suite: Candidate Compression and Ranking.
Tests:
1. candidate-ranker training example construction
2. no validation-label leakage
3. positive candidates only if present in raw blocking union
4. negative sampling correctness
5. hard-negative inclusion
6. deterministic feature computation
7. Recall@K computation
8. ranking determinism
9. candidate budget enforcement
10. candidate subset invariant
11. no Cartesian product materialization
12. batch/chunk equivalence to full small-batch scoring
"""

import pytest
import numpy as np
from typing import Dict, Set, List, Tuple

from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.features import (
    EntityProfile,
    extract_pair_features,
    extract_batch_features,
    FEATURE_NAMES,
    NUM_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.sampler import (
    HardNegativeSampler
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.model import (
    CandidateRanker
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.ranker import (
    rank_candidates_for_query,
    rank_candidates_batch
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.thresholding import (
    apply_candidate_budget,
    evaluate_compression_policy
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.evaluation import (
    compute_recall_at_k,
    categorize_unbiased_pair,
    evaluate_unbiased_categories
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.dataset_builder import (
    build_candidate_ranker_dataset
)


# Fixture profiles
@pytest.fixture
def sample_profiles():
    s1 = EntityProfile("S1-100", "Acme Logistics Inc", "123 Main St Ste 400 Springfield IL 62701", "US")
    # True match: slight typo
    t_true = EntityProfile("S2-001", "Acme Logistics Incorporated", "123 Main Street Suite 400 Springfield IL", "US")
    # Hard negative 1: same name, different address
    t_hn1 = EntityProfile("S2-002", "Acme Logistics", "999 Oak Avenue Chicago IL 60601", "US")
    # Hard negative 2: same address, different name (DBA/neighbor)
    t_hn2 = EntityProfile("S3-003", "Springfield Accounting LLC", "123 Main St Ste 400 Springfield IL 62701", "US")
    # Easy/random negative: completely different
    t_rand = EntityProfile("S3-004", "Zeta Technologies", "500 Pine Road Austin TX 78701", "US")
    return s1, t_true, t_hn1, t_hn2, t_rand


# 1. Test candidate-ranker training example construction
def test_training_example_construction(sample_profiles):
    s1, t_true, t_hn1, t_hn2, t_rand = sample_profiles
    targets = [t_true, t_hn1, t_hn2, t_rand]
    provs = [{"channels": ["ch1", "ch6"]}, {"channels": ["ch1"]}, {"channels": ["ch3", "ch4"]}, {"channels": ["ch5"]}]
    true_ids = {"S2-001"}

    sampler = HardNegativeSampler(negatives_per_positive=2, random_seed=42)
    s_targs, s_provs, s_labels, stats = sampler.sample_candidates_for_query(s1, targets, provs, true_ids)

    assert len(s_targs) == len(s_labels)
    assert 1 in s_labels
    assert 0 in s_labels
    assert stats["total_positives"] == 1
    assert stats["total_hard_negatives"] >= 1


# 2. Test no validation-label leakage
def test_no_validation_label_leakage(sample_profiles):
    s1, t_true, t_hn1, t_hn2, t_rand = sample_profiles
    val_s1_ids = {"S1-VAL-001", "S1-VAL-002"}

    # Training profiles contain only training S1
    train_profiles = {s1.eid: s1}
    # Ensure val_s1_ids are disjoint
    assert set(train_profiles.keys()).isdisjoint(val_s1_ids)

    cands_by_s1 = {s1.eid: [(t_true.eid, {}), (t_hn1.eid, {})]}
    gt = {s1.eid: {t_true.eid}, "S1-VAL-001": {"S2-999"}}
    targets = {t_true.eid: t_true, t_hn1.eid: t_hn1}

    X, y, groups, stats = build_candidate_ranker_dataset(
        train_profiles, targets, cands_by_s1, gt
    )

    # Verify no validation entities contributed to training rows
    assert len(X) > 0
    assert stats["total_queries_with_positives"] == 1


# 3. Test positive candidates only if present in raw blocking union
def test_positives_only_conditional_on_retrieval(sample_profiles):
    s1, t_true, t_hn1, t_hn2, t_rand = sample_profiles
    # Ground truth contains S2-999 which was NOT retrieved by blocker
    true_ids = {"S2-001", "S2-999"}
    # Retrieved candidates only contain S2-002 and S3-003 (missed S2-001 and S2-999)
    targets = [t_hn1, t_hn2]
    provs = [{}, {}]

    sampler = HardNegativeSampler()
    s_targs, s_provs, s_labels, stats = sampler.sample_candidates_for_query(s1, targets, provs, true_ids)

    # When blocker failed to retrieve true match, positive count must be 0
    assert stats["total_positives"] == 0
    assert len(s_labels) == 0


# 4. Test negative sampling correctness
def test_negative_sampling_correctness(sample_profiles):
    s1, t_true, t_hn1, t_hn2, t_rand = sample_profiles
    targets = [t_true, t_hn1, t_hn2, t_rand]
    provs = [{}, {}, {}, {}]
    true_ids = {"S2-001"}

    sampler = HardNegativeSampler(negatives_per_positive=2, random_seed=42)
    s_targs, s_provs, s_labels, stats = sampler.sample_candidates_for_query(s1, targets, provs, true_ids)

    neg_count = sum(1 for lab in s_labels if lab == 0)
    pos_count = sum(1 for lab in s_labels if lab == 1)
    assert pos_count == 1
    assert neg_count <= 2


# 5. Test hard-negative inclusion
def test_hard_negative_inclusion(sample_profiles):
    s1, t_true, t_hn1, t_hn2, t_rand = sample_profiles
    targets = [t_true, t_hn1, t_hn2, t_rand]
    # t_hn2 has identical address, t_hn1 has identical name
    provs = [{"channels": ["ch1"]}, {"channels": ["ch1"]}, {"channels": ["ch3", "ch4"]}, {"channels": ["ch5"]}]
    true_ids = {"S2-001"}

    sampler = HardNegativeSampler(negatives_per_positive=2, min_hard_ratio=1.0, random_seed=42)
    s_targs, s_provs, s_labels, stats = sampler.sample_candidates_for_query(s1, targets, provs, true_ids)

    sampled_tids = {t.eid for t in s_targs}
    # S2-002 or S3-003 (the hard negatives) must be sampled before S3-004 (easy negative)
    assert ("S2-002" in sampled_tids) or ("S3-003" in sampled_tids)


# 6. Test deterministic feature computation
def test_deterministic_features(sample_profiles):
    s1, t_true, t_hn1, _, _ = sample_profiles
    prov = {"channels": ["ch1_core_name", "ch6_distinctive_token"], "rarest_token_df": 150}

    f1 = extract_pair_features(s1, t_true, prov)
    f2 = extract_pair_features(s1, t_true, prov)
    f3 = extract_pair_features(s1, t_hn1, prov)

    assert np.array_equal(f1, f2), "Pair feature extraction must be strictly deterministic"
    assert not np.array_equal(f1, f3), "Distinct targets must yield different feature vectors"
    assert len(f1) == NUM_FEATURES
    assert not np.isnan(f1).any(), "Feature vectors must contain no NaN values"


# 7. Test Recall@K computation
def test_recall_at_k():
    ranked_cands = {
        "S1-1": [("T1", 0.9), ("T2", 0.8), ("T3", 0.5), ("T4", 0.1)],
        "S1-2": [("T5", 0.95), ("T6", 0.4), ("T7", 0.2)]
    }
    gt = {
        "S1-1": {"T1", "T3"},  # T1 at rank 1, T3 at rank 3
        "S1-2": {"T6"}          # T6 at rank 2
    }
    res = compute_recall_at_k(ranked_cands, gt, k_values=[1, 2, 3])

    # K=1: S1-1 captures T1 (1/3 total true) -> Recall = 1/3 = 0.33333
    assert abs(res["Recall@1"]["recall"] - (1.0 / 3.0)) < 1e-4
    # K=2: S1-1 captures T1, S1-2 captures T6 -> Recall = 2/3 = 0.66667
    assert abs(res["Recall@2"]["recall"] - (2.0 / 3.0)) < 1e-4
    # K=3: S1-1 captures T1, T3; S1-2 captures T6 -> Recall = 3/3 = 1.0
    assert abs(res["Recall@3"]["recall"] - 1.0) < 1e-4


# 8. Test ranking determinism
def test_ranking_determinism(sample_profiles):
    s1, t_true, t_hn1, t_hn2, t_rand = sample_profiles
    targets = [t_true, t_hn1, t_hn2, t_rand]
    provs = [{}, {}, {}, {}]

    # Simple mock ranker
    ranker = CandidateRanker(model_type="logistic", random_state=42)
    # Fit on mock data
    X_mock = np.random.RandomState(42).randn(10, NUM_FEATURES).astype(np.float32)
    y_mock = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int32)
    ranker.fit(X_mock, y_mock)

    run1 = rank_candidates_for_query(s1, targets, provs, ranker)
    run2 = rank_candidates_for_query(s1, targets, provs, ranker)

    assert run1 == run2, "rank_candidates_for_query must be deterministic across executions"
    assert len(run1) == len(targets)
    # Scores must be descending
    scores = [s for _, s in run1]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


# 9. Test candidate budget enforcement
def test_candidate_budget_enforcement():
    ranked = [("T1", 0.95), ("T2", 0.85), ("T3", 0.70), ("T4", 0.60), ("T5", 0.40)]

    cands_k2 = apply_candidate_budget(ranked, policy="fixed_k", k=2)
    assert len(cands_k2) == 2
    assert cands_k2 == ["T1", "T2"]

    cands_thresh = apply_candidate_budget(ranked, policy="score_threshold", threshold=0.75, max_candidates=10)
    assert len(cands_thresh) == 2
    assert cands_thresh == ["T1", "T2"]


# 10. Test candidate subset invariant
def test_candidate_subset_invariant():
    ranked = [("T1", 0.95), ("T2", 0.85), ("T3", 0.70), ("T4", 0.60)]
    k_small = apply_candidate_budget(ranked, policy="fixed_k", k=2)
    k_large = apply_candidate_budget(ranked, policy="fixed_k", k=3)

    # Top-2 must be a strict subset of Top-3
    assert set(k_small).issubset(set(k_large))
    assert set(k_small).issubset({t for t, _ in ranked})


# 11. Test no Cartesian product
def test_no_cartesian_product():
    # Verify candidate generator and dataset builder do not multiply queries x all targets
    s1_dict = {f"S1-{i}": EntityProfile(f"S1-{i}", f"Company {i}", f"Addr {i}") for i in range(10)}
    targets = {f"T-{i}": EntityProfile(f"T-{i}", f"Target {i}", f"Addr {i}") for i in range(100)}

    # Sparse candidate list: each S1 only has 2 candidates
    sparse_cands = {f"S1-{i}": [(f"T-{i}", {}), (f"T-{(i+1)%100}", {})] for i in range(10)}
    gt = {f"S1-{i}": {f"T-{i}"} for i in range(10)}

    X, y, groups, stats = build_candidate_ranker_dataset(s1_dict, targets, sparse_cands, gt)
    # Total examples must be bounded by sparse candidates, NEVER 10 * 100 = 1000
    assert len(X) <= 20
    assert len(X) < 1000


# 12. Test batch/chunk equivalence to individual scoring
def test_batch_equivalence_to_individual_scoring(sample_profiles):
    s1, t_true, t_hn1, t_hn2, _ = sample_profiles
    targets = [t_true, t_hn1, t_hn2]
    provs = [{}, {}, {}]

    ranker = CandidateRanker(model_type="logistic", random_state=42)
    X_mock = np.random.RandomState(42).randn(10, NUM_FEATURES).astype(np.float32)
    y_mock = np.array([1, 0, 1, 0, 1, 0, 1, 0, 0, 0], dtype=np.int32)
    ranker.fit(X_mock, y_mock)

    # Single query rank
    single_res = rank_candidates_for_query(s1, targets, provs, ranker)

    # Batch rank
    cands_by_s1 = {s1.eid: list(zip(targets, provs))}
    batch_res = rank_candidates_batch([s1], cands_by_s1, ranker)

    assert single_res == batch_res[s1.eid]
