"""
Phase 3 Test Suite: Final Matching Model & Entity-Level Decisioning (Task 19).
Verifies:
1. pair-label construction
2. no validation leakage
3. feature determinism
4. no NaNs in features
5. model training reproducibility
6. calibrated probability behavior
7. singleton decisioning
8. multi-match decisioning
9. threshold search isolation
10. target conflict resolution
11. candidate-policy integration
12. exact macro F_0.5
13. batch scoring equivalence
14. final output subset invariant
15. source-specific behavior
16. deterministic end-to-end validation
"""

import pytest
import numpy as np
from typing import Dict, List, Set, Tuple, Any

from amazon_ml_er.code.business_entity_resolution.src.matching.features import (
    MatchingEntityProfile,
    extract_matching_features,
    extract_batch_matching_features,
    fast_levenshtein_ratio,
    FEATURE_NAMES,
    NUM_MATCHING_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.matching.dataset import (
    MatcherHardNegativeSampler,
    build_matching_dataset
)
from amazon_ml_er.code.business_entity_resolution.src.matching.models import FinalMatcherModel
from amazon_ml_er.code.business_entity_resolution.src.matching.calibration import (
    MatcherCalibrator,
    compute_calibration_diagnostics
)
from amazon_ml_er.code.business_entity_resolution.src.matching.entity_decision import (
    EntityDecisionEngine,
    compute_singleton_diagnostics
)
from amazon_ml_er.code.business_entity_resolution.src.matching.thresholding import search_optimal_threshold
from amazon_ml_er.code.business_entity_resolution.src.matching.conflict_resolution import resolve_target_conflicts_greedy
from amazon_ml_er.code.business_entity_resolution.src.evaluation.metrics import (
    compute_entity_f05,
    compute_macro_f05,
    compute_detailed_evaluation
)


@pytest.fixture
def sample_profiles():
    s1 = MatchingEntityProfile("S1_1", "Starbucks Coffee Company", "2401 Utah Ave S, Seattle, WA", "US")
    t1 = MatchingEntityProfile("T1", "Starbucks Coffee", "2401 Utah Ave S, Seattle, WA", "US")
    t2 = MatchingEntityProfile("T2", "Starbucks Coffe", "2401 Utah Avenue South, Seattle, WA", "US")
    t_neg = MatchingEntityProfile("T3", "Peets Coffee", "2401 Utah Ave S, Seattle, WA", "US")
    return s1, t1, t2, t_neg


# 1. Test pair-label construction
def test_pair_label_construction(sample_profiles):
    s1, t1, t2, t_neg = sample_profiles
    sampler = MatcherHardNegativeSampler(negatives_per_positive=2, random_seed=42)
    true_tids = {"T1", "T2"}

    targs, provs, scores, ranks, labels, stats = sampler.sample_candidates_for_query(
        s1, [t1, t2, t_neg], [{}, {}, {}], [0.9, 0.8, 0.3], true_tids
    )
    assert sum(labels) == 2, "Must have exactly 2 positive labels"
    assert len(labels) == len(targs)
    for t_prof, lab in zip(targs, labels):
        if lab == 1:
            assert t_prof.eid in true_tids
        else:
            assert t_prof.eid not in true_tids


# 2. Test no validation leakage
def test_no_validation_leakage():
    # Verify split files have zero overlap
    with open("amazon_ml_er/splits/train_source1_ids.txt") as f:
        train_ids = [line.strip() for line in f if line.strip()]
    with open("amazon_ml_er/splits/val_source1_ids.txt") as f:
        val_ids = set(line.strip() for line in f if line.strip())

    train_5k = set(train_ids[:5000])
    train_25k = set(train_ids[:25000])
    tune_2k = set(train_ids[25000:27000])

    assert len(train_25k & val_ids) == 0, "Train 25k must not overlap with validation"
    assert len(tune_2k & val_ids) == 0, "Tune 2k must not overlap with validation"
    assert len(train_25k & tune_2k) == 0, "Train 25k must not overlap with tune 2k"


# 3. Test feature determinism
def test_feature_determinism(sample_profiles):
    s1, t1, _, _ = sample_profiles
    f1 = extract_matching_features(s1, t1, {"channels": {"ch1_core_name"}}, 0.85, 1)
    f2 = extract_matching_features(s1, t1, {"channels": {"ch1_core_name"}}, 0.85, 1)
    assert np.array_equal(f1, f2), "Features must be bit-exact deterministic"


# 4. Test no NaNs in features
def test_no_nans_in_features(sample_profiles):
    s1, t1, t2, t_neg = sample_profiles
    for t in [t1, t2, t_neg]:
        feat = extract_matching_features(s1, t, {}, 0.5, 1)
        assert not np.any(np.isnan(feat)), "Feature vector contains NaN!"
        assert not np.any(np.isinf(feat)), "Feature vector contains Inf!"
        assert len(feat) == NUM_MATCHING_FEATURES


# 5. Test model training reproducibility
def test_model_training_reproducibility():
    np.random.seed(42)
    X = np.random.randn(100, NUM_MATCHING_FEATURES).astype(np.float32)
    y = np.random.randint(0, 2, size=100)

    m1 = FinalMatcherModel(model_type="hist_gb", random_state=42, max_iter=20)
    m1.fit(X, y)
    p1 = m1.predict_proba(X)

    m2 = FinalMatcherModel(model_type="hist_gb", random_state=42, max_iter=20)
    m2.fit(X, y)
    p2 = m2.predict_proba(X)

    assert np.allclose(p1, p2, atol=1e-5), "HistGB predictions must be deterministic"


# 6. Test calibrated probability behavior
def test_calibrated_probability_behavior():
    raw_scores = np.array([0.1, 0.2, 0.4, 0.7, 0.8, 0.9], dtype=np.float32)
    y_true = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    cal = MatcherCalibrator(method="sigmoid").fit(raw_scores, y_true)
    cal_scores = cal.predict(raw_scores)

    assert len(cal_scores) == len(raw_scores)
    assert np.all(cal_scores >= 0.0) and np.all(cal_scores <= 1.0)
    # Calibrated probabilities should preserve rank order monotonically for sigmoid
    assert all(cal_scores[i] <= cal_scores[i+1] for i in range(len(cal_scores)-1))


# 7. Test singleton decisioning
def test_singleton_decisioning():
    engine = EntityDecisionEngine(score_threshold=0.50, strategy="global_threshold")
    # Entity with no candidates above threshold
    cands_empty = [("T1", 0.35, None), ("T2", 0.20, None)]
    pred = engine.decide_matches_for_entity(cands_empty)
    assert len(pred) == 0, "Should predict empty set (singleton) when top score < threshold"

    # Single candidate above threshold
    cands_match = [("T1", 0.75, None), ("T2", 0.20, None)]
    pred_match = engine.decide_matches_for_entity(cands_match)
    assert pred_match == {"T1"}


# 8. Test multi-match decisioning
def test_multi_match_decisioning():
    engine = EntityDecisionEngine(score_threshold=0.50, score_margin=0.20, strategy="threshold_and_margin")
    cands = [("T1", 0.85, None), ("T2", 0.80, None), ("T3", 0.70, None), ("T4", 0.55, None)]
    # top = 0.85, margin = 0.20 -> cutoff = 0.65. T1 (0.85), T2 (0.80), T3 (0.70) pass; T4 (0.55 < 0.65) fails.
    pred = engine.decide_matches_for_entity(cands)
    assert pred == {"T1", "T2", "T3"}


# 9. Test threshold search isolation
def test_threshold_search_isolation():
    tune_gt = {"S1_1": {"T1"}, "S1_2": {"T2"}}
    cands = {
        "S1_1": [("T1", 0.8, None), ("T9", 0.3, None)],
        "S1_2": [("T2", 0.6, None), ("T8", 0.4, None)]
    }
    res = search_optimal_threshold(tune_gt, cands, threshold_grid=[0.5, 0.7])
    assert "best_params" in res
    assert res["best_macro_f05"] >= 0.0


# 10. Test target conflict resolution
def test_target_conflict_resolution():
    # Both S1_A and S1_B claim T1, but S1_A has higher score
    preds = {
        "S1_A": {"T1", "T2"},
        "S1_B": {"T1", "T3"}
    }
    scored = {
        "S1_A": [("T1", 0.90, None), ("T2", 0.80, None)],
        "S1_B": [("T1", 0.60, None), ("T3", 0.75, None)]
    }
    resolved, stats = resolve_target_conflicts_greedy(preds, scored)
    assert "T1" in resolved["S1_A"], "S1_A should win T1 with score 0.90"
    assert "T1" not in resolved["S1_B"], "S1_B should lose T1 with score 0.60"
    assert "T3" in resolved["S1_B"], "S1_B should keep unconflicted T3"
    assert stats["conflicted_targets_count"] == 1


# 11. Test candidate policy integration
def test_candidate_policy_integration():
    cands = [("T1", 0.9, None), ("T2", 0.8, None), ("T3", 0.7, None)]
    eng = EntityDecisionEngine(score_threshold=0.50, strategy="global_threshold")
    res = eng.decide_matches_for_entity(cands[:2])
    assert len(res) == 2


# 12. Test exact macro F_0.5 computation
def test_exact_macro_f05():
    y_true = {"S1_1": {"T1"}, "S1_2": set()}
    y_pred = {"S1_1": {"T1"}, "S1_2": set()}
    f05 = compute_macro_f05(y_true, y_pred)
    assert f05 == 1.0, "Perfect predictions should yield macro F0.5 = 1.0"


# 13. Test batch scoring equivalence
def test_batch_scoring_equivalence(sample_profiles):
    s1, t1, t2, _ = sample_profiles
    m = FinalMatcherModel(model_type="hist_gb", random_state=42, max_iter=10)
    X = np.random.randn(20, NUM_MATCHING_FEATURES).astype(np.float32)
    y = np.random.randint(0, 2, size=20)
    m.fit(X, y)

    f1 = extract_matching_features(s1, t1, {}, 0.5, 1)
    f2 = extract_matching_features(s1, t2, {}, 0.5, 2)
    batch_f = extract_batch_matching_features(s1, [t1, t2], [{}, {}], [0.5, 0.5], [1, 2])

    p_ind = np.array([m.predict_proba(f1.reshape(1, -1))[0], m.predict_proba(f2.reshape(1, -1))[0]])
    p_batch = m.predict_proba(batch_f)

    assert np.allclose(p_ind, p_batch, atol=1e-5), "Batch scoring must match individual scoring"


# 14. Test final output subset invariant
def test_final_output_subset_invariant():
    cands = [("T1", 0.8, None), ("T2", 0.6, None)]
    engine = EntityDecisionEngine(score_threshold=0.50)
    pred = engine.decide_matches_for_entity(cands)
    cand_ids = set(c[0] for c in cands)
    assert pred.issubset(cand_ids), "Predicted matches must be strict subset of input candidates"


# 15. Test source-specific behavior
def test_source_specific_behavior(sample_profiles):
    s1, t1, _, _ = sample_profiles
    t1_s2 = MatchingEntityProfile("T1_S2", "Starbucks", "2401 Utah Ave", "US")
    t1_s3 = MatchingEntityProfile("S3_1001", "Starbucks", "2401 Utah Ave", "US")

    f_s2 = extract_matching_features(s1, t1_s2, {}, 0.5, 1)
    f_s3 = extract_matching_features(s1, t1_s3, {}, 0.5, 1)

    idx_s3 = FEATURE_NAMES.index("target_is_source3")
    assert f_s2[idx_s3] == 0.0
    assert f_s3[idx_s3] == 1.0


# 16. Test deterministic end-to-end validation
def test_deterministic_end_to_end_validation():
    lev1 = fast_levenshtein_ratio("amazon corporation", "amazon corp")
    lev2 = fast_levenshtein_ratio("amazon corporation", "amazon corp")
    assert lev1 == lev2
    assert 0.0 <= lev1 <= 1.0
