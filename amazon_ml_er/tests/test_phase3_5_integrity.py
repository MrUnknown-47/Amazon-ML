"""
Phase 3.5 Integrity Test Suite:
Validates evaluation integrity, decision engine edge cases, full singleton handling,
split disjointness, and conflict resolution determinism.
"""

import pytest
import csv
from typing import Dict, List, Set, Tuple, Any

from amazon_ml_er.code.business_entity_resolution.src.config import (
    TRAIN_GROUND_TRUTH,
    VALIDATION_SPLIT_S1_PATH,
    TRAIN_SPLIT_S1_PATH
)
from amazon_ml_er.code.business_entity_resolution.src.matching.entity_decision import EntityDecisionEngine
from amazon_ml_er.code.business_entity_resolution.src.matching.conflict_resolution import resolve_target_conflicts_greedy
from amazon_ml_er.code.business_entity_resolution.src.evaluation.metrics import (
    compute_entity_f05,
    compute_macro_f05
)


def test_train_tune_val_complete_disjointness():
    """Verify strict pairwise disjointness between TRAIN_25K, TUNE_2K, and VAL_50K."""
    with open(TRAIN_SPLIT_S1_PATH, "r", encoding="utf-8") as f:
        train_pool = [line.strip() for line in f if line.strip()]

    with open(VALIDATION_SPLIT_S1_PATH, "r", encoding="utf-8") as f:
        val50k = set(line.strip() for line in f if line.strip())

    train_25k = set(train_pool[:25000])
    tune_2k = set(train_pool[25000:27000])

    assert len(train_25k) == 25000
    assert len(tune_2k) == 2000
    assert len(val50k) == 50000

    # Strict pairwise disjointness
    assert len(train_25k & tune_2k) == 0, "TRAIN_25K and TUNE must be disjoint"
    assert len(train_25k & val50k) == 0, "TRAIN_25K and VAL_50K must be disjoint"
    assert len(tune_2k & val50k) == 0, "TUNE and VAL_50K must be disjoint"


def test_full_singleton_population_audit():
    """Verify ground truth singleton counts: exactly 2,768 singletons out of 50,000."""
    with open(VALIDATION_SPLIT_S1_PATH, "r", encoding="utf-8") as f:
        val_ids = set(line.strip() for line in f if line.strip())

    assert len(val_ids) == 50000

    gt_matches = {}
    with open(TRAIN_GROUND_TRUTH, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for row in r:
            if row:
                s1 = row[0].strip()
                tids = [t.strip() for t in row[1].split(",") if t.strip()]
                gt_matches[s1] = tids

    singletons = [eid for eid in val_ids if len(gt_matches.get(eid, [])) == 0]
    non_singletons = [eid for eid in val_ids if len(gt_matches.get(eid, [])) > 0]

    assert len(singletons) == 2768, f"Expected exactly 2,768 singletons, got {len(singletons)}"
    assert len(non_singletons) == 47232, f"Expected exactly 47,232 non-singletons, got {len(non_singletons)}"
    assert len(singletons) + len(non_singletons) == 50000


def test_decision_engine_margin_top1_rule():
    """
    Test that the margin rule is computed strictly against top-1 score.
    With tau=0.70, margin=0.10:
    - top_score = 0.95 -> min_allowed = max(0.70, 0.95 - 0.10) = 0.85
    - candidate with 0.88 survives
    - candidate with 0.82 is pruned (even though >= 0.70)
    """
    engine = EntityDecisionEngine(score_threshold=0.70, score_margin=0.10, strategy="threshold_and_margin")

    cands = [
        ("T1", 0.95, None),
        ("T2", 0.88, None),
        ("T3", 0.82, None), # Below 0.95 - 0.10 = 0.85
        ("T4", 0.72, None),
    ]

    selected = engine.decide_matches_for_entity(cands)
    assert selected == {"T1", "T2"}, f"Expected T1 and T2, got {selected}"


def test_decision_engine_empty_prediction_on_low_scores():
    """Test that when top candidate score < tau, entity prediction is strictly empty."""
    engine = EntityDecisionEngine(score_threshold=0.70, score_margin=0.10, strategy="threshold_and_margin")

    cands = [
        ("T1", 0.68, None), # Below 0.70
        ("T2", 0.65, None),
    ]

    selected = engine.decide_matches_for_entity(cands)
    assert selected == set(), f"Expected empty set, got {selected}"


def test_decision_engine_single_survivor_when_runner_up_violates_margin():
    """
    When top score >= tau, but runner-up score < (top_score - margin),
    exactly one match (the top match) survives.
    """
    engine = EntityDecisionEngine(score_threshold=0.70, score_margin=0.10, strategy="threshold_and_margin")

    cands = [
        ("T1", 0.85, None),
        ("T2", 0.71, None), # Margin drop is 0.14 > 0.10
    ]

    selected = engine.decide_matches_for_entity(cands)
    assert selected == {"T1"}, f"Expected exactly T1, got {selected}"


def test_decision_engine_multi_match_survival():
    """Test that multiple valid cluster targets within margin all survive."""
    engine = EntityDecisionEngine(score_threshold=0.70, score_margin=0.10, strategy="threshold_and_margin")

    cands = [
        ("T1", 0.98, None),
        ("T2", 0.96, None),
        ("T3", 0.94, None),
        ("T4", 0.91, None),
        ("T5", 0.85, None), # Pruned: 0.98 - 0.85 = 0.13 > 0.10
    ]

    selected = engine.decide_matches_for_entity(cands)
    assert selected == {"T1", "T2", "T3", "T4"}


def test_decision_engine_per_entity_cap():
    """Test that max_matches_cap enforces an upper bound when configured."""
    engine = EntityDecisionEngine(score_threshold=0.70, score_margin=0.20, max_matches_cap=2, strategy="threshold_and_margin")

    cands = [
        ("T1", 0.95, None),
        ("T2", 0.90, None),
        ("T3", 0.88, None),
    ]

    selected = engine.decide_matches_for_entity(cands)
    assert len(selected) <= 2


def test_conflict_resolution_reproducibility():
    """Test deterministic conflict resolution when two S1 entities claim the same target."""
    predictions = {
        "S1_A": {"T_SHARED", "T_A_ONLY"},
        "S1_B": {"T_SHARED", "T_B_ONLY"}
    }
    scored = {
        "S1_A": [("T_SHARED", 0.92, None), ("T_A_ONLY", 0.85, None)],
        "S1_B": [("T_SHARED", 0.78, None), ("T_B_ONLY", 0.88, None)]
    }

    resolved, stats = resolve_target_conflicts_greedy(predictions, scored, min_confidence=0.0)

    # S1_A had score 0.92 for T_SHARED, S1_B had score 0.78 -> S1_A must keep T_SHARED
    assert "T_SHARED" in resolved["S1_A"]
    assert "T_SHARED" not in resolved["S1_B"]
    assert "T_B_ONLY" in resolved["S1_B"]
    assert stats["conflicted_targets_count"] == 1
    assert stats["duplicate_claims_removed"] == 1


def test_independent_macro_f05_calculation():
    """Test official macro F0.5 formula independently."""
    # Test case 1: Singleton correct
    p, r, f = compute_entity_f05(set(), set())
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # Test case 2: Singleton false positive
    p, r, f = compute_entity_f05(set(), {"T1"})
    assert (p, r, f) == (0.0, 0.0, 0.0)

    # Test case 3: Non-singleton perfect match
    p, r, f = compute_entity_f05({"T1", "T2"}, {"T1", "T2"})
    assert (p, r, f) == (1.0, 1.0, 1.0)

    # Test case 4: Precision 1.0, Recall 0.5 (TP=1, FP=0, FN=1)
    # F_0.5 = (1.25 * 1.0 * 0.5) / (0.25 * 1.0 + 0.5) = 0.625 / 0.75 = 0.8333333333333334
    p, r, f = compute_entity_f05({"T1", "T2"}, {"T1"})
    assert abs(p - 1.0) < 1e-6
    assert abs(r - 0.5) < 1e-6
    assert abs(f - 5/6) < 1e-6
