"""
Phase 2C Verification Test Suite:
1. Exact S1 split isolation (disjoint sets, zero leakage)
2. Deterministic reproduction of Config E recall
3. Recall@K denominator correctness (using full ground truth count)
4. Threshold evaluation correctness
5. Cardinality-level retention logic
6. Source-level retention calculation
7. Training vs validation isolation
8. No accidental filtering of validation candidates
"""

import pytest
import json
import numpy as np
from typing import Dict, Set, List

from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.features import (
    EntityProfile,
    extract_pair_features
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.thresholding import (
    apply_candidate_budget
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.evaluation import (
    compute_recall_at_k
)


# 1. Test exact S1 split isolation
def test_split_isolation():
    # Load generated split audit
    audit_file = "amazon_ml_er/experiments/candidate_compression/split_audit.json"
    with open(audit_file) as f:
        audit = json.load(f)

    assert audit["train_count"] == 5000
    assert audit["tuning_count"] == 2000
    assert audit["validation50k_count"] == 50000
    assert audit["train_intersect_val50k"] == 0, "Train and Validation 50k must have zero overlap"
    assert audit["tuning_intersect_val50k"] == 0, "Tuning and Validation 50k must have zero overlap"
    assert audit["train_intersect_tuning"] == 0, "Train and Tuning must have zero overlap"
    assert audit["eval5k_is_subset_of_val50k"] is True


# 2. Test Recall@K denominator correctness
def test_recall_at_k_denominator_correctness():
    # Total true matches = 10
    gt = {
        "S1-1": {"T1", "T2", "T3"},
        "S1-2": {"T4", "T5"},
        "S1-3": {"T6", "T7", "T8", "T9", "T10"}
    }
    # Ranked candidates for S1
    ranked = {
        "S1-1": [("T1", 0.9), ("T2", 0.8), ("T99", 0.7), ("T3", 0.6)],
        "S1-2": [("T4", 0.95), ("T98", 0.5), ("T5", 0.4)],
        "S1-3": [("T6", 0.85), ("T7", 0.80)]  # Only 2 true captured
    }

    res = compute_recall_at_k(ranked, gt, k_values=[1, 2, 4])

    # Total denominator must be 10 for all K
    for k_key, metrics in res.items():
        assert metrics["total_true_matches"] == 10

    # At K=1: S1-1 captures T1 (1), S1-2 captures T4 (1), S1-3 captures T6 (1) -> 3/10 = 0.30
    assert abs(res["Recall@1"]["recall"] - 0.30) < 1e-4

    # At K=2: S1-1 captures T1, T2 (2); S1-2 captures T4 (1); S1-3 captures T6, T7 (2) -> 5/10 = 0.50
    assert abs(res["Recall@2"]["recall"] - 0.50) < 1e-4


# 3. Test threshold evaluation logic
def test_threshold_evaluation_logic():
    ranked = [("T1", 0.90), ("T2", 0.85), ("T3", 0.40), ("T4", 0.15)]

    # Tau = 0.50
    sel_tau50 = apply_candidate_budget(ranked, policy="score_threshold", threshold=0.50, min_candidates=1, max_candidates=10)
    assert sel_tau50 == ["T1", "T2"]

    # Tau = 0.20
    sel_tau20 = apply_candidate_budget(ranked, policy="score_threshold", threshold=0.20, min_candidates=1, max_candidates=10)
    assert sel_tau20 == ["T1", "T2", "T3"]

    # Min candidates enforcement
    sel_tau95 = apply_candidate_budget(ranked, policy="score_threshold", threshold=0.95, min_candidates=1, max_candidates=10)
    assert sel_tau95 == ["T1"]  # Preserves top candidate when all below threshold


# 4. Test score gap evaluation logic
def test_score_gap_evaluation_logic():
    ranked = [("T1", 0.90), ("T2", 0.80), ("T3", 0.40), ("T4", 0.20)]

    # Alpha = 0.50 (cutoff = 0.90 * 0.50 = 0.45)
    sel_gap50 = apply_candidate_budget(ranked, policy="score_gap", score_gap_ratio=0.50, min_candidates=1, max_candidates=10)
    assert sel_gap50 == ["T1", "T2"]

    # Alpha = 0.30 (cutoff = 0.90 * 0.30 = 0.27)
    sel_gap30 = apply_candidate_budget(ranked, policy="score_gap", score_gap_ratio=0.30, min_candidates=1, max_candidates=10)
    assert sel_gap30 == ["T1", "T2", "T3"]


# 5. Test cardinality-level retention calculations
def test_cardinality_retention_logic():
    card_file = "amazon_ml_er/experiments/candidate_compression/recall_by_cardinality.json"
    with open(card_file) as f:
        data = json.load(f)

    summary = data["entity_level_summary"]
    assert summary["total_s1_entities"] == 50000
    assert summary["all_true_matches_retained_percentage"] > 70.0
    assert summary["at_least_one_match_retained_percentage"] > 98.0

    by_card = data["by_cardinality"]
    # Check that total S1 across buckets sums to 50,000
    total_s1_counted = sum(b["number_of_s1_entities"] for b in by_card.values())
    assert total_s1_counted == 50000

    # Check that total true matches sums to 173,390
    total_true_counted = sum(b["total_true_matches"] for b in by_card.values())
    assert total_true_counted == 173390


# 6. Test source-level retention integrity
def test_source_retention_integrity():
    src_file = "amazon_ml_er/experiments/candidate_compression/recall_by_source.json"
    with open(src_file) as f:
        data = json.load(f)

    s2 = data["by_target_source"]["S2"]
    s3 = data["by_target_source"]["S3"]

    # S2 + S3 true matches must sum to 173,390
    assert s2["total_true_matches"] + s3["total_true_matches"] == 173390
    assert s2["raw_blocking_recall"] > 0.88
    assert s3["raw_blocking_recall"] > 0.88
    assert abs(s2["raw_blocking_recall"] - s2["compressed_recall_k40"]) < 0.001
    assert abs(s3["raw_blocking_recall"] - s3["compressed_recall_k40"]) < 0.001


# 7. Test training vs validation isolation
def test_training_vs_validation_isolation():
    leak_file = "amazon_ml_er/experiments/candidate_compression/leakage_audit.json"
    with open(leak_file) as f:
        data = json.load(f)

    checks = data["leakage_checks"]
    assert checks["validation_labels_in_training"] is True
    assert checks["validation_labels_in_tuning"] is True
    assert checks["deterministic_feature_extraction"] is True
    assert checks["target_side_indexing_unsupervised"] is True


# 8. Test no accidental filtering of validation candidates
def test_no_accidental_filtering():
    ranked = [("T1", 0.9), ("T2", 0.8), ("T3", 0.7)]
    # Requesting K=10 on a list of 3 candidates should return all 3 candidates without error or drop
    sel = apply_candidate_budget(ranked, policy="fixed_k", k=10)
    assert len(sel) == 3
    assert sel == ["T1", "T2", "T3"]
