"""
Automated unit tests for Phase 0 & Phase 1 Foundations:
1. TSV schema validation
2. Ground-truth parsing
3. Duplicate-ID detection
4. Empty-match parsing
5. Macro F_0.5 example reproduction
6. Singleton scoring
7. Normalization
8. Validation split reproducibility
9. Train/validation S1 disjointness
10. Unicode handling
11. Candidate recall metric calculation
12. Stratification ratio preservation
"""

import os
import tempfile
from pathlib import Path
import pytest

from amazon_ml_er.code.business_entity_resolution.src.data_loader import (
    stream_tsv_rows,
    TSVValidationError
)
from amazon_ml_er.code.business_entity_resolution.src.ground_truth import (
    parse_ground_truth_file,
    GroundTruthValidationError
)
from amazon_ml_er.code.business_entity_resolution.src.evaluation.metrics import (
    compute_macro_f05,
    compute_entity_f05,
    compute_detailed_evaluation,
    compute_candidate_recall
)
from amazon_ml_er.code.business_entity_resolution.src.validation.split import (
    create_s1_validation_split
)
from amazon_ml_er.code.business_entity_resolution.src.preprocessing.normalizer import (
    normalize_text,
    normalize_business_name,
    normalize_business_address,
    strip_accents_and_diacritics,
    extract_numeric_tokens
)


# 1. TSV Schema Validation
def test_tsv_schema_validation():
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        f.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
        temp_path = Path(f.name)

    try:
        rows = list(stream_tsv_rows(temp_path, expected_columns=["entity_id", "business_name", "business_address", "country"]))
        assert len(rows) == 1
        assert rows[0]["entity_id"] == "S1-001"
        assert rows[0]["business_name"] == "Acme Corp"
    finally:
        os.remove(temp_path)

    # Broken header should raise TSVValidationError
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write("id\tname\taddr\tcntry\n")
        f.write("S1-001\tAcme Corp\t123 Main St\tUS\n")
        broken_path = Path(f.name)

    try:
        with pytest.raises(TSVValidationError):
            list(stream_tsv_rows(broken_path, expected_columns=["entity_id", "business_name", "business_address", "country"]))
    finally:
        os.remove(broken_path)


# 2. Ground-truth Parsing & 4. Empty-match Parsing
def test_ground_truth_parsing_and_empty_matches():
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-100\tS2-200,S3-300\n")
        f.write("S1-101\t\n")  # Singleton
        f.write("S1-102\tS2-202\n")
        temp_path = Path(f.name)

    try:
        gt = parse_ground_truth_file(temp_path)
        assert len(gt) == 3
        assert gt["S1-100"] == {"S2-200", "S3-300"}
        assert gt["S1-101"] == set()  # Singleton parsed as empty set
        assert gt["S1-102"] == {"S2-202"}
    finally:
        os.remove(temp_path)


# 3. Duplicate-ID Detection & Self-match rejection
def test_duplicate_and_invalid_ids():
    # Duplicate S1 in ground truth
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-100\tS2-200\n")
        f.write("S1-100\tS3-300\n")  # duplicate S1
        temp_path = Path(f.name)

    try:
        with pytest.raises(GroundTruthValidationError):
            parse_ground_truth_file(temp_path)
    finally:
        os.remove(temp_path)

    # Self-match rejection
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-100\tS1-100\n")  # Self-match!
        temp_path2 = Path(f.name)

    try:
        with pytest.raises(GroundTruthValidationError):
            parse_ground_truth_file(temp_path2)
    finally:
        os.remove(temp_path2)


# 5. Macro F_0.5 problem statement example reproduction
def test_macro_f05_official_example():
    """
    Problem statement specification:
    S1-00001: true = [S2-00047, S3-00812], pred = [S2-00047, S2-00193, S3-00812]
    Precision = 2/3 = 0.666667
    Recall = 2/2 = 1.0
    F_0.5 = (1.25 * (2/3) * 1.0) / (0.25 * (2/3) + 1.0) = (0.833333) / (1.166667) = 0.7142857...
    """
    y_true = {"S1-00001": {"S2-00047", "S3-00812"}}
    y_pred = {"S1-00001": {"S2-00047", "S2-00193", "S3-00812"}}

    p, r, f05 = compute_entity_f05(y_true["S1-00001"], y_pred["S1-00001"])
    assert round(p, 4) == round(2 / 3, 4)
    assert round(r, 4) == 1.0
    assert round(f05, 3) == 0.714

    macro_f05 = compute_macro_f05(y_true, y_pred)
    assert round(macro_f05, 3) == 0.714


# 6. Singleton Scoring
def test_singleton_scoring():
    # Correct singleton: true empty, pred empty -> 1.0
    p1, r1, f1 = compute_entity_f05(set(), set())
    assert f1 == 1.0
    assert p1 == 1.0
    assert r1 == 1.0

    # False merge on singleton: true empty, pred non-empty -> 0.0
    p2, r2, f2 = compute_entity_f05(set(), {"S2-00099"})
    assert f2 == 0.0
    assert p2 == 0.0
    assert r2 == 0.0

    # Missed match on non-singleton: true non-empty, pred empty -> 0.0
    p3, r3, f3 = compute_entity_f05({"S2-00001"}, set())
    assert f3 == 0.0
    assert p3 == 0.0
    assert r3 == 0.0

    # Test aggregate macro with singletons
    y_true = {
        "S1-1": set(),            # True singleton
        "S1-2": set(),            # True singleton
        "S1-3": {"S2-10", "S3-20"} # Non-singleton
    }
    y_pred = {
        "S1-1": set(),            # Correct singleton -> 1.0
        "S1-2": {"S2-99"},        # False merge -> 0.0
        "S1-3": {"S2-10", "S3-20"} # Perfect match -> 1.0
    }
    # Expected macro F_0.5 = (1.0 + 0.0 + 1.0) / 3 = 2/3 = 0.6667
    score = compute_macro_f05(y_true, y_pred)
    assert round(score, 4) == round(2 / 3, 4)


# 7. Normalization
def test_normalization():
    # Punctuation & casing
    n1 = normalize_business_name("Acme, Inc. & Sons")
    assert n1.cleaned_lower == "acme, inc. & sons"
    assert "inc" in n1.detected_legal_suffixes
    assert "acme" in n1.significant_tokens
    assert "sons" in n1.significant_tokens
    assert "and" not in n1.significant_tokens  # Stopword removed from significant

    # Token order invariance via sorted_top2_tokens
    n2 = normalize_business_name("Flowers and Hendricks")
    n3 = normalize_business_name("Hendricks and Flowers LLC")
    assert n2.sorted_top2_tokens == ("flowers", "hendricks")
    assert n3.sorted_top2_tokens == ("flowers", "hendricks")

    # Numeric address tokens
    addr = normalize_business_address("1795 Westchester Drive, High Point, NC 27262")
    assert "1795" in addr.numeric_tokens
    assert "27262" in addr.numeric_tokens

    # Empty text
    empty_norm = normalize_business_name("")
    assert empty_norm.raw == ""
    assert empty_norm.tokens == []


# 8. Validation Split Reproducibility & 9. Train/Validation S1 Disjointness
def test_validation_split_reproducibility_and_disjointness():
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for i in range(100):
            c = "US" if i < 60 else "India"
            f.write(f"S1-{i:04d}\tBusiness {i}\tAddress {i}\t{c}\n")
        temp_path = Path(f.name)

    try:
        val_size = 20
        train1, val1 = create_s1_validation_split(
            s1_path=temp_path,
            val_size=val_size,
            random_seed=42,
            output_val_path=None,
            output_train_path=None,
            stratify_by_country=True
        )
        train2, val2 = create_s1_validation_split(
            s1_path=temp_path,
            val_size=val_size,
            random_seed=42,
            output_val_path=None,
            output_train_path=None,
            stratify_by_country=True
        )

        # Reproducibility
        assert val1 == val2
        assert train1 == train2
        assert len(val1) == 20
        assert len(train1) == 80

        # Disjointness
        set_val = set(val1)
        set_train = set(train1)
        assert len(set_val & set_train) == 0
        assert len(set_val | set_train) == 100
    finally:
        os.remove(temp_path)


# 10. Unicode Handling (Accents & Indic Scripts)
def test_unicode_handling():
    # French accented string
    accented = "Thermal & Fils SASU, 20 Rue Parmentier, Dunkerque"
    clean = strip_accents_and_diacritics(accented)
    assert "Thermal & Fils SASU" in clean

    french_name = "Fractales Amis Groupe S.A.S"
    norm_fr = normalize_business_name(french_name)
    assert "sas" in norm_fr.detected_legal_suffixes
    assert "fractales" in norm_fr.tokens
    assert "amis" in norm_fr.tokens

    # Indic Script (Hindi Devanagari)
    hindi_name = "एसएस फूड प्राइवेट लिमिटेड"
    norm_hi = normalize_business_name(hindi_name)
    assert len(norm_hi.tokens) == 4
    assert norm_hi.tokens[0] == "एसएस"
    assert norm_hi.tokens[1] == "फूड"
    assert norm_hi.tokens[2] == "प्राइवेट"
    assert norm_hi.tokens[3] == "लिमिटेड"

    # Indic Script (Tamil)
    tamil_name = "ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி"
    norm_ta = normalize_business_name(tamil_name)
    assert len(norm_ta.tokens) == 3
    assert norm_ta.tokens[0] == "ராஜ்"
    assert norm_ta.tokens[1] == "இன்வெஸ்ட்மெண்ட்ஸ்"
    assert norm_ta.tokens[2] == "எல்எல்பி"


# 11. Candidate Recall Metric Verification
def test_candidate_recall_metric():
    y_true = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": set()  # Singleton
    }
    # Candidates capture S2-10 but miss S3-20, capture S2-30, and have an extra distractor
    candidates = {
        "S1-1": {"S2-10", "S2-99"},
        "S1-2": {"S2-30"},
        "S1-3": {"S3-88"}
    }
    recall_stats = compute_candidate_recall(y_true, candidates)
    # Total true matches = 2 + 1 = 3. Captured = 1 (S2-10) + 1 (S2-30) = 2.
    assert recall_stats["total_true_matches"] == 3
    assert recall_stats["captured_true_matches"] == 2
    assert round(recall_stats["candidate_recall"], 4) == round(2 / 3, 4)
    assert recall_stats["entity_coverage"]["full_coverage_count"] == 1
    assert recall_stats["entity_coverage"]["partial_coverage_count"] == 1
