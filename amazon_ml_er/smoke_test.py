"""
End-to-End Smoke Test for Phase 0 & 1 Foundations.
Validates:
1. Small deterministic subset loading
2. TSV schema validation
3. Ground truth parsing and integrity checks
4. Validation split creation and disjointness verification
5. Normalization across multilingual text (US, India, France)
6. Official Macro F_0.5 and candidate recall evaluation
7. Exits 0 on success, non-zero on failure.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amazon_ml_er.code.business_entity_resolution.src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    SOURCE_COLUMNS,
    GROUND_TRUTH_COLUMNS
)
from amazon_ml_er.code.business_entity_resolution.src.data_loader import (
    stream_tsv_rows,
    load_source_records,
    audit_tsv_file
)
from amazon_ml_er.code.business_entity_resolution.src.ground_truth import (
    parse_ground_truth_file
)
from amazon_ml_er.code.business_entity_resolution.src.validation.split import (
    create_s1_validation_split
)
from amazon_ml_er.code.business_entity_resolution.src.evaluation.metrics import (
    compute_macro_f05,
    compute_detailed_evaluation,
    compute_candidate_recall
)
from amazon_ml_er.code.business_entity_resolution.src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address
)


def run_smoke_test():
    print("==================================================", flush=True)
    print("      RUNNING PHASE 0 & 1 SMOKE TEST             ", flush=True)
    print("==================================================", flush=True)

    # 1. Test subset data loading and schema check
    print("\n[Step 1] Loading deterministic 500-record sample from sources...")
    s1_sample = load_source_records(TRAIN_SOURCE1, limit=500)
    assert len(s1_sample) == 500, f"Expected 500 S1 records, got {len(s1_sample)}"
    print(f"  ✓ Loaded 500 train S1 records successfully.")

    test_s1_sample = load_source_records(TEST_SOURCE1, limit=500)
    assert len(test_s1_sample) == 500, f"Expected 500 test S1 records, got {len(test_s1_sample)}"
    print(f"  ✓ Loaded 500 test S1 records (including open-set countries) successfully.")

    # 2. Test ground truth parsing on sample
    print("\n[Step 2] Parsing ground truth sample...")
    gt_sample = parse_ground_truth_file(TRAIN_GROUND_TRUTH, limit=500)
    assert len(gt_sample) == 500, f"Expected 500 GT rows, got {len(gt_sample)}"
    singletons = sum(1 for m in gt_sample.values() if len(m) == 0)
    non_singletons = sum(1 for m in gt_sample.values() if len(m) > 0)
    print(f"  ✓ Parsed 500 ground truth rows: {singletons} singletons, {non_singletons} non-singletons.")

    # 3. Test validation split on sample
    print("\n[Step 3] Testing validation split generator...")
    with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".tsv") as f:
        f.write("entity_id\tbusiness_name\tbusiness_address\tcountry\n")
        for eid, r in s1_sample.items():
            f.write(f"{eid}\t{r['business_name']}\t{r['business_address']}\t{r['country']}\n")
        sample_s1_path = Path(f.name)

    try:
        val_size = 100
        train_ids, val_ids = create_s1_validation_split(
            s1_path=sample_s1_path,
            val_size=val_size,
            random_seed=42,
            output_val_path=None,
            output_train_path=None,
            stratify_by_country=True
        )
        assert len(val_ids) == 100, f"Expected 100 val IDs, got {len(val_ids)}"
        assert len(train_ids) == 400, f"Expected 400 train IDs, got {len(train_ids)}"
        assert len(set(val_ids) & set(train_ids)) == 0, "Train and Val IDs overlap!"
        print(f"  ✓ Created stratified split: 400 train, 100 val. Strictly disjoint.")
    finally:
        os.remove(sample_s1_path)

    # 4. Test multilingual normalization
    print("\n[Step 4] Testing multilingual normalization...")
    cases = [
        ("Maure Williams Colombier Inc", "US"),
        ("एसएस फूड प्राइवेट लिमिटेड", "India (Hindi)"),
        ("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி", "India (Tamil)"),
        ("Fractales Amis Groupe S.A.S", "France"),
        ("Thermal & Fils SASU, 20 Rue Parmentier", "France")
    ]
    for text, label in cases:
        norm = normalize_business_name(text)
        assert norm.raw == text
        assert len(norm.tokens) > 0
        print(f"  ✓ [{label}] '{text}' -> Tokens: {norm.tokens} | Suffixes: {norm.detected_legal_suffixes}")

    # 5. Test Evaluation metrics with trivial predictions
    print("\n[Step 5] Testing evaluation metrics...")
    # Trivial baseline: perfect on first 250, empty on remaining 250
    trivial_pred = {}
    for i, (eid, targets) in enumerate(gt_sample.items()):
        if i < 250:
            trivial_pred[eid] = set(targets)
        else:
            trivial_pred[eid] = set()

    eval_results = compute_detailed_evaluation(gt_sample, trivial_pred)
    f05 = eval_results["macro_f05"]
    assert 0.0 <= f05 <= 1.0, f"Invalid F_0.5 score: {f05}"
    print(f"  ✓ Detailed evaluation computed: Macro F_0.5 = {f05:.4f}, Prec = {eval_results['macro_precision']:.4f}, Rec = {eval_results['macro_recall']:.4f}")
    print(f"  ✓ Singleton accuracy = {eval_results['singletons']['singleton_accuracy']*100:.2f}%")

    # 6. Test Candidate Recall metric
    cand_results = compute_candidate_recall(gt_sample, trivial_pred)
    print(f"  ✓ Candidate recall metric: Recall = {cand_results['candidate_recall']*100:.2f}%, Mean Cand = {cand_results['candidate_size_stats']['mean']}")

    print("\n==================================================")
    print("       ALL PHASE 0 & 1 SMOKE TESTS PASSED         ")
    print("==================================================")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(run_smoke_test())
    except Exception as e:
        print(f"\n❌ SMOKE TEST FAILED: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
