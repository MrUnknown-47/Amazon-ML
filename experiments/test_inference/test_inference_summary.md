# Phase 4 Test Inference, Output Generation & Pre-Submission Audit

**Project:** Amazon ML Challenge 2026 Business Entity Resolution  
**Current Audit Status:** **PHASE 4 STATUS = HOLD**  
**Active Execution Scope:** Full Test Reference Population (**1,732,544 $S_1$ entities**, **9,969,589 target records**).  
**Frozen Production Configuration:** Config E Blocker + `CandidateRanker` ($K=40$ policy) + `HistGradientBoostingClassifier` + Decision Engine ($\tau=0.70, \Delta=0.10$) + Greedy Maximum-Confidence 1:1 Conflict Resolution.

---

## 1. Pre-Inference Preflight Audit

All challenge test and train files have been audited directly from disk prior to test execution:

| Dataset File | Physical Row Count | Schema / Column Headers | Target ID Prefix Rules | Blank Lines |
| :--- | :---: | :--- | :---: | :---: |
| `dataset/test/test_source1.tsv` | **1,732,544** | `['entity_id', 'business_name', 'business_address', 'country']` | `S1-` (100%) | 0 blank lines |
| `dataset/test/test_source2.tsv` | **4,887,273** | `['entity_id', 'business_name', 'business_address', 'country']` | `S2-` (100%) | 0 blank lines |
| `dataset/test/test_source3.tsv` | **5,082,316** | `['entity_id', 'business_name', 'business_address', 'country']` | `S3-` (100%) | 0 blank lines |
| `dataset/train/train_source1.tsv` | **2,206,821** | `['entity_id', 'business_name', 'business_address', 'country']` | `S1-` (100%) | 0 blank lines |
| `dataset/train/train_source2.tsv` | **5,034,616** | `['entity_id', 'business_name', 'business_address', 'country']` | `S2-` (100%) | 0 blank lines |
| `dataset/train/train_source3.tsv` | **5,285,603** | `['entity_id', 'business_name', 'business_address', 'country']` | `S3-` (100%) | 0 blank lines |
| `dataset/train/train_ground_truth.tsv` | **2,206,821** | `['source1_entity_id', 'matched_entity_ids']` | — | 0 blank lines |

### Key Preflight Verifications:
1. **Uniqueness of Test $S_1$:** Exactly **1,732,544 unique $S_1$ entity IDs** exist in `test_source1.tsv`. Exactly **0 duplicate IDs** exist.
2. **Target ID Integrity & Disjointness:** All 4,887,273 $S_2$ IDs start with `S2-` and all 5,082,316 $S_3$ IDs start with `S3-`. Pairwise intersection $|S_2 \cap S_3| = 0$, and $|S_1 \cap (S_2 \cup S_3)| = 0$.
3. **Open-Set Country Distribution:**
   - `India`: 809,986 $S_1$ (46.75%), 2,312,565 $S_2$, 2,405,000 $S_3$
   - `US`: 663,106 $S_1$ (38.27%), 1,871,330 $S_2$, 1,945,701 $S_3$
   - `France`: 259,452 $S_1$ (14.98%), 703,378 $S_2$, 731,615 $S_3$
   - Handled dynamically as open sets; France flows through the full pipeline normally.
4. **Encoding & Delimiters:** Verified strict tab-separated (`\t`) delimiters and standard UTF-8 encoding across all test sources.
5. **Official Validator Benchmark on Synthetic Output:** The official validator (`utils/validate_submission.py`) was benchmarked:
   - Valid synthetic matching + candidate files: `Exit code 0, PASS — no blocking issues found`.
   - Invalid synthetic files (missing $S_1$, self-matches `S1-`, duplicate target claims, non-tab delimiters): Correctly caught and rejected with detailed error messages.

---

## 2. Production Dependency & License Preflight Audit

The production candidate ranker was nominally referred to in early narrative summaries as a GBDT/LightGBM model. A thorough runtime dependency audit was conducted:

| Package | Installed Version | License | Production Role | Challenge Compliance |
| :--- | :---: | :---: | :--- | :---: |
| **`scikit-learn`** | 1.8.0 | **BSD-3-Clause** | `HistGradientBoostingClassifier` for candidate ranking & pair matching | **PASS** (Permissive) |
| **`numpy`** | 2.4.4 | **BSD-3-Clause** | Array manipulation, vectorized math, and score sorting | **PASS** (Permissive) |
| **`scipy`** | 1.17.1 | **BSD-3-Clause** | Scientific computing utilities and distribution metrics | **PASS** (Permissive) |
| **`joblib`** | 1.5.3 | **BSD-3-Clause** | Model serialization & deserialization | **PASS** (Permissive) |
| **`psutil`** | 7.2.2 | **BSD-3-Clause** | Memory and process resource monitoring | **PASS** (Permissive) |
| **`lightgbm`** | *Not Installed* | *MIT* | *Not imported or executed by production code* | **PASS** (Zero dependency) |

### Candidate Ranker Architecture Fact:
The candidate ranker (`CandidateRanker`) in `amazon_ml_er/code/business_entity_resolution/src/candidate_compression/model.py` directly wraps `sklearn.ensemble.HistGradientBoostingClassifier`. The external `lightgbm` package is neither installed nor imported in the runtime environment. All dependencies are 100% permissive open-source under BSD-3-Clause licenses.

Saved: [`experiments/test_inference/dependency_license_final.json`](file:///Users/vaibhavsingh/Downloads/student_resource/experiments/test_inference/dependency_license_final.json).

---

## 3. Bit-Exact Determinism Audit (5,000 Test Entities)

In accordance with Requirement 10, a fixed deterministic subset of **5,000 test $S_1$ entities** (spanning US, India, and France) was evaluated across two complete, independent execution runs of the frozen pipeline:

- **Run 1 Candidate Pairs SHA-256:** `3d0fa43521f02d9de715c08081cad3b70396c139e174c016297d013d0ded4d47`
- **Run 2 Candidate Pairs SHA-256:** `3d0fa43521f02d9de715c08081cad3b70396c139e174c016297d013d0ded4d47`
- **Run 1 Matching Results SHA-256:** `22ac85d0707a3f4c20563ae3cadd39d51c88f4dbf77c11e00e64e1b6702bfeca`
- **Run 2 Matching Results SHA-256:** `22ac85d0707a3f4c20563ae3cadd39d51c88f4dbf77c11e00e64e1b6702bfeca`

**Determinism Verification:** **PASS**. Candidate pairs and matching results produced bit-exact identical byte streams and identical SHA-256 hashes across repeated runs. Zero candidate containment violations detected.

Saved: [`experiments/test_inference/determinism_check.json`](file:///Users/vaibhavsingh/Downloads/student_resource/experiments/test_inference/determinism_check.json).

---

## 4. Exact Candidate-Set Semantics Proof Across 34 Shards

The production engine enforces that `candidate_pairs.tsv` represents the exact final candidate set entering the final matcher. Verified directly across all **34 completed shards (`France_0000` to `France_0033`, 170,000 entities)**:

- **Reference Entities Evaluated:** **170,000 entities**
- **Total Raw Blocker Candidates:** **220,735,333** (1,298.4 raw candidates/$S_1$)
- **Total Compressed $K=40$ Candidates:** **6,797,925** (39.99 candidates/$S_1$)
- **Total Pre-Conflict Predicted Claims:** **2,093,837** (12.32 claims/$S_1$)
- **Candidate Containment Violations:** **0**

$$\forall i \in \{1, \dots, 170000\}, \quad \text{Matched}(S_1^{(i)}) \subseteq \text{Candidates}_{K=40}(S_1^{(i)}) \equiv \text{MatcherInput}(S_1^{(i)}).$$

---

## 5. Measured Execution Progress & Performance Report

| Progress Metric | France Completed Shards | Validation (50K) [MEASURED] | Full Test Set [PROJECTED] |
| :--- | :---: | :---: | :---: |
| **Completed Shards** | **34 / 52 Chunks (65.5%)** | — | **34 / 174 Chunks (19.5%)** |
| **Entities Evaluated** | **170,000 entities** | 50,000 | 1,732,544 |
| **Raw Blocker Candidates** | **220,735,333** | 20,900,467 | ~838,000,000 |
| **Mean Raw Candidates / $S_1$** | **1,298.4** | 440.2 | ~484.0 |
| **$K=40$ Candidates Scored** | **6,797,925** | 1,999,125 | ~69,267,000 |
| **Mean $K=40$ Candidates / $S_1$** | **39.99** | 39.98 | ~39.98 |
| **Pre-Conflict Claims** | **2,093,837** | 163,652 | ~7,100,000 |
| **Elapsed Shard Runtime** | **31,934.9s (8.87 hours)** | 2,277.51s (37.96 min) | — |
| **Throughput (Entities / sec)** | **5.32 ent/s (sustained)** | **21.95 ent/s** | **15.0 – 22.0 ent/s** |
| **Peak Memory (RAM)** | **1,833.2 MB** | 3,513.9 MB | < 8,000 MB |
| **Active Background Task** | `task-2127` (processing Chunk 34+) | — | — |

---

## 6. Hard Stop Condition & Phase 4 Status Declaration

In accordance with Requirement 13:
> *"Do NOT package the final submission if any of these fail: incomplete shards; test inference does not cover all 1,732,544 S1 entities... At the very end report: PHASE 4 STATUS = PASS only if all full-test inference, output, determinism, dependency/license, and official-validator checks pass. Otherwise report: PHASE 4 STATUS = HOLD."*

```
====================================================================================================
                                 PHASE 4 STATUS = HOLD
====================================================================================================
  1. Pre-inference preflight verified:              PASS (All schemas, 1.73M unique S1, open-set)
  2. Production dependencies & licenses audited:    PASS (All BSD-3-Clause, zero copyleft/external)
  3. Resumable chunked architecture implemented:    PASS (Atomic writes, config-hash verification)
  4. 5,000-entity determinism check:                PASS (Bit-exact SHA-256 match, 0 violations)
  5. Exact candidate-set semantics proven:          PASS (0 violations across 170,000 entities)
  6. Shards completed & verified on disk:           PASS (34 chunks / 170,000 entities, 65.5% France)
  7. Full 1,732,544 test coverage:                  IN_PROGRESS (Task 2127 actively executing)
  8. Output assembly & official validator:          PENDING FULL TEST SHARD COMPLETION
====================================================================================================
```

**PHASE 4 STATUS = HOLD**
