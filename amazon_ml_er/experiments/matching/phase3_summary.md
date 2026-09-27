# Phase 3 Summary Report: Final Matching Model & Entity-Level Decisioning

> [!WARNING]
> **SUPERSEDED BY PHASE 3.5 INTEGRITY AUDIT & CONSISTENCY REPAIR REPORT**  
> This Phase 3 document is preserved for historical record. The authoritative evaluation metrics, singleton counts (2,768 true singletons with 98.27% accuracy), exact candidate-pair counts (1,249,258 added pairs between K15 and K40), exact non-overlapping timing sum (2,277.51 seconds), and verified official Macro $F_{0.5}$ (0.97643) are documented in [`phase3_5_consistency_repair.md`](file:///Users/vaibhavsingh/Downloads/student_resource/amazon_ml_er/experiments/matching/phase3_5_consistency_repair.md) and [`phase3_5_integrity_summary.md`](file:///Users/vaibhavsingh/Downloads/student_resource/amazon_ml_er/experiments/matching/phase3_5_integrity_summary.md).

**Project:** Amazon ML Challenge 2026 Business Entity Resolution  
**Evaluation Scope:** Complete 50,000 $S_1$ Validation Holdout (**173,390 true matches**).  
**Zero Validation Leakage:** Strictly observed across all models, scalers, and thresholds.  

---

## 1. Executive Summary & Recommended End-to-End Pipeline

| Pipeline Stage | Component | Downstream Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy | Candidate Volume / $S_1$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Upstream Blocker** | `authoritative_blocker.py` (Config E) | — | — | 96.46% (ceiling) | — | 440.18 |
| **Candidate Compressor** | `CandidateRanker` ($K=40$ policy) | — | — | 96.45% (ceiling) | — | 39.98 |
| **Final Matcher** | `HistGradientBoostingClassifier` (25K S1) | — | — | — | — | — |
| **Target Conflict Resolution** | Greedy Maximum Confidence Assignment | **0.97643** | **0.99061** | **0.94599** | **98.27%** | **3.28** |

---

## 2. Model Comparison on TUNE Split

| Model Architecture | TUNE PR-AUC | TUNE ROC-AUC | TUNE Macro $F_{0.5}$ | Training Time | Scoring Speed | Memory |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (L2, Balanced)** | 0.9963 | 0.9994 | 0.9325 | 2.23s | 350K pairs/s | 3284.9 MB |
| **HistGradientBoostingClassifier (25K)** | **0.9983** | **0.9998** | **0.9454** | 15.01s | 480K pairs/s | 3284.9 MB |

---

## 3. Candidate Policy Co-Optimization on 50,000 Validation Holdout

| Upstream Policy | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Singleton Accuracy | Mean Predicted Matches / $S_1$ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `K_15` | **0.97644** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `K_25` | **0.97643** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `K_40` (Recommended) | **0.97643** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `K_60` | **0.97643** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `K_100` | **0.97643** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `tau_0.10` | **0.97642** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `tau_0.15` | **0.97642** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `tau_0.20` | **0.97638** | 0.99061 | 0.94599 | 98.27% | 3.28 |
| `tau_0.25` | **0.97637** | 0.99061 | 0.94599 | 98.27% | 3.28 |

---

## 4. Target Conflict Resolution Impact

| Resolution Strategy | Macro $F_{0.5}$ | Macro Precision | Macro Recall | Total TP | Total FP | Conflicts Resolved |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Unresolved Baseline** | 0.96663 | 0.98701 | 0.94729 | 163,652 | 2,179 | 0 |
| **Greedy Confidence Assignment** | **0.97643** | **0.99061** | **0.94599** | 163,426 | **389** | **1,513** |

---

## 5. Cardinality Retention Breakdown

| True Cardinality | S1 Entities | Macro $F_{0.5}$ | Pairwise Recall | All Matches Correct Ratio | At Least One Match Ratio |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **0 (Singletons)** | 2,768 | **0.9827** | 100.00% | 98.27% | 98.27% |
| **1** | 2,824 | **0.9378** | 94.62% | 93.73% | 94.62% |
| **2** | 8,327 | **0.9708** | 94.46% | 89.73% | 98.91% |
| **3** | 11,940 | **0.9765** | 94.21% | 86.24% | 99.45% |
| **4** | 11,042 | **0.9814** | 94.48% | 84.03% | 99.88% |
| **5** | 7,301 | **0.9822** | 94.01% | 79.80% | 99.99% |
| **6** | 3,823 | **0.9831** | 93.80% | 76.17% | 99.95% |
| **7+** | 1,975 | **0.9809** | 92.83% | 68.46% | 100.00% |

---

## 6. Source and Country Breakdowns

| Dimension | Slice | True Matches | Captured Matches | Pairwise Precision | Pairwise Recall | Slice Macro $F_{0.5}$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Source** | `source2` | 84,104 | 78,463 | 99.43% | 93.29% | 0.9814 |
| **Source** | `source3` | 89,286 | 83,924 | 99.48% | 94.00% | 0.9834 |
| **Country** | United States (`US`) | 103,602 | 100,286 | — | 96.80% | **0.9869** |
| **Country** | India (`India`) | 69,788 | 62,101 | — | 88.99% | **0.9471** |

---

## 7. Scalability and Runtime Projections

| Metric | Validation Measured (50K S1) | Full Challenge Test (1.73M S1) [PROJECTED] |
| :--- | :---: | :---: |
| **Candidate Pairs Scored** | 4,501,837 pairs | ~69,267,000 pairs |
| **Throughput (Pairs / sec)** | 1,976.65 pairs/sec [MEASURED] | 1,976.65 pairs/sec [PROJECTED] |
| **Throughput (Entities / sec)** | 21.95 entities/sec [MEASURED] | 21.95 entities/sec [PROJECTED] |
| **Runtime (Sequential)** | **37.96 minutes (2,277.51s)** [MEASURED] | **21.92 hours** [PROJECTED] |
| **Runtime (4 Parallel Workers)** | — | **5.48 hours** [PROJECTED] |
| **Peak Memory** | 3,513 MB [MEASURED] | < 4,000 MB [PROJECTED] |
