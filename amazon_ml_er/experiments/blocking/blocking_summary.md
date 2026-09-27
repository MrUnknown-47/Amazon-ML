# Phase 2 Blocking & Candidate Generation Experiment Report

**Evaluation Scale:** 50,000 Validation S1 Entities (searched against FULL 10.32M target corpus)  
**Evaluation Runtime:** 377.68 seconds (132.4 queries/sec)  
**Peak Memory Usage:** 4278.52 MB  

---

## 1. Per-Channel Blocking Recall & Candidate Size

| Channel Name | Recall (%) | Captured Matches | Mean Cands/S1 | Median | P95 | P99 | Max | Total Candidates | Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `ch1_core_name` | 50.53% | 87,610 / 173,390 | 39.19 | 4.0 | 194.0 | 500.0 | 500 | 1,959,676 | 0.99999620 |
| `ch2_top1_token` | 28.23% | 48,943 / 173,390 | 405.13 | 500.0 | 500.0 | 500.0 | 500 | 20,256,508 | 0.99996074 |
| `ch2_top2_tokens` | 57.31% | 99,366 / 173,390 | 82.25 | 12.0 | 500.0 | 500.0 | 500 | 4,112,547 | 0.99999203 |
| `ch2_top3_tokens` | 54.13% | 93,850 / 173,390 | 46.08 | 5.0 | 257.0 | 500.0 | 500 | 2,303,819 | 0.99999554 |
| `ch3_street_num_token` | 48.78% | 84,575 / 173,390 | 28.64 | 3.0 | 175.0 | 500.0 | 500 | 1,431,850 | 0.99999723 |
| `ch4_address_location` | 23.58% | 40,878 / 173,390 | 107.58 | 4.0 | 500.0 | 500.0 | 500 | 5,378,886 | 0.99998958 |
| `ch5_posting_char_3gram` | 9.83% | 17,038 / 173,390 | 9.02 | 0.0 | 25.0 | 25.0 | 25 | 450,918 | 0.99999913 |

---

## 2. Unpruned Union Blocking (R_block)

- **R_block (Raw Union Recall):** **81.55%**
- **Total True Matches in Validation Ground Truth:** 173,390
- **Captured True Matches:** 141,402
- **Missed True Matches:** 31,988
- **Total Candidate Pairs Generated:** 11,579,300
- **Mean Candidates / S1:** 231.59
- **Median Candidates / S1:** 81.0
- **P95 Candidates / S1:** 791.0
- **P99 Candidates / S1:** 1037.0
- **Max Candidates / S1:** 1701
- **Reduction Ratio:** 0.99997756

### Recall Attribution & Incrementality

| Channel | Cumulative Recall | Incremental Recall Added | Exclusive Matches Found |
| :--- | :--- | :--- | :--- |
| `ch1_core_name` | 50.53% | +50.53% (87,610) | 0.86% (1,497) |
| `ch2_top2_tokens` | 59.43% | +8.90% (15,433) | 3.66% (6,344) |
| `ch3_street_num_token` | 78.80% | +19.37% (33,579) | 10.65% (18,463) |
| `ch4_address_location` | 80.35% | +1.56% (2,701) | 1.47% (2,550) |
| `ch5_posting_char_3gram` | 81.55% | +1.20% (2,079) | 1.20% (2,079) |

---

## 3. Candidate Pruning Benchmark: R_block vs R_pruned Across Budgets (K)

| Candidate Budget (K) | R_pruned (%) | Lost Matches to Pruning | S1 Entities Truncated | Mean Cands/S1 | P95 | P99 | Reduction Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **K = 10** | **71.86%** | 16,797 | 10,509 | 9.47 | 10.0 | 10.0 | 0.99999908 |
| **K = 15** | **74.15%** | 12,836 | 8,210 | 13.82 | 15.0 | 15.0 | 0.99999866 |
| **K = 25** | **76.70%** | 8,416 | 5,777 | 22.08 | 25.0 | 25.0 | 0.99999786 |
| **K = 40** | **77.65%** | 6,765 | 4,715 | 32.42 | 40.0 | 40.0 | 0.99999686 |
| **K = 60** | **78.48%** | 5,331 | 3,794 | 43.95 | 60.0 | 60.0 | 0.99999574 |
| **K = 100** | **79.93%** | 2,805 | 2,107 | 64.12 | 100.0 | 100.0 | 0.99999379 |

---

## 4. Progressive Scalability Benchmark

| Validation Scale | Runtime (s) | Throughput (queries/s) | Peak Memory (MB) | Total Candidates |
| :--- | :--- | :--- | :--- | :--- |
| 1,000 S1 queries | 8.07s | 123.9 q/s | 4033.58 MB | 217,552 |
| 5,000 S1 queries | 48.06s | 104.0 q/s | 3495.73 MB | 1,152,930 |
| 10,000 S1 queries | 76.64s | 130.5 q/s | 3358.34 MB | 2,302,118 |
| 50,000 S1 queries | 377.68s | 132.4 q/s | 4278.52 MB | 11,579,300 |

---

## 5. Failure Analysis of Missed Matches

- **Total Missed Ground Truth Matches:** 31,988

| Root Cause Category | Count | Percentage | Primary Noise Pattern |
| :--- | :--- | :--- | :--- |
| `missing_target_address` | 24,136 | 75.45% | Sample inspected in JSON |
| `name_typo_or_abbreviation` | 4,810 | 15.04% | Sample inspected in JSON |
| `transliteration_indic_name` | 1,663 | 5.20% | Sample inspected in JSON |
| `missing_street_number` | 636 | 1.99% | Sample inspected in JSON |
| `trade_name_or_dba` | 531 | 1.66% | Sample inspected in JSON |
| `unknown` | 118 | 0.37% | Sample inspected in JSON |
| `word_order_transposition` | 88 | 0.28% | Sample inspected in JSON |
| `legal_suffix_only` | 6 | 0.02% | Sample inspected in JSON |
