# Business Entity Resolution: Empirical Data & Schema Audit Report

**Audit Execution Timestamp:** 2026-09-25 18:08:55 UTC  
**Total Audit Duration:** 81.42 seconds  
**Peak Memory Usage:** 954.75 MB (Delta: 923.66 MB)  

---

## 1. Source TSV Overview & Schema Conformance

| Source | Total Rows | Unique IDs | Duplicate IDs | Null Names | Null Addresses (%) | Countries Found |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1` | 2,206,821 | 2,206,821 | 0 | 0 | 0 (0.00%) | US (1,323,633), India (883,188) |
| `train_source2` | 5,034,616 | 5,034,616 | 0 | 0 | 168,967 (3.36%) | India (2,017,799), US (3,016,817) |
| `train_source3` | 5,285,603 | 5,285,603 | 0 | 0 | 175,916 (3.33%) | US (3,170,056), India (2,115,547) |
| `test_source1` | 1,732,544 | 1,732,544 | 0 | 0 | 0 (0.00%) | US (663,106), France (259,452), India (809,986) |
| `test_source2` | 4,887,273 | 4,887,273 | 0 | 0 | 129,408 (2.65%) | India (2,312,565), France (703,378), US (1,871,330) |
| `test_source3` | 5,082,316 | 5,082,316 | 0 | 0 | 136,098 (2.68%) | India (2,405,000), France (731,615), US (1,945,701) |

---

## 2. Text Length Statistics (Characters)

| Source | Field | Min | Median | Mean | P95 | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `train_source1` | Name | 3 | 24.0 | 24.03 | 37.0 | 105 |
| `train_source1` | Address | 11 | 41.0 | 52.07 | 103.0 | 256 |
| `train_source2` | Name | 2 | 25.0 | 25.1 | 40.0 | 104 |
| `train_source2` | Address | 0 | 37.0 | 46.23 | 96.0 | 249 |
| `train_source3` | Name | 2 | 25.0 | 25.2 | 42.0 | 123 |
| `train_source3` | Address | 0 | 42.0 | 46.71 | 91.0 | 240 |
| `test_source1` | Name | 3 | 24.0 | 23.84 | 36.0 | 92 |
| `test_source1` | Address | 11 | 50.0 | 57.21 | 105.0 | 268 |
| `test_source2` | Name | 2 | 25.0 | 25.7 | 42.0 | 102 |
| `test_source2` | Address | 0 | 43.0 | 50.41 | 99.0 | 269 |
| `test_source3` | Name | 2 | 25.0 | 25.66 | 42.0 | 103 |
| `test_source3` | Address | 0 | 43.0 | 48.74 | 94.0 | 267 |

---

## 3. Ground Truth Analysis & Cardinality

- **Total S1 Entities in Ground Truth:** 2,206,821
- **Singletons (0 matches):** 123,247 (5.5848%)
- **Non-Singletons (>= 1 match):** 2,083,574 (94.4152%)
- **Matches in S2 Only:** 143,029 (6.48%)
- **Matches in S3 Only:** 164,498 (7.45%)
- **Matches in Both S2 & S3:** 1,776,047 (80.48%)
- **Unique Target IDs Matched:** 7,638,365
- **Target IDs Mapping to Multiple S1s:** 0 (0.0000%)

### Matches Per S1 Distribution
- **Min:** 0
- **P25:** 2.0
- **Median:** 3.0
- **Mean:** 3.4613
- **P75:** 5.0
- **P95:** 6.0
- **P99:** 8.0
- **Max:** 11

### Country Breakdown in Ground Truth

- **India:** Total S1 = 883,188 | Singletons = 49,351 (5.59%) | Total Matches = 3,059,843 | Avg Matches/S1 = 3.46
- **US:** Total S1 = 1,323,633 | Singletons = 73,896 (5.58%) | Total Matches = 4,578,522 | Avg Matches/S1 = 3.46
