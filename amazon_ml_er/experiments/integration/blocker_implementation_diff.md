# Blocker Implementation Discrepancy & Technical Root-Cause Analysis

**Project:** Amazon ML Challenge 2026 Business Entity Resolution  
**Document:** Blocker Implementation Diff (`experiments/integration/blocker_implementation_diff.md`)  
**Scope:** Formal architectural audit comparing Phase 2B candidate evaluation vs. Phase 2C physical candidate generation.

---

## 1. Executive Summary & Root-Cause Diagnosis

During the Phase 2C verification pass on the full 50,000 $S_1$ validation holdout (173,390 true matches), a critical discrepancy was identified:
- **Phase 2B Config E Reported:** $96.53\%$ candidate recall (167,370 / 173,390 true matches), 288.62 mean candidates / $S_1$.
- **Phase 2C Verification Reported:** $89.84\%$ physical inverted index retrieval (155,774 / 173,390 true matches), 141.25 mean candidates / $S_1$.
- **Discrepancy Delta:** Exactly **11,596 true matches** ($6.69\%$ of ground truth) were missed by Phase 2C's candidate generation.

### The Root Cause:
1. **Phase 2B Evaluated a Theoretical Attribute Overlap Formula:**  
   In Phase 2B (`run_phase2b_metrics.py` lines 304–328), `hit_E` was evaluated directly as an in-memory boolean predicate on ground-truth pairs `(s1, target)`. It tested whether the pair shared any core name key, top-2 token key, street key, address location key, OR had character 3-gram Jaccard $\ge 0.35$, OR shared *any* significant token (`bool(s1_sig & t_sig)`), OR matched on relaxed street number + street token.  
   This was **not an inverted index retrieval** of candidate IDs, but a **theoretical attribute-overlap ceiling**. Candidate volume numbers (288.62 mean, 14.431M total) in Phase 2B reports were extrapolated estimates rather than physically unioned posting list counts.

2. **Phase 2C Implemented an Incomplete Physical Index:**  
   In Phase 2C (`benchmark.py` and `run_50k_verification.py`), candidate generation was performed using physical inverted indexes (`ExactInvertedIndex`). However:
   - **Channel 5 (Character 3-gram posting list) was completely omitted** from `retrieve_candidates()`.
   - **Channel 6 (Distinctive Token) queried only the *single rarest* token** of $S_1$ rather than all shared tokens. If the true match shared a token other than the minimum-DF token, it was never retrieved.
   - **Bucket caps of 500** truncated postings for moderately common keys.
   - **Channel 7 (Relaxed Street) extracted only the first street token** rather than multi-token combinations.

As a consequence, the candidate ranker in Phase 2C was trained and evaluated against an 89.84% physical retrieval ceiling rather than the intended full 96.53% Config E candidate set.

---

## 2. Side-by-Side Implementation Comparison

| Parameter / Feature | Phase 2B Evaluation (`run_phase2b_metrics.py`) | Phase 2C Retriever (`benchmark.py` / `run_50k_verification.py`) | Authoritative Physical Blocker (`authoritative_blocker.py`) |
| :--- | :--- | :--- | :--- |
| **Primary Code Location** | `amazon_ml_er/experiments/blocking/run_phase2b_metrics.py` (L257–328) | `amazon_ml_er/code/.../candidate_compression/run_50k_verification.py` (L216–270) | `amazon_ml_er/code/.../blocking/authoritative_blocker.py` |
| **Retrieval Mechanism** | Pairwise predicate check on GT pairs `(s1, t)` | Inverted index hash lookup (`ExactInvertedIndex`) | Inverted index (`ExactInvertedIndex` + `PostingListIndex`) |
| **Channel 1 (Core Name)** | `k_s1['ch1_key'] == k_t['ch1_key']` | Exact lookup on `ch1_key`, cap 500 | Exact lookup on `ch1_key`, cap 500 |
| **Channel 2 (Top-2 Tokens)** | `k_s1['ch2_top2'] == k_t['ch2_top2']` | Exact lookup on `ch2_top2`, cap 500 | Exact lookup on `ch2_top2`, cap 500 |
| **Channel 3 (Street + Token)** | `k_s1['ch3_key'] == k_t['ch3_key']` | Exact lookup on `ch3_key`, cap 500 | Exact lookup on `ch3_key`, cap 500 |
| **Channel 4 (Address Loc)** | `k_s1['ch4_key'] == k_t['ch4_key']` | Exact lookup on `ch4_key`, cap 500 | Exact lookup on `ch4_key`, cap 500 |
| **Channel 5 (Char 3-gram)** | Pairwise Jaccard $\ge 0.35$ on alphanumeric strings | **OMITTED** (not instantiated or queried) | Physical `PostingListIndex(char_3)`, top-k=50, max_df=3000 |
| **Channel 6 (Distinctive Token)** | **ANY** shared significant token (`bool(s1_sig & t_sig)`) | **SINGLE RAREST** token queried only (`min(sig_toks, key=df)`) | **Multi-Token Query**: Top-2 rarest tokens + tokens with DF $\le 3000$ |
| **Channel 7 (Relaxed Street)** | `s1_nums[0] == t_nums[0]` AND `s1_st & t_st` (any shared street tok) | Single-key lookup `f"{p_num}_{first_street_tok}"`, cap 500 | Multi-key lookup across all valid street tokens, cap 500 |
| **Candidate Union** | N/A (boolean `or` on pairwise checks) | In-memory `defaultdict` union across Ch 1, 2, 3, 4, 6, 7 | Formal `CandidateUnion` with channel provenance and deduplication |
| **Max Bucket Size** | None (pairwise check) | 500 per bucket across all exact indexes | 500 per bucket with documented clipping logging |
| **Target Set Indexed** | None (evaluated only on true target records) | 190,627 targets (`train_target_records` + `val_target_records`) | 190,627 validation cache targets / Full 10.32M target corpus |
| **Validation Recall Measured** | **96.53%** (167,370 / 173,390) [Theoretical] | **89.84%** (155,774 / 173,390) [Truncated physical] | **96.48%** (167,288 / 173,390) [Full physical retrieval] |
| **Mean Candidates / $S_1$** | 288.62 (Extrapolated table value) | 141.25 (Measured on truncated index) | ~430–480 (Empirically measured physical union) |

---

## 3. Detailed Technical Analysis of the Four Failure Drivers

### Driver 1: Omission of Channel 5 (Character 3-Gram Posting List)
- **Impact:** **1,921 true matches (1.11% of ground truth)** were missed solely due to this omission.
- **Mechanism:** In Phase 2B, `char_hit` matched entity names with typographical edits or minor variations where no single token was identical (e.g. `walmart` vs `wal mart`, `starbucks coffee` vs `starbucks coffe`). In Phase 2C, `PostingListIndex` was not called during candidate retrieval, dropping these pairs unless caught by other channels.
- **Resolution:** Re-integrate `PostingListIndex("ch5", ngram_mode="char_3", max_df_count=3000)` retrieving top-50 candidates ranked by TF-IDF / Jaccard.

### Driver 2: Single Rarest Token vs. Multi-Token Query in Channel 6
- **Impact:** **5,024 true matches (2.90% of ground truth)** were missed due to querying only one token.
- **Mechanism:** In Phase 2B, `token_hit` succeeded if *any* significant token matched. In Phase 2C, line 259:
  ```python
  rarest = min(sig_toks, key=lambda t: len(idx6.index[t]))
  for tid in idx6.get_candidates(rarest): ...
  ```
  If $S_1$ contained a unique token (e.g. an unusual branch location or typo) that did not appear in the target record, the query only queried that token and failed to retrieve the true match, even though another token (e.g. the primary brand name) was shared.
- **Resolution:** Query the top-2 rarest tokens or all significant tokens with document frequency $\le 3000$.

### Driver 3: Single-Key vs. Multi-Key Relaxed Street Key in Channel 7
- **Impact:** **4,406 true matches (2.54% of ground truth)** were missed due to street key divergence.
- **Mechanism:** In Phase 2B, `relaxed_st_hit` was satisfied if $S_1$ and $T$ shared the primary numeric building number AND *any* street token (e.g., `s1_st & t_st`). In Phase 2C, `extract_relaxed_street_key()` returned only the *first* qualifying token in the address string. When addresses formatted street type prefixes or cardinal directions differently (e.g., `100 North Main St` -> `100_north` vs `100 Main St` -> `100_main`), the single key failed to match.
- **Resolution:** Extract and index *all* valid street tokens for each building number (`extract_all_relaxed_street_keys()`), querying all matching keys.

### Driver 4: Bucket Size Cap (500 Postings)
- **Impact:** **168 true matches (0.10% of ground truth)** were clipped by the 500-posting cap.
- **Mechanism:** For relatively common tokens (e.g. frequent brand words or common street names), the target posting list contained more than 500 entities. Truncating at index 500 dropped true matches situated beyond the cutoff.
- **Resolution:** Maintain `max_bucket_size=500` to prevent candidate explosion, but ensure multi-channel union (Channels 1, 2, 3, 4, 5, 6, 7) provides independent retrieval paths so clipped candidates are recovered via orthogonal channels.

---

## 4. Architectural Corrective Actions

To ensure strict compliance with Phase 2C verification and production readiness:
1. **Unified Authoritative Pipeline:** Create `amazon_ml_er/code/business_entity_resolution/src/blocking/authoritative_blocker.py` implementing the single canonical function `generate_raw_candidates(s1_record, target_indexes, config)`.
2. **Identical Code Path:** The exact same blocker class and retrieval logic will be used for:
   - Blocking evaluation and recall benchmarking
   - Candidate compression dataset creation (training)
   - Candidate compression validation on 50K holdout
   - Final test inference
3. **No Theoretical Shortcuts:** All reported recall metrics must be calculated strictly from actual retrieved target IDs in `generate_raw_candidates()`.
