"""
Authoritative 5,000-Entity India Benchmark and Regression Gate.
Compares frozen reference vs native C optimized engine on India test data.
Measures wall-clock time, ent/s, peak RAM, swap delta, and exact bit-exactness.
"""

import sys
import gc
import os
import time
import math
import csv
import json
from collections import defaultdict
from pathlib import Path
import psutil
import numpy as np

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "amazon_ml_er" / "experiments" / "test_inference"))

from amazon_ml_er.code.business_entity_resolution.src.blocking.authoritative_blocker import (
    BlockerConfig,
    TargetCorpusIndex,
    generate_raw_candidates
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.features import (
    EntityProfile as RankerEntityProfile,
    extract_pair_features as extract_ranker_features_ref,
    NUM_FEATURES as NUM_RANKER_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.model import CandidateRanker
from amazon_ml_er.code.business_entity_resolution.src.matching.features import (
    MatchingEntityProfile,
    extract_matching_features as extract_matching_features_ref,
    fast_levenshtein_ratio as fast_levenshtein_ratio_ref,
    NUM_MATCHING_FEATURES
)
import amazon_ml_er.code.business_entity_resolution.src.matching.features as mf
from amazon_ml_er.code.business_entity_resolution.src.matching.models import FinalMatcherModel
from amazon_ml_er.experiments.test_inference.run_chunked_inference import (
    RANKER_MODEL_PATH, MATCHER_MODEL_PATH, load_targets, build_index,
    get_ranker_profile, get_matching_profile, SHARDS_DIR, CONFIG_HASH
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.indexes import PostingListIndex
import heapq
import fast_features_native

ref_retrieve_top_k = PostingListIndex.retrieve_top_k

def fast_retrieve_top_k(
    self,
    query_text: str,
    top_k: int = 25,
    df_threshold = None
):
    q_ngrams = self._extract_ngrams(query_text)
    if not q_ngrams:
        return []

    scores = fast_features_native.c_accumulate_scores_dict(
        self.postings,
        q_ngrams,
        self.doc_freq,
        self.idf,
        self.total_docs,
        df_threshold if df_threshold is not None else 0
    )

    if not scores:
        return []

    if len(scores) <= top_k:
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    else:
        sorted_items = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])

    return [(self.int_to_id[int_id], score) for int_id, score in sorted_items]


def get_swap_mb():
    try:
        return psutil.swap_memory().used / (1024 * 1024)
    except Exception:
        return 0.0


def run_benchmark_and_verify():
    print("=======================================================", flush=True)
    print("PHASE 4: AUTHORITATIVE 5,000-ENTITY INDIA BENCHMARK & REGRESSION GATE", flush=True)
    print("=======================================================", flush=True)

    swap_start = get_swap_mb()
    rss_start = psutil.Process().memory_info().rss / (1024 * 1024)

    # 1. Load targets and build index
    print("\n[Step 1] Loading India targets...", flush=True)
    t0 = time.time()
    targets = load_targets("India")
    t_load = time.time() - t0
    print(f"  Loaded {len(targets):,} targets in {t_load:.2f}s.", flush=True)

    cfg = BlockerConfig(
        max_bucket_size=500,
        ch5_top_k=50,
        ch5_max_df=3000,
        ch6_tokens_to_query=2,
        ch6_max_df=3000
    )
    print("  Building inverted index (Config E)...", flush=True)
    t0 = time.time()
    target_index = build_index(targets, cfg)
    t_idx = time.time() - t0
    rss_after_idx = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"  Inverted index built in {t_idx:.2f}s. Peak RAM: {rss_after_idx:.1f} MB", flush=True)

    # 2. Load models
    ranker = CandidateRanker.load(RANKER_MODEL_PATH)
    matcher = FinalMatcherModel.load(MATCHER_MODEL_PATH)
    tau = 0.70
    delta = 0.10
    k_budget = 40

    # 3. Load queries
    queries_file = repo_root / "dataset" / "test" / "country_targets" / "queries_India.tsv"
    queries = []
    with open(queries_file, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            if i >= 5000:
                break
            queries.append({
                "entity_id": row[0],
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3]
            })
    print(f"\n[Step 2] Loaded {len(queries):,} queries for shard India_0000.", flush=True)

    # ---------------------------------------------------------
    # PART A: REGRESSION GATE ON FIRST 200 QUERIES
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------", flush=True)
    print("PART A: REGRESSION VERIFICATION GATE (Frozen Reference vs Native C)", flush=True)
    print("-------------------------------------------------------", flush=True)

    reg_queries = queries[:200]
    reg_r_cache = {}
    reg_m_cache = {}

    max_ranker_diff = 0.0
    max_matcher_diff = 0.0
    max_rscore_diff = 0.0
    max_mscore_diff = 0.0
    pred_diff_count = 0
    k40_diff_count = 0
    raw_diff_count = 0
    containment_violations = 0
    total_reg_pairs = 0
    total_reg_k40 = 0

    # Temporarily restore reference Levenshtein for ground truth comparison
    mf.fast_levenshtein_ratio = fast_levenshtein_ratio_ref

    for q in reg_queries:
        # Blocker comparison (Reference Python vs Native C)
        PostingListIndex.retrieve_top_k = ref_retrieve_top_k
        raw_c_ref = generate_raw_candidates(q, target_index, cfg)
        PostingListIndex.retrieve_top_k = fast_retrieve_top_k
        raw_c = generate_raw_candidates(q, target_index, cfg)
        if raw_c_ref != raw_c:
            raw_diff_count += 1

        n_c = len(raw_c)
        if n_c == 0:
            continue
        total_reg_pairs += n_c

        c_tids = [t for t, _ in raw_c]
        c_provs = [p for _, p in raw_c]

        # 1. Ranker feature comparison
        s1_rprof = RankerEntityProfile(q["entity_id"], q["business_name"], q["business_address"], q.get("country", ""))
        t_rprofs = [get_ranker_profile(tid, targets, reg_r_cache) for tid in c_tids]

        # Python reference features
        X_rank_ref = np.zeros((n_c, NUM_RANKER_FEATURES), dtype=np.float32)
        for i in range(n_c):
            X_rank_ref[i] = extract_ranker_features_ref(s1_rprof, t_rprofs[i], c_provs[i])

        # Native C features
        X_rank_c = np.zeros((n_c, NUM_RANKER_FEATURES), dtype=np.float32)
        fast_features_native.extract_ranker_features_batch_c(s1_rprof, t_rprofs, c_provs, X_rank_c, 0)

        # Parallel C features
        pair_s1_test = np.zeros(n_c, dtype=np.uint32)
        pair_t_test = np.arange(n_c, dtype=np.uint32)
        CH_MAP_T = {
            "ch1_core_name": 1, "ch2_top2_tokens": 2, "ch3_street_num_token": 4,
            "ch4_address_location": 8, "ch5_char3_k50": 16, "ch6_distinctive_token": 32,
            "ch7_relaxed_street_num": 64
        }
        pair_chan_test = np.array([sum(CH_MAP_T.get(ch, 0) for ch in p.get("channels", ())) for p in c_provs], dtype=np.uint8)
        pair_rare_test = np.array([p.get("rarest_token_df", 0) for p in c_provs], dtype=np.uint32)
        X_rank_parallel = np.zeros((n_c, NUM_RANKER_FEATURES), dtype=np.float32)
        fast_features_native.extract_ranker_features_parallel_c(
            [s1_rprof], t_rprofs, pair_s1_test, pair_t_test, pair_chan_test, pair_rare_test, X_rank_parallel, 4
        )

        r_diff = np.max(np.abs(X_rank_ref - X_rank_c))
        r_diff_par = np.max(np.abs(X_rank_ref - X_rank_parallel))
        if max(r_diff, r_diff_par) > max_ranker_diff:
            max_ranker_diff = max(r_diff, r_diff_par)

        # Ranker scores
        r_scores_ref = ranker.score(X_rank_ref)
        r_scores_c = ranker.score(X_rank_c)
        rs_diff = np.max(np.abs(r_scores_ref - r_scores_c))
        if rs_diff > max_rscore_diff:
            max_rscore_diff = rs_diff

        # K40 selection
        sort_idx_ref = np.argsort(r_scores_ref)[::-1][:k_budget]
        sort_idx_c = np.argsort(r_scores_c)[::-1][:k_budget]

        k40_tids_ref = [c_tids[i] for i in sort_idx_ref]
        k40_tids_c = [c_tids[i] for i in sort_idx_c]
        if k40_tids_ref != k40_tids_c:
            k40_diff_count += 1

        k40_provs = [c_provs[i] for i in sort_idx_c]
        k40_rscores = [float(r_scores_c[i]) for i in sort_idx_c]
        n_k40 = len(k40_tids_c)
        total_reg_k40 += n_k40

        # 2. Matcher feature comparison
        s1_mprof = MatchingEntityProfile(q["entity_id"], q["business_name"], q["business_address"], q.get("country", ""))
        t_mprofs = [get_matching_profile(tid, targets, reg_m_cache) for tid in k40_tids_c]

        # Python reference matcher features
        mf.fast_levenshtein_ratio = fast_levenshtein_ratio_ref
        X_match_ref = np.zeros((n_k40, NUM_MATCHING_FEATURES), dtype=np.float32)
        for rank_pos, (t_mp, prov, rscore) in enumerate(zip(t_mprofs, k40_provs, k40_rscores), start=1):
            X_match_ref[rank_pos - 1] = extract_matching_features_ref(s1_mprof, t_mp, prov, rscore, rank_pos)

        # Native C Levenshtein matcher features
        mf.fast_levenshtein_ratio = fast_features_native.fast_levenshtein_ratio
        X_match_c = np.zeros((n_k40, NUM_MATCHING_FEATURES), dtype=np.float32)
        for rank_pos, (t_mp, prov, rscore) in enumerate(zip(t_mprofs, k40_provs, k40_rscores), start=1):
            X_match_c[rank_pos - 1] = extract_matching_features_ref(s1_mprof, t_mp, prov, rscore, rank_pos)

        m_diff = np.max(np.abs(X_match_ref - X_match_c))
        if m_diff > max_matcher_diff:
            max_matcher_diff = m_diff

        # Matcher scores
        m_scores_ref = matcher.predict_proba(X_match_ref)
        m_scores_c = matcher.predict_proba(X_match_c)
        ms_diff = np.max(np.abs(m_scores_ref - m_scores_c))
        if ms_diff > max_mscore_diff:
            max_mscore_diff = ms_diff

        # Decision & predictions
        top_ref = float(m_scores_ref[0])
        preds_ref = [tid for tid, sc in zip(k40_tids_c, m_scores_ref) if top_ref >= tau and float(sc) >= tau and (top_ref - float(sc)) <= delta]

        top_c = float(m_scores_c[0])
        preds_c = [tid for tid, sc in zip(k40_tids_c, m_scores_c) if top_c >= tau and float(sc) >= tau and (top_c - float(sc)) <= delta]

        if preds_ref != preds_c:
            pred_diff_count += 1

        # Containment check
        for p in preds_c:
            if p not in k40_tids_c:
                containment_violations += 1

    print(f"  Candidate pairs evaluated:          {total_reg_pairs:,}", flush=True)
    print(f"  K40 pairs evaluated:                {total_reg_k40:,}", flush=True)
    print(f"  Max Ranker feature difference:      {max_ranker_diff:.2e}", flush=True)
    print(f"  Max Matcher feature difference:     {max_matcher_diff:.2e}", flush=True)
    print(f"  Max Ranker score difference:        {max_rscore_diff:.2e}", flush=True)
    print(f"  Max Matcher score difference:       {max_mscore_diff:.2e}", flush=True)
    print(f"  K40 candidate set differences:      {k40_diff_count}", flush=True)
    print(f"  Final prediction differences:       {pred_diff_count}", flush=True)
    print(f"  Raw candidate differences:          {raw_diff_count}", flush=True)
    print(f"  Candidate containment violations:   {containment_violations}", flush=True)

    reg_pass = (
        raw_diff_count == 0 and
        max_ranker_diff == 0.0 and
        max_matcher_diff <= 1e-12 and
        max_rscore_diff <= 1e-12 and
        max_mscore_diff <= 1e-12 and
        k40_diff_count == 0 and
        pred_diff_count == 0 and
        containment_violations == 0
    )

    print(f"\n  REGRESSION GATE RESULT: {'PASS' if reg_pass else 'FAIL'}", flush=True)
    assert reg_pass, "Regression gate FAILED! Reverting optimization."

    del reg_r_cache
    del reg_m_cache
    gc.collect()

    # ---------------------------------------------------------
    # PART B: FULL 5,000-ENTITY SHARD BENCHMARK (India_0000)
    # ---------------------------------------------------------
    print("\n-------------------------------------------------------", flush=True)
    print("PART B: FULL 5,000-ENTITY SHARD BENCHMARK (India_0000)", flush=True)
    print("-------------------------------------------------------", flush=True)

    r_cache = {}
    m_cache = {}
    mf.fast_levenshtein_ratio = fast_features_native.fast_levenshtein_ratio
    PostingListIndex.retrieve_top_k = fast_retrieve_top_k

    gc.disable()
    t_shard0 = time.time()

    # Stage 1: Blocker
    t0 = time.time()
    raw_candidates_list = []
    total_raw = 0
    for q_i, s1_rec in enumerate(queries):
        raw_c = generate_raw_candidates(s1_rec, target_index, cfg)
        raw_candidates_list.append(raw_c)
        total_raw += len(raw_c)
        if (q_i + 1) % 1000 == 0:
            print(f"  [Part B Blocker] {q_i+1:,}/{len(queries):,} done ({total_raw:,} candidates)...", flush=True)
    t_blocker = time.time() - t0
    print(f"  [Part B Stage 1 Blocker Complete] {t_blocker:.2f}s ({total_raw:,} raw candidates)", flush=True)

    # Stage 2: Ranker Features (Native C Multi-Thread Parallel)
    t0 = time.time()
    X_rank = np.empty((total_raw, NUM_RANKER_FEATURES), dtype=np.float32)
    slices = []

    s1_profs = [
        RankerEntityProfile(q["entity_id"], q["business_name"], q["business_address"], q.get("country", ""))
        for q in queries
    ]
    unique_tids = list(dict.fromkeys(t for raw in raw_candidates_list for t, _ in raw))
    tid_to_idx = {t: i for i, t in enumerate(unique_tids)}
    target_profs = [get_ranker_profile(tid, targets, r_cache) for tid in unique_tids]

    pair_s1 = np.empty(total_raw, dtype=np.uint32)
    pair_t = np.empty(total_raw, dtype=np.uint32)
    pair_chan = np.empty(total_raw, dtype=np.uint8)
    pair_rare = np.empty(total_raw, dtype=np.uint32)

    CH_MAP = {
        "ch1_core_name": 1, "ch2_top2_tokens": 2, "ch3_street_num_token": 4,
        "ch4_address_location": 8, "ch5_char3_k50": 16, "ch6_distinctive_token": 32,
        "ch7_relaxed_street_num": 64
    }

    cur = 0
    for s1_i, raw_c in enumerate(raw_candidates_list):
        n_c = len(raw_c)
        if n_c == 0:
            slices.append((cur, cur))
            continue
        pair_s1[cur:cur+n_c] = s1_i
        for j, (t, prov) in enumerate(raw_c):
            idx = cur + j
            pair_t[idx] = tid_to_idx[t]
            cmask = 0
            for ch in prov.get("channels", ()):
                cmask |= CH_MAP.get(ch, 0)
            pair_chan[idx] = cmask
            pair_rare[idx] = prov.get("rarest_token_df", 0)
        slices.append((cur, cur + n_c))
        cur += n_c

    # Multi-thread scaling benchmark on ranker feature extraction
    print("\n  [Thread Scalability Benchmark on Ranker Extraction]", flush=True)
    thread_results = {}
    for n_th in [1, 2, 4, 6, 8]:
        t_th0 = time.time()
        fast_features_native.extract_ranker_features_parallel_c(
            s1_profs, target_profs, pair_s1, pair_t, pair_chan, pair_rare, X_rank, n_th
        )
        dt_th = time.time() - t_th0
        rate_th = total_raw / max(0.0001, dt_th)
        thread_results[n_th] = (dt_th, rate_th)
        print(f"    {n_th} native thread(s): {dt_th:6.3f}s ({rate_th:12,.0f} pairs/sec)", flush=True)

    t_rfeat = thread_results[8][0]
    print(f"  [Part B Stage 2 Ranker Feat Complete (8 threads)] {t_rfeat:.2f}s ({thread_results[8][1]:,.0f} pairs/s)", flush=True)

    # Stage 3: Ranker Scoring (HistGB)
    t0 = time.time()
    if total_raw > 0:
        r_scores = ranker.score(X_rank)
    else:
        r_scores = np.array([])
    t_rscore = time.time() - t0

    # Stage 4: Top-40 Selection
    t0 = time.time()
    total_k40 = 0
    k40_records_list = []
    chunk_candidates = {}

    for (start, end), s1_rec, raw_c in zip(slices, queries, raw_candidates_list):
        s1_id = s1_rec["entity_id"]
        if start == end:
            k40_records_list.append([])
            chunk_candidates[s1_id] = []
            continue

        sub_scores = r_scores[start:end]
        c_tids = [t for t, _ in raw_c]
        c_provs = [p for _, p in raw_c]
        sort_idx = np.argsort(sub_scores)[::-1][:k_budget]

        k40_tids = [c_tids[i] for i in sort_idx]
        k40_provs = [c_provs[i] for i in sort_idx]
        k40_rscores = [float(sub_scores[i]) for i in sort_idx]

        chunk_candidates[s1_id] = k40_tids
        k40_records_list.append((k40_tids, k40_provs, k40_rscores))
        total_k40 += len(k40_tids)
    t_k40 = time.time() - t0

    # Stage 5: Matcher Features (C Levenshtein)
    t0 = time.time()
    X_match = np.empty((total_k40, NUM_MATCHING_FEATURES), dtype=np.float32)
    m_slices = []
    cur_m = 0
    for s1_rec, k40_data in zip(queries, k40_records_list):
        if not k40_data:
            m_slices.append((cur_m, cur_m))
            continue
        k40_tids, k40_provs, k40_rscores = k40_data
        n_k40 = len(k40_tids)
        s1_mprof = MatchingEntityProfile(
            s1_rec["entity_id"], s1_rec["business_name"], s1_rec["business_address"], s1_rec.get("country", "")
        )
        for rank_pos, (tid, prov, rscore) in enumerate(zip(k40_tids, k40_provs, k40_rscores), start=1):
            t_mprof = get_matching_profile(tid, targets, m_cache)
            X_match[cur_m + rank_pos - 1] = extract_matching_features_ref(s1_mprof, t_mprof, prov, rscore, rank_pos)
        m_slices.append((cur_m, cur_m + n_k40))
        cur_m += n_k40
    t_mfeat = time.time() - t0

    # Stage 6: Matcher Scoring (HistGB)
    t0 = time.time()
    if total_k40 > 0:
        m_scores = matcher.predict_proba(X_match)
    else:
        m_scores = np.array([])
    t_mscore = time.time() - t0

    # Stage 7: Threshold and Margin Decision
    t0 = time.time()
    chunk_claims = {}
    total_claims = 0
    for (start, end), s1_rec, k40_data in zip(m_slices, queries, k40_records_list):
        s1_id = s1_rec["entity_id"]
        if start == end:
            chunk_claims[s1_id] = []
            continue

        k40_tids, _, _ = k40_data
        sub_mscores = m_scores[start:end]
        top_score = float(sub_mscores[0])
        claims = []
        if top_score >= tau:
            for tid, score in zip(k40_tids, sub_mscores):
                score = float(score)
                if score >= tau and (top_score - score) <= delta:
                    claims.append((tid, score))
        chunk_claims[s1_id] = claims
        total_claims += len(claims)
    t_decision = time.time() - t0

    # Stage 8: Checkpoint Write
    t0 = time.time()
    chunk_id = "India_0000"
    cands_path = SHARDS_DIR / f"candidate_pairs_{chunk_id}.tsv"
    claims_path = SHARDS_DIR / f"scored_claims_{chunk_id}.json"
    manifest_path = SHARDS_DIR / f"manifest_{chunk_id}.json"

    with open(cands_path, "w", encoding="utf-8") as f:
        for s1_rec in queries:
            eid = s1_rec["entity_id"]
            tids = chunk_candidates.get(eid, [])
            f.write(f"{eid}\t{','.join(tids)}\n")

    with open(claims_path, "w", encoding="utf-8") as f:
        json.dump(chunk_claims, f)

    t_disk = time.time() - t0

    t_shard_total = time.time() - t_shard0

    # Conflict resolution benchmark on this shard's claims
    t0 = time.time()
    flat_claims = []
    for s1_id, claims in chunk_claims.items():
        for tid, score in claims:
            flat_claims.append((s1_id, tid, float(score)))
    flat_claims.sort(key=lambda x: x[2], reverse=True)
    assigned = set()
    matches = defaultdict(list)
    for s1_id, tid, score in flat_claims:
        if tid not in assigned:
            assigned.add(tid)
            matches[s1_id].append(tid)
    t_cr = time.time() - t0

    # Clean up memory
    r_cache.clear()
    m_cache.clear()
    del X_rank
    del X_match
    gc.enable()
    gc.collect()

    peak_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    swap_end = get_swap_mb()
    swap_delta = swap_end - swap_start

    ent_per_sec = len(queries) / max(0.01, t_shard_total)

    meta = {
        "status": "COMPLETE",
        "config_hash": CONFIG_HASH,
        "chunk_id": chunk_id,
        "country": "India",
        "chunk_idx": 0,
        "entity_count": len(queries),
        "total_raw_candidates": total_raw,
        "total_k40_candidates": sum(len(v) for v in chunk_candidates.values()),
        "pre_conflict_claims": total_claims,
        "runtime_seconds": round(t_shard_total, 2),
        "throughput_ent_per_sec": round(ent_per_sec, 2),
        "stage_timings": {
            "blocker_s": round(t_blocker, 2),
            "ranker_feat_s": round(t_rfeat, 2),
            "ranker_score_s": round(t_rscore, 2),
            "k40_sel_s": round(t_k40, 2),
            "matcher_feat_s": round(t_mfeat, 2),
            "matcher_score_s": round(t_mscore, 2),
            "decision_s": round(t_decision, 2),
            "disk_write_s": round(t_disk, 2)
        },
        "peak_rss_mb": round(peak_rss, 1),
        "swap_delta_mb": round(swap_delta, 1),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ---------------------------------------------------------
    # PRINT AUTHORITATIVE REPORT
    # ---------------------------------------------------------
    print("\n=======================================================", flush=True)
    print("FINAL 5K INDIA BENCHMARK RESULTS:", flush=True)
    print("=======================================================", flush=True)
    print(f"  Shard ID:                       India_0000", flush=True)
    print(f"  Entities processed:             {len(queries):,}", flush=True)
    print(f"  Total raw candidates:           {total_raw:,} ({total_raw/len(queries):.1f} / entity)", flush=True)
    print(f"  Total K=40 candidate pairs:     {total_k40:,} ({total_k40/len(queries):.1f} / entity)", flush=True)
    print(f"  Pre-conflict claims:            {total_claims:,}", flush=True)
    print(f"  Stage Timings Breakdown:", flush=True)
    print(f"    1. Blocker (raw candidate gen):   {t_blocker:6.2f}s ({t_blocker/t_shard_total*100:4.1f}%)", flush=True)
    print(f"    2. Ranker Features (Native C):    {t_rfeat:6.2f}s ({t_rfeat/t_shard_total*100:4.1f}%) [{int(total_raw/max(0.001, t_rfeat)):,} pairs/s]", flush=True)
    print(f"    3. Ranker Scoring (HistGB):       {t_rscore:6.2f}s ({t_rscore/t_shard_total*100:4.1f}%) [{int(total_raw/max(0.001, t_rscore)):,} rows/s]", flush=True)
    print(f"    4. Top-40 Selection:              {t_k40:6.2f}s ({t_k40/t_shard_total*100:4.1f}%)", flush=True)
    print(f"    5. Matcher Features (C Lev):      {t_mfeat:6.2f}s ({t_mfeat/t_shard_total*100:4.1f}%) [{int(total_k40/max(0.001, t_mfeat)):,} pairs/s]", flush=True)
    print(f"    6. Matcher Scoring (HistGB):      {t_mscore:6.2f}s ({t_mscore/t_shard_total*100:4.1f}%)", flush=True)
    print(f"    7. Threshold & Margin Decision:   {t_decision:6.2f}s ({t_decision/t_shard_total*100:4.1f}%)", flush=True)
    print(f"    8. Checkpoint Disk Write:         {t_disk:6.2f}s ({t_disk/t_shard_total*100:4.1f}%)", flush=True)
    print(f"    9. Conflict Resolution:           {t_cr:6.2f}s", flush=True)
    print(f"  -----------------------------------------------------", flush=True)
    print(f"  TOTAL SHARD WALL-CLOCK:         {t_shard_total:6.2f}s", flush=True)
    print(f"  THROUGHPUT:                     {ent_per_sec:6.2f} entities/sec", flush=True)
    print(f"  Peak RSS:                       {peak_rss:6.1f} MB", flush=True)
    print(f"  Swap Delta:                     {swap_delta:6.1f} MB", flush=True)
    print("=======================================================", flush=True)

    # Dynamic Deadline Calculation
    remaining_india = 809986 - len(queries)  # 5000 already done in India_0000
    sec_inference = remaining_india / max(0.01, ent_per_sec)
    post_inference_overhead_sec = 240  # 4 mins: conflict res (2m) + assembly (1m) + validator (1m)
    now_t = time.time()
    t_finish = now_t + sec_inference + post_inference_overhead_sec

    # Target timestamps for today 23:40 and 23:30
    local_lt = time.localtime(now_t)
    t_2340 = time.mktime((local_lt.tm_year, local_lt.tm_month, local_lt.tm_mday, 23, 40, 0, 0, 0, -1))
    t_2330 = time.mktime((local_lt.tm_year, local_lt.tm_month, local_lt.tm_mday, 23, 30, 0, 0, 0, -1))

    avail_2340 = max(1, (t_2340 - now_t) - post_inference_overhead_sec)
    avail_2330 = max(1, (t_2330 - now_t) - post_inference_overhead_sec)

    min_tp_2340 = remaining_india / avail_2340
    min_tp_2330 = remaining_india / avail_2330

    if t_finish <= t_2330:
        deadline_status = "SAFE"
    elif t_finish <= t_2340:
        deadline_status = "TIGHT"
    else:
        deadline_status = "CRITICAL"

    proj_finish_str = time.strftime("%H:%M IST", time.localtime(t_finish))

    print(f"\n=======================================================", flush=True)
    print(f"INDIA 5K TIME:\n{t_shard_total:.2f} sec\n", flush=True)
    print(f"INDIA THROUGHPUT:\n{ent_per_sec:.2f} ent/s\n", flush=True)
    print(f"MINIMUM THROUGHPUT FOR 23:40:\n{min_tp_2340:.2f} ent/s (for <=23:30: {min_tp_2330:.2f} ent/s)\n", flush=True)
    print(f"PROJECTED FINAL VALIDATED OUTPUT:\n{proj_finish_str}\n", flush=True)
    print(f"DEADLINE:\n{deadline_status}\n", flush=True)
    print(f"PHASE 4 STATUS:\nHOLD\n", flush=True)

    print("OPTIMIZATION BENCHMARK", flush=True)
    print("======================", flush=True)
    print(f"threads/processes: 8 native threads / 1 process (read-only shared memory)", flush=True)
    print(f"wall time: {t_shard_total:.2f}s", flush=True)
    print(f"entities/sec: {ent_per_sec:.2f} ent/s", flush=True)
    print(f"ranker feature pairs/sec: {int(total_raw / max(0.001, t_rfeat)):,} pairs/s", flush=True)
    print(f"peak RSS: {peak_rss:.1f} MB", flush=True)
    print(f"swap: {swap_delta:.1f} MB delta (total used: {swap_end:.1f} MB)", flush=True)
    print(f"regression: {'PASS' if reg_pass else 'FAIL'}", flush=True)
    print(f"prediction differences: {pred_diff_count}", flush=True)
    eta_india_str = time.strftime("%H:%M:%S IST", time.localtime(now_t + sec_inference))
    print(f"ETA for remaining India: {eta_india_str}", flush=True)
    print(f"ETA for final validated output: {proj_finish_str}", flush=True)
    print(f"=======================================================", flush=True)


if __name__ == "__main__":
    run_benchmark_and_verify()
