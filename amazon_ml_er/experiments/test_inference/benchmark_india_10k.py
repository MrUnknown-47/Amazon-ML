"""
10,000 India Entity Production Benchmark for Accelerated Feature Extraction Pipeline.
Measures wall-clock time, ent/s, pairs/s, peak RSS, swap, candidates per entity.
"""

import time
import os
import sys
import gc
import json
import csv
import psutil
from pathlib import Path
import numpy as np

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from amazon_ml_er.code.business_entity_resolution.src.blocking.authoritative_blocker import (
    BlockerConfig,
    TargetCorpusIndex,
    generate_raw_candidates
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.features import (
    EntityProfile as RankerEntityProfile,
    NUM_FEATURES as NUM_RANKER_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.model import CandidateRanker
from amazon_ml_er.code.business_entity_resolution.src.matching.features import (
    MatchingEntityProfile,
    extract_matching_features,
    NUM_MATCHING_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.matching.models import FinalMatcherModel
from amazon_ml_er.experiments.test_inference.run_chunked_inference import (
    RANKER_MODEL_PATH, MATCHER_MODEL_PATH, load_targets, build_index,
    get_ranker_profile, get_matching_profile, SHARDS_DIR, CONFIG_HASH
)
from amazon_ml_er.experiments.test_inference.benchmark_optimized_india import (
    extract_ranker_features_fast_into
)

def benchmark_10k_india():
    print("=======================================================", flush=True)
    print("STARTING 10,000 INDIA ENTITY BENCHMARK", flush=True)
    print("=======================================================", flush=True)

    # 1. Load first 10,000 queries
    query_file = repo_root / "dataset" / "test" / "country_targets" / "queries_India.tsv"
    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            if i >= 10000: break
            queries.append({"entity_id": row[0], "business_name": row[1], "business_address": row[2], "country": row[3]})

    print(f"Loaded {len(queries):,} queries for India benchmark.", flush=True)

    # 2. Build index
    cfg = BlockerConfig(max_bucket_size=500, ch5_top_k=50, ch5_max_df=3000, ch6_tokens_to_query=2, ch6_max_df=3000)
    t0 = time.time()
    targets = load_targets("India")
    print(f"Loaded {len(targets):,} targets in {time.time()-t0:.2f}s. Building index...", flush=True)
    t0 = time.time()
    idx = build_index(targets, cfg)
    print(f"Index built in {time.time()-t0:.2f}s. RAM: {psutil.Process().memory_info().rss/(1024*1024):.1f} MB", flush=True)

    ranker = CandidateRanker.load(RANKER_MODEL_PATH)
    matcher = FinalMatcherModel.load(MATCHER_MODEL_PATH)

    r_cache = {}
    m_cache = {}

    sm_start = psutil.swap_memory()
    t_start_all = time.time()
    total_raw_all = 0
    total_k40_all = 0

    chunk_size = 5000
    for chunk_idx in range(2):
        chunk_slice = queries[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
        chunk_id = f"India_bench_{chunk_idx:04d}"
        print(f"\nProcessing {chunk_id} ({len(chunk_slice):,} entities)...", flush=True)
        t_c0 = time.time()
        gc.disable()

        # Step 1: Blocking
        t_b0 = time.time()
        raw_list = [generate_raw_candidates(q, idx, cfg) for q in chunk_slice]
        t_block = time.time() - t_b0
        total_raw = sum(len(c) for c in raw_list)
        total_raw_all += total_raw

        # Step 2: Ranker features
        t_rf0 = time.time()
        X_rank = np.empty((total_raw, NUM_RANKER_FEATURES), dtype=np.float32)
        slices = []
        cur = 0
        for q, raw_c in zip(chunk_slice, raw_list):
            n_c = len(raw_c)
            if n_c == 0:
                slices.append((cur, cur))
                continue
            s1_rprof = RankerEntityProfile(q["entity_id"], q["business_name"], q["business_address"], q.get("country", ""))
            c_tids = [t for t, _ in raw_c]
            c_provs = [p for _, p in raw_c]
            for i in range(n_c):
                t_prof = get_ranker_profile(c_tids[i], targets, r_cache)
                extract_ranker_features_fast_into(X_rank, cur + i, s1_rprof, t_prof, c_provs[i])
            slices.append((cur, cur + n_c))
            cur += n_c
        t_rfeats = time.time() - t_rf0

        # Step 3: Ranker score
        t_rs0 = time.time()
        if total_raw > 0:
            r_scores = ranker.score(X_rank)
        else:
            r_scores = np.array([])
        t_rscore = time.time() - t_rs0

        # Step 4: Top-40 & Matcher feats
        t_mf0 = time.time()
        k40_records_list = []
        total_k40 = 0
        for (start, end), q, raw_c in zip(slices, chunk_slice, raw_list):
            if start == end:
                k40_records_list.append([])
                continue
            sub_scores = r_scores[start:end]
            c_tids = [t for t, _ in raw_c]
            c_provs = [p for _, p in raw_c]
            sort_idx = np.argsort(sub_scores)[::-1][:40]
            k40_tids = [c_tids[i] for i in sort_idx]
            k40_provs = [c_provs[i] for i in sort_idx]
            k40_rscores = [float(sub_scores[i]) for i in sort_idx]
            k40_records_list.append((k40_tids, k40_provs, k40_rscores))
            total_k40 += len(k40_tids)
        total_k40_all += total_k40

        X_match = np.empty((total_k40, NUM_MATCHING_FEATURES), dtype=np.float32)
        m_slices = []
        cur_m = 0
        for q, k40_data in zip(chunk_slice, k40_records_list):
            if not k40_data:
                m_slices.append((cur_m, cur_m))
                continue
            k40_tids, k40_provs, k40_rscores = k40_data
            n_k40 = len(k40_tids)
            s1_mprof = MatchingEntityProfile(q["entity_id"], q["business_name"], q["business_address"], q.get("country", ""))
            for rank_pos, (tid, prov, rscore) in enumerate(zip(k40_tids, k40_provs, k40_rscores), start=1):
                t_mprof = get_matching_profile(tid, targets, m_cache)
                X_match[cur_m + rank_pos - 1] = extract_matching_features(s1_mprof, t_mprof, prov, rscore, rank_pos)
            m_slices.append((cur_m, cur_m + n_k40))
            cur_m += n_k40
        t_mfeats = time.time() - t_mf0

        # Step 5: Matcher score
        t_ms0 = time.time()
        if total_k40 > 0:
            m_scores = matcher.predict_proba(X_match)
        else:
            m_scores = np.array([])
        t_mscore = time.time() - t_ms0

        t_elapsed = time.time() - t_c0
        if len(r_cache) > 200000:
            r_cache.clear()
        if len(m_cache) > 50000:
            m_cache.clear()
        del X_rank
        del X_match
        gc.enable()
        gc.collect()

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        print(f"  [{chunk_id}] Finished in {t_elapsed:.2f}s ({len(chunk_slice)/t_elapsed:.1f} ent/s, {total_raw/t_elapsed:,.0f} pairs/s):", flush=True)
        print(f"    - Blocking:      {t_block:.2f}s", flush=True)
        print(f"    - Ranker Feats:  {t_rfeats:.2f}s", flush=True)
        print(f"    - Ranker Score:  {t_rscore:.2f}s", flush=True)
        print(f"    - Matcher Feats: {t_mfeats:.2f}s", flush=True)
        print(f"    - Matcher Score: {t_mscore:.2f}s", flush=True)
        print(f"    - Process RSS:   {rss_mb:.1f} MB", flush=True)

    t_total_10k = time.time() - t_start_all
    sm_end = psutil.swap_memory()
    swap_diff_mb = (sm_end.used - sm_start.used) / (1024 * 1024)

    ent_s = 10000 / t_total_10k
    pairs_s = total_raw_all / t_total_10k
    speedup = ent_s / 11.0

    print("\n=======================================================", flush=True)
    print("10,000 INDIA ENTITY BENCHMARK RESULTS", flush=True)
    print("=======================================================", flush=True)
    print(f"Wall-clock Time:         {t_total_10k:.2f}s")
    print(f"Overall Throughput:      {ent_s:.2f} entities/sec")
    print(f"Candidate Pair Rate:     {pairs_s:,.0f} pairs/sec")
    print(f"Total Raw Candidates:    {total_raw_all:,} ({total_raw_all/10000:.1f} per S1)")
    print(f"Total K40 Candidates:    {total_k40_all:,} ({total_k40_all/10000:.1f} per S1)")
    print(f"Peak RSS:                {psutil.Process().memory_info().rss/(1024*1024):.1f} MB")
    print(f"Swap Delta:              {swap_diff_mb:+.1f} MB")
    print(f"Speedup vs 11 ent/s:     {speedup:.2f}x")
    print("=======================================================", flush=True)

if __name__ == "__main__":
    benchmark_10k_india()
