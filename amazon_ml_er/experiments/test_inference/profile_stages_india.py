"""
Profile exact stage timings on India test entities using native C acceleration.
"""

import sys
import time
import csv
import json
from pathlib import Path
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
    NUM_FEATURES as NUM_RANKER_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.model import CandidateRanker
from amazon_ml_er.code.business_entity_resolution.src.matching.features import (
    MatchingEntityProfile,
    extract_matching_features,
    NUM_MATCHING_FEATURES
)
import amazon_ml_er.code.business_entity_resolution.src.matching.features as mf
from amazon_ml_er.code.business_entity_resolution.src.matching.models import FinalMatcherModel
from amazon_ml_er.experiments.test_inference.run_chunked_inference import (
    RANKER_MODEL_PATH, MATCHER_MODEL_PATH, load_targets, build_index,
    get_ranker_profile, get_matching_profile
)
import fast_features_native

# Monkey-patch C Levenshtein into matcher module
mf.fast_levenshtein_ratio = fast_features_native.fast_levenshtein_ratio


def run_profile(num_queries=500):
    print(f"=======================================================", flush=True)
    print(f"PROFILING INDIA STAGES ON {num_queries} QUERIES", flush=True)
    print(f"=======================================================", flush=True)

    queries = []
    with open("dataset/test/country_targets/queries_India.tsv") as f:
        r = csv.reader(f, delimiter="\t")
        next(r)
        for i, row in enumerate(r):
            if i >= num_queries:
                break
            queries.append({"entity_id": row[0], "business_name": row[1], "business_address": row[2], "country": row[3]})

    print(f"Loading India targets...", flush=True)
    t_load0 = time.time()
    targets = load_targets("India")
    print(f"Loaded {len(targets):,} targets in {time.time()-t_load0:.2f}s. Building index...", flush=True)

    cfg = BlockerConfig(max_bucket_size=500, ch5_top_k=50, ch5_max_df=3000, ch6_tokens_to_query=2, ch6_max_df=3000)
    t_idx0 = time.time()
    target_index = build_index(targets, cfg)
    print(f"Built index in {time.time()-t_idx0:.2f}s.", flush=True)

    ranker = CandidateRanker.load(RANKER_MODEL_PATH)
    matcher = FinalMatcherModel.load(MATCHER_MODEL_PATH)
    tau = 0.70
    delta = 0.10
    k_budget = 40
    r_cache = {}
    m_cache = {}

    print(f"\nProfiling {num_queries} queries through all pipeline stages...", flush=True)

    # Stage 1: Blocker
    t0 = time.time()
    raw_candidates_list = []
    total_raw = 0
    for s1_rec in queries:
        raw_c = generate_raw_candidates(s1_rec, target_index, cfg)
        raw_candidates_list.append(raw_c)
        total_raw += len(raw_c)
    t_blocker = time.time() - t0

    # Stage 2: Ranker features (Native C)
    t0 = time.time()
    X_rank = np.empty((total_raw, NUM_RANKER_FEATURES), dtype=np.float32)
    slices = []
    cur = 0
    for s1_rec, raw_c in zip(queries, raw_candidates_list):
        n_c = len(raw_c)
        if n_c == 0:
            slices.append((cur, cur))
            continue
        s1_rprof = RankerEntityProfile(
            s1_rec["entity_id"], s1_rec["business_name"], s1_rec["business_address"], s1_rec.get("country", "")
        )
        c_tids = [t for t, _ in raw_c]
        c_provs = [p for _, p in raw_c]
        t_profs = [get_ranker_profile(tid, targets, r_cache) for tid in c_tids]

        fast_features_native.extract_ranker_features_batch_c(s1_rprof, t_profs, c_provs, X_rank, cur)
        slices.append((cur, cur + n_c))
        cur += n_c
    t_rank_feats = time.time() - t0

    # Stage 3: Ranker scoring (HistGB)
    t0 = time.time()
    if total_raw > 0:
        r_scores = ranker.score(X_rank)
    else:
        r_scores = np.array([])
    t_rank_score = time.time() - t0

    # Stage 4: K=40 selection
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
    t_k40_sel = time.time() - t0

    # Stage 5: Matcher feature extraction (with C Levenshtein)
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
            X_match[cur_m + rank_pos - 1] = extract_matching_features(s1_mprof, t_mprof, prov, rscore, rank_pos)
        m_slices.append((cur_m, cur_m + n_k40))
        cur_m += n_k40
    t_match_feats = time.time() - t0

    # Stage 6: Matcher scoring (HistGB)
    t0 = time.time()
    if total_k40 > 0:
        m_scores = matcher.predict_proba(X_match)
    else:
        m_scores = np.array([])
    t_match_score = time.time() - t0

    # Stage 7: Threshold and margin decisioning
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

    total_time = t_blocker + t_rank_feats + t_rank_score + t_k40_sel + t_match_feats + t_match_score + t_decision
    ent_per_sec = num_queries / total_time

    print(f"\n=======================================================", flush=True)
    print(f"STAGE BREAKDOWN FOR INDIA ({num_queries} queries, {total_raw:,} raw candidates, {total_k40:,} K40 pairs):", flush=True)
    print(f"  1. Blocker (raw candidate gen):     {t_blocker:6.2f}s ({t_blocker/total_time*100:4.1f}%)", flush=True)
    print(f"  2. Ranker Features (Native C):      {t_rank_feats:6.2f}s ({t_rank_feats/total_time*100:4.1f}%) [{int(total_raw/max(0.001, t_rank_feats)):,} pairs/s]", flush=True)
    print(f"  3. Ranker Scoring (HistGB):         {t_rank_score:6.2f}s ({t_rank_score/total_time*100:4.1f}%) [{int(total_raw/max(0.001, t_rank_score)):,} rows/s]", flush=True)
    print(f"  4. Top-40 Selection:                {t_k40_sel:6.2f}s ({t_k40_sel/total_time*100:4.1f}%)", flush=True)
    print(f"  5. Matcher Features (C Levenshtein):{t_match_feats:6.2f}s ({t_match_feats/total_time*100:4.1f}%) [{int(total_k40/max(0.001, t_match_feats)):,} pairs/s]", flush=True)
    print(f"  6. Matcher Scoring (HistGB):        {t_match_score:6.2f}s ({t_match_score/total_time*100:4.1f}%)", flush=True)
    print(f"  7. Threshold & Margin Decision:     {t_decision:6.2f}s ({t_decision/total_time*100:4.1f}%)", flush=True)
    print(f"-------------------------------------------------------", flush=True)
    print(f"TOTAL PIPELINE TIME:                  {total_time:6.2f}s (100.0%)", flush=True)
    print(f"THROUGHPUT:                           {ent_per_sec:6.1f} entities/sec", flush=True)
    print(f"PROJECTED 5,000 CHUNK TIME:           {total_time * (5000 / num_queries):6.1f}s", flush=True)
    print(f"PROJECTED INDIA FULL TIME (809,986):  {(809986 / ent_per_sec) / 3600:6.2f} hours ({(809986 / ent_per_sec) / 60:6.1f} minutes)", flush=True)
    print(f"=======================================================", flush=True)


if __name__ == "__main__":
    run_profile(num_queries=500)
