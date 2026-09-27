import sys
import gc
import os
import time
import math
import csv
import json
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
from amazon_ml_er.experiments.test_inference.run_accelerated_inference import (
    RANKER_MODEL_PATH, MATCHER_MODEL_PATH, load_targets, build_index
)
def get_ranker_profile(tid: str, targets: dict, cache: dict) -> RankerEntityProfile:
    p = cache.get(tid)
    if p is None:
        if len(cache) >= 30000:
            cache.clear()
        rec = targets[tid]
        p = RankerEntityProfile(tid, rec["business_name"], rec["business_address"], rec.get("country", ""))
        cache[tid] = p
    return p

def get_matching_profile(tid: str, targets: dict, cache: dict) -> MatchingEntityProfile:
    p = cache.get(tid)
    if p is None:
        if len(cache) >= 30000:
            cache.clear()
        rec = targets[tid]
        p = MatchingEntityProfile(tid, rec["business_name"], rec["business_address"], rec.get("country", ""))
        cache[tid] = p
    return p
from amazon_ml_er.code.business_entity_resolution.src.blocking.indexes import PostingListIndex
import heapq
import fast_features_native

# Monkey patch C accelerations
mf.fast_levenshtein_ratio = fast_features_native.fast_levenshtein_ratio

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

PostingListIndex.retrieve_top_k = fast_retrieve_top_k

def run_test():
    print("[1] Loading models and index...")
    cfg = BlockerConfig(
        max_bucket_size=500,
        ch5_top_k=50,
        ch5_max_df=3000,
        ch6_tokens_to_query=2,
        ch6_max_df=3000
    )
    ranker = CandidateRanker.load(RANKER_MODEL_PATH)
    matcher = FinalMatcherModel.load(MATCHER_MODEL_PATH)

    targets = load_targets("India")
    print(f"Loaded {len(targets):,} targets.")

    t0 = time.time()
    t_idx = build_index(targets, cfg)
    print(f"Index built in {time.time()-t0:.2f}s.")

    s1_path = repo_root / "dataset" / "test" / "test_source1.tsv"
    queries = []
    with open(s1_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)
        for row in reader:
            if len(row) >= 4 and row[3] == "India":
                queries.append({
                    "entity_id": row[0],
                    "business_name": row[1],
                    "business_address": row[2],
                    "country": row[3]
                })
                if len(queries) == 500:
                    break

    print(f"Testing 500 entities with zero swap...")
    r_cache = {}
    m_cache = {}
    tau = 0.70
    delta = 0.10
    k_budget = 40

    t_start = time.time()
    raw_candidates_list = []
    total_raw = 0
    t_b0 = time.time()
    for s1_rec in queries:
        raw_c = generate_raw_candidates(s1_rec, t_idx, cfg)
        raw_candidates_list.append(raw_c)
        total_raw += len(raw_c)
    t_block = time.time() - t_b0

    t_rf0 = time.time()
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
    t_rfeat = time.time() - t_rf0

    t_rs0 = time.time()
    if total_raw > 0:
        r_scores = ranker.score(X_rank)
    else:
        r_scores = np.array([])
    t_rscore = time.time() - t_rs0

    t_k0 = time.time()
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
    t_k40 = time.time() - t_k0

    t_mf0 = time.time()
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
    t_mfeat = time.time() - t_mf0

    t_ms0 = time.time()
    if total_k40 > 0:
        m_scores = matcher.predict_proba(X_match)
    else:
        m_scores = np.array([])
    t_mscore = time.time() - t_ms0

    t_d0 = time.time()
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
    t_dec = time.time() - t_d0

    t_total = time.time() - t_start
    t_rate = len(queries) / max(0.01, t_total)

    peak_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    print("="*50)
    print(f"500 Queries Time:       {t_total:.2f}s")
    print(f"Throughput:             {t_rate:.2f} ent/s")
    print(f"  Blocker:              {t_block:.2f}s ({total_raw:,} raw cands)")
    print(f"  Ranker Feat (C):      {t_rfeat:.2f}s ({int(total_raw/max(0.001, t_rfeat)):,} pairs/s)")
    print(f"  Ranker Score (HistGB):{t_rscore:.2f}s")
    print(f"  Top-40:               {t_k40:.2f}s")
    print(f"  Matcher Feat (C Lev): {t_mfeat:.2f}s ({int(total_k40/max(0.001, t_mfeat)):,} pairs/s)")
    print(f"  Matcher Score (HistGB):{t_mscore:.2f}s")
    print(f"  Decision:             {t_dec:.2f}s ({total_claims:,} claims)")
    print(f"Peak RSS:               {peak_rss:.1f} MB")
    print(f"Targets cached in r_cache: {len(r_cache):,}")
    print("="*50)

if __name__ == "__main__":
    run_test()
