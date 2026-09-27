"""
Production Resumable Chunked Inference Engine for Business Entity Resolution.
Phase 4 Production Implementation:
- Blocker: Config E (authoritative_blocker.py)
- Candidate Ranker: CandidateRanker (K=40 policy)
- Final Matcher: HistGradientBoostingClassifier (final_matcher_histgb.pkl)
- Decision Engine: threshold_and_margin (tau=0.70, delta=0.10)
- Conflict Resolution: Greedy Maximum Confidence 1:1 Target Assignment
- Fully Resumable Deterministic Chunks with Atomic Checkpoints
"""

import os
import sys
import time
import json
import csv
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
import psutil

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
    extract_pair_features as extract_ranker_features,
    NUM_FEATURES as NUM_RANKER_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.candidate_compression.model import CandidateRanker
from amazon_ml_er.code.business_entity_resolution.src.matching.features import (
    MatchingEntityProfile,
    extract_matching_features,
    NUM_MATCHING_FEATURES
)
from amazon_ml_er.code.business_entity_resolution.src.matching.models import FinalMatcherModel

RANKER_MODEL_PATH = repo_root / "amazon_ml_er" / "experiments" / "integration" / "candidate_ranker_authoritative.pkl"
MATCHER_MODEL_PATH = repo_root / "amazon_ml_er" / "experiments" / "matching" / "final_matcher_histgb.pkl"
TEST_SOURCE1_PATH = repo_root / "dataset" / "test" / "test_source1.tsv"

SHARDS_DIR = repo_root / "experiments" / "work" / "shards"
COUNTRY_DIR = repo_root / "experiments" / "work" / "country_results"
OUTPUT_DIR = repo_root / "output"

CONFIG_HASH = "PHASE4_FROZEN_CONFIG_E_K40_TAU070_DELTA010"


def load_targets(country: str):
    """Load target records for a country from pre-partitioned target file."""
    targets = {}
    country_file = repo_root / "dataset" / "test" / "country_targets" / f"targets_{country}.tsv"
    with open(country_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                targets[row[0]] = {
                    "business_name": row[1],
                    "business_address": row[2],
                    "country": row[3]
                }
    return targets


def build_index(targets: dict, config: BlockerConfig):
    """Build authoritative Config E inverted index."""
    t_idx = TargetCorpusIndex(config=config)
    for tid, trec in targets.items():
        t_idx.add_target_record(tid, trec["business_name"], trec["business_address"], trec["country"])
    t_idx.finalize()
    return t_idx


def get_ranker_profile(tid: str, targets: dict, cache: dict) -> RankerEntityProfile:
    p = cache.get(tid)
    if p is None:
        rec = targets[tid]
        p = RankerEntityProfile(tid, rec["business_name"], rec["business_address"], rec.get("country", ""))
        cache[tid] = p
    return p


def get_matching_profile(tid: str, targets: dict, cache: dict) -> MatchingEntityProfile:
    p = cache.get(tid)
    if p is None:
        rec = targets[tid]
        p = MatchingEntityProfile(tid, rec["business_name"], rec["business_address"], rec.get("country", ""))
        cache[tid] = p
    return p


def process_chunk(
    country: str,
    chunk_idx: int,
    s1_chunk: list,
    target_index: TargetCorpusIndex,
    targets: dict,
    r_cache: dict,
    m_cache: dict,
    ranker: CandidateRanker,
    matcher: FinalMatcherModel,
    cfg: BlockerConfig,
    tau: float = 0.70,
    delta: float = 0.10,
    k_budget: int = 40
):
    """
    Process a single deterministic chunk and write atomic checkpoints.
    """
    chunk_id = f"{country}_{chunk_idx:04d}"
    manifest_path = SHARDS_DIR / f"manifest_{chunk_id}.json"
    cands_path = SHARDS_DIR / f"candidate_pairs_{chunk_id}.tsv"
    claims_path = SHARDS_DIR / f"scored_claims_{chunk_id}.json"

    # Check if already complete
    if manifest_path.exists() and cands_path.exists() and claims_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("status") == "COMPLETE" and meta.get("config_hash") == CONFIG_HASH:
                print(f"  [Chunk {chunk_id}] Already complete ({meta.get('entity_count'):,} entities). Skipping.", flush=True)
                return meta
        except Exception:
            pass  # Corrupt, recompute

    print(f"  [Chunk {chunk_id}] Processing {len(s1_chunk):,} entities...", flush=True)
    t0 = time.time()

    # 1. Blocking
    raw_candidates_list = []
    total_raw = 0
    for s1_rec in s1_chunk:
        raw_c = generate_raw_candidates(s1_rec, target_index, cfg)
        raw_candidates_list.append(raw_c)
        total_raw += len(raw_c)

    # 2. Ranker feature extraction & scoring
    all_rfeats = []
    slices = []
    cur = 0
    for s1_rec, raw_c in zip(s1_chunk, raw_candidates_list):
        if not raw_c:
            slices.append((cur, cur))
            continue
        s1_rprof = RankerEntityProfile(
            s1_rec["entity_id"], s1_rec["business_name"], s1_rec["business_address"], s1_rec.get("country", "")
        )
        c_tids = [t for t, _ in raw_c]
        c_provs = [p for _, p in raw_c]
        n_c = len(raw_c)
        feats = np.zeros((n_c, NUM_RANKER_FEATURES), dtype=np.float32)
        for i in range(n_c):
            t_prof = get_ranker_profile(c_tids[i], targets, r_cache)
            feats[i] = extract_ranker_features(s1_rprof, t_prof, c_provs[i])
        all_rfeats.append(feats)
        slices.append((cur, cur + n_c))
        cur += n_c

    if all_rfeats:
        X_rank = np.vstack(all_rfeats)
        r_scores = ranker.score(X_rank)
    else:
        r_scores = np.array([])

    # 3. K=40 candidates & Matcher feature extraction
    all_mfeats = []
    m_slices = []
    cur_m = 0
    k40_records_list = []
    chunk_candidates = {}

    for (start, end), s1_rec, raw_c in zip(slices, s1_chunk, raw_candidates_list):
        s1_id = s1_rec["entity_id"]
        if start == end:
            m_slices.append((cur_m, cur_m))
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
        n_k40 = len(k40_tids)

        s1_mprof = MatchingEntityProfile(
            s1_id, s1_rec["business_name"], s1_rec["business_address"], s1_rec.get("country", "")
        )
        m_f = np.zeros((n_k40, NUM_MATCHING_FEATURES), dtype=np.float32)
        for rank_pos, (tid, prov, rscore) in enumerate(zip(k40_tids, k40_provs, k40_rscores), start=1):
            t_mprof = get_matching_profile(tid, targets, m_cache)
            m_f[rank_pos - 1] = extract_matching_features(s1_mprof, t_mprof, prov, rscore, rank_pos)
        all_mfeats.append(m_f)
        m_slices.append((cur_m, cur_m + n_k40))
        cur_m += n_k40

    # 4. Batch matcher scoring
    if all_mfeats:
        X_match = np.vstack(all_mfeats)
        m_scores = matcher.predict_proba(X_match)
    else:
        m_scores = np.array([])

    # 5. Threshold and margin decisioning
    chunk_claims = {}
    total_claims = 0
    for (start, end), s1_rec, k40_data in zip(m_slices, s1_chunk, k40_records_list):
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

    t_elapsed = time.time() - t0

    # 6. Atomic checkpoint writes
    cands_tmp = SHARDS_DIR / f"candidate_pairs_{chunk_id}.tsv.tmp"
    with open(cands_tmp, "w", encoding="utf-8") as f:
        for s1_rec in s1_chunk:
            eid = s1_rec["entity_id"]
            tids = chunk_candidates.get(eid, [])
            f.write(f"{eid}\t{','.join(tids)}\n")
    os.replace(cands_tmp, cands_path)

    claims_tmp = SHARDS_DIR / f"scored_claims_{chunk_id}.json.tmp"
    with open(claims_tmp, "w", encoding="utf-8") as f:
        json.dump(chunk_claims, f)
    os.replace(claims_tmp, claims_path)

    meta = {
        "status": "COMPLETE",
        "config_hash": CONFIG_HASH,
        "chunk_id": chunk_id,
        "country": country,
        "chunk_idx": chunk_idx,
        "entity_count": len(s1_chunk),
        "total_raw_candidates": total_raw,
        "total_k40_candidates": sum(len(v) for v in chunk_candidates.values()),
        "pre_conflict_claims": total_claims,
        "runtime_seconds": round(t_elapsed, 2),
        "throughput_ent_per_sec": round(len(s1_chunk) / max(0.01, t_elapsed), 2),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    manifest_tmp = SHARDS_DIR / f"manifest_{chunk_id}.json.tmp"
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    os.replace(manifest_tmp, manifest_path)

    print(f"  [Chunk {chunk_id}] Completed in {t_elapsed:.1f}s ({len(s1_chunk)/t_elapsed:.1f} ent/s). Claims: {total_claims:,}", flush=True)
    return meta


def run_country_inference(country: str, chunk_size: int = 5000, max_chunks: int = None):
    """
    Run full chunked inference for a country.
    """
    print(f"\n=======================================================", flush=True)
    print(f"STARTING COUNTRY INFERENCE: {country.upper()}", flush=True)
    print(f"=======================================================", flush=True)

    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    COUNTRY_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load queries for country
    query_file = repo_root / "dataset" / "test" / "country_targets" / f"queries_{country}.tsv"
    queries = []
    with open(query_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row:
                queries.append({
                    "entity_id": row[0],
                    "business_name": row[1],
                    "business_address": row[2],
                    "country": row[3]
                })

    print(f"Loaded {len(queries):,} test queries for {country}.", flush=True)

    # 2. Load targets and build index
    cfg = BlockerConfig(
        max_bucket_size=500,
        ch5_top_k=50,
        ch5_max_df=3000,
        ch6_tokens_to_query=2,
        ch6_max_df=3000
    )
    t_idx0 = time.time()
    print(f"Loading target records for {country}...", flush=True)
    targets = load_targets(country)
    print(f"Loaded {len(targets):,} targets in {time.time()-t_idx0:.2f}s. Building inverted index...", flush=True)
    t_b0 = time.time()
    idx = build_index(targets, cfg)
    print(f"Inverted index built in {time.time()-t_b0:.2f}s. RAM: {psutil.Process().memory_info().rss/(1024*1024):.1f} MB", flush=True)

    # 3. Load models
    ranker = CandidateRanker.load(RANKER_MODEL_PATH)
    matcher = FinalMatcherModel.load(MATCHER_MODEL_PATH)

    r_cache = {}
    m_cache = {}

    # 4. Process in deterministic chunks
    num_chunks = (len(queries) + chunk_size - 1) // chunk_size
    if max_chunks:
        num_chunks = min(num_chunks, max_chunks)

    print(f"Total chunks to process: {num_chunks} (chunk size = {chunk_size:,})", flush=True)

    chunk_metas = []
    for c_idx in range(num_chunks):
        chunk_slice = queries[c_idx * chunk_size : (c_idx + 1) * chunk_size]
        meta = process_chunk(
            country=country,
            chunk_idx=c_idx,
            s1_chunk=chunk_slice,
            target_index=idx,
            targets=targets,
            r_cache=r_cache,
            m_cache=m_cache,
            ranker=ranker,
            matcher=matcher,
            cfg=cfg,
            tau=0.70,
            delta=0.10,
            k_budget=40
        )
        chunk_metas.append(meta)

    # 5. If all chunks complete, run greedy conflict resolution for country
    if num_chunks == (len(queries) + chunk_size - 1) // chunk_size:
        print(f"\nAll chunks complete for {country}. Running country-wide 1:1 conflict resolution...", flush=True)
        t_cr0 = time.time()
        country_claims = {}
        for c_idx in range(num_chunks):
            chunk_id = f"{country}_{c_idx:04d}"
            claims_path = SHARDS_DIR / f"scored_claims_{chunk_id}.json"
            with open(claims_path, "r", encoding="utf-8") as f:
                c_data = json.load(f)
                country_claims.update(c_data)

        # Flatten claims and sort descending
        flat_claims = []
        for s1_id, claims in country_claims.items():
            for tid, score in claims:
                flat_claims.append((s1_id, tid, float(score)))

        flat_claims.sort(key=lambda x: x[2], reverse=True)

        assigned_targets = set()
        s1_final_matches = defaultdict(list)
        conflicts_count = 0
        dupes_removed = 0

        tid_counts = Counter(t for _, t, _ in flat_claims)
        conflicts_count = sum(1 for c in tid_counts.values() if c > 1)

        for s1_id, tid, score in flat_claims:
            if tid not in assigned_targets:
                assigned_targets.add(tid)
                s1_final_matches[s1_id].append(tid)
            else:
                dupes_removed += 1

        print(f"Conflict resolution complete in {time.time()-t_cr0:.2f}s:")
        print(f"  Conflicted targets: {conflicts_count:,}")
        print(f"  Duplicate claims removed: {dupes_removed:,}")
        print(f"  Total matched entities: {sum(len(v) for v in s1_final_matches.values()):,}")

        # Write country results
        c_match_path = COUNTRY_DIR / f"matching_{country}.tsv"
        with open(c_match_path, "w", encoding="utf-8") as f:
            for q in queries:
                eid = q["entity_id"]
                preds = s1_final_matches.get(eid, [])
                f.write(f"{eid}\t{','.join(preds)}\n")

        c_cands_path = COUNTRY_DIR / f"candidates_{country}.tsv"
        with open(c_cands_path, "w", encoding="utf-8") as f:
            for c_idx in range(num_chunks):
                chunk_id = f"{country}_{c_idx:04d}"
                c_part = SHARDS_DIR / f"candidate_pairs_{chunk_id}.tsv"
                with open(c_part, "r", encoding="utf-8") as pf:
                    for line in pf:
                        f.write(line)

        print(f"Country {country} artifacts assembled successfully.")


def assemble_final_submission():
    """
    Merge country artifacts in the exact sequence of test_source1.tsv.
    Validates schemas and generates official output files.
    """
    print("\n=======================================================", flush=True)
    print("ASSEMBLING FINAL SUBMISSION TSV FILES", flush=True)
    print("=======================================================", flush=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_matching_path = OUTPUT_DIR / "matching_results.tsv"
    final_candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    # Load country results mappings
    matching_map = {}
    candidate_map = {}

    for country in ["France", "US", "India"]:
        m_file = COUNTRY_DIR / f"matching_{country}.tsv"
        c_file = COUNTRY_DIR / f"candidates_{country}.tsv"
        if not m_file.exists() or not c_file.exists():
            print(f"ERROR: Missing country outputs for {country}. Assembly aborted.")
            return False

        with open(m_file, "r", encoding="utf-8") as f:
            for line in f:
                s1, tab, rest = line.partition("\t")
                if tab:
                    matching_map[s1.strip()] = rest.strip()

        with open(c_file, "r", encoding="utf-8") as f:
            for line in f:
                s1, tab, rest = line.partition("\t")
                if tab:
                    candidate_map[s1.strip()] = rest.strip()

    print(f"Loaded {len(matching_map):,} matching records and {len(candidate_map):,} candidate records.")

    # Write in exact order of test_source1.tsv
    total_written = 0
    t0 = time.time()
    with open(TEST_SOURCE1_PATH, "r", encoding="utf-8") as s1_f, \
         open(final_matching_path, "w", encoding="utf-8") as m_out, \
         open(final_candidate_path, "w", encoding="utf-8") as c_out:

        reader = csv.reader(s1_f, delimiter="\t")
        next(reader)  # skip header

        # Write official headers
        m_out.write("source1_entity_id\tmatched_entity_ids\n")
        c_out.write("source1_entity_id\tcandidate_entity_ids\n")

        for row in reader:
            if not row:
                continue
            eid = row[0]
            m_preds = matching_map.get(eid, "")
            c_cands = candidate_map.get(eid, "")

            m_out.write(f"{eid}\t{m_preds}\n")
            c_out.write(f"{eid}\t{c_cands}\n")
            total_written += 1

    print(f"Assembled {total_written:,} rows in {time.time()-t0:.2f}s.")
    print(f"  Matching Results: {final_matching_path} ({final_matching_path.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  Candidate Pairs:  {final_candidate_path} ({final_candidate_path.stat().st_size / (1024*1024):.1f} MB)")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", type=str, default="France")
    parser.add_argument("--chunk-size", type=int, default=5000)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()

    if args.assemble:
        assemble_final_submission()
    else:
        run_country_inference(country=args.country, chunk_size=args.chunk_size, max_chunks=args.max_chunks)
