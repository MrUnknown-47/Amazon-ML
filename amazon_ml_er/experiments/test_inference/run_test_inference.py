"""
Authoritative Test Inference Engine for Business Entity Resolution.
Phase 4 Production Pipeline:
- Blocker: Config E (authoritative_blocker.py)
- Candidate Ranker: CandidateRanker (K=40 policy)
- Final Matcher: HistGradientBoostingClassifier
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

# Ensure repo root is on sys.path
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

# Authoritative model paths
RANKER_MODEL_PATH = repo_root / "amazon_ml_er" / "experiments" / "integration" / "candidate_ranker_authoritative.pkl"
MATCHER_MODEL_PATH = repo_root / "amazon_ml_er" / "experiments" / "matching" / "final_matcher_histgb.pkl"
TEST_SOURCE1_PATH = repo_root / "dataset" / "test" / "test_source1.tsv"
TEST_SOURCE2_PATH = repo_root / "dataset" / "test" / "test_source2.tsv"
TEST_SOURCE3_PATH = repo_root / "dataset" / "test" / "test_source3.tsv"

CONFIG_HASH = "PHASE4_FROZEN_CONFIG_E_K40_TAU070_DELTA010"


def load_target_records_for_country(country: str, max_records: int = None):
    """Load target records for a given country from pre-partitioned file."""
    targets = {}
    country_file = repo_root / "dataset" / "test" / "country_targets" / f"targets_{country}.tsv"
    if not country_file.exists():
        raise FileNotFoundError(f"Country target file not found: {country_file}")

    with open(country_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if not row:
                continue
            targets[row[0]] = {
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3]
            }
            if max_records and len(targets) >= max_records:
                break
    return targets


def build_target_index(targets: dict, config: BlockerConfig):
    """Build authoritative Config E inverted index for target records."""
    t_idx = TargetCorpusIndex(config=config)
    for tid, trec in targets.items():
        t_idx.add_target_record(
            entity_id=tid,
            business_name=trec["business_name"],
            business_address=trec["business_address"],
            country=trec["country"]
        )
    t_idx.finalize()
    return t_idx


def get_ranker_profile(tid: str, targets: dict, cache: dict) -> RankerEntityProfile:
    """Retrieve or lazily create a RankerEntityProfile for a target entity."""
    p = cache.get(tid)
    if p is None:
        rec = targets[tid]
        p = RankerEntityProfile(tid, rec["business_name"], rec["business_address"], rec.get("country", ""))
        cache[tid] = p
    return p


def get_matching_profile(tid: str, targets: dict, cache: dict) -> MatchingEntityProfile:
    """Retrieve or lazily create a MatchingEntityProfile for a target entity."""
    p = cache.get(tid)
    if p is None:
        rec = targets[tid]
        p = MatchingEntityProfile(tid, rec["business_name"], rec["business_address"], rec.get("country", ""))
        cache[tid] = p
    return p


def process_entity_chunk(
    s1_chunk: list,
    target_index: TargetCorpusIndex,
    targets: dict,
    target_rprofs_cache: dict,
    target_mprofs_cache: dict,
    candidate_ranker: CandidateRanker,
    final_matcher: FinalMatcherModel,
    blocker_config: BlockerConfig,
    tau: float = 0.70,
    delta: float = 0.10,
    k_budget: int = 40
):
    """
    Process a chunk of S1 reference entities.
    Returns:
    - candidates_map: {s1_id: [candidate_target_ids]} (exact K40 candidate set)
    - unconflict_preds: {s1_id: [(target_id, matcher_score)]} (pre-conflict predictions)
    - chunk_stats: dict of diagnostic counts
    """
    candidates_map = {}
    unconflict_preds = {}

    total_raw_pairs = 0
    total_k40_pairs = 0

    # 1. Generate raw candidates for each entity in chunk
    raw_candidates_list = []
    for s1_rec in s1_chunk:
        raw_c = generate_raw_candidates(s1_rec, target_index, blocker_config)
        raw_candidates_list.append(raw_c)
        total_raw_pairs += len(raw_c)

    # 2. Extract ranker features & batch score
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
            t_prof = get_ranker_profile(c_tids[i], targets, target_rprofs_cache)
            feats[i] = extract_ranker_features(s1_rprof, t_prof, c_provs[i])
        all_rfeats.append(feats)
        slices.append((cur, cur + n_c))
        cur += n_c

    if all_rfeats:
        X_rank = np.vstack(all_rfeats)
        r_scores = candidate_ranker.score(X_rank)
    else:
        r_scores = np.array([])

    # 3. Form exact K=40 candidate sets & extract matcher features
    all_mfeats = []
    m_slices = []
    cur_m = 0
    k40_records_list = []

    for (start, end), s1_rec, raw_c in zip(slices, s1_chunk, raw_candidates_list):
        s1_id = s1_rec["entity_id"]
        if start == end:
            m_slices.append((cur_m, cur_m))
            k40_records_list.append([])
            candidates_map[s1_id] = []
            continue

        sub_scores = r_scores[start:end]
        c_tids = [t for t, _ in raw_c]
        c_provs = [p for _, p in raw_c]
        sort_idx = np.argsort(sub_scores)[::-1][:k_budget]

        k40_tids = [c_tids[i] for i in sort_idx]
        k40_provs = [c_provs[i] for i in sort_idx]
        k40_rscores = [float(sub_scores[i]) for i in sort_idx]

        candidates_map[s1_id] = k40_tids
        total_k40_pairs += len(k40_tids)

        k40_records_list.append((k40_tids, k40_provs, k40_rscores))
        n_k40 = len(k40_tids)

        s1_mprof = MatchingEntityProfile(
            s1_id, s1_rec["business_name"], s1_rec["business_address"], s1_rec.get("country", "")
        )
        m_f = np.zeros((n_k40, NUM_MATCHING_FEATURES), dtype=np.float32)
        for rank_pos, (tid, prov, rscore) in enumerate(zip(k40_tids, k40_provs, k40_rscores), start=1):
            t_mprof = get_matching_profile(tid, targets, target_mprofs_cache)
            m_f[rank_pos - 1] = extract_matching_features(s1_mprof, t_mprof, prov, rscore, rank_pos)
        all_mfeats.append(m_f)
        m_slices.append((cur_m, cur_m + n_k40))
        cur_m += n_k40

    # 4. Batch matcher scoring
    if all_mfeats:
        X_match = np.vstack(all_mfeats)
        m_scores = final_matcher.predict_proba(X_match)
    else:
        m_scores = np.array([])

    # 5. Entity-level decisioning (threshold tau=0.70, delta=0.10)
    for (start, end), s1_rec, k40_data in zip(m_slices, s1_chunk, k40_records_list):
        s1_id = s1_rec["entity_id"]
        if start == end:
            unconflict_preds[s1_id] = []
            continue

        k40_tids, _, _ = k40_data
        sub_mscores = m_scores[start:end]

        top_score = float(sub_mscores[0])
        entity_preds = []
        if top_score >= tau:
            for tid, score in zip(k40_tids, sub_mscores):
                score = float(score)
                if score >= tau and (top_score - score) <= delta:
                    entity_preds.append((tid, score))

        unconflict_preds[s1_id] = entity_preds

    chunk_stats = {
        "entities_processed": len(s1_chunk),
        "total_raw_pairs": total_raw_pairs,
        "total_k40_pairs": total_k40_pairs,
        "unconflict_claims": sum(len(v) for v in unconflict_preds.values())
    }
    return candidates_map, unconflict_preds, chunk_stats


def resolve_conflicts_greedy_1to1(predictions_map: dict):
    """
    Apply greedy maximum-confidence 1:1 target conflict resolution.
    If multiple S1 entities claim the same target ID, the pair with the highest score wins.
    Returns:
    - final_predictions: {s1_id: [matched_target_ids]}
    - conflict_stats: dict of resolved conflicts and duplicate claims removed
    """
    all_claims = []
    for s1_id, claims in predictions_map.items():
        for tid, score in claims:
            all_claims.append((s1_id, tid, float(score)))

    # Sort claims descending by score
    all_claims.sort(key=lambda x: x[2], reverse=True)

    assigned_targets = set()
    s1_assigned = defaultdict(list)
    conflicts_detected = 0
    duplicate_claims_removed = 0

    target_claim_counts = Counter(tid for _, tid, _ in all_claims)
    conflicted_target_ids = {tid for tid, count in target_claim_counts.items() if count > 1}
    conflicts_detected = len(conflicted_target_ids)

    for s1_id, tid, score in all_claims:
        if tid not in assigned_targets:
            assigned_targets.add(tid)
            s1_assigned[s1_id].append(tid)
        else:
            duplicate_claims_removed += 1

    final_predictions = {}
    for s1_id in predictions_map.keys():
        final_predictions[s1_id] = s1_assigned.get(s1_id, [])

    conflict_stats = {
        "conflicted_targets_count": conflicts_detected,
        "duplicate_claims_removed": duplicate_claims_removed,
        "total_final_matches": sum(len(v) for v in final_predictions.values()),
        "empty_predictions_count": sum(1 for v in final_predictions.values() if len(v) == 0)
    }
    return final_predictions, conflict_stats


def run_determinism_check(subset_size: int = 5000):
    """
    Select fixed 5,000 test S1 entities and run frozen pipeline twice.
    Verifies bit-exact identical candidates, predictions, and SHA-256 hashes.
    """
    print(f"\n=======================================================", flush=True)
    print(f"RUNNING DETERMINISM CHECK ON {subset_size:,} TEST ENTITIES", flush=True)
    print(f"=======================================================", flush=True)

    # 1. Load first subset_size entities
    s1_queries = []
    with open(TEST_SOURCE1_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            s1_queries.append({
                "entity_id": row[0],
                "business_name": row[1],
                "business_address": row[2],
                "country": row[3]
            })
            if len(s1_queries) >= subset_size:
                break

    # Determine countries present in subset
    country_counts = Counter(q["country"] for q in s1_queries)
    print(f"Subset Countries: {dict(country_counts)}")

    # Load targets for each country present
    cfg = BlockerConfig(
        max_bucket_size=500,
        ch5_top_k=50,
        ch5_max_df=3000,
        ch6_tokens_to_query=2,
        ch6_max_df=3000
    )
    ranker = CandidateRanker.load(RANKER_MODEL_PATH)
    matcher = FinalMatcherModel.load(MATCHER_MODEL_PATH)

    # Group queries by country
    queries_by_country = defaultdict(list)
    for q in s1_queries:
        queries_by_country[q["country"]].append(q)

    # Pre-load targets and build indexes once for all countries present
    print("Pre-loading targets and building inverted indexes for subset countries...", flush=True)
    country_targets = {}
    country_indexes = {}
    for country in sorted(queries_by_country.keys()):
        c_len = len(queries_by_country[country])
        print(f"  Indexing targets for {country} (serving {c_len:,} queries)...", flush=True)
        t_c0 = time.time()
        targets = load_target_records_for_country(country)
        idx = build_target_index(targets, cfg)
        country_targets[country] = targets
        country_indexes[country] = idx
        print(f"    {country} index built in {time.time()-t_c0:.2f}s ({len(targets):,} targets).", flush=True)

    runs_output = []
    for run_idx in [1, 2]:
        print(f"\n--- Determinism Run {run_idx} / 2 ---", flush=True)
        t_start = time.time()
        run_candidates = {}
        run_predictions = {}

        for country in sorted(queries_by_country.keys()):
            c_queries = queries_by_country[country]
            targets = country_targets[country]
            idx = country_indexes[country]
            r_cache = {}
            m_cache = {}

            c_cands, c_unconflict, _ = process_entity_chunk(
                c_queries, idx, targets, r_cache, m_cache, ranker, matcher, cfg, tau=0.70, delta=0.10, k_budget=40
            )
            c_final, _ = resolve_conflicts_greedy_1to1(c_unconflict)

            run_candidates.update(c_cands)
            run_predictions.update(c_final)

        # Serialize in exact original query order
        cands_lines = ["source1_entity_id\tcandidate_entity_ids\n"]
        match_lines = ["source1_entity_id\tmatched_entity_ids\n"]
        for q in s1_queries:
            eid = q["entity_id"]
            cands_lines.append(f"{eid}\t{','.join(run_candidates.get(eid, []))}\n")
            match_lines.append(f"{eid}\t{','.join(run_predictions.get(eid, []))}\n")

        cands_str = "".join(cands_lines).encode("utf-8")
        match_str = "".join(match_lines).encode("utf-8")

        cands_hash = hashlib.sha256(cands_str).hexdigest()
        match_hash = hashlib.sha256(match_str).hexdigest()

        print(f"  Run {run_idx} complete in {time.time()-t_start:.2f}s.")
        print(f"  Candidate pairs SHA-256: {cands_hash}")
        print(f"  Matching results SHA-256: {match_hash}")

        runs_output.append({
            "run": run_idx,
            "candidate_hash": cands_hash,
            "matching_hash": match_hash,
            "candidates": run_candidates,
            "predictions": run_predictions
        })

    # Compare Run 1 vs Run 2
    hashes_match = (
        runs_output[0]["candidate_hash"] == runs_output[1]["candidate_hash"] and
        runs_output[0]["matching_hash"] == runs_output[1]["matching_hash"]
    )
    assert hashes_match, "DETERMINISM FAILURE: Hashes between Run 1 and Run 2 did not match!"

    # Candidate containment check: predictions subset of candidates
    containment_violations = 0
    for q in s1_queries:
        eid = q["entity_id"]
        preds = set(runs_output[0]["predictions"].get(eid, []))
        cands = set(runs_output[0]["candidates"].get(eid, []))
        if not preds.issubset(cands):
            containment_violations += 1

    assert containment_violations == 0, f"CONTAINMENT FAILURE: {containment_violations} entities had predictions not in candidates!"

    det_result = {
        "status": "PASS",
        "subset_entities_evaluated": subset_size,
        "run1_candidate_sha256": runs_output[0]["candidate_hash"],
        "run2_candidate_sha256": runs_output[1]["candidate_hash"],
        "run1_matching_sha256": runs_output[0]["matching_hash"],
        "run2_matching_sha256": runs_output[1]["matching_hash"],
        "bit_exact_match": True,
        "containment_violations": 0
    }
    with open(repo_root / "experiments" / "test_inference" / "determinism_check.json", "w", encoding="utf-8") as f:
        json.dump(det_result, f, indent=2)

    print("\n>>> DETERMINISM CHECK: PASS! Bit-exact reproducibility confirmed. <<<")
    return det_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Inference Pipeline")
    parser.add_argument("--mode", choices=["determinism", "country", "all"], default="determinism")
    parser.add_argument("--subset", type=int, default=5000)
    parser.add_argument("--country", type=str, default="France")
    args = parser.parse_args()

    if args.mode == "determinism":
        run_determinism_check(subset_size=args.subset)
