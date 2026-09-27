"""
Multiprocessing Benchmark for Phase 3.5 (Issue 5).
Benchmarks real worker pools (1 worker, 2 workers, 4 workers) on validation entities.
Measures wall-clock runtime, entity throughput, and peak RAM consumption.
"""

import sys
import os
import time
import json
import psutil
from pathlib import Path
from multiprocessing import get_context
import numpy as np

repo_root = Path(__file__).resolve().parents[2].parent
sys.path.insert(0, str(repo_root))

from amazon_ml_er.code.business_entity_resolution.src.config import (
    TRAIN_GROUND_TRUTH,
    VALIDATION_SPLIT_S1_PATH,
    SPLITS_DIR
)
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


# Global references for worker processes (shared after fork or loaded in worker)
_worker_target_index = None
_worker_ranker = None
_worker_matcher = None
_worker_target_rprofs = None
_worker_target_mprofs = None
_worker_blocker_config = None


def init_worker(target_records_raw, ranker_path, matcher_path):
    global _worker_target_index, _worker_ranker, _worker_matcher
    global _worker_target_rprofs, _worker_target_mprofs, _worker_blocker_config

    _worker_blocker_config = BlockerConfig(
        max_bucket_size=500,
        ch5_top_k=50,
        ch5_max_df=3000,
        ch6_tokens_to_query=2,
        ch6_max_df=3000
    )
    _worker_target_index = TargetCorpusIndex(config=_worker_blocker_config)
    for tid, trec in target_records_raw.items():
        _worker_target_index.add_target_record(
            entity_id=tid,
            business_name=trec["business_name"],
            business_address=trec["business_address"],
            country=trec.get("country", "")
        )
    _worker_target_index.finalize()

    _worker_target_rprofs = {
        tid: RankerEntityProfile(tid, trec["business_name"], trec["business_address"], trec.get("country", ""))
        for tid, trec in target_records_raw.items()
    }
    _worker_target_mprofs = {
        tid: MatchingEntityProfile(tid, trec["business_name"], trec["business_address"], trec.get("country", ""))
        for tid, trec in target_records_raw.items()
    }

    _worker_ranker = CandidateRanker.load(ranker_path)
    _worker_matcher = FinalMatcherModel.load(matcher_path)


def process_query_batch(s1_records_batch):
    global _worker_target_index, _worker_ranker, _worker_matcher
    global _worker_target_rprofs, _worker_target_mprofs, _worker_blocker_config

    results = {}
    for s1_id, s1_rec in s1_records_batch:
        raw_cands = generate_raw_candidates(s1_rec, _worker_target_index, _worker_blocker_config)
        if not raw_cands:
            results[s1_id] = []
            continue

        s1_r_prof = RankerEntityProfile(s1_id, s1_rec["business_name"], s1_rec["business_address"], s1_rec.get("country", ""))
        s1_m_prof = MatchingEntityProfile(s1_id, s1_rec["business_name"], s1_rec["business_address"], s1_rec.get("country", ""))

        c_tids = [tid for tid, _ in raw_cands]
        provs = [prov for _, prov in raw_cands]
        n_c = len(raw_cands)

        r_feats = np.zeros((n_c, NUM_RANKER_FEATURES), dtype=np.float32)
        for idx in range(n_c):
            r_feats[idx] = extract_ranker_features(s1_r_prof, _worker_target_rprofs[c_tids[idx]], provs[idx])

        r_scores = _worker_ranker.score(r_feats)
        top_k_indices = np.argsort(r_scores)[::-1][:40]

        filtered_tids = [c_tids[idx] for idx in top_k_indices]
        filtered_provs = [provs[idx] for idx in top_k_indices]
        filtered_r_scores = [float(r_scores[idx]) for idx in top_k_indices]

        n_m = len(filtered_tids)
        m_feats = np.zeros((n_m, NUM_MATCHING_FEATURES), dtype=np.float32)
        for idx in range(n_m):
            m_feats[idx] = extract_matching_features(
                s1_m_prof,
                _worker_target_mprofs[filtered_tids[idx]],
                filtered_provs[idx],
                ranker_score=filtered_r_scores[idx],
                ranker_rank=idx + 1
            )

        m_scores = _worker_matcher.predict_proba(m_feats) if n_m > 0 else np.array([])
        results[s1_id] = [(filtered_tids[i], filtered_r_scores[i], float(m_scores[i])) for i in range(n_m)]

    return results


def run_benchmark():
    print("Loading test data for multiprocessing benchmark...", flush=True)
    with open("amazon_ml_er/splits/val_s1_records.json", "r", encoding="utf-8") as f:
        val_s1_all = json.load(f)

    with open("amazon_ml_er/splits/val_target_records.json", "r", encoding="utf-8") as f:
        val_targets_all = json.load(f)

    # Use first 500 validation entities for safe scaling measurement
    sample_size = 500
    val_keys = sorted(val_s1_all.keys())[:sample_size]
    sample_records = [(k, val_s1_all[k]) for k in val_keys]

    ranker_path = "amazon_ml_er/experiments/integration/candidate_ranker_authoritative.pkl"
    matcher_path = "amazon_ml_er/experiments/matching/final_matcher_histgb.pkl"

    results_table = {}
    proc = psutil.Process(os.getpid())

    for n_workers in [1, 2, 4]:
        print(f"\n--- Testing with {n_workers} worker(s) on {sample_size} entities ---", flush=True)
        t0 = time.time()
        ram_before = proc.memory_info().rss / (1024 * 1024)

        batch_size = (sample_size + n_workers - 1) // n_workers
        batches = [sample_records[i * batch_size: (i + 1) * batch_size] for i in range(n_workers)]

        ctx = get_context("spawn")
        with ctx.Pool(
            processes=n_workers,
            initializer=init_worker,
            initargs=(val_targets_all, ranker_path, matcher_path)
        ) as pool:
            outputs = pool.map(process_query_batch, batches)

        elapsed = time.time() - t0
        ram_after = proc.memory_info().rss / (1024 * 1024)
        throughput = sample_size / elapsed

        print(f"  {n_workers} worker(s): {elapsed:.2f}s ({throughput:.1f} entities/sec). RAM delta: {ram_after - ram_before:.1f} MB")
        results_table[f"{n_workers}_workers"] = {
            "workers": n_workers,
            "entities": sample_size,
            "runtime_seconds": float(round(elapsed, 2)),
            "throughput_entities_per_sec": float(round(throughput, 2)),
            "speedup_vs_single_worker": float(round((sample_size / elapsed) / (sample_size / results_table.get("1_workers", {}).get("runtime_seconds", elapsed)), 2)),
            "peak_ram_mb": float(round(ram_after, 2))
        }

    # Save multiprocessing benchmark
    out_dir = Path("amazon_ml_er/experiments/matching")
    with open(out_dir / "multiprocessing_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print("\nMultiprocessing benchmark saved to multiprocessing_benchmark.json")


if __name__ == "__main__":
    run_benchmark()
