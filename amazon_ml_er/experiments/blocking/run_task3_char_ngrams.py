"""
Phase 2B: Task 3 - Character N-Gram Variants and DF Policy Benchmarking.
Evaluates character 2-gram, 3-gram, 4-gram, 2+3 gram, 3+4 gram across
DF thresholds (500, 1000, 3000, 10000, 50000) and top-K budgets (25, 50, 100)
on 5,000 validation S1 entities against full 10.32M target corpus.
"""

import sys
import os
import time
import json
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from amazon_ml_er.code.business_entity_resolution.src.config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    VALIDATION_SPLIT_S1_PATH,
    EXPERIMENTS_DIR
)
from amazon_ml_er.code.business_entity_resolution.src.data_loader import stream_tsv_rows
from amazon_ml_er.code.business_entity_resolution.src.evaluation.metrics import compute_candidate_recall
from amazon_ml_er.code.business_entity_resolution.src.blocking.keys import extract_all_blocking_keys
from amazon_ml_er.code.business_entity_resolution.src.blocking.indexes import PostingListIndex


def run_task3(scale: int = 5000):
    proc = psutil.Process(os.getpid())
    print(f"=== Starting Task 3: Character N-Gram & DF Policy Benchmark (Scale: {scale:,} S1) ===", flush=True)

    with open(VALIDATION_SPLIT_S1_PATH, "r", encoding="utf-8") as f:
        all_val_ids = [line.strip() for line in f if line.strip()]
    eval_ids = set(all_val_ids[:scale])

    val_queries = {}
    for row in stream_tsv_rows(TRAIN_SOURCE1):
        eid = row.get("entity_id")
        if eid in eval_ids:
            val_queries[eid] = row

    eval_gt = defaultdict(set)
    with open(TRAIN_GROUND_TRUTH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row and row[0].strip() in eval_ids:
                s1_id = row[0].strip()
                for tid in row[1].strip().split(","):
                    if tid.strip():
                        eval_gt[s1_id].add(tid.strip())

    queries_by_country = defaultdict(dict)
    for eid, q in val_queries.items():
        queries_by_country[q["country"]][eid] = q

    # Task 3 test configurations
    configs = [
        {"name": "char_2gram_df3000_k25", "idx": "char_2", "df": 3000, "k": 25},
        {"name": "char_3gram_df500_k25", "idx": "char_3", "df": 500, "k": 25},
        {"name": "char_3gram_df1000_k25", "idx": "char_3", "df": 1000, "k": 25},
        {"name": "char_3gram_df3000_k25", "idx": "char_3", "df": 3000, "k": 25}, # baseline Channel 5
        {"name": "char_3gram_df10000_k25", "idx": "char_3", "df": 10000, "k": 25},
        {"name": "char_3gram_df50000_k25", "idx": "char_3", "df": 50000, "k": 25},
        {"name": "char_4gram_df3000_k25", "idx": "char_4", "df": 3000, "k": 25},
        {"name": "char_2_3gram_df3000_k25", "idx": "char_2_3", "df": 3000, "k": 25},
        {"name": "char_3_4gram_df3000_k25", "idx": "char_3_4", "df": 3000, "k": 25},
        {"name": "char_3gram_df3000_k50", "idx": "char_3", "df": 3000, "k": 50},
        {"name": "char_3gram_df3000_k100", "idx": "char_3", "df": 3000, "k": 100},
        {"name": "char_2_3gram_df10000_k50", "idx": "char_2_3", "df": 10000, "k": 50},
    ]

    base_modes = ["char_2", "char_3", "char_4", "char_2_3", "char_3_4"]
    results_by_variant = defaultdict(lambda: defaultdict(set))
    timings = defaultdict(float)
    total_target_records = 10320219

    for country, c_queries in queries_by_country.items():
        print(f"\n--- Indexing & Querying Country: {country} ({len(c_queries):,} S1 queries) ---", flush=True)
        indexes = {
            m: PostingListIndex(name=f"{country}_{m}", ngram_mode=m, max_df_count=50000, max_df_ratio=0.1)
            for m in base_modes
        }

        t0 = time.time()
        n_indexed = 0

        for src_path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
            with open(src_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                for row in reader:
                    if not row or row[3].strip() != country:
                        continue
                    n_indexed += 1
                    eid = row[0].strip()
                    keys = extract_all_blocking_keys(row[1], row[2])
                    text = keys["ch1_key"]
                    if not text:
                        continue

                    # Pre-extract character ngrams in single pass
                    clean = text.lower()
                    c2 = {clean[j:j+2] for j in range(len(clean)-1)} if len(clean) >= 2 else ({clean} if clean else set())
                    c3 = {clean[j:j+3] for j in range(len(clean)-2)} if len(clean) >= 3 else ({clean} if clean else set())
                    c4 = {clean[j:j+4] for j in range(len(clean)-3)} if len(clean) >= 4 else ({clean} if clean else set())

                    c2_3 = {f"2g_{x}" for x in c2} | {f"3g_{x}" for x in c3}
                    c3_4 = {f"3g_{x}" for x in c3} | {f"4g_{x}" for x in c4}

                    indexes["char_2"].index_document_features(eid, c2)
                    indexes["char_3"].index_document_features(eid, c3)
                    indexes["char_4"].index_document_features(eid, c4)
                    indexes["char_2_3"].index_document_features(eid, c2_3)
                    indexes["char_3_4"].index_document_features(eid, c3_4)

                    if n_indexed % 2000000 == 0:
                        m_curr = proc.memory_info().rss / (1024 * 1024)
                        print(f"  Indexed {n_indexed:,} targets in {time.time()-t0:.1f}s (RAM: {m_curr:.1f} MB)...", flush=True)

        for idx in indexes.values():
            idx.finalize_index()
        t_index_done = time.time()
        print(f"  Finalized 5 base char indexes for {country} in {t_index_done - t0:.2f}s (RAM: {proc.memory_info().rss / (1024 * 1024):.1f} MB).")

        # Query all configurations
        print(f"  Querying {len(c_queries):,} S1 queries across {len(configs)} char-ngram configs...", flush=True)
        for cfg in configs:
            var_name = cfg["name"]
            idx = indexes[cfg["idx"]]
            df = cfg["df"]
            k = cfg["k"]

            t_q0 = time.time()
            for qid, qrec in c_queries.items():
                keys = extract_all_blocking_keys(qrec["business_name"], qrec["business_address"])
                text = keys["ch1_key"]
                hits = idx.retrieve_top_k(text, top_k=k, df_threshold=df)
                results_by_variant[var_name][qid].update(h[0] for h in hits)

            timings[var_name] += time.time() - t_q0

        print(f"  Completed queries for country {country} in {time.time() - t_index_done:.2f}s.")

    # Compute metrics for Task 3
    print("\n" + "=" * 105)
    print("  TASK 3: CHARACTER N-GRAM & DF POLICY BENCHMARK RESULTS")
    print("=" * 105)
    print(f"{'Variant Name':30s} | {'Recall':8s} | {'Mean Cands':10s} | {'P95':5s} | {'P99':5s} | {'Max':5s} | {'Total Cands':11s} | {'Time (s)':8s}")
    print("-" * 105)

    results_out = {}
    for cfg in configs:
        var_name = cfg["name"]
        cand_dict = results_by_variant[var_name]
        recall_stats = compute_candidate_recall(eval_gt, cand_dict)
        lens = [len(s) for s in cand_dict.values()]
        arr = np.array(lens) if lens else np.array([0])
        tot = int(arr.sum())
        elapsed = round(timings[var_name], 2)

        metrics = {
            "variant_name": var_name,
            "category": "char_ngram",
            "evaluated_queries": scale,
            "candidate_recall": float(round(recall_stats["candidate_recall"], 5)),
            "captured_true_matches": recall_stats["captured_true_matches"],
            "total_true_matches": recall_stats["total_true_matches"],
            "missed_matches": recall_stats["missed_matches"],
            "mean_candidates_per_S1": float(round(arr.mean(), 2)),
            "median_candidates_per_S1": float(np.median(arr)),
            "P95_candidates_per_S1": float(np.percentile(arr, 95)),
            "P99_candidates_per_S1": float(np.percentile(arr, 99)),
            "max_candidates_per_S1": int(arr.max()),
            "total_candidates": tot,
            "reduction_ratio": float(1.0 - (tot / (scale * total_target_records))),
            "query_runtime_seconds": elapsed,
            "queries_per_second": float(round(scale / elapsed, 1)) if elapsed > 0 else 0.0,
            "peak_memory_mb": float(round(proc.memory_info().rss / (1024 * 1024), 2))
        }
        results_out[var_name] = metrics
        print(f"{var_name:30s} | {metrics['candidate_recall']*100:6.2f}% | {metrics['mean_candidates_per_S1']:10.2f} | {metrics['P95_candidates_per_S1']:5.0f} | {metrics['P99_candidates_per_S1']:5.0f} | {metrics['max_candidates_per_S1']:5.0f} | {tot:11,d} | {elapsed:8.2f}")

    out_file = Path("amazon_ml_er/experiments/blocking/task3_char_ngram_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results_out, f, indent=2)
    print(f"\nSaved Task 3 results to {out_file}")
    return results_out


if __name__ == "__main__":
    run_task3(scale=5000)
