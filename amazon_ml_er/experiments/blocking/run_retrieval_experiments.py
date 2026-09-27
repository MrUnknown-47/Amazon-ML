"""
Phase 2B: Retrieval Channel Benchmarks (Tasks 3, 4, 5, 7).
Benchmarks character n-gram variants, document-frequency policies, token retrieval,
prefix retrieval, and multi-stage re-ranking against full target corpus on 5,000 S1 queries.
Saves results to experiments/blocking/retrieval_variants.json.
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


def run_experiments(scale: int = 5000):
    proc = psutil.Process(os.getpid())
    print(f"=== Running Phase 2B Retrieval Channel Experiments (Scale: {scale:,} S1 Queries) ===", flush=True)

    # 1. Load validation S1 IDs
    with open(VALIDATION_SPLIT_S1_PATH, "r", encoding="utf-8") as f:
        all_val_ids = [line.strip() for line in f if line.strip()]
    eval_ids = set(all_val_ids[:scale])
    print(f"Loaded {len(eval_ids):,} evaluation S1 IDs.")

    # 2. Load S1 query records
    val_queries = {}
    for row in stream_tsv_rows(TRAIN_SOURCE1):
        eid = row.get("entity_id")
        if eid in eval_ids:
            val_queries[eid] = row
    print(f"Loaded {len(val_queries):,} query records.")

    # 3. Load ground truth for evaluated queries
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
    total_true_matches = sum(len(v) for v in eval_gt.values())
    print(f"Evaluation ground truth: {total_true_matches:,} true targets across {len(eval_gt):,} entities.")

    # Partition queries by country
    queries_by_country = defaultdict(dict)
    for eid, q in val_queries.items():
        queries_by_country[q["country"]][eid] = q

    # Define test configurations
    configs = [
        # Task 3: Character N-Grams and DF Policies
        {"name": "char_2gram_df3000_k25", "idx": "char_2", "df": 3000, "k": 25, "stage": "single"},
        {"name": "char_3gram_df500_k25", "idx": "char_3", "df": 500, "k": 25, "stage": "single"},
        {"name": "char_3gram_df1000_k25", "idx": "char_3", "df": 1000, "k": 25, "stage": "single"},
        {"name": "char_3gram_df3000_k25", "idx": "char_3", "df": 3000, "k": 25, "stage": "single"}, # Baseline
        {"name": "char_3gram_df10000_k25", "idx": "char_3", "df": 10000, "k": 25, "stage": "single"},
        {"name": "char_3gram_df50000_k25", "idx": "char_3", "df": 50000, "k": 25, "stage": "single"},
        {"name": "char_4gram_df3000_k25", "idx": "char_4", "df": 3000, "k": 25, "stage": "single"},
        {"name": "char_2_3gram_df3000_k25", "idx": "char_2_3", "df": 3000, "k": 25, "stage": "single"},
        {"name": "char_3_4gram_df3000_k25", "idx": "char_3_4", "df": 3000, "k": 25, "stage": "single"},
        {"name": "char_3gram_df3000_k50", "idx": "char_3", "df": 3000, "k": 50, "stage": "single"},
        {"name": "char_3gram_df3000_k100", "idx": "char_3", "df": 3000, "k": 100, "stage": "single"},
        {"name": "char_2_3gram_df10000_k50", "idx": "char_2_3", "df": 10000, "k": 50, "stage": "single"},
        # Task 4: Token Retrieval
        {"name": "token_1_df1000_k25", "idx": "token_1", "df": 1000, "k": 25, "stage": "single"},
        {"name": "token_1_df3000_k25", "idx": "token_1", "df": 3000, "k": 25, "stage": "single"},
        {"name": "token_1_df10000_k25", "idx": "token_1", "df": 10000, "k": 25, "stage": "single"},
        {"name": "token_bigram_df3000_k25", "idx": "token_2", "df": 3000, "k": 25, "stage": "single"},
        {"name": "token_1_df3000_k50", "idx": "token_1", "df": 3000, "k": 50, "stage": "single"},
        {"name": "token_1_df3000_k100", "idx": "token_1", "df": 3000, "k": 100, "stage": "single"},
        # Task 5: Prefix Keys
        {"name": "prefix_4_df3000_k25", "idx": "prefix_4", "df": 3000, "k": 25, "stage": "single"},
        # Task 7: Two-Stage Re-Ranking
        {"name": "twostage_char3_pool50_k25", "idx": "char_3", "df": 3000, "pool": 50, "k": 25, "stage": "twostage"},
        {"name": "twostage_char3_pool100_k25", "idx": "char_3", "df": 3000, "pool": 100, "k": 25, "stage": "twostage"},
        {"name": "twostage_char3_pool250_k25", "idx": "char_3", "df": 3000, "pool": 250, "k": 25, "stage": "twostage"},
        {"name": "twostage_char3_pool250_k50", "idx": "char_3", "df": 3000, "pool": 250, "k": 50, "stage": "twostage"},
        {"name": "twostage_token1_pool100_k25", "idx": "token_1", "df": 3000, "pool": 100, "k": 25, "stage": "twostage"},
        {"name": "twostage_char2_3_pool100_k25", "idx": "char_2_3", "df": 3000, "pool": 100, "k": 25, "stage": "twostage"},
    ]

    needed_indexes = ["char_2", "char_3", "char_4", "char_2_3", "char_3_4", "token_1", "token_2", "prefix_4"]

    results_by_variant = defaultdict(lambda: defaultdict(set))
    timings = defaultdict(float)
    total_target_records = 10320219

    # Stream target corpora per country partition
    for country, c_queries in queries_by_country.items():
        print(f"\n=======================================================", flush=True)
        print(f"  Indexing & Querying Country: {country} ({len(c_queries):,} queries)", flush=True)
        print(f"=======================================================", flush=True)

        indexes = {
            m: PostingListIndex(name=f"{country}_{m}", ngram_mode=m, max_df_count=50000, max_df_ratio=0.05)
            for m in needed_indexes
        }

        print(f"  Streaming target records for {country} into 8 base indexes...", flush=True)
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

                    # Index document into each base index
                    for idx in indexes.values():
                        idx.index_document(eid, text)

                    if n_indexed % 2000000 == 0:
                        m_curr = proc.memory_info().rss / (1024 * 1024)
                        print(f"    Indexed {n_indexed:,} targets in {time.time()-t0:.1f}s (RAM: {m_curr:.1f} MB)...", flush=True)

        print(f"  Finalizing 8 base indexes for {country} (total {n_indexed:,} targets)...", flush=True)
        for idx in indexes.values():
            idx.finalize_index()
        t_index_done = time.time()
        print(f"  Indexing completed in {t_index_done - t0:.2f}s (RAM: {proc.memory_info().rss / (1024 * 1024):.1f} MB).")

        # Query all configurations
        print(f"  Querying {len(c_queries):,} S1 queries across {len(configs)} test configurations...", flush=True)

        for cfg in configs:
            var_name = cfg["name"]
            idx = indexes[cfg["idx"]]
            df = cfg["df"]
            k = cfg["k"]
            stage = cfg["stage"]

            t_q0 = time.time()
            if stage == "single":
                for qid, qrec in c_queries.items():
                    keys = extract_all_blocking_keys(qrec["business_name"], qrec["business_address"])
                    text = keys["ch1_key"]
                    hits = idx.retrieve_top_k(text, top_k=k, df_threshold=df)
                    results_by_variant[var_name][qid].update(h[0] for h in hits)
            else:
                pool = cfg["pool"]
                for qid, qrec in c_queries.items():
                    keys = extract_all_blocking_keys(qrec["business_name"], qrec["business_address"])
                    text = keys["ch1_key"]
                    hits = idx.retrieve_two_stage(text, pool_size=pool, top_k=k, df_threshold=df, rank_metric="jaccard")
                    results_by_variant[var_name][qid].update(h[0] for h in hits)

            timings[var_name] += time.time() - t_q0

        print(f"  Completed all test queries for country {country} in {time.time() - t_index_done:.2f}s.")

    # 4. Compute metrics for each variant
    print("\n=======================================================", flush=True)
    print("  COMPUTING RETRIEVAL VARIANT METRICS & RECALL", flush=True)
    print("=======================================================", flush=True)

    variant_summary = {}
    print(f"{'Variant Name':32s} | {'Recall':8s} | {'Mean Cands':10s} | {'P95':5s} | {'P99':5s} | {'Max':5s} | {'Total Cands':11s} | {'Time (s)':8s}")
    print("-" * 105)

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
            "category": "char_ngram" if "char" in var_name and "twostage" not in var_name else ("token" if "token" in var_name and "twostage" not in var_name else ("prefix" if "prefix" in var_name else "two_stage")),
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
            "queries_per_second": float(round(scale / elapsed, 1)) if elapsed > 0 else 0.0
        }
        variant_summary[var_name] = metrics

        print(f"{var_name:32s} | {metrics['candidate_recall']*100:6.2f}% | {metrics['mean_candidates_per_S1']:10.2f} | {metrics['P95_candidates_per_S1']:5.0f} | {metrics['P99_candidates_per_S1']:5.0f} | {metrics['max_candidates_per_S1']:5.0f} | {tot:11,d} | {elapsed:8.2f}")

    # Save to experiments/blocking/retrieval_variants.json
    out_path = Path("amazon_ml_er/experiments/blocking/retrieval_variants.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(variant_summary, f, indent=2)

    print(f"\nSuccessfully saved retrieval experiment results to {out_path}")
    return variant_summary


if __name__ == "__main__":
    run_experiments(scale=5000)
