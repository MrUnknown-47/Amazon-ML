"""
Optimized Orchestrator for Phase 2 Blocking Benchmark Experiments.
Streams and indexes target corpora once per country partition,
benchmarks Channel 2 token variants (top-1, top-2, top-3),
runs progressive query scaling (1K, 5K, 10K, 50K),
evaluates candidate budgets (K in 10, 15, 25, 40, 60, 100),
performs failure analysis on missed true matches, and exports structured reports.
"""

import time
import os
import json
import csv
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import numpy as np
import psutil

from ..config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    VALIDATION_SPLIT_S1_PATH,
    EXPERIMENTS_DIR
)
from ..ground_truth import parse_ground_truth_file
from ..evaluation.metrics import compute_candidate_recall
from .keys import extract_all_blocking_keys
from .indexes import ExactInvertedIndex, PostingListIndex
from .union import (
    union_channel_candidates,
    analyze_union_and_contributions
)
from .pruning import (
    extract_compact_record_features,
    evaluate_candidate_budgets
)
from .benchmark import run_failure_analysis


def load_validation_queries(
    val_ids: Set[str],
    s1_path: Path = TRAIN_SOURCE1
) -> Dict[str, Dict[str, str]]:
    """Load query records for validation S1 IDs."""
    queries = {}
    with open(s1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if not row:
                continue
            eid = row[0].strip()
            if eid in val_ids:
                queries[eid] = {
                    "entity_id": eid,
                    "business_name": row[1],
                    "business_address": row[2],
                    "country": row[3]
                }
    return queries


def run_blocking_experiment(
    sample_sizes: List[int] = [1000, 5000, 10000, 50000],
    budgets: List[int] = [10, 15, 25, 40, 60, 100],
    max_bucket_size: int = 500
) -> Dict[str, Any]:
    """
    Execute progressive blocking benchmarks across country partitions.
    """
    proc = psutil.Process(os.getpid())
    t_start = time.time()
    mem_start = proc.memory_info().rss / (1024 * 1024)

    print("=== Starting Phase 2 Blocking & Candidate Generation Experiment ===", flush=True)

    # 1. Load validation S1 IDs
    print(f"Loading validation S1 IDs from {VALIDATION_SPLIT_S1_PATH}...", flush=True)
    with open(VALIDATION_SPLIT_S1_PATH, "r", encoding="utf-8") as f:
        all_val_ids = [line.strip() for line in f if line.strip()]
    val_id_set = set(all_val_ids)
    print(f"  Loaded {len(all_val_ids):,} validation S1 IDs.")

    # 2. Load query records
    print("Loading validation S1 records...", flush=True)
    val_queries = load_validation_queries(val_id_set, TRAIN_SOURCE1)
    print(f"  Loaded {len(val_queries):,} query records.")

    # 3. Load ground truth for validation entities
    print("Loading ground truth for validation entities...", flush=True)
    full_gt = parse_ground_truth_file(TRAIN_GROUND_TRUTH)
    val_gt = {eid: full_gt.get(eid, set()) for eid in val_queries.keys()}
    total_true_matches = sum(len(m) for m in val_gt.values())
    print(f"  Validation ground truth matches: {total_true_matches:,} true targets across {len(val_gt):,} entities.")

    # Partition validation queries by country
    queries_by_country: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for eid, q in val_queries.items():
        queries_by_country[q["country"]][eid] = q

    print(f"Validation queries by country: { {c: len(qs) for c, qs in queries_by_country.items()} }")

    total_target_corpus_size = 10320219  # Train S2 (5.03M) + S3 (5.29M)

    channel_keys = [
        "ch1_core_name",
        "ch2_top1_token",
        "ch2_top2_tokens",
        "ch2_top3_tokens",
        "ch3_street_num_token",
        "ch4_address_location",
        "ch5_posting_char_3gram"
    ]

    full_ch_candidates: Dict[str, Dict[str, Set[str]]] = {
        name: {eid: set() for eid in val_queries} for name in channel_keys
    }
    all_retrieved_target_ids: Set[str] = set()
    country_indexing_times: Dict[str, float] = {}

    # Stream & Index targets per country partition ONCE
    for country, c_queries in queries_by_country.items():
        print(f"\n=======================================================", flush=True)
        print(f"   INDEXING & QUERYING COUNTRY: {country} ({len(c_queries):,} queries)", flush=True)
        print(f"=======================================================", flush=True)

        # Build channels for this country
        idx1 = ExactInvertedIndex("ch1_core_name", max_bucket_size=max_bucket_size)
        idx2_top1 = ExactInvertedIndex("ch2_top1_token", max_bucket_size=max_bucket_size)
        idx2_top2 = ExactInvertedIndex("ch2_top2_tokens", max_bucket_size=max_bucket_size)
        idx2_top3 = ExactInvertedIndex("ch2_top3_tokens", max_bucket_size=max_bucket_size)
        idx3 = ExactInvertedIndex("ch3_street_num_token", max_bucket_size=max_bucket_size)
        idx4 = ExactInvertedIndex("ch4_address_location", max_bucket_size=max_bucket_size)
        pl5 = PostingListIndex("ch5_posting_char_3gram", n=3, ngram_type="char", max_df_ratio=0.005, max_df_count=3000)

        # Single streaming pass over S2 and S3 for this country
        print(f"  Streaming target records for {country} from S2 and S3...", flush=True)
        n_targets = 0
        t_idx0 = time.time()

        for source_path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
            with open(source_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f, delimiter="\t")
                next(reader)
                for row in reader:
                    if not row:
                        continue
                    if row[3].strip() != country:
                        continue
                    n_targets += 1
                    eid = row[0].strip()

                    # Extract all channel keys in a single normalization pass
                    keys = extract_all_blocking_keys(row[1], row[2])

                    idx1.add(keys["ch1_key"], eid)
                    idx2_top1.add(keys["ch2_top1"], eid)
                    idx2_top2.add(keys["ch2_top2"], eid)
                    idx2_top3.add(keys["ch2_top3"], eid)
                    if keys["ch3_key"]:
                        idx3.add(keys["ch3_key"], eid)
                    if keys["ch4_key"]:
                        idx4.add(keys["ch4_key"], eid)
                    pl5.index_document(eid, keys["ch5_text"])

                    if n_targets % 1000000 == 0:
                        elapsed = time.time() - t_idx0
                        rate = int(n_targets / elapsed) if elapsed > 0 else 0
                        print(f"    Indexed {n_targets:,} records ({rate:,} records/sec)...", flush=True)

        print(f"  Finalizing posting list index for {country}...", flush=True)
        pl5.finalize_index()
        t_idx1 = time.time()
        country_indexing_times[country] = round(t_idx1 - t_idx0, 2)
        print(f"  Indexed total {n_targets:,} {country} target records in {country_indexing_times[country]}s.")

        # Query all queries for this country across channels
        print(f"  Querying {len(c_queries):,} {country} S1 queries across channels...", flush=True)
        t_q0 = time.time()

        for qid, qrec in c_queries.items():
            qkeys = extract_all_blocking_keys(qrec["business_name"], qrec["business_address"])

            # Ch 1
            c1 = set(idx1.get_candidates(qkeys["ch1_key"]))
            full_ch_candidates["ch1_core_name"][qid].update(c1)
            all_retrieved_target_ids.update(c1)

            # Ch 2 variants
            c2_1 = set(idx2_top1.get_candidates(qkeys["ch2_top1"]))
            full_ch_candidates["ch2_top1_token"][qid].update(c2_1)
            all_retrieved_target_ids.update(c2_1)

            c2_2 = set(idx2_top2.get_candidates(qkeys["ch2_top2"]))
            full_ch_candidates["ch2_top2_tokens"][qid].update(c2_2)
            all_retrieved_target_ids.update(c2_2)

            c2_3 = set(idx2_top3.get_candidates(qkeys["ch2_top3"]))
            full_ch_candidates["ch2_top3_tokens"][qid].update(c2_3)
            all_retrieved_target_ids.update(c2_3)

            # Ch 3
            if qkeys["ch3_key"]:
                c3 = set(idx3.get_candidates(qkeys["ch3_key"]))
                full_ch_candidates["ch3_street_num_token"][qid].update(c3)
                all_retrieved_target_ids.update(c3)

            # Ch 4
            if qkeys["ch4_key"]:
                c4 = set(idx4.get_candidates(qkeys["ch4_key"]))
                full_ch_candidates["ch4_address_location"][qid].update(c4)
                all_retrieved_target_ids.update(c4)

            # Ch 5 Posting List
            top5 = pl5.retrieve_top_k(qkeys["ch5_text"], top_k=25)
            c5 = {eid for eid, _ in top5}
            full_ch_candidates["ch5_posting_char_3gram"][qid].update(c5)
            all_retrieved_target_ids.update(c5)

        t_q1 = time.time()
        print(f"  All channels queried for {len(c_queries):,} {country} entities in {t_q1 - t_q0:.2f}s.", flush=True)

    # 4. Stream target features for pruning (only for the retrieved candidates)
    print(f"\nCaching features for {len(all_retrieved_target_ids):,} unique retrieved candidates...", flush=True)
    t_feat0 = time.time()
    retrieved_target_records = {}
    for source_path in [TRAIN_SOURCE2, TRAIN_SOURCE3]:
        with open(source_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f, delimiter="\t")
            next(reader)
            for row in reader:
                if not row:
                    continue
                eid = row[0].strip()
                if eid in all_retrieved_target_ids:
                    retrieved_target_records[eid] = {
                        "business_name": row[1],
                        "business_address": row[2],
                        "country": row[3]
                    }
    t_feat1 = time.time()
    print(f"  Cached {len(retrieved_target_records):,} target records in {t_feat1 - t_feat0:.2f}s.")

    # Main production channel subset: ch1, ch2_top2, ch3, ch4, ch5
    prod_channels = [
        "ch1_core_name",
        "ch2_top2_tokens",
        "ch3_street_num_token",
        "ch4_address_location",
        "ch5_posting_char_3gram"
    ]

    # Progressive benchmarks on subset scales
    benchmark_scales_results = {}
    final_scale_results = {}

    for scale in sample_sizes:
        print(f"\n=======================================================", flush=True)
        print(f"   EVALUATING SCALE: {scale:,} Validation S1 Queries", flush=True)
        print(f"=======================================================", flush=True)

        scale_val_ids = all_val_ids[:scale]
        scale_queries = {eid: val_queries[eid] for eid in scale_val_ids}
        scale_gt = {eid: val_gt[eid] for eid in scale_val_ids}

        t_eval0 = time.time()

        # Channel metrics at this scale
        ch_metrics_at_scale = {}
        ch_contributions_input = []

        for ch_name in channel_keys:
            cands = {eid: full_ch_candidates[ch_name][eid] for eid in scale_val_ids}
            recall_stats = compute_candidate_recall(scale_gt, cands)
            cand_lens = [len(c) for c in cands.values()]
            arr_lens = np.array(cand_lens) if cand_lens else np.array([0])
            tot_c = int(arr_lens.sum())

            ch_m = {
                "channel_name": ch_name,
                "query_count": scale,
                "candidate_count_total": tot_c,
                "mean_candidates_per_S1": float(round(arr_lens.mean(), 2)),
                "median_candidates_per_S1": float(np.median(arr_lens)),
                "P95_candidates_per_S1": float(np.percentile(arr_lens, 95)),
                "P99_candidates_per_S1": float(np.percentile(arr_lens, 99)),
                "max_candidates_per_S1": int(arr_lens.max()),
                "candidate_recall": float(round(recall_stats["candidate_recall"], 5)),
                "captured_true_matches": recall_stats["captured_true_matches"],
                "total_true_matches": recall_stats["total_true_matches"],
                "reduction_ratio": float(1.0 - (tot_c / (scale * total_target_corpus_size)))
            }
            ch_metrics_at_scale[ch_name] = ch_m
            if ch_name in prod_channels:
                ch_contributions_input.append((ch_name, cands))

            print(f"  {ch_name:25s} | Recall: {ch_m['candidate_recall']*100:.2f}% | Mean Cands: {ch_m['mean_candidates_per_S1']:6.2f} | P95: {ch_m['P95_candidates_per_S1']:4.1f}")

        # Union of production channels (R_block)
        union_cands, union_metrics = analyze_union_and_contributions(
            ch_contributions_input,
            scale_queries,
            scale_gt,
            total_target_corpus_size
        )
        print(f"\n  Production Union R_block: {union_metrics['R_block']*100:.2f}% ({union_metrics['captured_true_matches']:,} / {union_metrics['total_true_matches']:,})")
        print(f"  Mean Candidates per S1: {union_metrics['mean_candidates_per_S1']:.2f} | P95: {union_metrics['P95_candidates_per_S1']} | Reduction Ratio: {union_metrics['reduction_ratio']:.8f}")

        # Compute channel hits for pruning
        channel_hit_counts = defaultdict(int)
        for _, cands in ch_contributions_input:
            for s1_id, t_set in cands.items():
                for tid in t_set:
                    channel_hit_counts[(s1_id, tid)] += 1

        # Pruning budget evaluation
        budget_results = evaluate_candidate_budgets(
            union_candidates=union_cands,
            query_records=scale_queries,
            target_records=retrieved_target_records,
            ground_truth=scale_gt,
            channel_hit_counts=channel_hit_counts,
            budgets=budgets,
            total_target_corpus_size=total_target_corpus_size
        )

        for k_str, k_res in budget_results["budgets_evaluated"].items():
            print(f"  Budget K = {int(k_str):3d} | R_pruned = {k_res['R_pruned']*100:.2f}% | Lost Matches: {k_res['true_matches_lost_to_pruning']:4d} | P95: {k_res['P95_candidates_per_S1']:4.1f}")

        # Failure analysis on missed pairs
        missed_pairs = []
        for s1_id, true_set in scale_gt.items():
            if not true_set:
                continue
            captured = true_set & union_cands.get(s1_id, set())
            for tid in (true_set - captured):
                missed_pairs.append((s1_id, tid))

        failure_report = run_failure_analysis(
            missed_pairs=missed_pairs,
            query_records=scale_queries,
            target_records=retrieved_target_records
        )

        elapsed_scale = round(time.time() - t_eval0, 2)
        peak_mem_scale = round(proc.memory_info().rss / (1024 * 1024), 2)
        throughput = round(scale / elapsed_scale, 1)

        scale_record = {
            "scale_queries": scale,
            "runtime_seconds": elapsed_scale,
            "throughput_queries_per_sec": throughput,
            "peak_memory_mb": peak_mem_scale,
            "channels": ch_metrics_at_scale,
            "union": union_metrics,
            "budgets": budget_results,
            "failure_analysis": failure_report
        }
        benchmark_scales_results[str(scale)] = scale_record

        if scale == sample_sizes[-1]:
            final_scale_results = scale_record

    # Save artifacts to experiments/blocking/
    blocking_exp_dir = EXPERIMENTS_DIR / "blocking"
    os.makedirs(blocking_exp_dir, exist_ok=True)

    with open(blocking_exp_dir / "channel_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_scale_results.get("channels", {}), f, indent=2)

    with open(blocking_exp_dir / "candidate_budget_results.json", "w", encoding="utf-8") as f:
        json.dump(final_scale_results.get("budgets", {}), f, indent=2)

    with open(blocking_exp_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_scales_results, f, indent=2)

    # Generate Markdown Summary
    md_summary = generate_blocking_markdown_report(benchmark_scales_results, sample_sizes[-1])
    with open(blocking_exp_dir / "blocking_summary.md", "w", encoding="utf-8") as f:
        f.write(md_summary)

    print(f"\n=== Phase 2 Blocking Experiment Complete ===", flush=True)
    print(f"Artifacts successfully saved to {blocking_exp_dir}/")
    return benchmark_scales_results


def generate_blocking_markdown_report(benchmark_results: Dict[str, Any], final_scale: int) -> str:
    """Generate comprehensive markdown summary for Phase 2 blocking results."""
    final = benchmark_results[str(final_scale)]
    channels = final["channels"]
    union = final["union"]
    budgets = final["budgets"]
    failures = final["failure_analysis"]

    lines = [
        "# Phase 2 Blocking & Candidate Generation Experiment Report",
        "",
        f"**Evaluation Scale:** {final_scale:,} Validation S1 Entities (searched against FULL 10.32M target corpus)  ",
        f"**Evaluation Runtime:** {final['runtime_seconds']} seconds ({final['throughput_queries_per_sec']} queries/sec)  ",
        f"**Peak Memory Usage:** {final['peak_memory_mb']} MB  ",
        "",
        "---",
        "",
        "## 1. Per-Channel Blocking Recall & Candidate Size",
        "",
        "| Channel Name | Recall (%) | Captured Matches | Mean Cands/S1 | Median | P95 | P99 | Max | Total Candidates | Reduction Ratio |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for name, m in channels.items():
        lines.append(
            f"| `{name}` | {m['candidate_recall']*100:.2f}% | {m['captured_true_matches']:,} / {m['total_true_matches']:,} | "
            f"{m['mean_candidates_per_S1']:.2f} | {m['median_candidates_per_S1']} | {m['P95_candidates_per_S1']} | "
            f"{m['P99_candidates_per_S1']} | {m['max_candidates_per_S1']} | {m['candidate_count_total']:,} | {m['reduction_ratio']:.8f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Unpruned Union Blocking (R_block)",
        "",
        f"- **R_block (Raw Union Recall):** **{union['R_block']*100:.2f}%**",
        f"- **Total True Matches in Validation Ground Truth:** {union['total_true_matches']:,}",
        f"- **Captured True Matches:** {union['captured_true_matches']:,}",
        f"- **Missed True Matches:** {union['missed_true_matches']:,}",
        f"- **Total Candidate Pairs Generated:** {union['candidate_count_total']:,}",
        f"- **Mean Candidates / S1:** {union['mean_candidates_per_S1']:.2f}",
        f"- **Median Candidates / S1:** {union['median_candidates_per_S1']}",
        f"- **P95 Candidates / S1:** {union['P95_candidates_per_S1']}",
        f"- **P99 Candidates / S1:** {union['P99_candidates_per_S1']}",
        f"- **Max Candidates / S1:** {union['max_candidates_per_S1']}",
        f"- **Reduction Ratio:** {union['reduction_ratio']:.8f}",
        "",
        "### Recall Attribution & Incrementality",
        "",
        "| Channel | Cumulative Recall | Incremental Recall Added | Exclusive Matches Found |",
        "| :--- | :--- | :--- | :--- |"
    ])

    cum = union["recall_attribution"]["cumulative_recall_by_channel"]
    inc = union["recall_attribution"]["incremental_recall_by_channel"]
    exc = union["recall_attribution"]["exclusive_recall_by_channel"]

    for (c_name, c_rec), (_, i_rec, i_cnt), (_, e_rec, e_cnt) in zip(cum, inc, exc):
        lines.append(f"| `{c_name}` | {c_rec*100:.2f}% | +{i_rec*100:.2f}% ({i_cnt:,}) | {e_rec*100:.2f}% ({e_cnt:,}) |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Candidate Pruning Benchmark: R_block vs R_pruned Across Budgets (K)",
        "",
        "| Candidate Budget (K) | R_pruned (%) | Lost Matches to Pruning | S1 Entities Truncated | Mean Cands/S1 | P95 | P99 | Reduction Ratio |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for k_str, k_res in budgets["budgets_evaluated"].items():
        lines.append(
            f"| **K = {k_str}** | **{k_res['R_pruned']*100:.2f}%** | {k_res['true_matches_lost_to_pruning']:,} | "
            f"{k_res['entities_with_truncated_matches']:,} | {k_res['mean_candidates_per_S1']:.2f} | "
            f"{k_res['P95_candidates_per_S1']} | {k_res['P99_candidates_per_S1']} | {k_res['reduction_ratio']:.8f} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 4. Progressive Scalability Benchmark",
        "",
        "| Validation Scale | Runtime (s) | Throughput (queries/s) | Peak Memory (MB) | Total Candidates |",
        "| :--- | :--- | :--- | :--- | :--- |"
    ])

    for s_str, s_data in benchmark_results.items():
        lines.append(
            f"| {int(s_str):,} S1 queries | {s_data['runtime_seconds']}s | {s_data['throughput_queries_per_sec']} q/s | "
            f"{s_data['peak_memory_mb']} MB | {s_data['union']['candidate_count_total']:,} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 5. Failure Analysis of Missed Matches",
        "",
        f"- **Total Missed Ground Truth Matches:** {failures['total_missed_pairs']:,}",
        "",
        "| Root Cause Category | Count | Percentage | Primary Noise Pattern |",
        "| :--- | :--- | :--- | :--- |"
    ])

    for cat, c_info in failures["category_breakdown"].items():
        lines.append(f"| `{cat}` | {c_info['count']:,} | {c_info['percentage']:.2f}% | Sample inspected in JSON |")

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_blocking_experiment()
