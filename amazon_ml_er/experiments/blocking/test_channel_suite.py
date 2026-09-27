"""
Test Channel Suite on 5,000 Validation S1 Entities against FULL 10.32M target corpus.
Evaluates:
- ch1_core_name
- ch2_top2_tokens
- ch3_street_num_token
- ch4_address_location
- ch5_char_3gram (k=50)
- ch6_distinctive_token (rarest token query with bucket cap 500)
- ch7_relaxed_street_num (ordinal-cleaned street num with bucket cap 500)
And measures individual recall, cumulative recall, candidates/S1, and reduction ratio.
"""

import sys
import os
import time
import json
import csv
import re
from pathlib import Path
from collections import defaultdict, Counter
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
from amazon_ml_er.code.business_entity_resolution.src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    extract_numeric_tokens
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.keys import extract_all_blocking_keys
from amazon_ml_er.code.business_entity_resolution.src.blocking.indexes import ExactInvertedIndex, PostingListIndex


def clean_street_tok(tok):
    return re.sub(r'(\d+)(st|nd|rd|th)$', r'\1', tok.lower())


def extract_relaxed_street_key(addr):
    if not addr or not addr.strip():
        return ''
    nums = extract_numeric_tokens(addr)
    if not nums:
        return ''
    p_num = str(int(nums[0]))
    norm_a = normalize_business_address(addr)
    for tok in norm_a.tokens:
        if not tok.isdigit() and len(tok) >= 3 and tok not in {'st', 'street', 'rd', 'road', 'ave', 'avenue', 'dr', 'drive', 'blvd', 'ln', 'fl', 'floor', 'ste', 'suite', 'apt', 'box'}:
            return f"{p_num}_{clean_street_tok(tok)}"
    return ''


def run_suite(scale: int = 5000):
    proc = psutil.Process(os.getpid())
    print(f"=== Running Test Channel Suite (Scale: {scale:,} S1 Queries) ===", flush=True)

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
    total_true = sum(len(v) for v in eval_gt.values())
    print(f"Validation Ground Truth: {total_true:,} true matches across {len(eval_gt):,} entities.")

    queries_by_country = defaultdict(dict)
    for eid, q in val_queries.items():
        queries_by_country[q["country"]][eid] = q

    channels = [
        "ch1_core_name",
        "ch2_top2_tokens",
        "ch3_street_num_token",
        "ch4_address_location",
        "ch5_char3_k50",
        "ch6_distinctive_token",
        "ch7_relaxed_street_num"
    ]

    all_cands = {c: defaultdict(set) for c in channels}
    all_union_cands = defaultdict(set)

    for country, c_queries in queries_by_country.items():
        print(f"\n--- Indexing & Querying Country: {country} ({len(c_queries):,} S1 queries) ---", flush=True)
        idx1 = ExactInvertedIndex("ch1", max_bucket_size=500)
        idx2 = ExactInvertedIndex("ch2", max_bucket_size=500)
        idx3 = ExactInvertedIndex("ch3", max_bucket_size=500)
        idx4 = ExactInvertedIndex("ch4", max_bucket_size=500)
        pl5 = PostingListIndex("ch5", ngram_mode="char_3", max_df_count=3000, max_df_ratio=0.01)
        idx6 = ExactInvertedIndex("ch6", max_bucket_size=500)
        idx7 = ExactInvertedIndex("ch7", max_bucket_size=500)

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

                    idx1.add(keys["ch1_key"], eid)
                    idx2.add(keys["ch2_top2"], eid)
                    if keys["ch3_key"]:
                        idx3.add(keys["ch3_key"], eid)
                    if keys["ch4_key"]:
                        idx4.add(keys["ch4_key"], eid)
                    if keys["ch1_key"]:
                        pl5.index_document(eid, keys["ch1_key"])

                    # Distinctive tokens
                    norm_n = normalize_business_name(row[1])
                    for tok in norm_n.significant_tokens:
                        if len(tok) >= 3 and not tok.isdigit():
                            idx6.add(tok, eid)

                    # Relaxed street key
                    r_key = extract_relaxed_street_key(row[2])
                    if r_key:
                        idx7.add(r_key, eid)

                    if n_indexed % 2000000 == 0:
                        m_curr = proc.memory_info().rss / (1024 * 1024)
                        print(f"  Indexed {n_indexed:,} targets in {time.time()-t0:.1f}s (RAM: {m_curr:.1f} MB)...", flush=True)

        pl5.finalize_index()
        t_idx_done = time.time()
        print(f"  Finalized indexes for {country} in {t_idx_done - t0:.2f}s (RAM: {proc.memory_info().rss / (1024 * 1024):.1f} MB).")

        # Query all S1 queries
        print(f"  Querying {len(c_queries):,} S1 queries across channels...", flush=True)
        t_q0 = time.time()

        for qid, qrec in c_queries.items():
            keys = extract_all_blocking_keys(qrec["business_name"], qrec["business_address"])

            # Ch1
            c1 = set(idx1.get_candidates(keys["ch1_key"]))
            all_cands["ch1_core_name"][qid].update(c1)

            # Ch2
            c2 = set(idx2.get_candidates(keys["ch2_top2"]))
            all_cands["ch2_top2_tokens"][qid].update(c2)

            # Ch3
            if keys["ch3_key"]:
                c3 = set(idx3.get_candidates(keys["ch3_key"]))
                all_cands["ch3_street_num_token"][qid].update(c3)

            # Ch4
            if keys["ch4_key"]:
                c4 = set(idx4.get_candidates(keys["ch4_key"]))
                all_cands["ch4_address_location"][qid].update(c4)

            # Ch5
            if keys["ch1_key"]:
                top5 = pl5.retrieve_top_k(keys["ch1_key"], top_k=50)
                c5 = {eid for eid, _ in top5}
                all_cands["ch5_char3_k50"][qid].update(c5)

            # Ch6: Distinctive token query (query the rarest token)
            norm_n = normalize_business_name(qrec["business_name"])
            sig_toks = [t for t in norm_n.significant_tokens if len(t) >= 3 and not t.isdigit() and t in idx6.index]
            if sig_toks:
                # Find rarest token
                rarest_tok = min(sig_toks, key=lambda t: len(idx6.index[t]))
                c6 = set(idx6.get_candidates(rarest_tok))
                all_cands["ch6_distinctive_token"][qid].update(c6)

            # Ch7: Relaxed street num
            r_key = extract_relaxed_street_key(qrec["business_address"])
            if r_key:
                c7 = set(idx7.get_candidates(r_key))
                all_cands["ch7_relaxed_street_num"][qid].update(c7)

        print(f"  Completed queries for country {country} in {time.time() - t_q0:.2f}s.")

    # Compute individual channel metrics
    print("\n" + "=" * 105)
    print("  INDIVIDUAL CHANNEL METRICS (Scale: 5,000 S1)")
    print("=" * 105)
    print(f"{'Channel Name':25s} | {'Recall':8s} | {'Captured':10s} | {'Mean Cands':10s} | {'P95':5s} | {'P99':5s} | {'Max':5s}")
    print("-" * 105)

    for ch_name in channels:
        cd = all_cands[ch_name]
        stats = compute_candidate_recall(eval_gt, cd)
        lens = [len(s) for s in cd.values()]
        arr = np.array(lens) if lens else np.array([0])
        print(f"{ch_name:25s} | {stats['candidate_recall']*100:6.2f}% | {stats['captured_true_matches']:5,d}/{stats['total_true_matches']:5,d} | {arr.mean():10.2f} | {np.percentile(arr, 95):5.0f} | {np.percentile(arr, 99):5.0f} | {arr.max():5d}")

    # Compute Progressive Union
    print("\n" + "=" * 105)
    print("  PROGRESSIVE UNION RECALL (R_block Progression)")
    print("=" * 105)
    cum_union = defaultdict(set)
    for ch_name in channels:
        cd = all_cands[ch_name]
        for qid, s in cd.items():
            cum_union[qid].update(s)
        stats = compute_candidate_recall(eval_gt, cum_union)
        lens = [len(s) for s in cum_union.values()]
        arr = np.array(lens) if lens else np.array([0])
        print(f"Union through `{ch_name:25s}` -> R_block: {stats['candidate_recall']*100:6.2f}% ({stats['captured_true_matches']:,}/{stats['total_true_matches']:,}) | Mean Cands: {arr.mean():6.2f} | P95: {np.percentile(arr, 95):5.0f} | P99: {np.percentile(arr, 99):5.0f}")

    print("\n=== Test Channel Suite Completed ===", flush=True)


if __name__ == "__main__":
    run_suite(scale=5000)
