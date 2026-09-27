import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
"""
Task 1 & Task 2 Execution Script:
1. Extract and format representative missed ground truth pairs across failure categories.
2. Generate experiments/blocking/missed_examples.csv and missed_examples.md.
3. Compute offline name similarity distributions comparing missed pairs vs captured pairs.
"""

import csv
import json
import os
import re
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np

from amazon_ml_er.code.business_entity_resolution.src.config import (
    TRAIN_SOURCE1,
    TRAIN_GROUND_TRUTH,
    VALIDATION_SPLIT_S1_PATH,
    EXPERIMENTS_DIR
)
from amazon_ml_er.code.business_entity_resolution.src.data_loader import stream_tsv_rows
from amazon_ml_er.code.business_entity_resolution.src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    extract_numeric_tokens
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.keys import extract_all_blocking_keys


def levenshtein_distance(s1: str, s2: str) -> int:
    """Standard Levenshtein distance."""
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    # Keep only previous and current row
    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)
    
    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0[:] = v1
    return v0[len(s2)]


def edit_similarity(s1: str, s2: str) -> float:
    """Normalized Levenshtein similarity: 1 - dist / max(len1, len2)."""
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


def extract_char_ngrams(text: str, n: int) -> set:
    s = text.strip().lower()
    if len(s) < n:
        return {s} if s else set()
    return {s[i:i+n] for i in range(len(s) - n + 1)}


def jaccard_overlap(set1: set, set2: set) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def common_prefix_len(s1: str, s2: str) -> int:
    i = 0
    while i < len(s1) and i < len(s2) and s1[i] == s2[i]:
        i += 1
    return i


def common_suffix_len(s1: str, s2: str) -> int:
    i = 0
    while i < len(s1) and i < len(s2) and s1[-(i+1)] == s2[-(i+1)]:
        i += 1
    return i


def run_task1_and_2():
    print("=== Executing Phase 2B Task 1 & Task 2 ===", flush=True)

    # 1. Load cached target records
    target_cache_path = Path("amazon_ml_er/splits/val_target_records.json")
    print(f"Loading target records from {target_cache_path}...", flush=True)
    with open(target_cache_path, "r", encoding="utf-8") as f:
        target_records = json.load(f)
    print(f"  Loaded {len(target_records):,} target records.")

    # 2. Load validation S1 queries
    val_ids = set()
    with open(VALIDATION_SPLIT_S1_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                val_ids.add(line.strip())

    print(f"Loading {len(val_ids):,} validation S1 queries...", flush=True)
    val_queries = {}
    for row in stream_tsv_rows(TRAIN_SOURCE1):
        eid = row.get("entity_id")
        if eid in val_ids:
            val_queries[eid] = row
    print(f"  Loaded {len(val_queries):,} S1 queries.")

    # 3. Load ground truth
    val_gt_pairs = []
    with open(TRAIN_GROUND_TRUTH, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader)
        for row in reader:
            if row and row[0].strip() in val_ids:
                s1_id = row[0].strip()
                for tid in row[1].strip().split(","):
                    tid = tid.strip()
                    if tid in target_records:
                        val_gt_pairs.append((s1_id, tid))
    print(f"  Loaded {len(val_gt_pairs):,} ground truth pairs.")

    # 4. Determine channel matches and separate captured vs missed
    print("Evaluating baseline channel matches for all ground truth pairs...", flush=True)
    captured_pairs = []
    missed_pairs = []

    for s1_id, tid in val_gt_pairs:
        s1 = val_queries[s1_id]
        t = target_records[tid]
        k_s1 = extract_all_blocking_keys(s1["business_name"], s1["business_address"])
        k_t = extract_all_blocking_keys(t["business_name"], t["business_address"])

        failed_channels = []
        if not (k_s1["ch1_key"] and k_s1["ch1_key"] == k_t["ch1_key"]):
            failed_channels.append("ch1_core_name")
        if not (k_s1["ch2_top2"] and k_s1["ch2_top2"] == k_t["ch2_top2"]):
            failed_channels.append("ch2_top2_tokens")
        if not (k_s1["ch3_key"] and k_s1["ch3_key"] == k_t["ch3_key"]):
            failed_channels.append("ch3_street_num_token")
        if not (k_s1["ch4_key"] and k_s1["ch4_key"] == k_t["ch4_key"]):
            failed_channels.append("ch4_address_location")

        is_captured = len(failed_channels) < 4  # At least one channel matched
        pair_info = {
            "s1_id": s1_id,
            "target_id": tid,
            "s1": s1,
            "target": t,
            "k_s1": k_s1,
            "k_t": k_t,
            "failed_channels": failed_channels
        }

        if is_captured:
            captured_pairs.append(pair_info)
        else:
            missed_pairs.append(pair_info)

    print(f"  Baseline Captured Pairs: {len(captured_pairs):,} ({len(captured_pairs)/len(val_gt_pairs)*100:.2f}%)")
    print(f"  Baseline Missed Pairs:   {len(missed_pairs):,} ({len(missed_pairs)/len(val_gt_pairs)*100:.2f}%)")

    # 5. Categorize missed pairs
    categorized_missed = defaultdict(list)
    for p in missed_pairs:
        s1_name = p["s1"].get("business_name", "")
        t_name = p["target"].get("business_name", "")
        s1_addr = p["s1"].get("business_address", "").strip()
        t_addr = p["target"].get("business_address", "").strip()

        # Primary categories
        if not t_addr:
            cat = "missing_target_address"
        elif any("\u0900" <= c <= "\u0d7f" for c in t_name) and not any("\u0900" <= c <= "\u0d7f" for c in s1_name):
            cat = "transliteration"
        else:
            norm_s1_n = normalize_business_name(s1_name)
            norm_t_n = normalize_business_name(t_name)
            s1_nums = extract_numeric_tokens(s1_addr)
            t_nums = extract_numeric_tokens(t_addr)

            if norm_s1_n.significant_tokens and norm_s1_n.significant_tokens == norm_t_n.significant_tokens:
                cat = "legal_suffix"
            elif sorted(norm_s1_n.significant_tokens) == sorted(norm_t_n.significant_tokens) and norm_s1_n.significant_tokens:
                cat = "word_order"
            elif s1_nums and not t_nums:
                cat = "missing_street_number"
            else:
                s1_toks = set(norm_s1_n.tokens)
                t_toks = set(norm_t_n.tokens)
                norm_s1_a = normalize_business_address(s1_addr)
                norm_t_a = normalize_business_address(t_addr)
                a_overlap = len(set(norm_s1_a.tokens) & set(norm_t_a.tokens)) / max(1, len(set(norm_s1_a.tokens) | set(norm_t_a.tokens)))
                n_overlap = len(s1_toks & t_toks)
                if n_overlap == 0 and a_overlap >= 0.25:
                    cat = "trade_name_or_dba"
                elif n_overlap > 0:
                    cat = "name_typo_or_abbreviation"
                else:
                    cat = "unknown"

        p["category"] = cat
        categorized_missed[cat].append(p)

    print("\nCategorized Missed Pairs:")
    for cat, items in categorized_missed.items():
        print(f"  {cat:30s}: {len(items):6,d} ({len(items)/len(missed_pairs)*100:5.2f}%)")

    # 6. Extract representative samples: at least 100 per required category
    target_sample_cats = [
        "missing_target_address",
        "name_typo_or_abbreviation",
        "transliteration",
        "trade_name_or_dba",
        "missing_street_number",
        "word_order",
        "legal_suffix",
        "unknown"
    ]

    sampled_examples = []
    # Deterministic sampling: take first 120 of each category
    for cat in target_sample_cats:
        items = categorized_missed.get(cat, [])
        sample_count = min(120, len(items))
        sampled_examples.extend(items[:sample_count])

    print(f"\nTotal sampled missed examples for inspection: {len(sampled_examples)}")

    # 7. Write experiments/blocking/missed_examples.csv
    exp_dir = Path("amazon_ml_er/experiments/blocking")
    exp_dir.mkdir(parents=True, exist_ok=True)
    csv_path = exp_dir / "missed_examples.csv"

    fieldnames = [
        "failure_category",
        "s1_id",
        "s1_country",
        "s1_name",
        "s1_address",
        "target_id",
        "target_source",
        "target_name",
        "target_address",
        "norm_s1_name",
        "norm_target_name",
        "norm_s1_address",
        "norm_target_address",
        "failed_blocking_channels",
        "s1_has_address",
        "target_has_address",
        "s1_numeric_tokens",
        "target_numeric_tokens",
        "s1_name_token_count",
        "target_name_token_count",
        "s1_name_char_len",
        "target_name_char_len"
    ]

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in sampled_examples:
            s1 = p["s1"]
            t = p["target"]
            norm_s1_n = normalize_business_name(s1.get("business_name", ""))
            norm_t_n = normalize_business_name(t.get("business_name", ""))
            norm_s1_a = normalize_business_address(s1.get("business_address", ""))
            norm_t_a = normalize_business_address(t.get("business_address", ""))

            s1_nums = extract_numeric_tokens(s1.get("business_address", ""))
            t_nums = extract_numeric_tokens(t.get("business_address", ""))

            writer.writerow({
                "failure_category": p["category"],
                "s1_id": p["s1_id"],
                "s1_country": s1.get("country", ""),
                "s1_name": s1.get("business_name", ""),
                "s1_address": s1.get("business_address", ""),
                "target_id": p["target_id"],
                "target_source": t.get("source", ""),
                "target_name": t.get("business_name", ""),
                "target_address": t.get("business_address", ""),
                "norm_s1_name": " ".join(norm_s1_n.significant_tokens) if norm_s1_n.significant_tokens else norm_s1_n.alphanumeric,
                "norm_target_name": " ".join(norm_t_n.significant_tokens) if norm_t_n.significant_tokens else norm_t_n.alphanumeric,
                "norm_s1_address": norm_s1_a.cleaned_lower,
                "norm_target_address": norm_t_a.cleaned_lower,
                "failed_blocking_channels": "; ".join(p["failed_channels"]),
                "s1_has_address": bool(s1.get("business_address", "").strip()),
                "target_has_address": bool(t.get("business_address", "").strip()),
                "s1_numeric_tokens": "; ".join(s1_nums),
                "target_numeric_tokens": "; ".join(t_nums),
                "s1_name_token_count": len(norm_s1_n.tokens),
                "target_name_token_count": len(norm_t_n.tokens),
                "s1_name_char_len": len(s1.get("business_name", "")),
                "target_name_char_len": len(t.get("business_name", ""))
            })
    print(f"Saved {len(sampled_examples)} rows to {csv_path}")

    # 8. Write experiments/blocking/missed_examples.md
    md_path = exp_dir / "missed_examples.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Phase 2B: Inspection of Real Missed Ground-Truth Pairs\n\n")
        f.write("This report provides deterministic samples and failure diagnostics for ground-truth entity pairs that failed all 4 raw baseline blocking channels (`ch1_core_name`, `ch2_top2_tokens`, `ch3_street_num_token`, `ch4_address_location`).\n\n")
        f.write(f"**Total Validation True Pairs:** {len(val_gt_pairs):,}  \n")
        f.write(f"**Baseline Captured Pairs:** {len(captured_pairs):,} ({len(captured_pairs)/len(val_gt_pairs)*100:.2f}%)  \n")
        f.write(f"**Baseline Missed Pairs:** {len(missed_pairs):,} ({len(missed_pairs)/len(val_gt_pairs)*100:.2f}%)  \n\n")
        
        f.write("## 1. Distribution of Failure Categories (Empirically Measured on Real Target Records)\n\n")
        f.write("| Category | Count | % of Missed | Root Cause Summary |\n")
        f.write("| :--- | :--- | :--- | :--- |\n")
        f.write(f"| `name_typo_or_abbreviation` | {len(categorized_missed['name_typo_or_abbreviation']):,} | {len(categorized_missed['name_typo_or_abbreviation'])/len(missed_pairs)*100:.2f}% | Name has spelling typo, abbreviation, or extra tokens that break exact token hashing. Address numbers match or partially match. |\n")
        f.write(f"| `transliteration` | {len(categorized_missed['transliteration']):,} | {len(categorized_missed['transliteration'])/len(missed_pairs)*100:.2f}% | S1 name is Latin script, target name is Devanagari/Telugu/Gujarati/Bengali script. Zero Latin name match possible. |\n")
        f.write(f"| `trade_name_or_dba` | {len(categorized_missed['trade_name_or_dba']):,} | {len(categorized_missed['trade_name_or_dba'])/len(missed_pairs)*100:.2f}% | Entity uses different legal name vs trade name / website domain (e.g., `felixrubio.com`), but physical address matches. |\n")
        f.write(f"| `missing_street_number` | {len(categorized_missed['missing_street_number']):,} | {len(categorized_missed['missing_street_number'])/len(missed_pairs)*100:.2f}% | Target address lacks house/street numbers (e.g. market street only), breaking numeric address channels. |\n")
        f.write(f"| `missing_target_address` | {len(categorized_missed['missing_target_address']):,} | {len(categorized_missed['missing_target_address'])/len(missed_pairs)*100:.2f}% | Target record has empty address field `\"\"`. Only name signals can resolve these. |\n")
        f.write(f"| `unknown` | {len(categorized_missed['unknown']):,} | {len(categorized_missed['unknown'])/len(missed_pairs)*100:.2f}% | Highly distorted or acronym-only records with weak shared textual tokens. |\n")
        f.write(f"| `word_order` | {len(categorized_missed['word_order']):,} | {len(categorized_missed['word_order'])/len(missed_pairs)*100:.2f}% | Tokens transposed across positions beyond top-2 sorted window. |\n")
        f.write(f"| `legal_suffix` | {len(categorized_missed['legal_suffix']):,} | {len(categorized_missed['legal_suffix'])/len(missed_pairs)*100:.2f}% | Unstandardized legal abbreviations (e.g. rare punctuation/brackets). |\n\n")

        f.write("--- \n\n")
        f.write("## 2. Detailed Sample Inspection by Failure Category\n\n")

        for cat in ["missing_target_address", "name_typo_or_abbreviation", "transliteration", "trade_name_or_dba", "missing_street_number"]:
            items = categorized_missed.get(cat, [])
            f.write(f"### Category: `{cat}` (Total: {len(items):,} pairs; Showing 10 Detailed Examples)\n\n")
            f.write("| # | S1 ID & Name | S1 Address | Target ID & Name | Target Address | Failed Channels | Diagnostic Notes |\n")
            f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
            for idx, p in enumerate(items[:10], 1):
                s1 = p["s1"]
                t = p["target"]
                fc = ", ".join(p["failed_channels"])
                f.write(f"| {idx} | **{p['s1_id']}**<br>`{s1.get('business_name')}` | `{s1.get('business_address')}` | **{p['target_id']}** ({t.get('source')})<br>`{t.get('business_name')}` | `{t.get('business_address')}` | `{fc}` | Name tokens: `{normalize_business_name(s1.get('business_name')).significant_tokens}` vs `{normalize_business_name(t.get('business_name')).significant_tokens}` |\n")
            f.write("\n")
    print(f"Saved markdown report to {md_path}")

    # =========================================================================
    # Task 2: Compute Offline Name Similarity Distributions
    # =========================================================================
    print("\n--- Computing Task 2 Name Similarity Distributions ---", flush=True)

    def compute_pair_similarities(pair_list, max_eval=5000):
        # Deterministically sample if list is large to keep compute fast
        eval_items = pair_list[:max_eval]
        metrics = defaultdict(list)
        for p in eval_items:
            s1_name = p["s1"].get("business_name", "")
            t_name = p["target"].get("business_name", "")

            norm_s1 = normalize_business_name(s1_name)
            norm_t = normalize_business_name(t_name)

            s1_str = " ".join(norm_s1.tokens)
            t_str = " ".join(norm_t.tokens)

            # 1. Exact equality
            exact_eq = 1.0 if s1_str == t_str and s1_str else 0.0
            metrics["exact_name_equality"].append(exact_eq)

            # 2. Token Jaccard
            s1_toks = set(norm_s1.tokens)
            t_toks = set(norm_t.tokens)
            metrics["token_jaccard"].append(jaccard_overlap(s1_toks, t_toks))

            # 3. Char 2-gram overlap
            c2_s1 = extract_char_ngrams(s1_str, 2)
            c2_t = extract_char_ngrams(t_str, 2)
            metrics["char_2gram_overlap"].append(jaccard_overlap(c2_s1, c2_t))

            # 4. Char 3-gram overlap
            c3_s1 = extract_char_ngrams(s1_str, 3)
            c3_t = extract_char_ngrams(t_str, 3)
            metrics["char_3gram_overlap"].append(jaccard_overlap(c3_s1, c3_t))

            # 5. Char 4-gram overlap
            c4_s1 = extract_char_ngrams(s1_str, 4)
            c4_t = extract_char_ngrams(t_str, 4)
            metrics["char_4gram_overlap"].append(jaccard_overlap(c4_s1, c4_t))

            # 6. Normalized edit similarity
            metrics["edit_similarity"].append(edit_similarity(s1_str, t_str))

            # 7. Prefix overlap
            max_len = max(len(s1_str), len(t_str))
            prefix_rat = common_prefix_len(s1_str, t_str) / max_len if max_len > 0 else 1.0
            metrics["prefix_overlap"].append(prefix_rat)

            # 8. Suffix overlap
            suffix_rat = common_suffix_len(s1_str, t_str) / max_len if max_len > 0 else 1.0
            metrics["suffix_overlap"].append(suffix_rat)

            # 9. Length ratio
            min_len = min(len(s1_str), len(t_str))
            len_ratio = min_len / max_len if max_len > 0 else 1.0
            metrics["length_ratio"].append(len_ratio)

            # 10. Token containment
            if s1_toks and t_toks:
                containment = 1.0 if (s1_toks.issubset(t_toks) or t_toks.issubset(s1_toks)) else 0.0
            else:
                containment = 0.0
            metrics["token_containment"].append(containment)

        # Compute summary distribution stats
        summary = {}
        for feature, vals in metrics.items():
            arr = np.array(vals)
            summary[feature] = {
                "mean": float(round(arr.mean(), 4)),
                "median": float(round(np.median(arr), 4)),
                "p25": float(round(np.percentile(arr, 25), 4)),
                "p75": float(round(np.percentile(arr, 75), 4)),
                "p90": float(round(np.percentile(arr, 90), 4)),
                "p95": float(round(np.percentile(arr, 95), 4))
            }
        return summary

    # (A) Missed pairs with empty target address
    missed_empty_addr = [p for p in missed_pairs if not p["target"].get("business_address", "").strip()]
    print(f"Computing similarity for missed pairs with empty target address ({len(missed_empty_addr):,} pairs)...")
    stats_missed_empty = compute_pair_similarities(missed_empty_addr)

    # (B) All missed pairs
    print(f"Computing similarity for all missed pairs ({len(missed_pairs):,} pairs, sample 5,000)...")
    stats_all_missed = compute_pair_similarities(missed_pairs, max_eval=5000)

    # (C) Correctly blocked true matches
    print(f"Computing similarity for correctly blocked true matches ({len(captured_pairs):,} pairs, sample 5,000)...")
    stats_captured = compute_pair_similarities(captured_pairs, max_eval=5000)

    task2_results = {
        "missed_empty_target_address": {
            "evaluated_pairs": len(missed_empty_addr),
            "distribution": stats_missed_empty
        },
        "all_missed_pairs": {
            "evaluated_pairs": min(5000, len(missed_pairs)),
            "total_missed_pairs": len(missed_pairs),
            "distribution": stats_all_missed
        },
        "correctly_blocked_matches": {
            "evaluated_pairs": min(5000, len(captured_pairs)),
            "total_captured_pairs": len(captured_pairs),
            "distribution": stats_captured
        }
    }

    task2_out = exp_dir / "name_similarity_analysis.json"
    with open(task2_out, "w", encoding="utf-8") as f:
        json.dump(task2_results, f, indent=2)
    print(f"Saved Task 2 similarity results to {task2_out}")

    print("\nTask 2 Summary Table Comparison (Mean Values):")
    print(f"{'Feature':25s} | {'Missed (Empty Addr)':20s} | {'All Missed':15s} | {'Correctly Blocked':18s}")
    print("-" * 85)
    for feat in stats_captured.keys():
        m_emp = stats_missed_empty[feat]["mean"]
        m_all = stats_all_missed[feat]["mean"]
        m_cap = stats_captured[feat]["mean"]
        print(f"{feat:25s} | {m_emp:20.4f} | {m_all:15.4f} | {m_cap:18.4f}")

    print("\n=== Task 1 & Task 2 Execution Completed Successfully ===", flush=True)


if __name__ == "__main__":
    run_task1_and_2()
