"""
Progressive Scalability Benchmark and Failure Analysis Harness.
Orchestrates progressive validation benchmarks (1K, 5K, 10K, 50K),
categorizes missed true matches, and exports structured reports.
"""

import time
import os
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict, Counter
import psutil

from ..config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    VALIDATION_SPLIT_S1_PATH,
    EXPERIMENTS_DIR
)
from ..data_loader import stream_tsv_rows
from ..ground_truth import parse_ground_truth_file
from ..preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    extract_numeric_tokens
)
from .channels import (
    build_channel_1_normalized_name,
    build_channel_2_sorted_tokens,
    build_channel_3_street_num_token,
    build_channel_4_address_location,
    build_channel_5_posting_list,
    evaluate_single_channel
)
from .union import analyze_union_and_contributions
from .pruning import evaluate_candidate_budgets


def categorize_missed_pair(
    s1_rec: Dict[str, str],
    target_rec: Dict[str, str]
) -> str:
    """
    Empirically categorize the primary root cause for a missed ground-truth pair.
    """
    s1_name = s1_rec.get("business_name", "")
    t_name = target_rec.get("business_name", "")
    s1_addr = s1_rec.get("business_address", "")
    t_addr = target_rec.get("business_address", "")

    # 1. Missing target address
    if not t_addr.strip():
        return "missing_target_address"

    # 2. Check for non-Latin script in target (transliteration)
    if any("\u0900" <= c <= "\u0d7f" for c in t_name) and not any("\u0900" <= c <= "\u0d7f" for c in s1_name):
        return "transliteration_indic_name"

    # 3. Legal suffix discrepancy
    norm_s1 = normalize_business_name(s1_name)
    norm_t = normalize_business_name(t_name)
    if norm_s1.significant_tokens and norm_t.significant_tokens:
        if norm_s1.significant_tokens == norm_t.significant_tokens:
            return "legal_suffix_only"

    # 4. Word-order difference
    if sorted(norm_s1.significant_tokens) == sorted(norm_t.significant_tokens):
        return "word_order_transposition"

    # 5. Missing number in address
    s1_nums = extract_numeric_tokens(s1_addr)
    t_nums = extract_numeric_tokens(t_addr)
    if not t_nums and s1_nums:
        return "missing_street_number"

    # 6. Trade name / DBA (name token overlap == 0, but address overlap > 0.5)
    s1_toks = set(norm_s1.tokens)
    t_toks = set(norm_t.tokens)
    norm_s1_addr = normalize_business_address(s1_addr)
    norm_t_addr = normalize_business_address(t_addr)
    s1_atoks = set(norm_s1_addr.tokens)
    t_atoks = set(norm_t_addr.tokens)

    name_overlap = len(s1_toks & t_toks)
    addr_overlap = len(s1_atoks & t_atoks) / max(1, len(s1_atoks | t_atoks))

    if name_overlap == 0 and addr_overlap >= 0.4:
        return "trade_name_or_dba"

    # 7. Name typo (character overlap high, but token overlap low)
    if name_overlap > 0:
        return "name_typo_or_abbreviation"

    return "unknown"


def run_failure_analysis(
    missed_pairs: List[Tuple[str, str]],
    query_records: Dict[str, Dict[str, str]],
    target_records: Dict[str, Dict[str, str]]
) -> Dict[str, Any]:
    """Analyze root causes for all missed true match pairs."""
    categories = Counter()
    sample_offenders = defaultdict(list)

    for s1_id, tid in missed_pairs:
        s1_rec = query_records.get(s1_id, {})
        t_rec = target_records.get(tid, {})
        cat = categorize_missed_pair(s1_rec, t_rec)
        categories[cat] += 1
        if len(sample_offenders[cat]) < 3:
            sample_offenders[cat].append({
                "s1_id": s1_id,
                "s1_name": s1_rec.get("business_name", ""),
                "s1_addr": s1_rec.get("business_address", ""),
                "target_id": tid,
                "target_name": t_rec.get("business_name", ""),
                "target_addr": t_rec.get("business_address", "")
            })

    total_missed = len(missed_pairs)
    breakdown = {}
    for cat, count in categories.most_common():
        breakdown[cat] = {
            "count": count,
            "percentage": float(round(count / total_missed * 100, 2)) if total_missed > 0 else 0.0,
            "samples": sample_offenders[cat]
        }

    return {
        "total_missed_pairs": total_missed,
        "category_breakdown": breakdown
    }
