"""
Ground truth parser, verification, and audit module for Business Entity Resolution.
"""

import csv
import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import Counter
import numpy as np

from .config import GROUND_TRUTH_COLUMNS


class GroundTruthValidationError(Exception):
    """Raised when ground truth data violates competition integrity rules."""
    pass


def parse_ground_truth_file(
    file_path: Path,
    limit: Optional[int] = None
) -> Dict[str, Set[str]]:
    """
    Parse train_ground_truth.tsv into a dictionary mapping:
        source1_entity_id -> set of matched entity_ids {S2-..., S3-...}
    Singletons are mapped to empty sets: set().
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")

    ground_truth = {}
    seen_s1 = set()

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            header = next(reader)
        except StopIteration:
            raise GroundTruthValidationError(f"Ground truth file is empty: {file_path}")

        if header != GROUND_TRUTH_COLUMNS:
            raise GroundTruthValidationError(
                f"Invalid ground truth header. Got {header}, expected {GROUND_TRUTH_COLUMNS}"
            )

        for line_num, row in enumerate(reader, start=2):
            if limit is not None and len(ground_truth) >= limit:
                break

            if not row:
                continue

            s1_id = row[0].strip()
            if not s1_id.startswith("S1-"):
                raise GroundTruthValidationError(
                    f"Line {line_num}: source1_entity_id '{s1_id}' does not have 'S1-' prefix."
                )

            if s1_id in seen_s1:
                raise GroundTruthValidationError(
                    f"Line {line_num}: Duplicate source1_entity_id '{s1_id}' in ground truth."
                )
            seen_s1.add(s1_id)

            matched_str = row[1].strip() if len(row) > 1 else ""
            if not matched_str:
                ground_truth[s1_id] = set()
                continue

            matched_ids = [m.strip() for m in matched_str.split(",") if m.strip()]

            # Intra-list duplicate check
            if len(matched_ids) != len(set(matched_ids)):
                raise GroundTruthValidationError(
                    f"Line {line_num}: Duplicate target ID found within match list for '{s1_id}'"
                )

            # Check target prefixes
            for mid in matched_ids:
                if mid.startswith("S1-"):
                    raise GroundTruthValidationError(
                        f"Line {line_num}: Self-match '{mid}' found for S1 entity '{s1_id}'."
                    )
                if not mid.startswith(("S2-", "S3-")):
                    raise GroundTruthValidationError(
                        f"Line {line_num}: Target ID '{mid}' lacks valid S2- or S3- prefix."
                    )

            ground_truth[s1_id] = set(matched_ids)

    return ground_truth


def audit_ground_truth(
    gt_path: Path,
    s1_path: Path,
    s2_path: Optional[Path] = None,
    s3_path: Optional[Path] = None,
    verify_target_ids: bool = False
) -> Dict[str, Any]:
    """
    Perform deep verification and statistical audit of train_ground_truth.tsv.
    Validates:
    - Exactly 1 row per S1 in train_source1
    - No self-matches
    - No intra-list duplicates
    - Valid target ID existence (optional full check)
    - Cardinality and singleton metrics
    """
    # 1. Load S1 IDs and country mapping
    s1_countries = {}
    with open(s1_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)
        for row in reader:
            if row:
                s1_countries[row[0].strip()] = row[3].strip() if len(row) > 3 else "UNKNOWN"

    total_s1_expected = len(s1_countries)

    # 2. Parse GT and gather stats
    gt_map = parse_ground_truth_file(gt_path)

    # Check S1 set equality
    gt_s1_set = set(gt_map.keys())
    expected_s1_set = set(s1_countries.keys())

    missing_in_gt = expected_s1_set - gt_s1_set
    extra_in_gt = gt_s1_set - expected_s1_set

    if missing_in_gt:
        raise GroundTruthValidationError(
            f"{len(missing_in_gt)} S1 entities from {s1_path} are missing in {gt_path}."
        )
    if extra_in_gt:
        raise GroundTruthValidationError(
            f"{len(extra_in_gt)} unexpected S1 entities found in {gt_path} not in {s1_path}."
        )

    # Metrics computation
    total_s1 = len(gt_map)
    singletons = 0
    non_singletons = 0
    s2_only = 0
    s3_only = 0
    both = 0

    match_counts = []
    target_to_s1: Dict[str, str] = {}
    target_multi_s1 = set()

    country_stats: Dict[str, Dict[str, Any]] = {}
    for c in set(s1_countries.values()):
        country_stats[c] = {"total_s1": 0, "singletons": 0, "total_matches": 0}

    for s1_id, targets in gt_map.items():
        cnt = len(targets)
        match_counts.append(cnt)
        country = s1_countries.get(s1_id, "UNKNOWN")
        country_stats[country]["total_s1"] += 1
        country_stats[country]["total_matches"] += cnt

        if cnt == 0:
            singletons += 1
            country_stats[country]["singletons"] += 1
        else:
            non_singletons += 1
            has_s2 = any(t.startswith("S2-") for t in targets)
            has_s3 = any(t.startswith("S3-") for t in targets)
            if has_s2 and not has_s3:
                s2_only += 1
            elif has_s3 and not has_s2:
                s3_only += 1
            elif has_s2 and has_s3:
                both += 1

            for t in targets:
                if t in target_to_s1:
                    target_multi_s1.add(t)
                else:
                    target_to_s1[t] = s1_id

    # Target ID existence verification (if paths provided)
    if verify_target_ids and s2_path and s3_path:
        valid_targets = set()
        for p in [s2_path, s3_path]:
            with open(p, "r", encoding="utf-8", newline="") as f:
                r = csv.reader(f, delimiter="\t")
                next(r, None)
                for row in r:
                    if row:
                        valid_targets.add(row[0].strip())
        unknown_targets = set(target_to_s1.keys()) - valid_targets
        if unknown_targets:
            raise GroundTruthValidationError(
                f"{len(unknown_targets)} matched targets in ground truth do not exist in S2 or S3!"
            )

    mc_arr = np.array(match_counts)
    audit_results = {
        "total_s1_entities": total_s1,
        "singletons_count": singletons,
        "singletons_rate": float(singletons / total_s1) if total_s1 else 0.0,
        "non_singletons_count": non_singletons,
        "non_singletons_rate": float(non_singletons / total_s1) if total_s1 else 0.0,
        "s2_only_count": s2_only,
        "s3_only_count": s3_only,
        "both_s2_and_s3_count": both,
        "unique_targets_matched": len(target_to_s1),
        "targets_mapping_to_multiple_s1": len(target_multi_s1),
        "target_multiple_s1_rate": float(len(target_multi_s1) / len(target_to_s1)) if target_to_s1 else 0.0,
        "match_count_summary": {
            "min": int(mc_arr.min()),
            "p25": float(np.percentile(mc_arr, 25)),
            "median": float(np.median(mc_arr)),
            "mean": float(round(mc_arr.mean(), 4)),
            "p75": float(np.percentile(mc_arr, 75)),
            "p95": float(np.percentile(mc_arr, 95)),
            "p99": float(np.percentile(mc_arr, 99)),
            "max": int(mc_arr.max())
        },
        "country_breakdown": {
            c: {
                "total_s1": s["total_s1"],
                "singletons": s["singletons"],
                "singleton_rate": float(s["singletons"] / s["total_s1"]) if s["total_s1"] else 0.0,
                "total_matches": s["total_matches"],
                "avg_matches_per_s1": float(round(s["total_matches"] / s["total_s1"], 4)) if s["total_s1"] else 0.0
            }
            for c, s in country_stats.items()
        }
    }
    return audit_results
