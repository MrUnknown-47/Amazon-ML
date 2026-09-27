"""
Full Data Audit and Schema Verification Runner.
Executes an empirical audit across all train and test files,
records execution time and peak memory, and exports both
machine-readable (JSON) and human-readable (Markdown) reports.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any
import psutil

from .config import (
    TRAIN_SOURCE1,
    TRAIN_SOURCE2,
    TRAIN_SOURCE3,
    TRAIN_GROUND_TRUTH,
    TEST_SOURCE1,
    TEST_SOURCE2,
    TEST_SOURCE3,
    DATA_AUDIT_JSON,
    DATA_AUDIT_MD,
    REPORTS_DIR,
    SOURCE_COLUMNS
)
from .data_loader import audit_tsv_file
from .ground_truth import audit_ground_truth


def run_full_dataset_audit() -> Dict[str, Any]:
    """
    Execute streaming audit on all challenge files and return consolidated audit report.
    """
    start_time = time.time()
    process = psutil.Process(os.getpid())
    start_mem_mb = process.memory_info().rss / (1024 * 1024)

    print("=== Starting Full Dataset and Schema Audit ===", flush=True)

    files_to_audit = [
        ("train_source1", TRAIN_SOURCE1, "S1-"),
        ("train_source2", TRAIN_SOURCE2, "S2-"),
        ("train_source3", TRAIN_SOURCE3, "S3-"),
        ("test_source1", TEST_SOURCE1, "S1-"),
        ("test_source2", TEST_SOURCE2, "S2-"),
        ("test_source3", TEST_SOURCE3, "S3-")
    ]

    source_audits = {}
    for name, path, prefix in files_to_audit:
        print(f"Auditing {name} ({path.name})...", flush=True)
        t0 = time.time()
        res = audit_tsv_file(path, expected_columns=SOURCE_COLUMNS, id_prefix=prefix)
        elapsed = round(time.time() - t0, 2)
        res["audit_duration_seconds"] = elapsed
        source_audits[name] = res
        print(f"  Done in {elapsed}s | Rows: {res['total_rows']:,} | Null Addr: {res['empty_string_counts']['business_address']:,}", flush=True)

    # Ground truth audit
    print("Auditing train_ground_truth.tsv...", flush=True)
    t0 = time.time()
    gt_audit = audit_ground_truth(
        gt_path=TRAIN_GROUND_TRUTH,
        s1_path=TRAIN_SOURCE1,
        verify_target_ids=False  # Keep lightweight
    )
    elapsed_gt = round(time.time() - t0, 2)
    gt_audit["audit_duration_seconds"] = elapsed_gt
    print(f"  Done in {elapsed_gt}s | Singletons: {gt_audit['singletons_count']:,} ({gt_audit['singletons_rate']*100:.2f}%)", flush=True)

    end_time = time.time()
    peak_mem_mb = process.memory_info().rss / (1024 * 1024)
    total_duration = round(end_time - start_time, 2)

    consolidated = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
            "total_audit_duration_seconds": total_duration,
            "start_memory_mb": round(start_mem_mb, 2),
            "peak_memory_mb": round(peak_mem_mb, 2),
            "memory_delta_mb": round(peak_mem_mb - start_mem_mb, 2)
        },
        "sources": source_audits,
        "ground_truth": gt_audit
    }

    # Ensure reports directory exists
    os.makedirs(REPORTS_DIR, exist_ok=True)

    # Save JSON report
    with open(DATA_AUDIT_JSON, "w", encoding="utf-8") as f:
        json.dump(consolidated, f, indent=2)
    print(f"Saved machine-readable audit report to {DATA_AUDIT_JSON}")

    # Generate Markdown report
    md_content = generate_markdown_audit_report(consolidated)
    with open(DATA_AUDIT_MD, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Saved human-readable audit report to {DATA_AUDIT_MD}")

    return consolidated


def generate_markdown_audit_report(data: Dict[str, Any]) -> str:
    """Generate clean Markdown summary from audit dictionary."""
    meta = data["metadata"]
    srcs = data["sources"]
    gt = data["ground_truth"]

    lines = [
        "# Business Entity Resolution: Empirical Data & Schema Audit Report",
        "",
        f"**Audit Execution Timestamp:** {meta['timestamp']}  ",
        f"**Total Audit Duration:** {meta['total_audit_duration_seconds']} seconds  ",
        f"**Peak Memory Usage:** {meta['peak_memory_mb']} MB (Delta: {meta['memory_delta_mb']} MB)  ",
        "",
        "---",
        "",
        "## 1. Source TSV Overview & Schema Conformance",
        "",
        "| Source | Total Rows | Unique IDs | Duplicate IDs | Null Names | Null Addresses (%) | Countries Found |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for name, s in srcs.items():
        n_rows = s["total_rows"]
        null_addr = s["empty_string_counts"]["business_address"]
        null_addr_pct = (null_addr / n_rows * 100) if n_rows else 0.0
        countries_str = ", ".join(f"{k} ({v:,})" for k, v in s["country_counts"].items())
        lines.append(
            f"| `{name}` | {n_rows:,} | {s['unique_ids']:,} | {s['duplicate_ids']} | "
            f"{s['empty_string_counts']['business_name']} | {null_addr:,} ({null_addr_pct:.2f}%) | {countries_str} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Text Length Statistics (Characters)",
        "",
        "| Source | Field | Min | Median | Mean | P95 | Max |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ])

    for name, s in srcs.items():
        nl = s.get("name_length_stats", {})
        al = s.get("addr_length_stats", {})
        if nl:
            lines.append(f"| `{name}` | Name | {nl.get('min')} | {nl.get('median')} | {nl.get('mean')} | {nl.get('p95')} | {nl.get('max')} |")
        if al:
            lines.append(f"| `{name}` | Address | {al.get('min')} | {al.get('median')} | {al.get('mean')} | {al.get('p95')} | {al.get('max')} |")

    lines.extend([
        "",
        "---",
        "",
        "## 3. Ground Truth Analysis & Cardinality",
        "",
        f"- **Total S1 Entities in Ground Truth:** {gt['total_s1_entities']:,}",
        f"- **Singletons (0 matches):** {gt['singletons_count']:,} ({gt['singletons_rate']*100:.4f}%)",
        f"- **Non-Singletons (>= 1 match):** {gt['non_singletons_count']:,} ({gt['non_singletons_rate']*100:.4f}%)",
        f"- **Matches in S2 Only:** {gt['s2_only_count']:,} ({gt['s2_only_count']/gt['total_s1_entities']*100:.2f}%)",
        f"- **Matches in S3 Only:** {gt['s3_only_count']:,} ({gt['s3_only_count']/gt['total_s1_entities']*100:.2f}%)",
        f"- **Matches in Both S2 & S3:** {gt['both_s2_and_s3_count']:,} ({gt['both_s2_and_s3_count']/gt['total_s1_entities']*100:.2f}%)",
        f"- **Unique Target IDs Matched:** {gt['unique_targets_matched']:,}",
        f"- **Target IDs Mapping to Multiple S1s:** {gt['targets_mapping_to_multiple_s1']} ({gt['target_multiple_s1_rate']*100:.4f}%)",
        "",
        "### Matches Per S1 Distribution",
        f"- **Min:** {gt['match_count_summary']['min']}",
        f"- **P25:** {gt['match_count_summary']['p25']}",
        f"- **Median:** {gt['match_count_summary']['median']}",
        f"- **Mean:** {gt['match_count_summary']['mean']}",
        f"- **P75:** {gt['match_count_summary']['p75']}",
        f"- **P95:** {gt['match_count_summary']['p95']}",
        f"- **P99:** {gt['match_count_summary']['p99']}",
        f"- **Max:** {gt['match_count_summary']['max']}",
        "",
        "### Country Breakdown in Ground Truth",
        ""
    ])

    for country, cs in gt.get("country_breakdown", {}).items():
        lines.append(
            f"- **{country}:** Total S1 = {cs['total_s1']:,} | Singletons = {cs['singletons']:,} "
            f"({cs['singleton_rate']*100:.2f}%) | Total Matches = {cs['total_matches']:,} "
            f"| Avg Matches/S1 = {cs['avg_matches_per_s1']:.2f}"
        )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run_full_dataset_audit()
