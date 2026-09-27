"""
End-to-End Autonomous Pipeline Orchestrator for Phase 4 Test Inference.
Monitors US completion -> Launches India -> Assembles Final TSVs -> Validates -> Audits.
"""

import os
import sys
import time
import json
import csv
import hashlib
import subprocess
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SHARDS_DIR = repo_root / "experiments" / "work" / "shards"
COUNTRY_DIR = repo_root / "experiments" / "work" / "country_results"
OUTPUT_DIR = repo_root / "output"
TEST_SOURCE1 = repo_root / "dataset" / "test" / "test_source1.tsv"

def log(msg: str):
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)

def wait_for_us():
    us_matching = COUNTRY_DIR / "matching_US.tsv"
    log("Monitoring US inference execution...")
    while not us_matching.exists():
        manifests = list(SHARDS_DIR.glob("manifest_US_*.json"))
        log(f"  US Progress: {len(manifests)}/133 chunks complete ({len(manifests)*5000:,} entities).")
        time.sleep(30)
    log(f"US Inference COMPLETE: {us_matching} generated ({us_matching.stat().st_size / (1024*1024):.1f} MB).")

def run_india():
    india_matching = COUNTRY_DIR / "matching_India.tsv"
    if india_matching.exists():
        log(f"India already complete: {india_matching}.")
        return

    log("Starting India inference under optimized single-worker policy...")
    cmd = [
        "caffeinate", "-dimsu",
        "taskpolicy", "-a",
        sys.executable, "-u",
        str(repo_root / "amazon_ml_er" / "experiments" / "test_inference" / "run_chunked_inference.py"),
        "--country", "India",
        "--chunk-size", "5000"
    ]
    india_log = repo_root / "experiments" / "work" / "india_inference.log"
    with open(india_log, "w", encoding="utf-8") as out_f:
        proc = subprocess.Popen(cmd, stdout=out_f, stderr=subprocess.STDOUT)
    
    log(f"India process spawned (PID={proc.pid}). Streaming to {india_log}...")
    while proc.poll() is None:
        manifests = list(SHARDS_DIR.glob("manifest_India_*.json"))
        log(f"  India Progress: {len(manifests)}/162 chunks complete ({len(manifests)*5000:,} entities).")
        time.sleep(60)
    
    if proc.returncode != 0:
        log(f"ERROR: India inference exited with code {proc.returncode}!")
        sys.exit(proc.returncode)
    log("India inference completed successfully.")

def assemble_and_validate():
    log("Assembling final submission files...")
    cmd = [
        sys.executable,
        str(repo_root / "amazon_ml_er" / "experiments" / "test_inference" / "run_chunked_inference.py"),
        "--assemble"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.returncode != 0:
        log(f"Assembly failed: {res.stderr}")
        sys.exit(1)

    matching_file = OUTPUT_DIR / "matching_results.tsv"
    candidate_file = OUTPUT_DIR / "candidate_pairs.tsv"

    log("Running official validator...")
    val_cmd = [
        sys.executable,
        str(repo_root / "utils" / "validate_submission.py"),
        "--matching", str(matching_file),
        "--candidate", str(candidate_file),
        "--test-dir", str(repo_root / "dataset" / "test"),
        "--check-ids"
    ]
    val_res = subprocess.run(val_cmd, capture_output=True, text=True)
    val_log_file = repo_root / "experiments" / "test_inference" / "validator_result.txt"
    val_log_file.write_text(val_res.stdout + "\n" + val_res.stderr, encoding="utf-8")
    print(val_res.stdout)
    if val_res.returncode != 0:
        log(f"CRITICAL: Validator failed with code {val_res.returncode}!")
        sys.exit(val_res.returncode)
    log("OFFICIAL VALIDATOR PASSED (Exit Code = 0)!")

    # Checksums
    log("Computing SHA-256 hashes...")
    m_sha = hashlib.sha256(matching_file.read_bytes()).hexdigest()
    c_sha = hashlib.sha256(candidate_file.read_bytes()).hexdigest()
    log(f"matching_results.tsv SHA-256: {m_sha}")
    log(f"candidate_pairs.tsv  SHA-256: {c_sha}")

    audit = {
        "status": "PASS",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_test_s1_entities": 1732544,
        "matching_results_sha256": m_sha,
        "candidate_pairs_sha256": c_sha,
        "matching_results_bytes": matching_file.stat().st_size,
        "candidate_pairs_bytes": candidate_file.stat().st_size,
        "validator_exit_code": 0
    }
    audit_file = repo_root / "experiments" / "test_inference" / "final_submission_audit.json"
    audit_file.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log("PHASE 4 STATUS = PASS! All outputs assembled and verified.")

if __name__ == "__main__":
    wait_for_us()
    run_india()
    assemble_and_validate()
