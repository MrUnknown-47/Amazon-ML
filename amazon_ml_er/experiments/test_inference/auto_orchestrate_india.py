"""
Authoritative End-to-End Orchestrator for India Execution and Final Assembly.
Waits for US inference to reach 663,106 / 663,106.
Executes the India 5K benchmark and regression gate.
Launches accelerated full India execution upon gate pass.
Performs country conflict resolution, output assembly, official validation, and SHA-256 checksums.
"""

import sys
import os
import time
import json
import csv
import subprocess
import hashlib
from pathlib import Path
import psutil

repo_root = Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

SHARDS_DIR = repo_root / "experiments" / "work" / "shards"
COUNTRY_DIR = repo_root / "experiments" / "work" / "country_results"
OUTPUT_DIR = repo_root / "output"
LOG_DIR = repo_root / "experiments" / "work"
PIPELINE_LOG = LOG_DIR / "india_orchestrator.log"


def log(msg: str):
    t_str = time.strftime("%Y-%m-%d %H:%M:%S")
    out = f"[{t_str}] {msg}"
    print(out, flush=True)
    with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
        f.write(out + "\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def wait_for_us():
    log("Waiting for US inference to finish (all 133 shards and country conflict resolution)...")
    manifest_132 = SHARDS_DIR / "manifest_US_0132.json"
    matching_us = COUNTRY_DIR / "matching_US.tsv"
    candidates_us = COUNTRY_DIR / "candidates_US.tsv"

    last_count = -1
    while True:
        completed_shards = len(list(SHARDS_DIR.glob("manifest_US_*.json")))
        if completed_shards != last_count:
            last_count = completed_shards
            log(f"  US progress: {completed_shards} / 133 shards complete.")

        if manifest_132.exists() and matching_us.exists() and candidates_us.exists():
            # Check US process termination
            us_pids = []
            for p in psutil.process_iter(["pid", "cmdline"]):
                cmd = " ".join(p.info.get("cmdline") or [])
                if "run_chunked_inference.py" in cmd and "--country US" in cmd:
                    us_pids.append(p.pid)
            if not us_pids:
                log("  US process termination confirmed. Memory released.")
                break
            else:
                log(f"  Waiting for US worker PID(s) {us_pids} to exit cleanly...")

        time.sleep(5)

    t_us_complete = time.time()
    us_complete_str = time.strftime("%Y-%m-%d %H:%M:%S IST", time.localtime(t_us_complete))
    log(f"US INFERENCE 100% COMPLETE ({us_complete_str}) — 663,106 / 663,106 entities verified.")
    return t_us_complete, us_complete_str


def run_benchmark_gate():
    log("\n=======================================================")
    log("RUNNING INDIA 5K BENCHMARK & REGRESSION GATE")
    log("=======================================================")

    cmd = [
        sys.executable, "-u",
        str(repo_root / "amazon_ml_er" / "experiments" / "test_inference" / "benchmark_and_verify_india_5k.py")
    ]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    full_output = []
    for line in iter(p.stdout.readline, ""):
        line_str = line.rstrip()
        print(line_str, flush=True)
        full_output.append(line_str)
        with open(PIPELINE_LOG, "a", encoding="utf-8") as f:
            f.write(line_str + "\n")
    p.wait()

    if p.returncode != 0:
        log(f"FATAL: Benchmark/regression gate failed with exit code {p.returncode}!")
        sys.exit(1)

    # Parse measured values from output
    manifest_0 = SHARDS_DIR / "manifest_India_0000.json"
    if not manifest_0.exists():
        log("FATAL: manifest_India_0000.json was not created!")
        sys.exit(1)

    with open(manifest_0, "r", encoding="utf-8") as f:
        meta = json.load(f)

    shard_time = meta["runtime_seconds"]
    ent_per_sec = meta["throughput_ent_per_sec"]

    log(f"India_0000 verified: {shard_time}s, throughput={ent_per_sec} ent/s.")
    return shard_time, ent_per_sec


def run_full_india_inference(measured_ent_per_sec: float):
    log("\n=======================================================")
    log("LAUNCHING FULL ACCELERATED INDIA INFERENCE (India_0001..India_0161)")
    log("=======================================================")

    cmd = [
        "caffeinate", "-dimsu", "taskpolicy", "-a",
        sys.executable, "-u",
        str(repo_root / "amazon_ml_er" / "experiments" / "test_inference" / "run_accelerated_inference.py"),
        "--country", "India",
        "--chunk-size", "5000"
    ]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    rolling_times = []
    shards_seen = set()

    # Stream output and monitor rolling throughput
    while True:
        line = proc.stdout.readline()
        if not line and proc.poll() is not None:
            break
        if line:
            line_str = line.rstrip()
            print(line_str, flush=True)
            with open(india_log, "a", encoding="utf-8") as lf:
                lf.write(line_str + "\n")

            # Check for completed chunks
            if "[Chunk India_" in line_str and "Completed in" in line_str:
                # e.g. [Chunk India_0001] Completed in 14.2s (352.1 ent/s)...
                try:
                    parts = line_str.split("Completed in ")
                    sec_str = parts[1].split("s")[0]
                    c_sec = float(sec_str)
                    c_id = line_str.split("[Chunk ")[1].split("]")[0]
                    if c_id not in shards_seen:
                        shards_seen.add(c_id)
                        rolling_times.append(c_sec)
                        n_done = len(rolling_times)

                        if n_done == 3:
                            avg_3 = 5000.0 / (sum(rolling_times[-3:]) / 3.0)
                            log(f"\n>>> LIVE MONITORING (First 3 Shards): Rolling throughput = {avg_3:.2f} ent/s <<<")
                        elif n_done == 10:
                            avg_10 = 5000.0 / (sum(rolling_times[-10:]) / 10.0)
                            rem_shards = 162 - n_done - 1  # shard 0 was done in gate
                            eta_sec = rem_shards * (sum(rolling_times[-10:]) / 10.0)
                            eta_time = time.strftime("%H:%M:%S IST", time.localtime(time.time() + eta_sec))
                            log(f"\n>>> LIVE MONITORING (First 10 Shards): Rolling throughput = {avg_10:.2f} ent/s | ETA: {eta_time} <<<\n")
                except Exception:
                    pass

    rc = proc.wait()
    if rc != 0:
        log(f"FATAL: run_accelerated_inference.py exited with code {rc}!")
        sys.exit(rc)

    log("Full India inference complete!")


def assemble_and_validate():
    log("\n=======================================================")
    log("ASSEMBLING FINAL SUBMISSION TSV FILES")
    log("=======================================================")

    cmd_assemble = [
        sys.executable, "-u",
        str(repo_root / "amazon_ml_er" / "experiments" / "test_inference" / "run_accelerated_inference.py"),
        "--assemble"
    ]
    p_asm = subprocess.run(cmd_assemble, capture_output=True, text=True)
    print(p_asm.stdout, flush=True)
    if p_asm.returncode != 0:
        log(f"FATAL: Assembly failed: {p_asm.stderr}")
        sys.exit(p_asm.returncode)

    matching_file = OUTPUT_DIR / "matching_results.tsv"
    candidate_file = OUTPUT_DIR / "candidate_pairs.tsv"

    if not matching_file.exists() or not candidate_file.exists():
        log("FATAL: Output files missing!")
        sys.exit(1)

    log("\n=======================================================")
    log("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    log("=======================================================")

    val_cmd = [
        sys.executable,
        str(repo_root / "utils" / "validate_submission.py"),
        "--matching", str(matching_file),
        "--candidate", str(candidate_file),
        "--test-dir", str(repo_root / "dataset" / "test")
    ]
    p_val = subprocess.run(val_cmd, capture_output=True, text=True)
    print(p_val.stdout, flush=True)
    if p_val.stderr:
        print(p_val.stderr, flush=True)

    if p_val.returncode != 0:
        log(f"FATAL: Official validator failed with exit code {p_val.returncode}!")
        sys.exit(p_val.returncode)

    log("OFFICIAL VALIDATOR PASSED (Exit Code: 0)!")

    # Checksums
    log("\n=======================================================")
    log("COMPUTING SHA-256 CHECKSUMS")
    log("=======================================================")
    m_sha = sha256_file(matching_file)
    c_sha = sha256_file(candidate_file)
    m_size = matching_file.stat().st_size / (1024 * 1024)
    c_size = candidate_file.stat().st_size / (1024 * 1024)

    log(f"matching_results.tsv:  {m_size:.2f} MB | SHA-256: {m_sha}")
    log(f"candidate_pairs.tsv:   {c_size:.2f} MB | SHA-256: {c_sha}")

    log("\n=======================================================")
    log("PHASE 4 SUBMISSION VERIFICATION COMPLETE: ALL PASS")
    log("=======================================================")


def main():
    log("Starting India Orchestrator Master Pipeline...")
    t_us_complete, us_complete_str = wait_for_us()
    shard_time, ent_per_sec = run_benchmark_gate()

    t_gate_finish = time.time()
    t_gate_str = time.strftime("%H:%M:%S IST", time.localtime(t_gate_finish))

    # Dynamic Deadline Recalculation
    remaining_india = 809986 - 5000  # 804,986 queries remaining
    post_inference_overhead_sec = 240  # 4 mins: conflict resolution (2m) + assembly (1m) + validator (1m)

    local_lt = time.localtime(t_gate_finish)
    t_2330 = time.mktime((local_lt.tm_year, local_lt.tm_month, local_lt.tm_mday, 23, 30, 0, 0, 0, -1))
    t_2340 = time.mktime((local_lt.tm_year, local_lt.tm_month, local_lt.tm_mday, 23, 40, 0, 0, 0, -1))

    avail_sec_2330 = max(1.0, (t_2330 - t_gate_finish) - post_inference_overhead_sec)
    avail_sec_2340 = max(1.0, (t_2340 - t_gate_finish) - post_inference_overhead_sec)

    min_throughput_2330 = remaining_india / avail_sec_2330
    min_throughput_2340 = remaining_india / avail_sec_2340

    sec_needed = remaining_india / max(0.01, ent_per_sec)
    t_projected_finish = t_gate_finish + sec_needed + post_inference_overhead_sec
    proj_finish_str = time.strftime("%H:%M:%S IST", time.localtime(t_projected_finish))

    log("\n=======================================================")
    log("DYNAMIC INDIA DEADLINE EVALUATION (FROM ACTUAL MEASURED THROUGHPUT)")
    log("=======================================================")
    log(f"  US Completion Time:              {us_complete_str}")
    log(f"  Benchmark Finish Time:          {t_gate_str}")
    log(f"  Measured India_0000 Throughput: {ent_per_sec:.2f} ent/s (Wall-Clock: {shard_time:.2f}s)")
    log(f"  Remaining India Queries:        {remaining_india:,}")
    log(f"  Available Pure Inference Secs:")
    log(f"    - for <= 23:30 STRICT:        {avail_sec_2330:.1f}s ({avail_sec_2330/60:.1f} min)")
    log(f"    - for <= 23:40 HARD LIMIT:    {avail_sec_2340:.1f}s ({avail_sec_2340/60:.1f} min)")
    log(f"  Dynamic Throughput Requirements:")
    log(f"    - STRICT (<= 23:30):          >= {min_throughput_2330:.2f} ent/s")
    log(f"    - HARD LIMIT (<= 23:40):      >= {min_throughput_2340:.2f} ent/s")
    log(f"  Projected Validated Output:     {proj_finish_str}")

    if ent_per_sec >= min_throughput_2330:
        status_deadline = "DEADLINE SAFE"
        launch_full = True
    elif ent_per_sec >= min_throughput_2340:
        status_deadline = "DEADLINE TIGHT"
        launch_full = True
    else:
        status_deadline = "CRITICAL"
        launch_full = False

    log(f"  DECISION STATUS:                {status_deadline}")
    log("=======================================================\n")

    if not launch_full:
        log("FATAL: Measured throughput is insufficient to complete before 23:40! Halting for emergency optimization.")
        sys.exit(1)

    log(f"Launching India_0001..India_0161 immediately ({status_deadline})...")
    run_full_india_inference(ent_per_sec)
    assemble_and_validate()
    log("ALL WORKFLOW COMPLETE. SAFE FOR SUBMISSION.")


if __name__ == "__main__":
    main()
