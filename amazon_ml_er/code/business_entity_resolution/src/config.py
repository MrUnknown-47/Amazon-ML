"""
Configuration and environment path resolution for Amazon ML Challenge 2026.
Supports execution in both containerized environments (/mnt/data) and local workspaces.
"""

import os
from pathlib import Path

# Resolve data directory: check environment, /mnt/data, or local fallback
_ENV_DATA_DIR = os.environ.get("AMAZON_ML_DATA_DIR")
if _ENV_DATA_DIR and os.path.exists(_ENV_DATA_DIR):
    DATA_DIR = Path(_ENV_DATA_DIR)
elif os.path.exists("/mnt/data/dataset/train"):
    DATA_DIR = Path("/mnt/data")
else:
    # Walk upwards from current file to find directory containing 'dataset'
    current = Path(__file__).resolve()
    candidate = None
    for parent in [current] + list(current.parents):
        if (parent / "dataset" / "train").exists():
            candidate = parent
            break
    if candidate is not None:
        DATA_DIR = candidate
    else:
        DATA_DIR = Path("/mnt/data")  # Canonical default

# Project root path
_ENV_PROJECT_DIR = os.environ.get("AMAZON_ML_PROJECT_DIR")
if _ENV_PROJECT_DIR and os.path.exists(_ENV_PROJECT_DIR):
    PROJECT_ROOT = Path(_ENV_PROJECT_DIR)
elif (DATA_DIR / "amazon_ml_er").exists():
    PROJECT_ROOT = DATA_DIR / "amazon_ml_er"
elif os.path.exists("/mnt/data/amazon_ml_er"):
    PROJECT_ROOT = Path("/mnt/data/amazon_ml_er")
else:
    PROJECT_ROOT = DATA_DIR / "amazon_ml_er"

# File paths
TRAIN_DIR = DATA_DIR / "dataset" / "train"
TEST_DIR = DATA_DIR / "dataset" / "test"
UTILS_DIR = DATA_DIR / "utils"

TRAIN_SOURCE1 = TRAIN_DIR / "train_source1.tsv"
TRAIN_SOURCE2 = TRAIN_DIR / "train_source2.tsv"
TRAIN_SOURCE3 = TRAIN_DIR / "train_source3.tsv"
TRAIN_GROUND_TRUTH = TRAIN_DIR / "train_ground_truth.tsv"

TEST_SOURCE1 = TEST_DIR / "test_source1.tsv"
TEST_SOURCE2 = TEST_DIR / "test_source2.tsv"
TEST_SOURCE3 = TEST_DIR / "test_source3.tsv"

VALIDATOR_SCRIPT = UTILS_DIR / "validate_submission.py"

# Project artifact paths
OUTPUT_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
SPLITS_DIR = PROJECT_ROOT / "splits"

MATCHING_RESULTS_PATH = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_PAIRS_PATH = OUTPUT_DIR / "candidate_pairs.tsv"
DATA_AUDIT_JSON = REPORTS_DIR / "data_audit.json"
DATA_AUDIT_MD = REPORTS_DIR / "data_audit.md"
VALIDATION_SPLIT_S1_PATH = SPLITS_DIR / "val_source1_ids.txt"
TRAIN_SPLIT_S1_PATH = SPLITS_DIR / "train_source1_ids.txt"

# Schemas
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]

# Randomness and validation defaults
DEFAULT_RANDOM_SEED = 42
DEFAULT_VAL_SIZE = 50000
