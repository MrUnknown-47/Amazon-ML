"""
Reproducible Source 1 Validation Split Generator.
Splits strictly at the S1 entity level, stratified by country, to prevent data leakage.
"""

import os
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from collections import defaultdict

from ..config import (
    TRAIN_SOURCE1,
    DEFAULT_RANDOM_SEED,
    DEFAULT_VAL_SIZE,
    VALIDATION_SPLIT_S1_PATH,
    TRAIN_SPLIT_S1_PATH
)
from ..data_loader import stream_tsv_rows


def create_s1_validation_split(
    s1_path: Path = TRAIN_SOURCE1,
    val_size: int = DEFAULT_VAL_SIZE,
    random_seed: int = DEFAULT_RANDOM_SEED,
    output_val_path: Optional[Path] = VALIDATION_SPLIT_S1_PATH,
    output_train_path: Optional[Path] = TRAIN_SPLIT_S1_PATH,
    stratify_by_country: bool = True
) -> Tuple[List[str], List[str]]:
    """
    Generate a stratified split of Source 1 entity IDs into Train and Validation partitions.

    Guarantees:
    - Exactly S1 entity-level split
    - Fixed random seed for strict reproducibility
    - Exact disjointness: train_s1_ids ∩ val_s1_ids == ∅
    - Stratified country representation
    - Does NOT alter or sample S2/S3: validation S1 entities are tested against full S2/S3 corpora.
    """
    # 1. Collect S1 IDs grouped by country
    country_to_ids: Dict[str, List[str]] = defaultdict(list)
    total_s1 = 0

    for row in stream_tsv_rows(s1_path):
        eid = row["entity_id"]
        c = row.get("country", "UNKNOWN")
        country_to_ids[c].append(eid)
        total_s1 += 1

    if val_size >= total_s1:
        raise ValueError(f"val_size ({val_size}) must be strictly less than total S1 records ({total_s1}).")

    rng = np.random.RandomState(random_seed)

    val_ids: List[str] = []
    train_ids: List[str] = []

    if stratify_by_country:
        # Proportionally allocate val_size to each country
        for country, ids in country_to_ids.items():
            ids_arr = np.array(ids)
            rng.shuffle(ids_arr)

            # Country share
            share = len(ids) / total_s1
            k_val = int(round(val_size * share))
            # Ensure at least 1 if ids exist, but not exceeding len(ids)
            k_val = min(max(k_val, 1 if len(ids) > 1 else 0), len(ids) - 1)

            val_subset = ids_arr[:k_val].tolist()
            train_subset = ids_arr[k_val:].tolist()

            val_ids.extend(val_subset)
            train_ids.extend(train_subset)

        # Adjust in case rounding differed slightly from val_size
        diff = len(val_ids) - val_size
        if diff > 0:
            # Move diff from val to train
            rng.shuffle(val_ids)
            train_ids.extend(val_ids[:diff])
            val_ids = val_ids[diff:]
        elif diff < 0:
            # Move abs(diff) from train to val
            needed = abs(diff)
            rng.shuffle(train_ids)
            val_ids.extend(train_ids[:needed])
            train_ids = train_ids[needed:]
    else:
        all_ids = []
        for ids in country_to_ids.values():
            all_ids.extend(ids)
        all_arr = np.array(all_ids)
        rng.shuffle(all_arr)
        val_ids = all_arr[:val_size].tolist()
        train_ids = all_arr[val_size:].tolist()

    # Sort for deterministic file output
    val_ids.sort()
    train_ids.sort()

    # Sanity checks
    set_val = set(val_ids)
    set_train = set(train_ids)
    assert len(set_val & set_train) == 0, "Train and Validation S1 IDs must be strictly disjoint!"
    assert len(set_val) + len(set_train) == total_s1, "Sum of split sizes must equal total S1 entities!"

    # Write to files if paths provided
    if output_val_path:
        os.makedirs(output_val_path.parent, exist_ok=True)
        with open(output_val_path, "w", encoding="utf-8") as f:
            for eid in val_ids:
                f.write(f"{eid}\n")

    if output_train_path:
        os.makedirs(output_train_path.parent, exist_ok=True)
        with open(output_train_path, "w", encoding="utf-8") as f:
            for eid in train_ids:
                f.write(f"{eid}\n")

    return train_ids, val_ids


def load_validation_split_ids(val_path: Path = VALIDATION_SPLIT_S1_PATH) -> Set[str]:
    """
    Load the pre-computed validation S1 entity IDs.
    """
    if not os.path.exists(val_path):
        raise FileNotFoundError(f"Validation split file not found: {val_path}")
    with open(val_path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}
