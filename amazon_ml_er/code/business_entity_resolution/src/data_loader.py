"""
Robust TSV data loading and streaming module for Business Entity Resolution.
Preserves Unicode, respects explicit tab delimiters, preserves empty strings,
and never alters original source files.
"""

import csv
import os
from pathlib import Path
from typing import Generator, Dict, List, Optional, Set, Tuple, Any

from .config import SOURCE_COLUMNS


class TSVValidationError(Exception):
    """Raised when TSV structure violates challenge constraints."""
    pass


def stream_tsv_rows(
    file_path: Path,
    expected_columns: Optional[List[str]] = None,
    limit: Optional[int] = None,
    validate_header: bool = True
) -> Generator[Dict[str, str], None, None]:
    """
    Stream rows from a TSV file as dictionaries of strings.
    Guarantees:
    - Tab separation only
    - UTF-8 encoding
    - Empty strings preserved (not cast to None/NaN)
    - Original file is opened read-only
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        try:
            raw_header = next(reader)
        except StopIteration:
            raise TSVValidationError(f"File is empty: {file_path}")

        header = [col.strip() for col in raw_header]

        # Check for common CSV mistake
        if len(header) == 1 and "," in header[0]:
            raise TSVValidationError(
                f"File {file_path} appears to be comma-separated, expected tab-separated."
            )

        if validate_header and expected_columns is not None:
            if header != expected_columns:
                raise TSVValidationError(
                    f"Invalid header in {file_path}. Got {header}, expected {expected_columns}"
                )

        count = 0
        for line_num, row in enumerate(reader, start=2):
            if limit is not None and count >= limit:
                break

            # Handle possible row length mismatch
            if len(row) < len(header):
                row = row + [""] * (len(header) - len(row))
            elif expected_columns is not None and len(row) > len(header):
                raise TSVValidationError(
                    f"Row {line_num} in {file_path} has {len(row)} columns, expected {len(header)}."
                )

            row_dict = {col: val for col, val in zip(header, row)}
            yield row_dict
            count += 1


def load_source_records(
    file_path: Path,
    limit: Optional[int] = None,
    check_unique_ids: bool = True
) -> Dict[str, Dict[str, str]]:
    """
    Load an entire source TSV into an in-memory dictionary mapping entity_id -> record.
    Returns:
        {entity_id: {"business_name": ..., "business_address": ..., "country": ...}}
    """
    records = {}
    seen_ids = set()

    for row in stream_tsv_rows(file_path, expected_columns=SOURCE_COLUMNS, limit=limit):
        eid = row["entity_id"]
        if check_unique_ids:
            if eid in seen_ids:
                raise TSVValidationError(f"Duplicate entity_id found: {eid} in {file_path}")
            seen_ids.add(eid)

        records[eid] = {
            "business_name": row["business_name"],
            "business_address": row["business_address"],
            "country": row["country"]
        }

    return records


def load_entity_ids(file_path: Path, limit: Optional[int] = None) -> List[str]:
    """
    Extract only entity IDs from a source TSV efficiently with minimal memory.
    """
    ids = []
    seen = set()
    with open(file_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f, delimiter="\t")
        next(reader, None)  # skip header
        for row in reader:
            if not row or not row[0].strip():
                continue
            eid = row[0].strip()
            if eid in seen:
                raise TSVValidationError(f"Duplicate entity_id found: {eid} in {file_path}")
            seen.add(eid)
            ids.append(eid)
            if limit is not None and len(ids) >= limit:
                break
    return ids


def audit_tsv_file(
    file_path: Path,
    expected_columns: Optional[List[str]] = None,
    id_prefix: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform a comprehensive integrity and statistical audit of a single TSV file.
    Does not hold the entire file in RAM; computes statistics in a single streaming pass.
    """
    stats = {
        "file_path": str(file_path),
        "total_rows": 0,
        "unique_ids": 0,
        "duplicate_ids": 0,
        "missing_values": {},
        "empty_string_counts": {},
        "country_counts": {},
        "name_length_stats": {},
        "addr_length_stats": {}
    }

    if expected_columns is None:
        expected_columns = SOURCE_COLUMNS

    for col in expected_columns:
        stats["empty_string_counts"][col] = 0

    seen_ids: Set[str] = set()
    name_lens: List[int] = []
    addr_lens: List[int] = []

    for row in stream_tsv_rows(file_path, expected_columns=expected_columns):
        stats["total_rows"] += 1
        eid = row.get("entity_id", "")

        if id_prefix and not eid.startswith(id_prefix):
            raise TSVValidationError(f"Entity ID {eid} does not start with expected prefix {id_prefix}")

        if eid in seen_ids:
            stats["duplicate_ids"] += 1
        else:
            seen_ids.add(eid)

        for col in expected_columns:
            val = row.get(col, "")
            if val == "":
                stats["empty_string_counts"][col] += 1

        country = row.get("country", "")
        if country:
            stats["country_counts"][country] = stats["country_counts"].get(country, 0) + 1

        if "business_name" in row:
            name_lens.append(len(row["business_name"]))
        if "business_address" in row:
            addr_lens.append(len(row["business_address"]))

    stats["unique_ids"] = len(seen_ids)
    
    # Summary stats if numerical lengths collected
    if name_lens:
        import numpy as np
        arr = np.array(name_lens)
        stats["name_length_stats"] = {
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(round(arr.mean(), 2)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max())
        }
    if addr_lens:
        import numpy as np
        arr = np.array(addr_lens)
        stats["addr_length_stats"] = {
            "min": int(arr.min()),
            "median": float(np.median(arr)),
            "mean": float(round(arr.mean(), 2)),
            "p95": float(np.percentile(arr, 95)),
            "max": int(arr.max())
        }

    return stats
