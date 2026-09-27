"""
Individual Blocking Channel Implementations and Benchmark Harness.
Measures per-channel candidate recall, candidate set size distribution,
index bucket distributions, and latency.
"""

import time
import os
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
from collections import defaultdict
import numpy as np
import psutil

from .keys import (
    extract_normalized_name_key,
    extract_sorted_tokens_key,
    extract_street_num_token_key,
    extract_address_location_key
)
from .indexes import ExactInvertedIndex, PostingListIndex
from ..evaluation.metrics import compute_candidate_recall


class BlockingChannel:
    """
    Abstract interface for a single candidate generation channel.
    """

    def __init__(self, name: str):
        self.name = name

    def index_target_records(self, target_records: Dict[str, Dict[str, str]]) -> None:
        raise NotImplementedError

    def generate_candidates(self, query_record: Dict[str, str]) -> List[str]:
        raise NotImplementedError

    def get_index_statistics(self) -> Dict[str, Any]:
        return {}


class ExactKeyBlockingChannel(BlockingChannel):
    """
    Channel based on exact key hash-matching using ExactInvertedIndex.
    """

    def __init__(self, name: str, key_fn: Callable[[Dict[str, str]], str], max_bucket_size: int = 500):
        super().__init__(name)
        self.key_fn = key_fn
        self.max_bucket_size = max_bucket_size
        self.index = ExactInvertedIndex(name, max_bucket_size=max_bucket_size)

    def index_target_records(self, target_records: Dict[str, Dict[str, str]]) -> None:
        for eid, rec in target_records.items():
            key = self.key_fn(rec)
            self.index.add(key, eid)

    def generate_candidates(self, query_record: Dict[str, str]) -> List[str]:
        key = self.key_fn(query_record)
        return self.index.get_candidates(key)

    def get_index_statistics(self) -> Dict[str, Any]:
        return self.index.compute_bucket_statistics()


class PostingListBlockingChannel(BlockingChannel):
    """
    Channel based on IDF-weighted posting list retrieval.
    """

    def __init__(
        self,
        name: str,
        text_fn: Callable[[Dict[str, str]], str],
        n: int = 3,
        ngram_type: str = "char",
        top_k: int = 25,
        max_df_ratio: float = 0.02
    ):
        super().__init__(name)
        self.text_fn = text_fn
        self.top_k = top_k
        self.index = PostingListIndex(
            name,
            n=n,
            ngram_type=ngram_type,
            max_df_ratio=max_df_ratio
        )

    def index_target_records(self, target_records: Dict[str, Dict[str, str]]) -> None:
        for eid, rec in target_records.items():
            text = self.text_fn(rec)
            self.index.index_document(eid, text)
        self.index.finalize_index()

    def generate_candidates(self, query_record: Dict[str, str]) -> List[str]:
        text = self.text_fn(query_record)
        results = self.index.retrieve_top_k(text, top_k=self.top_k)
        return [eid for eid, _ in results]

    def get_index_statistics(self) -> Dict[str, Any]:
        return self.index.get_index_stats()


def build_channel_1_normalized_name(max_bucket_size: int = 500) -> ExactKeyBlockingChannel:
    """Channel 1: Core normalized name key (legal suffixes stripped)."""
    return ExactKeyBlockingChannel(
        name="ch1_core_name",
        key_fn=lambda r: extract_normalized_name_key(r.get("business_name", "")),
        max_bucket_size=max_bucket_size
    )


def build_channel_2_sorted_tokens(n_tokens: int = 2, max_bucket_size: int = 500) -> ExactKeyBlockingChannel:
    """Channel 2: Sorted top-N significant name tokens."""
    return ExactKeyBlockingChannel(
        name=f"ch2_sorted_tokens_top{n_tokens}",
        key_fn=lambda r: extract_sorted_tokens_key(r.get("business_name", ""), n_tokens=n_tokens),
        max_bucket_size=max_bucket_size
    )


def build_channel_3_street_num_token(max_bucket_size: int = 500) -> ExactKeyBlockingChannel:
    """Channel 3: Street number + first street token."""
    return ExactKeyBlockingChannel(
        name="ch3_street_num_token",
        key_fn=lambda r: extract_street_num_token_key(r.get("business_address", "")),
        max_bucket_size=max_bucket_size
    )


def build_channel_4_address_location(max_bucket_size: int = 500) -> ExactKeyBlockingChannel:
    """Channel 4: Street number + city/locality token."""
    return ExactKeyBlockingChannel(
        name="ch4_address_location",
        key_fn=lambda r: extract_address_location_key(r.get("business_address", "")),
        max_bucket_size=max_bucket_size
    )


def build_channel_5_posting_list(
    n: int = 3,
    ngram_type: str = "char",
    top_k: int = 25,
    max_df_ratio: float = 0.02
) -> PostingListBlockingChannel:
    """Channel 5: Scalable posting list retrieval on normalized name."""
    return PostingListBlockingChannel(
        name=f"ch5_posting_{ngram_type}_{n}gram",
        text_fn=lambda r: extract_normalized_name_key(r.get("business_name", "")),
        n=n,
        ngram_type=ngram_type,
        top_k=top_k,
        max_df_ratio=max_df_ratio
    )


def evaluate_single_channel(
    channel: BlockingChannel,
    query_records: Dict[str, Dict[str, str]],
    ground_truth: Dict[str, Set[str]],
    total_target_corpus_size: int
) -> Tuple[Dict[str, Set[str]], Dict[str, Any]]:
    """
    Execute candidate retrieval for one channel across query records and record metrics.
    Returns:
        (candidates_dict, channel_metrics_dict)
    """
    proc = psutil.Process(os.getpid())
    m0 = proc.memory_info().rss / (1024 * 1024)
    t0 = time.time()

    candidates: Dict[str, Set[str]] = {}
    cand_counts = []

    for s1_id, q_rec in query_records.items():
        cands = channel.generate_candidates(q_rec)
        c_set = set(cands)
        candidates[s1_id] = c_set
        cand_counts.append(len(c_set))

    elapsed = round(time.time() - t0, 3)
    m1 = proc.memory_info().rss / (1024 * 1024)

    # Compute recall against ground truth
    recall_stats = compute_candidate_recall(ground_truth, candidates)

    arr = np.array(cand_counts) if cand_counts else np.array([0])
    total_candidates = int(arr.sum())

    total_possible_pairs = len(query_records) * total_target_corpus_size
    reduction_ratio = 1.0 - (total_candidates / total_possible_pairs) if total_possible_pairs > 0 else 1.0

    metrics = {
        "channel_name": channel.name,
        "query_count": len(query_records),
        "candidate_count_total": total_candidates,
        "mean_candidates_per_S1": float(round(arr.mean(), 2)),
        "median_candidates_per_S1": float(np.median(arr)),
        "P95_candidates_per_S1": float(np.percentile(arr, 95)),
        "P99_candidates_per_S1": float(np.percentile(arr, 99)),
        "max_candidates_per_S1": int(arr.max()),
        "candidate_recall": float(round(recall_stats["candidate_recall"], 5)),
        "total_true_matches": recall_stats["total_true_matches"],
        "captured_true_matches": recall_stats["captured_true_matches"],
        "missed_true_matches": recall_stats["missed_matches"],
        "reduction_ratio": float(reduction_ratio),
        "runtime_seconds": elapsed,
        "throughput_queries_per_sec": float(round(len(query_records) / elapsed, 1)) if elapsed > 0 else 0.0,
        "peak_memory_mb": float(round(m1, 2)),
        "index_statistics": channel.get_index_statistics()
    }

    return candidates, metrics
