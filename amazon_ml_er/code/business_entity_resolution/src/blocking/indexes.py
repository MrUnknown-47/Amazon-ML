"""
Scalable index data structures for Business Entity Resolution blocking.
Implements exact hash-bucket inverted indexes and scalable IDF-weighted posting list indexes
with compact integer ID mappings and document-frequency noise pruning.
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np


class ExactInvertedIndex:
    """
    Inverted index mapping deterministic string keys to lists of target entity IDs.
    Monitors bucket distribution and prevents candidate explosion on high-frequency keys.
    """

    def __init__(self, name: str, max_bucket_size: int = 500):
        self.name = name
        self.max_bucket_size = max_bucket_size
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.total_indexed_records = 0

    def add(self, key: str, entity_id: str) -> None:
        """Add an entity ID under a key. Ignores empty keys."""
        if not key or not key.strip():
            return
        self.index[key].append(entity_id)
        self.total_indexed_records += 1

    def get_candidates(self, key: str) -> List[str]:
        """
        Retrieve candidate target IDs for a query key.
        Applies oversized-bucket pruning if bucket exceeds max_bucket_size.
        """
        if not key or key not in self.index:
            return []
        bucket = self.index[key]
        if len(bucket) > self.max_bucket_size:
            return bucket[:self.max_bucket_size]
        return bucket

    def compute_bucket_statistics(self) -> Dict[str, Any]:
        """Compute statistical distribution of bucket sizes across the index."""
        if not self.index:
            return {
                "unique_keys": 0,
                "total_postings": 0,
                "mean_bucket_size": 0.0,
                "median_bucket_size": 0.0,
                "p95_bucket_size": 0.0,
                "p99_bucket_size": 0.0,
                "max_bucket_size": 0,
                "oversized_buckets_count": 0
            }

        sizes = np.array([len(b) for b in self.index.values()])
        oversized = int(np.sum(sizes > self.max_bucket_size))

        return {
            "unique_keys": len(self.index),
            "total_postings": int(sizes.sum()),
            "mean_bucket_size": float(round(sizes.mean(), 2)),
            "median_bucket_size": float(np.median(sizes)),
            "p95_bucket_size": float(np.percentile(sizes, 95)),
            "p99_bucket_size": float(np.percentile(sizes, 99)),
            "max_bucket_size": int(sizes.max()),
            "oversized_buckets_count": oversized
        }


class PostingListIndex:
    """
    Scalable character or token n-gram posting list index.
    Uses compact integer IDs, prunes high document-frequency noise terms,
    and accumulates query-side IDF weights without dense Cartesian product materialization.
    """

    def __init__(
        self,
        name: str,
        n: int = 3,
        ngram_type: str = "char",
        ngram_mode: Optional[str] = None,
        max_df_ratio: float = 0.005,
        max_df_count: int = 3000
    ):
        self.name = name
        self.n = n
        self.ngram_type = ngram_type
        self.ngram_mode = ngram_mode or (f"{ngram_type}_{n}" if ngram_type == "char" else f"token_{n}")
        self.max_df_ratio = max_df_ratio
        self.max_df_count = max_df_count
        self.postings: Dict[str, List[int]] = defaultdict(list)
        self.doc_freq: Counter = Counter()
        self.idf: Dict[str, float] = {}
        self.id_to_int: Dict[str, int] = {}
        self.int_to_id: List[str] = []
        self.doc_lengths: List[int] = []
        self.total_docs = 0
        self.pruned_ngrams: Set[str] = set()

    def _extract_ngrams(self, text: str) -> Set[str]:
        from .keys import (
            extract_char_ngrams,
            extract_combined_char_ngrams,
            extract_token_ngrams,
            extract_token_set,
            extract_prefix_substring_keys
        )
        mode = self.ngram_mode
        if mode == "char_2":
            return extract_char_ngrams(text, n=2)
        elif mode == "char_3":
            return extract_char_ngrams(text, n=3)
        elif mode == "char_4":
            return extract_char_ngrams(text, n=4)
        elif mode == "char_2_3":
            return extract_combined_char_ngrams(text, ns=(2, 3))
        elif mode == "char_3_4":
            return extract_combined_char_ngrams(text, ns=(3, 4))
        elif mode == "token_1":
            return extract_token_set(text, min_len=2)
        elif mode in ("token_2", "token_bigram"):
            return extract_token_ngrams(text, n=2)
        elif mode == "prefix_4":
            return extract_prefix_substring_keys(text, prefix_len=4)
        else:
            if self.ngram_type == "char":
                return extract_char_ngrams(text, n=self.n)
            else:
                return extract_token_ngrams(text, n=self.n)

    def index_document(self, entity_id: str, text: str) -> None:
        """Add a document's n-grams to posting lists using compact integer indices."""
        ngrams = self._extract_ngrams(text)
        self.index_document_features(entity_id, ngrams)

    def index_document_features(self, entity_id: str, ngrams: Set[str]) -> None:
        """Directly index pre-extracted ngrams for performance."""
        if not ngrams:
            return

        if entity_id in self.id_to_int:
            int_id = self.id_to_int[entity_id]
        else:
            int_id = len(self.int_to_id)
            self.id_to_int[entity_id] = int_id
            self.int_to_id.append(entity_id)
            self.doc_lengths.append(len(ngrams))
            self.total_docs += 1

        for ng in ngrams:
            self.doc_freq[ng] += 1
            self.postings[ng].append(int_id)

    def finalize_index(self) -> None:
        """
        Compute IDF weights and prune high-frequency n-grams that exceed max_df_count.
        """
        ratio_cap = int(self.total_docs * self.max_df_ratio) if self.max_df_ratio > 0 else self.max_df_count
        effective_max_df = min(self.max_df_count, max(20, ratio_cap))
        self.pruned_ngrams = {
            ng for ng, df in self.doc_freq.items() if df > effective_max_df
        }

        # Calculate IDF for retained ngrams
        self.idf = {
            ng: math.log(1.0 + (self.total_docs / df))
            for ng, df in self.doc_freq.items()
            if ng not in self.pruned_ngrams
        }

    def retrieve_top_k(
        self,
        query_text: str,
        top_k: int = 25,
        df_threshold: Optional[int] = None
    ) -> List[Tuple[str, float]]:
        """
        Query the posting lists, accumulate IDF scores, and return top-K candidates.
        Optionally apply dynamic query-time df_threshold.
        """
        q_ngrams = self._extract_ngrams(query_text)
        if not q_ngrams:
            return []

        scores: Dict[int, float] = defaultdict(float)

        for ng in q_ngrams:
            df = self.doc_freq.get(ng, 0)
            if df == 0:
                continue
            if df_threshold is not None:
                if df > df_threshold:
                    continue
                w = math.log(1.0 + (self.total_docs / df))
            else:
                if ng in self.idf:
                    w = self.idf[ng]
                else:
                    continue

            for int_id in self.postings[ng]:
                scores[int_id] += w

        if not scores:
            return []

        if len(scores) <= top_k:
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        else:
            import heapq
            sorted_items = heapq.nlargest(top_k, scores.items(), key=lambda x: x[1])

        return [(self.int_to_id[int_id], score) for int_id, score in sorted_items]

    def retrieve_two_stage(
        self,
        query_text: str,
        pool_size: int = 100,
        top_k: int = 25,
        df_threshold: Optional[int] = None,
        rank_metric: str = "jaccard"
    ) -> List[Tuple[str, float]]:
        """
        Task 7 Two-Stage Name Retrieval:
        Stage A: Accumulate IDF scores to fetch candidate pool of size pool_size.
        Stage B: Fast Jaccard ranking using pre-computed doc_lengths and overlap counts.
        """
        q_ngrams = self._extract_ngrams(query_text)
        if not q_ngrams:
            return []
        len_q = len(q_ngrams)

        scores: Dict[int, float] = defaultdict(float)
        overlap_counts: Dict[int, int] = defaultdict(int)

        for ng in q_ngrams:
            df = self.doc_freq.get(ng, 0)
            if df == 0:
                continue
            if df_threshold is not None:
                if df > df_threshold:
                    continue
                w = math.log(1.0 + (self.total_docs / df))
            else:
                if ng in self.idf:
                    w = self.idf[ng]
                else:
                    continue

            for int_id in self.postings[ng]:
                scores[int_id] += w
                overlap_counts[int_id] += 1

        if not scores:
            return []

        # Stage A: Select top pool_size candidates by IDF score
        import heapq
        if len(scores) <= pool_size:
            pool = list(scores.keys())
        else:
            pool = [int_id for int_id, _ in heapq.nlargest(pool_size, scores.items(), key=lambda x: x[1])]

        # Stage B: Re-rank pool using Jaccard / Overlap similarity
        ranked_pool = []
        for int_id in pool:
            overlap = overlap_counts[int_id]
            doc_len = self.doc_lengths[int_id]
            if rank_metric == "jaccard":
                union = len_q + doc_len - overlap
                sim = overlap / union if union > 0 else 0.0
            else:
                sim = overlap / min(len_q, doc_len) if min(len_q, doc_len) > 0 else 0.0
            ranked_pool.append((int_id, sim, scores[int_id]))

        # Sort by similarity descending, then IDF score descending
        ranked_pool.sort(key=lambda x: (x[1], x[2]), reverse=True)
        chosen = ranked_pool[:top_k]
        return [(self.int_to_id[int_id], sim) for int_id, sim, _ in chosen]

    def get_index_stats(self) -> Dict[str, Any]:
        """Report indexing statistics, vocabulary size, and pruning metrics."""
        return {
            "total_docs_indexed": self.total_docs,
            "total_unique_ngrams": len(self.doc_freq),
            "retained_ngrams": len(self.idf),
            "pruned_high_df_ngrams": len(self.pruned_ngrams),
            "top_common_ngrams": self.doc_freq.most_common(5)
        }

