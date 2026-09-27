"""
Authoritative Production Blocker for Business Entity Resolution.
Implements the single canonical, leak-safe, reproducible multi-channel candidate generator.
Used identically across blocking evaluation, candidate compression training,
candidate compression validation, and test inference.

Channels:
- Channel 1 (ch1_core_name): Exact normalized business name key
- Channel 2 (ch2_top2_tokens): Top-2 alphabetically sorted significant tokens
- Channel 3 (ch3_street_num_token): Primary numeric building token + first significant street token
- Channel 4 (ch4_address_location): Primary numeric building token + locality/city token
- Channel 5 (ch5_char3_k50): Character 3-gram IDF-weighted posting list with noise pruning (top-k=50)
- Channel 6 (ch6_distinctive_token): Multi-token inverted index querying top rarest significant tokens
- Channel 7 (ch7_relaxed_street_num): Relaxed numeric building number + all cleaned street tokens
"""

import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Any

from ..preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    extract_numeric_tokens,
)
from .keys import extract_all_blocking_keys, ADDRESS_STOP_TOKENS
from .indexes import ExactInvertedIndex, PostingListIndex


def clean_street_token(tok: str) -> str:
    """Normalize ordinal suffixes from street tokens (e.g. 1st -> 1, 2nd -> 2, 42nd -> 42)."""
    return re.sub(r"(\d+)(st|nd|rd|th)$", r"\1", tok.lower())


def extract_all_relaxed_street_keys(address: str) -> List[str]:
    """
    Extract all relaxed street keys from an address string.
    Pairing the primary numeric building number with EVERY valid non-stop street token.
    Prevents miss-matches caused by address token ordering differences or prefixes.
    """
    if not address or not address.strip():
        return []
    nums = extract_numeric_tokens(address)
    if not nums:
        return []
    primary_num = str(int(nums[0]))
    norm_addr = normalize_business_address(address)
    keys = []
    seen = set()
    for tok in norm_addr.tokens:
        if not tok.isdigit() and len(tok) >= 3 and tok not in ADDRESS_STOP_TOKENS:
            cleaned = clean_street_token(tok)
            if cleaned not in seen:
                seen.add(cleaned)
                keys.append(f"{primary_num}_{cleaned}")
    return keys


@dataclass
class BlockerConfig:
    """Configuration parameters for the authoritative blocker."""
    max_bucket_size: int = 500
    ch5_top_k: int = 50
    ch5_max_df: int = 3000
    ch5_max_df_ratio: float = 0.01
    ch6_tokens_to_query: int = 2
    ch6_max_df: int = 3000
    channels_enabled: Tuple[str, ...] = (
        "ch1_core_name",
        "ch2_top2_tokens",
        "ch3_street_num_token",
        "ch4_address_location",
        "ch5_char3_k50",
        "ch6_distinctive_token",
        "ch7_relaxed_street_num",
    )


class TargetCorpusIndex:
    """
    In-memory indexing structure holding all 7 physical channels for target entities.
    Enforces deterministic indexing order and bucket-size governance.
    """

    def __init__(self, config: Optional[BlockerConfig] = None):
        self.config = config or BlockerConfig()
        cap = self.config.max_bucket_size
        self.idx1 = ExactInvertedIndex("ch1_core_name", max_bucket_size=cap)
        self.idx2 = ExactInvertedIndex("ch2_top2_tokens", max_bucket_size=cap)
        self.idx3 = ExactInvertedIndex("ch3_street_num_token", max_bucket_size=cap)
        self.idx4 = ExactInvertedIndex("ch4_address_location", max_bucket_size=cap)
        self.pl5 = PostingListIndex(
            "ch5_char3_k50",
            ngram_mode="char_3",
            max_df_count=self.config.ch5_max_df,
            max_df_ratio=self.config.ch5_max_df_ratio
        )
        self.idx6 = ExactInvertedIndex("ch6_distinctive_token", max_bucket_size=cap)
        self.idx7 = ExactInvertedIndex("ch7_relaxed_street_num", max_bucket_size=cap)
        self.total_records_indexed = 0
        self.is_finalized = False

    def add_target_record(
        self,
        entity_id: str,
        business_name: str,
        business_address: str,
        country: Optional[str] = None
    ) -> None:
        """Indexes a single target record across all physical channels."""
        keys = extract_all_blocking_keys(business_name, business_address)

        # Ch1 & Ch2
        if keys["ch1_key"]:
            self.idx1.add(keys["ch1_key"], entity_id)
            self.pl5.index_document(entity_id, keys["ch1_key"])
        if keys["ch2_top2"]:
            self.idx2.add(keys["ch2_top2"], entity_id)

        # Ch3 & Ch4
        if keys["ch3_key"]:
            self.idx3.add(keys["ch3_key"], entity_id)
        if keys["ch4_key"]:
            self.idx4.add(keys["ch4_key"], entity_id)

        # Ch6: Distinctive significant tokens
        norm_name = normalize_business_name(business_name)
        for tok in norm_name.significant_tokens:
            if len(tok) >= 3 and not tok.isdigit():
                self.idx6.add(tok, entity_id)

        # Ch7: Relaxed street keys (multi-key)
        for r_key in extract_all_relaxed_street_keys(business_address):
            self.idx7.add(r_key, entity_id)

        self.total_records_indexed += 1

    def finalize(self) -> None:
        """Finalize posting lists and IDF calculation."""
        self.pl5.finalize_index()
        self.is_finalized = True

    def get_index_statistics(self) -> Dict[str, Any]:
        """Return diagnostic metrics across all 7 physical channels."""
        return {
            "total_records_indexed": self.total_records_indexed,
            "ch1": self.idx1.compute_bucket_statistics(),
            "ch2": self.idx2.compute_bucket_statistics(),
            "ch3": self.idx3.compute_bucket_statistics(),
            "ch4": self.idx4.compute_bucket_statistics(),
            "ch5_unique_ngrams": len(self.pl5.postings),
            "ch5_pruned_ngrams": len(self.pl5.pruned_ngrams),
            "ch6": self.idx6.compute_bucket_statistics(),
            "ch7": self.idx7.compute_bucket_statistics(),
        }


def generate_raw_candidates(
    s1_record: Dict[str, Any],
    target_indexes: TargetCorpusIndex,
    config: Optional[BlockerConfig] = None
) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Generate candidate target records for a single S1 reference entity.
    Returns a deduplicated list of tuples: (target_entity_id, provenance_dict).

    Provenance dictionary fields:
    - channels: Set[str] of channels that retrieved this target record
    - rarest_token_df: int, minimum DF among matching significant tokens
    - ch5_score: float, character 3-gram score if retrieved by Channel 5, else 0.0
    """
    cfg = config or target_indexes.config
    name = s1_record.get("business_name", "")
    address = s1_record.get("business_address", "")

    keys = extract_all_blocking_keys(name, address)
    norm_name = normalize_business_name(name)

    candidates: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"channels": set(), "rarest_token_df": 0, "ch5_score": 0.0}
    )

    # Ch1: Core Name
    if "ch1_core_name" in cfg.channels_enabled and keys["ch1_key"]:
        for tid in target_indexes.idx1.get_candidates(keys["ch1_key"]):
            candidates[tid]["channels"].add("ch1_core_name")

    # Ch2: Top-2 Tokens
    if "ch2_top2_tokens" in cfg.channels_enabled and keys["ch2_top2"]:
        for tid in target_indexes.idx2.get_candidates(keys["ch2_top2"]):
            candidates[tid]["channels"].add("ch2_top2_tokens")

    # Ch3: Street Number + Street Token
    if "ch3_street_num_token" in cfg.channels_enabled and keys["ch3_key"]:
        for tid in target_indexes.idx3.get_candidates(keys["ch3_key"]):
            candidates[tid]["channels"].add("ch3_street_num_token")

    # Ch4: Address Location
    if "ch4_address_location" in cfg.channels_enabled and keys["ch4_key"]:
        for tid in target_indexes.idx4.get_candidates(keys["ch4_key"]):
            candidates[tid]["channels"].add("ch4_address_location")

    # Ch5: Character 3-Gram Posting List (top-k=50)
    if "ch5_char3_k50" in cfg.channels_enabled and keys["ch1_key"]:
        top_hits = target_indexes.pl5.retrieve_top_k(
            keys["ch1_key"],
            top_k=cfg.ch5_top_k,
            df_threshold=cfg.ch5_max_df
        )
        for tid, score in top_hits:
            candidates[tid]["channels"].add("ch5_char3_k50")
            candidates[tid]["ch5_score"] = float(score)

    # Ch6: Distinctive Significant Tokens (Multi-Token Strategy)
    if "ch6_distinctive_token" in cfg.channels_enabled:
        sig_toks = [
            tok for tok in norm_name.significant_tokens
            if len(tok) >= 3 and not tok.isdigit() and tok in target_indexes.idx6.index
        ]
        if sig_toks:
            # Sort tokens by ascending document frequency (rarest first)
            sorted_by_rarity = sorted(sig_toks, key=lambda t: len(target_indexes.idx6.index[t]))
            # Query up to ch6_tokens_to_query tokens (default 2) having df <= ch6_max_df
            queried = 0
            for tok in sorted_by_rarity:
                df = len(target_indexes.idx6.index[tok])
                if df <= cfg.ch6_max_df:
                    for tid in target_indexes.idx6.get_candidates(tok):
                        candidates[tid]["channels"].add("ch6_distinctive_token")
                        prev_df = candidates[tid]["rarest_token_df"]
                        if prev_df == 0 or df < prev_df:
                            candidates[tid]["rarest_token_df"] = df
                    queried += 1
                    if queried >= cfg.ch6_tokens_to_query:
                        break

    # Ch7: Relaxed Street Number (Multi-Key Strategy)
    if "ch7_relaxed_street_num" in cfg.channels_enabled:
        relaxed_keys = extract_all_relaxed_street_keys(address)
        for r_key in relaxed_keys:
            for tid in target_indexes.idx7.get_candidates(r_key):
                candidates[tid]["channels"].add("ch7_relaxed_street_num")

    return list(candidates.items())


def generate_raw_candidates_batch(
    s1_batch: List[Dict[str, Any]],
    target_indexes: TargetCorpusIndex,
    config: Optional[BlockerConfig] = None
) -> Dict[str, List[Tuple[str, Dict[str, Any]]]]:
    """
    Batch wrapper generating raw candidates for a collection of S1 entity records.
    Maps s1_entity_id -> list of (target_entity_id, provenance_dict).
    """
    results: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {}
    for s1_rec in s1_batch:
        eid = s1_rec.get("entity_id")
        results[eid] = generate_raw_candidates(s1_rec, target_indexes, config)
    return results
