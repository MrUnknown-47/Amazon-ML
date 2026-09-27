"""
Comprehensive Pairwise Matching Feature Extraction for Business Entity Resolution.
Implements fine-grained name similarities, address overlaps, numeric agreements,
interaction terms, candidate-ranker signals, and source/country indicators.
Zero external lookups, zero hardcoded country lists, fully open-set compatible.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from ..preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    extract_numeric_tokens
)
from ..blocking.keys import ADDRESS_STOP_TOKENS


def fast_levenshtein_ratio(s1: str, s2: str) -> float:
    """
    Compute normalized Levenshtein similarity ratio between two strings:
    sim = 1.0 - (lev_distance / max(len(s1), len(s2)))
    Optimized with length-difference pruning and two-row dynamic programming.
    """
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    max_len = max(len1, len2)
    if abs(len1 - len2) > max_len * 0.7:
        return 0.0

    if len1 > len2:
        s1, s2 = s2, s1
        len1, len2 = len2, len1

    current = list(range(len1 + 1))
    for i, c2 in enumerate(s2):
        previous, current = current, [i + 1] + [0] * len1
        for j, c1 in enumerate(s1):
            insert = previous[j + 1] + 1
            delete = current[j] + 1
            substitute = previous[j] + (0 if c1 == c2 else 1)
            current[j + 1] = min(insert, delete, substitute)

    dist = current[len1]
    return max(0.0, 1.0 - (dist / max_len))


FEATURE_NAMES = [
    # --- NAME FEATURES (0 to 14) ---
    "name_norm_exact_match",         # 0: Alphanumeric name exact match
    "name_raw_exact_match",          # 1: Raw case-insensitive exact match
    "name_token_jaccard",            # 2: Jaccard similarity on significant tokens
    "name_token_containment",        # 3: Token containment min(|s1|, |t|)
    "name_char2_sim",                # 4: Character 2-gram Jaccard
    "name_char3_sim",                # 5: Character 3-gram Jaccard
    "name_char4_sim",                # 6: Character 4-gram Jaccard
    "name_edit_sim",                 # 7: Normalized Levenshtein ratio on alphanumeric name
    "name_len_ratio",                # 8: Min length / Max length
    "name_token_count_diff",         # 9: Absolute difference in token counts
    "name_rare_token_overlap",       # 10: Inverse log DF of rarest shared token
    "name_shared_token_count",       # 11: Number of shared significant tokens
    "name_sig_token_overlap",        # 12: Overlap of non-digit significant tokens
    "name_containment_s1_in_t",      # 13: |S1 & T| / |S1|
    "name_containment_t_in_s1",      # 14: |S1 & T| / |T|

    # --- ADDRESS FEATURES (15 to 29) ---
    "addr_norm_exact_match",         # 15: Cleaned address exact match
    "addr_token_jaccard",            # 16: Address token Jaccard
    "addr_token_containment",        # 17: Address token containment
    "addr_char3_sim",                # 18: Address character 3-gram similarity
    "addr_numeric_overlap_count",    # 19: Number of shared numeric tokens
    "addr_numeric_jaccard",          # 20: Jaccard similarity on numeric tokens
    "addr_street_num_match",         # 21: Primary building number agreement
    "addr_postal_match",             # 22: Postal/PIN code agreement
    "addr_locality_overlap",         # 23: Shared locality/city token indicator
    "addr_region_overlap",           # 24: Shared region/state token indicator
    "addr_len_ratio",                # 25: Address length ratio
    "addr_missing_s1",               # 26: S1 address missing indicator
    "addr_missing_t",                # 27: T address missing indicator
    "addr_shared_tokens_count",      # 28: Count of shared non-stop address tokens
    "addr_strongest_token_weight",   # 29: Length of longest shared address token

    # --- NAME x ADDRESS INTERACTIONS (30 to 45) ---
    "name_x_addr_sim",               # 30: name_token_jaccard * addr_token_jaccard
    "name_exact_addr_partial",       # 31: name exact and addr jaccard >= 0.25
    "high_name_low_addr",            # 32: name >= 0.70 and addr < 0.20 (diff branch / name collision)
    "low_name_high_addr",            # 33: name < 0.30 and addr >= 0.60 (trade name / DBA co-location)
    "exact_numeric_low_name",        # 34: street num match and name < 0.30
    "exact_addr_low_name",           # 35: addr exact match and name < 0.30
    "field_consensus_count",         # 36: Total exact field agreements (name, addr, num, pin, country)
    "blocking_channel_hit_count",    # 37: Number of blocking channels retrieving pair
    "ch1_core_name_hit",             # 38: Channel 1 hit
    "ch2_top2_tokens_hit",           # 39: Channel 2 hit
    "ch3_street_num_hit",            # 40: Channel 3 hit
    "ch4_address_hit",               # 41: Channel 4 hit
    "ch5_char3_hit",                 # 42: Channel 5 hit
    "ch6_distinctive_token_hit",     # 43: Channel 6 hit
    "ch7_relaxed_street_hit",        # 44: Channel 7 hit
    "ch5_retrieval_score",           # 45: Channel 5 TF-IDF retrieval score

    # --- RANKER SIGNALS & METADATA (46 to 51) ---
    "candidate_ranker_score",        # 46: Candidate ranker model probability
    "candidate_ranker_rank",         # 47: Candidate ranker position (1, 2, ...)
    "candidate_ranker_reciprocal",   # 48: 1.0 / rank
    "target_is_source3",             # 49: Target belongs to Source 3
    "same_country",                  # 50: S1 and Target share country string
    "is_transliteration_pair"        # 51: Non-Latin script mismatch indicator
]

NUM_MATCHING_FEATURES = len(FEATURE_NAMES)


class MatchingEntityProfile:
    """
    Cached, pre-tokenized multi-field representation of an entity profile
    specifically optimized for fast pairwise matching feature extraction.
    """
    __slots__ = (
        "eid", "country", "raw_name", "raw_addr",
        "alphanumeric_name", "raw_name_clean",
        "sig_name_tokens", "all_name_tokens",
        "char2_name", "char3_name", "char4_name",
        "addr_clean_lower", "addr_tokens", "addr_non_stop_tokens",
        "addr_numeric_tokens", "addr_street_num", "addr_postal",
        "addr_locality_tokens", "addr_char3",
        "is_non_latin", "has_address"
    )

    def __init__(self, eid: str, name: str, address: str, country: str = ""):
        self.eid = eid
        self.country = country or ""
        self.raw_name = name or ""
        self.raw_addr = address or ""

        # Normalize name
        norm_n = normalize_business_name(self.raw_name)
        self.alphanumeric_name = norm_n.alphanumeric
        self.raw_name_clean = " ".join(norm_n.tokens)
        self.sig_name_tokens = set(norm_n.significant_tokens)
        self.all_name_tokens = set(norm_n.tokens)

        alpha = self.alphanumeric_name
        len_alpha = len(alpha)
        self.char2_name = {alpha[i:i+2] for i in range(len_alpha - 1)} if len_alpha >= 2 else set()
        self.char3_name = {alpha[i:i+3] for i in range(len_alpha - 2)} if len_alpha >= 3 else set()
        self.char4_name = {alpha[i:i+4] for i in range(len_alpha - 3)} if len_alpha >= 4 else set()

        # Normalize address
        norm_a = normalize_business_address(self.raw_addr)
        self.addr_clean_lower = norm_a.cleaned_lower
        self.addr_tokens = set(norm_a.tokens)
        self.addr_non_stop_tokens = {t for t in norm_a.tokens if t not in ADDRESS_STOP_TOKENS and len(t) >= 3 and not t.isdigit()}
        self.addr_numeric_tokens = set(norm_a.numeric_tokens)
        nums = norm_a.numeric_tokens
        self.addr_street_num = str(int(nums[0])) if nums else ""
        self.addr_postal = nums[-1] if len(nums) >= 2 else ""

        # Locality tokens (tail of address)
        self.addr_locality_tokens = set(norm_a.tokens[-3:]) if len(norm_a.tokens) >= 3 else set(norm_a.tokens)

        a_clean = self.addr_clean_lower
        len_a = len(a_clean)
        self.addr_char3 = {a_clean[i:i+3] for i in range(len_a - 2)} if len_a >= 3 else set()

        self.is_non_latin = any("\u0900" <= c <= "\u0d7f" for c in self.raw_name)
        self.has_address = bool(self.raw_addr and self.raw_addr.strip())


def extract_matching_features(
    s1: MatchingEntityProfile,
    t: MatchingEntityProfile,
    provenance: Optional[Dict[str, Any]] = None,
    ranker_score: float = 0.0,
    ranker_rank: int = 1
) -> np.ndarray:
    """
    Extract a dense 1D float32 vector of 52 features for candidate pair (s1, t).
    """
    prov = provenance or {}
    feat = np.zeros(NUM_MATCHING_FEATURES, dtype=np.float32)

    # 1. NAME FEATURES (0 to 14)
    s1_alpha = s1.alphanumeric_name
    t_alpha = t.alphanumeric_name
    exact_match = (s1_alpha == t_alpha and bool(s1_alpha))
    feat[0] = 1.0 if exact_match else 0.0
    feat[1] = 1.0 if (s1.raw_name_clean == t.raw_name_clean and bool(s1.raw_name_clean)) else 0.0

    s1_sig = s1.sig_name_tokens
    t_sig = t.sig_name_tokens
    inter_sig = len(s1_sig & t_sig)
    union_sig = len(s1_sig | t_sig)
    min_sig = min(len(s1_sig), len(t_sig))
    feat[2] = inter_sig / max(1, union_sig)
    feat[3] = inter_sig / max(1, min_sig)

    c2_inter = len(s1.char2_name & t.char2_name)
    c2_union = len(s1.char2_name | t.char2_name)
    feat[4] = c2_inter / max(1, c2_union)

    c3_inter = len(s1.char3_name & t.char3_name)
    c3_union = len(s1.char3_name | t.char3_name)
    feat[5] = c3_inter / max(1, c3_union)

    c4_inter = len(s1.char4_name & t.char4_name)
    c4_union = len(s1.char4_name | t.char4_name)
    feat[6] = c4_inter / max(1, c4_union)

    feat[7] = fast_levenshtein_ratio(s1_alpha, t_alpha)

    l1 = len(s1_alpha)
    l2 = len(t_alpha)
    feat[8] = min(l1, l2) / max(1, max(l1, l2))
    feat[9] = float(abs(len(s1.all_name_tokens) - len(t.all_name_tokens)))

    rare_df = prov.get("rarest_token_df", 0)
    feat[10] = 1.0 / math.log(1 + rare_df) if rare_df > 0 else 0.0

    feat[11] = float(inter_sig)
    feat[12] = feat[2]  # significant token Jaccard
    feat[13] = inter_sig / max(1, len(s1_sig))
    feat[14] = inter_sig / max(1, len(t_sig))

    # 2. ADDRESS FEATURES (15 to 29)
    s1_a_clean = s1.addr_clean_lower
    t_a_clean = t.addr_clean_lower
    addr_exact = (s1_a_clean == t_a_clean and bool(s1_a_clean))
    feat[15] = 1.0 if addr_exact else 0.0

    s1_atoks = s1.addr_tokens
    t_atoks = t.addr_tokens
    inter_a = len(s1_atoks & t_atoks)
    union_a = len(s1_atoks | t_atoks)
    feat[16] = inter_a / max(1, union_a)
    feat[17] = inter_a / max(1, min(len(s1_atoks), len(t_atoks)))

    ac3_inter = len(s1.addr_char3 & t.addr_char3)
    ac3_union = len(s1.addr_char3 | t.addr_char3)
    feat[18] = ac3_inter / max(1, ac3_union)

    shared_nums = s1.addr_numeric_tokens & t.addr_numeric_tokens
    feat[19] = float(len(shared_nums))
    union_nums = s1.addr_numeric_tokens | t.addr_numeric_tokens
    feat[20] = len(shared_nums) / max(1, len(union_nums))

    street_num_match = (s1.addr_street_num and s1.addr_street_num == t.addr_street_num)
    feat[21] = 1.0 if street_num_match else 0.0

    postal_match = (s1.addr_postal and s1.addr_postal == t.addr_postal)
    feat[22] = 1.0 if postal_match else 0.0

    loc_shared = s1.addr_locality_tokens & t.addr_locality_tokens
    feat[23] = 1.0 if loc_shared else 0.0
    feat[24] = len(loc_shared) / max(1, len(s1.addr_locality_tokens | t.addr_locality_tokens))

    la1 = len(s1_a_clean)
    la2 = len(t_a_clean)
    feat[25] = min(la1, la2) / max(1, max(la1, la2))
    feat[26] = 0.0 if s1.has_address else 1.0
    feat[27] = 0.0 if t.has_address else 1.0

    shared_non_stop = s1.addr_non_stop_tokens & t.addr_non_stop_tokens
    feat[28] = float(len(shared_non_stop))
    feat[29] = float(max((len(tok) for tok in shared_non_stop), default=0))

    # 3. NAME x ADDRESS INTERACTIONS (30 to 45)
    feat[30] = feat[2] * feat[16]  # name Jaccard * addr Jaccard
    feat[31] = 1.0 if (exact_match and feat[16] >= 0.25) else 0.0
    feat[32] = 1.0 if (feat[2] >= 0.70 and feat[16] < 0.20) else 0.0
    feat[33] = 1.0 if (feat[2] < 0.30 and feat[16] >= 0.60) else 0.0
    feat[34] = 1.0 if (street_num_match and feat[2] < 0.30) else 0.0
    feat[35] = 1.0 if (addr_exact and feat[2] < 0.30) else 0.0

    # Consensus count
    consensus = 0
    if exact_match:
        consensus += 1
    if addr_exact:
        consensus += 1
    if street_num_match:
        consensus += 1
    if postal_match:
        consensus += 1
    if s1.country and s1.country == t.country:
        consensus += 1
    feat[36] = float(consensus)

    channels = prov.get("channels", set())
    feat[37] = float(len(channels))
    feat[38] = 1.0 if "ch1_core_name" in channels else 0.0
    feat[39] = 1.0 if "ch2_top2_tokens" in channels else 0.0
    feat[40] = 1.0 if "ch3_street_num_token" in channels else 0.0
    feat[41] = 1.0 if "ch4_address_location" in channels else 0.0
    feat[42] = 1.0 if "ch5_char3_k50" in channels else 0.0
    feat[43] = 1.0 if "ch6_distinctive_token" in channels else 0.0
    feat[44] = 1.0 if "ch7_relaxed_street_num" in channels else 0.0
    feat[45] = float(prov.get("ch5_score", 0.0))

    # 4. RANKER SIGNALS & METADATA (46 to 51)
    feat[46] = float(ranker_score)
    feat[47] = float(ranker_rank)
    feat[48] = 1.0 / max(1, ranker_rank)
    feat[49] = 1.0 if (t.eid.startswith("S3") or "source3" in t.eid) else 0.0
    feat[50] = 1.0 if (s1.country and s1.country == t.country) else 0.0
    feat[51] = 1.0 if (s1.is_non_latin != t.is_non_latin) else 0.0

    return feat


def extract_batch_matching_features(
    s1: MatchingEntityProfile,
    target_profiles: List[MatchingEntityProfile],
    provenances: Optional[List[Dict[str, Any]]] = None,
    ranker_scores: Optional[List[float]] = None,
    ranker_ranks: Optional[List[int]] = None
) -> np.ndarray:
    """
    Extract a dense 2D float32 matrix of shape (N, NUM_MATCHING_FEATURES) for candidates of s1.
    """
    n = len(target_profiles)
    matrix = np.zeros((n, NUM_MATCHING_FEATURES), dtype=np.float32)
    prov_list = provenances or [{}] * n
    score_list = ranker_scores or [0.0] * n
    rank_list = ranker_ranks or list(range(1, n + 1))

    for i in range(n):
        matrix[i] = extract_matching_features(
            s1,
            target_profiles[i],
            prov_list[i],
            score_list[i],
            rank_list[i]
        )

    return matrix
