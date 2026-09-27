"""
Lightweight and fast feature extraction for candidate ranking and compression.
Computes name similarities, address overlaps, numeric agreement, interaction terms,
and blocking channel provenance indicators without external lookups.
"""

import math
from typing import Dict, List, Set, Tuple, Optional, Any
import numpy as np

from amazon_ml_er.code.business_entity_resolution.src.preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address
)

FEATURE_NAMES = [
    # Name features
    "name_exact_match",
    "name_token_jaccard",
    "name_token_containment",
    "name_char2_sim",
    "name_char3_sim",
    "name_char4_sim",
    "name_len_ratio",
    "name_token_count_diff",
    # Address features
    "addr_token_jaccard",
    "addr_token_containment",
    "addr_char3_sim",
    "addr_num_agreement",
    "addr_street_num_match",
    "addr_postal_match",
    "addr_missing_s1",
    "addr_missing_t",
    # Interaction & Provenance
    "name_x_addr_sim",
    "exact_agreement_count",
    "blocking_channel_hit_count",
    "ch1_core_name_hit",
    "ch2_top2_tokens_hit",
    "ch3_street_num_hit",
    "ch4_address_hit",
    "ch5_char3_hit",
    "ch6_distinctive_token_hit",
    "ch7_relaxed_street_hit",
    "rare_token_score",
    "is_transliteration_pair",
    "target_is_source3"
]

NUM_FEATURES = len(FEATURE_NAMES)


class EntityProfile:
    """
    Cached, pre-tokenized multi-field representation of a business entity.
    Avoids re-normalizing and re-tokenizing repeatedly during candidate ranking.
    """
    __slots__ = (
        "eid", "country", "raw_name", "raw_addr",
        "alphanumeric_name", "sig_name_tokens", "all_name_tokens",
        "char2_name", "char3_name", "char4_name",
        "addr_tokens", "addr_numeric_tokens", "addr_street_num", "addr_postal",
        "addr_char3", "addr_clean_lower",
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
        self.addr_numeric_tokens = norm_a.numeric_tokens
        self.addr_street_num = self.addr_numeric_tokens[0] if self.addr_numeric_tokens else ""
        self.addr_postal = self.addr_numeric_tokens[-1] if len(self.addr_numeric_tokens) >= 2 else ""

        a_clean = self.addr_clean_lower
        len_a = len(a_clean)
        self.addr_char3 = {a_clean[i:i+3] for i in range(len_a - 2)} if len_a >= 3 else set()

        self.is_non_latin = any("\u0900" <= c <= "\u0d7f" for c in self.raw_name)
        self.has_address = bool(self.raw_addr and self.raw_addr.strip())


def extract_pair_features(
    s1: EntityProfile,
    t: EntityProfile,
    provenance: Optional[Dict[str, Any]] = None
) -> np.ndarray:
    """
    Extract a dense 1D float32 feature vector for candidate pair (s1, t).
    """
    prov = provenance or {}
    feat = np.zeros(NUM_FEATURES, dtype=np.float32)

    # 1. NAME FEATURES
    # name_exact_match
    s1_alpha = s1.alphanumeric_name
    t_alpha = t.alphanumeric_name
    exact_match = (s1_alpha == t_alpha and bool(s1_alpha))
    feat[0] = 1.0 if exact_match else 0.0

    # name_token_jaccard & containment
    s1_sig = s1.sig_name_tokens
    t_sig = t.sig_name_tokens
    inter_sig = len(s1_sig & t_sig)
    union_sig = len(s1_sig | t_sig)
    feat[1] = inter_sig / max(1, union_sig)
    feat[2] = inter_sig / max(1, min(len(s1_sig), len(t_sig)))

    # char n-gram similarities
    c2_inter = len(s1.char2_name & t.char2_name)
    c2_union = len(s1.char2_name | t.char2_name)
    feat[3] = c2_inter / max(1, c2_union)

    c3_inter = len(s1.char3_name & t.char3_name)
    c3_union = len(s1.char3_name | t.char3_name)
    feat[4] = c3_inter / max(1, c3_union)

    c4_inter = len(s1.char4_name & t.char4_name)
    c4_union = len(s1.char4_name | t.char4_name)
    feat[5] = c4_inter / max(1, c4_union)

    # length ratio & token count diff
    l1 = len(s1_alpha)
    l2 = len(t_alpha)
    feat[6] = min(l1, l2) / max(1, max(l1, l2))
    feat[7] = abs(len(s1.all_name_tokens) - len(t.all_name_tokens))

    # 2. ADDRESS FEATURES
    s1_atoks = s1.addr_tokens
    t_atoks = t.addr_tokens
    inter_a = len(s1_atoks & t_atoks)
    union_a = len(s1_atoks | t_atoks)
    feat[8] = inter_a / max(1, union_a)
    feat[9] = inter_a / max(1, min(len(s1_atoks), len(t_atoks)))

    ac3_inter = len(s1.addr_char3 & t.addr_char3)
    ac3_union = len(s1.addr_char3 | t.addr_char3)
    feat[10] = ac3_inter / max(1, ac3_union)

    # numeric & street number agreement
    s1_nums = s1.addr_numeric_tokens
    t_nums = t.addr_numeric_tokens
    if s1_nums and t_nums:
        feat[11] = 1.0 if (s1_nums[0] == t_nums[0]) else 0.0
    else:
        feat[11] = 0.0

    feat[12] = 1.0 if (s1.addr_street_num and s1.addr_street_num == t.addr_street_num) else 0.0
    feat[13] = 1.0 if (s1.addr_postal and s1.addr_postal == t.addr_postal) else 0.0
    feat[14] = 0.0 if s1.has_address else 1.0
    feat[15] = 0.0 if t.has_address else 1.0

    # 3. INTERACTION & PROVENANCE
    feat[16] = feat[1] * feat[8]  # name_x_addr_sim

    # exact agreement count
    eq_count = 0
    if exact_match:
        eq_count += 1
    if feat[12] == 1.0:
        eq_count += 1
    if feat[13] == 1.0:
        eq_count += 1
    if s1.country and s1.country == t.country:
        eq_count += 1
    feat[17] = float(eq_count)

    # Channel provenance
    channels = prov.get("channels", set())
    feat[18] = float(len(channels))
    feat[19] = 1.0 if "ch1_core_name" in channels else 0.0
    feat[20] = 1.0 if "ch2_top2_tokens" in channels else 0.0
    feat[21] = 1.0 if "ch3_street_num_token" in channels else 0.0
    feat[22] = 1.0 if "ch4_address_location" in channels else 0.0
    feat[23] = 1.0 if "ch5_char3_k50" in channels else 0.0
    feat[24] = 1.0 if "ch6_distinctive_token" in channels else 0.0
    feat[25] = 1.0 if "ch7_relaxed_street_num" in channels else 0.0

    rare_df = prov.get("rarest_token_df", 0)
    feat[26] = 1.0 / math.log(1 + rare_df) if rare_df > 0 else 0.0

    feat[27] = 1.0 if (s1.is_non_latin != t.is_non_latin) else 0.0
    feat[28] = 1.0 if t.eid.startswith("S3") else 0.0

    return feat


def extract_batch_features(
    s1: EntityProfile,
    target_profiles: List[EntityProfile],
    provenances: Optional[List[Dict[str, Any]]] = None
) -> np.ndarray:
    """
    Extract a dense 2D float32 matrix of shape (N, NUM_FEATURES) for all candidates of s1.
    """
    n = len(target_profiles)
    matrix = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    prov_list = provenances or [{}] * n

    for i in range(n):
        matrix[i] = extract_pair_features(s1, target_profiles[i], prov_list[i])

    return matrix
