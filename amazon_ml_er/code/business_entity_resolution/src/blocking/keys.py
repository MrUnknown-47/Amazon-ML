"""
Blocking key extractors for Business Entity Resolution.
Derives deterministic and fuzzy indexing keys from normalized text and address components.
Zero hardcoding of countries; fully open-set compatible.
"""

import re
from typing import List, Tuple, Optional, Set, Dict, Any

from ..preprocessing.normalizer import (
    normalize_business_name,
    normalize_business_address,
    extract_numeric_tokens,
    NormalizedRepresentation
)

# Common address street-type / noise tokens across US, India, and France
ADDRESS_STOP_TOKENS = {
    # English / US / General
    "st", "street", "rd", "road", "ave", "avenue", "dr", "drive", "blvd", "boulevard",
    "ln", "lane", "way", "ct", "court", "pl", "place", "pkwy", "parkway", "cir", "circle",
    "fl", "floor", "ste", "suite", "unit", "apt", "apartment", "bldg", "building",
    "po", "box", "pobox", "near", "opp", "opposite", "behind", "beside", "adjacent",
    "block", "sector", "phase", "plot", "no", "shop", "flat",
    # French
    "rue", "r", "bd", "av", "allee", "route", "chemin", "impasse", "cours", "place",
    "quai", "zone", "zi", "za", "zac", "bis", "ter"
}


def extract_normalized_name_key(name: str) -> str:
    """
    Channel 1 Key: Core normalized name with legal suffixes removed.
    Returns cleaned alphanumeric string, or empty string if uninformative.
    """
    norm = normalize_business_name(name)
    if norm.significant_tokens:
        return " ".join(norm.significant_tokens)
    return norm.alphanumeric


def extract_sorted_tokens_key(name: str, n_tokens: int = 2) -> str:
    """
    Channel 2 Key: Alphabetically sorted top-N significant tokens.
    Permutation-invariant to word transpositions across all token positions.
    """
    norm = normalize_business_name(name)
    sig = [t for t in norm.significant_tokens if not t.isdigit() and len(t) >= 2]
    if not sig:
        sig = [t for t in norm.tokens if not t.isdigit() and len(t) >= 2]
    if not sig:
        return ""

    sorted_sig = sorted(sig)
    chosen = sorted_sig[:n_tokens]
    return "_".join(chosen)


def extract_street_num_token_key(address: str) -> str:
    """
    Channel 3 Key: Primary numeric address token + first significant street token.
    E.g. '85 Wayne Avenue, Ticonderoga, NY' -> '85_wayne'
    '3315 Fremont St, Peoria, IL' -> '3315_fremont'
    'AF-0684, Nandgram, Ghaziabad' -> '684_nandgram'
    Returns empty string if address is missing or has no numeric tokens.
    """
    if not address or not address.strip():
        return ""

    num_tokens = extract_numeric_tokens(address)
    if not num_tokens:
        return ""

    primary_num = str(int(num_tokens[0]))

    norm_addr = normalize_business_address(address)
    street_word = ""
    for tok in norm_addr.tokens:
        if tok.isdigit():
            continue
        if tok in ADDRESS_STOP_TOKENS:
            continue
        if len(tok) >= 3:
            street_word = tok
            break

    if not street_word:
        return ""

    return f"{primary_num}_{street_word}"


def extract_address_location_key(address: str) -> str:
    """
    Channel 4 Key: Primary numeric token + locality/city token.
    Extracts the last 1-2 address tokens (which in standardized addresses represent city/state/PIN).
    E.g. '85 Wayne Avenue, Ticonderoga, NY' -> '85_ticonderoga'
    '3315 Fremont Street, Peoria, IL' -> '3315_peoria'
    '20 Rue Parmentier, Dunkerque' -> '20_dunkerque'
    """
    if not address or not address.strip():
        return ""

    num_tokens = extract_numeric_tokens(address)
    if not num_tokens:
        return ""

    primary_num = str(int(num_tokens[0]))

    norm_addr = normalize_business_address(address)
    city_word = ""
    for tok in reversed(norm_addr.tokens):
        if tok.isdigit():
            continue
        if tok in ADDRESS_STOP_TOKENS:
            continue
        if len(tok) >= 3:
            city_word = tok
            break

    if not city_word:
        return ""

    return f"{primary_num}_{city_word}"


def extract_all_blocking_keys(name: str, address: str) -> Dict[str, Any]:
    """
    High-performance combined key extractor that normalizes name and address ONCE,
    deriving all channel keys in a single pass.
    """
    norm_name = normalize_business_name(name)
    norm_addr = normalize_business_address(address)

    # Ch 1 key
    ch1_key = " ".join(norm_name.significant_tokens) if norm_name.significant_tokens else norm_name.alphanumeric

    # Ch 2 key variants
    sig = [t for t in norm_name.significant_tokens if not t.isdigit() and len(t) >= 2]
    if not sig:
        sig = [t for t in norm_name.tokens if not t.isdigit() and len(t) >= 2]

    if sig:
        sorted_sig = sorted(sig)
        ch2_top1 = sorted_sig[0]
        ch2_top2 = "_".join(sorted_sig[:2])
        ch2_top3 = "_".join(sorted_sig[:3])
    else:
        ch2_top1 = ch2_top2 = ch2_top3 = ""

    # Ch 3 & Ch 4 address keys
    ch3_key = ""
    ch4_key = ""
    num_tokens = norm_addr.numeric_tokens
    if num_tokens and address and address.strip():
        primary_num = str(int(num_tokens[0]))
        for tok in norm_addr.tokens:
            if not tok.isdigit() and tok not in ADDRESS_STOP_TOKENS and len(tok) >= 3:
                ch3_key = f"{primary_num}_{tok}"
                break
        for tok in reversed(norm_addr.tokens):
            if not tok.isdigit() and tok not in ADDRESS_STOP_TOKENS and len(tok) >= 3:
                ch4_key = f"{primary_num}_{tok}"
                break

    return {
        "ch1_key": ch1_key,
        "ch2_top1": ch2_top1,
        "ch2_top2": ch2_top2,
        "ch2_top3": ch2_top3,
        "ch3_key": ch3_key,
        "ch4_key": ch4_key,
        "ch5_text": ch1_key
    }


def extract_char_ngrams(text: str, n: int = 3) -> Set[str]:
    """
    Extract set of character n-grams from normalized text.
    """
    clean = re.sub(r"\s+", " ", text.strip().lower())
    if len(clean) < n:
        return {clean} if clean else set()
    return {clean[i:i + n] for i in range(len(clean) - n + 1)}


def extract_combined_char_ngrams(text: str, ns: Tuple[int, ...] = (2, 3)) -> Set[str]:
    """
    Extract multiple character n-gram sizes (e.g. 2+3 or 3+4) from text.
    """
    clean = re.sub(r"\s+", " ", text.strip().lower())
    result = set()
    for n in ns:
        if len(clean) < n:
            if clean:
                result.add(f"{n}g_{clean}")
        else:
            for i in range(len(clean) - n + 1):
                result.add(f"{n}g_{clean[i:i + n]}")
    return result


def extract_token_ngrams(text: str, n: int = 2) -> Set[str]:
    """
    Extract token-level contiguous n-grams from text.
    """
    tokens = [t for t in re.findall(r"\w+", text.lower()) if t]
    if len(tokens) < n:
        return {"_".join(tokens)} if tokens else set()
    return {"_".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}


def extract_token_set(text: str, min_len: int = 2) -> Set[str]:
    """
    Extract unique single significant alphanumeric tokens.
    """
    tokens = [t for t in re.findall(r"\w+", text.lower()) if len(t) >= min_len]
    return set(tokens)


def extract_prefix_substring_keys(text: str, prefix_len: int = 4) -> Set[str]:
    """
    Extract prefix keys from significant tokens and start of name.
    """
    clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    keys = set()
    if clean:
        keys.add(f"pref_{clean[:prefix_len]}")
    tokens = [t for t in clean.split() if len(t) >= prefix_len]
    for t in tokens:
        keys.add(f"tokpref_{t[:prefix_len]}")
    return keys

