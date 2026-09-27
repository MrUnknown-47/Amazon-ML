"""
Conservative multi-representation normalization engine for names and addresses.
Preserves raw values while generating clean, normalized representations for indexing and matching.
Handles Unicode NFKD, diacritic stripping, case folding, and numeric token extraction.
"""

import re
import unicodedata
from dataclasses import dataclass
from typing import List, Set, Optional, Tuple


# Standard legal suffixes across jurisdictions (US, India, UK, France, Germany, etc.)
DEFAULT_LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "co", "company",
    "ltd", "limited", "pvt", "private", "llc", "llp", "lp", "pc",
    "gmbh", "ag", "sa", "sas", "sasu", "sarl", "sci", "eurl", "snc",
    "bhd", "pty", "pllc"
}

# Generic business noise words
DEFAULT_STOPWORDS = {
    "and", "&", "the", "of", "in", "at", "for", "by", "on", "a", "an",
    "et", "du", "de", "des", "la", "le", "les", "d"
}


@dataclass(frozen=True)
class NormalizedRepresentation:
    """
    Multi-faceted representation of a text field to avoid destructive over-normalization.
    """
    raw: str
    cleaned_lower: str
    ascii_clean: str
    alphanumeric: str
    tokens: List[str]
    significant_tokens: List[str]
    detected_legal_suffixes: List[str]
    numeric_tokens: List[str]
    sorted_top2_tokens: Tuple[str, ...]


def strip_accents_and_diacritics(text: str) -> str:
    """
    Decompose Unicode characters and remove combining diacritical marks for Latin text.
    Preserves Indic scripts (Devanagari, Tamil, Kannada, Bengali, etc. in 0x0900 - 0x0D7F)
    where combining characters represent essential vowel matras rather than optional accents.
    """
    nfkd_form = unicodedata.normalize("NFKD", text)
    return "".join(
        c for c in nfkd_form
        if not (unicodedata.combining(c) and not (0x0900 <= ord(c) <= 0x0D7F))
    )


def extract_numeric_tokens(text: str) -> List[str]:
    """
    Extract discrete numeric sequences from text (e.g. street numbers, PIN codes).
    Preserves numbers like '45' from '45th' or '684' from 'AF-0684'.
    """
    return re.findall(r"\b\d+\b", text)


def normalize_text(
    text: Optional[str],
    legal_suffixes: Optional[Set[str]] = None,
    stopwords: Optional[Set[str]] = None,
    strip_legal: bool = True
) -> NormalizedRepresentation:
    """
    Produce a comprehensive NormalizedRepresentation for a given string.
    Never alters irreversible original information.
    """
    if text is None:
        raw_text = ""
    else:
        raw_text = str(text)

    if legal_suffixes is None:
        legal_suffixes = DEFAULT_LEGAL_SUFFIXES
    if stopwords is None:
        stopwords = DEFAULT_STOPWORDS

    if not raw_text.strip():
        return NormalizedRepresentation(
            raw=raw_text,
            cleaned_lower="",
            ascii_clean="",
            alphanumeric="",
            tokens=[],
            significant_tokens=[],
            detected_legal_suffixes=[],
            numeric_tokens=[],
            sorted_top2_tokens=()
        )

    # 1. Unicode decomposition & Latin accent stripping (preserving Indic matras)
    ascii_clean = strip_accents_and_diacritics(raw_text)

    # 2. Lowercasing
    cleaned_lower = ascii_clean.lower().strip()

    # 3. Collapse dotted abbreviations: s.a.s. -> sas, p.v.t. -> pvt, u.s.a. -> usa
    collapsed = re.sub(r"\b([a-zA-Z0-9])\.", r"\1", cleaned_lower)

    # 4. Replace punctuation with space (& -> and, punctuation marks to space)
    normalized_punc = re.sub(r"[&]", " and ", collapsed)
    normalized_punc = re.sub(r"[\.,;:!\?\"'\/\\(\)\[\]\{\}<>~`@#\$%\^*_+=|-]", " ", normalized_punc)
    normalized_punc = re.sub(r"\s+", " ", normalized_punc).strip()

    # 5. Tokenization: split on whitespace to preserve Indic words with vowel marks
    tokens = [t.strip() for t in normalized_punc.split() if t.strip()]

    # 6. Extract numeric tokens
    numeric_tokens = extract_numeric_tokens(raw_text)

    # 7. Alphanumeric joined string
    alphanumeric = " ".join(tokens)

    # 8. Identify legal suffixes and significant tokens
    detected_suffixes = []
    significant_tokens = []

    for tok in tokens:
        if tok in legal_suffixes:
            detected_suffixes.append(tok)
            if not strip_legal:
                significant_tokens.append(tok)
        elif tok in stopwords:
            continue
        else:
            significant_tokens.append(tok)

    # 9. Sorted top-2 significant tokens (for permutation-invariant keys)
    sig_non_num = [t for t in significant_tokens if not t.isdigit() and len(t) >= 2]
    if len(sig_non_num) >= 2:
        sorted_top2 = tuple(sorted(sig_non_num[:2]))
    elif len(sig_non_num) == 1:
        sorted_top2 = (sig_non_num[0],)
    else:
        sorted_top2 = tuple(sorted(tokens[:2])) if len(tokens) >= 2 else (tokens[0],) if tokens else ()

    return NormalizedRepresentation(
        raw=raw_text,
        cleaned_lower=cleaned_lower,
        ascii_clean=ascii_clean,
        alphanumeric=alphanumeric,
        tokens=tokens,
        significant_tokens=significant_tokens,
        detected_legal_suffixes=detected_suffixes,
        numeric_tokens=numeric_tokens,
        sorted_top2_tokens=sorted_top2
    )


def normalize_business_name(name: str) -> NormalizedRepresentation:
    """Specialized wrapper for business names."""
    return normalize_text(name, strip_legal=True)


def normalize_business_address(address: str) -> NormalizedRepresentation:
    """Specialized wrapper for business addresses."""
    return normalize_text(address, strip_legal=False)
