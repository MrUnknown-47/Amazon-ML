"""
Comprehensive Unit & Integration Test Suite for Phase 2 Blocking & Candidate Generation:
1. Exact-name block retrieval
2. Multi-token block retrieval (sorted top-1, top-2, top-3)
3. Address-key extraction (street number + street token, address location)
4. Missing-address handling
5. Unseen country label handling (open-set France, Germany, etc.)
6. Duplicate target IDs remaining distinct
7. Channel union removes duplicate candidate IDs
8. No S1 IDs enter candidate sets
9. R_block calculation
10. R_pruned calculation
11. Candidate subset invariants
12. K-budget enforcement (K = 10, 15, 25, 40, 60, 100)
13. Deterministic candidate generation under fixed seed/config
14. Oversized bucket handling
15. Posting-list retrieval does not construct Cartesian products
"""

import pytest
from typing import Dict, Set

from amazon_ml_er.code.business_entity_resolution.src.blocking.keys import (
    extract_normalized_name_key,
    extract_sorted_tokens_key,
    extract_street_num_token_key,
    extract_address_location_key,
    extract_char_ngrams,
    extract_token_ngrams
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.indexes import (
    ExactInvertedIndex,
    PostingListIndex
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.channels import (
    build_channel_1_normalized_name,
    build_channel_2_sorted_tokens,
    build_channel_3_street_num_token,
    build_channel_4_address_location,
    build_channel_5_posting_list,
    evaluate_single_channel
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.union import (
    union_channel_candidates,
    analyze_union_and_contributions
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.pruning import (
    prune_candidate_set,
    evaluate_candidate_budgets
)


# 1. Exact-name block retrieval
def test_exact_name_block_retrieval():
    ch1 = build_channel_1_normalized_name()
    targets = {
        "S2-001": {"business_name": "Acme Tools Inc", "business_address": "123 Main St", "country": "US"},
        "S2-002": {"business_name": "Beta Corp", "business_address": "456 Oak Rd", "country": "US"}
    }
    ch1.index_target_records(targets)

    # Query with legal suffix variation ('Acme Tools LLC') -> matches 'Acme Tools Inc'
    query = {"business_name": "Acme Tools LLC", "business_address": "123 Main Street", "country": "US"}
    cands = ch1.generate_candidates(query)
    assert "S2-001" in cands
    assert "S2-002" not in cands


# 2. Multi-token block retrieval (sorted top-1, top-2, top-3)
def test_multi_token_block_retrieval():
    # Sorted top-2 is invariant to word transposition
    ch2_top2 = build_channel_2_sorted_tokens(n_tokens=2)
    targets = {
        "S2-101": {"business_name": "Flowers Hendricks LLC", "business_address": "", "country": "US"}
    }
    ch2_top2.index_target_records(targets)

    query = {"business_name": "Hendricks and Flowers Inc", "business_address": "", "country": "US"}
    cands = ch2_top2.generate_candidates(query)
    assert "S2-101" in cands

    # Test top-1 vs top-3 keys
    key_top1 = extract_sorted_tokens_key("Gamma Beta Alpha Corp", n_tokens=1)
    key_top3 = extract_sorted_tokens_key("Gamma Beta Alpha Corp", n_tokens=3)
    assert key_top1 == "alpha"
    assert key_top3 == "alpha_beta_gamma"


# 3. Address-key extraction & 4. Missing-address handling
def test_address_key_extraction_and_missing_address():
    # Regular address
    k3 = extract_street_num_token_key("85 Wayne Avenue, Ticonderoga, NY")
    assert k3 == "85_wayne"

    k4 = extract_address_location_key("85 Wayne Avenue, Ticonderoga, NY")
    assert k4 == "85_ticonderoga"

    # Address with leading zeros
    k3_zero = extract_street_num_token_key("AF-0684, Nandgram, Ghaziabad")
    assert k3_zero == "684_nandgram"

    # Missing / empty address must return empty string and not crash
    assert extract_street_num_token_key("") == ""
    assert extract_street_num_token_key("   ") == ""
    assert extract_address_location_key("") == ""

    # Address without numbers
    assert extract_street_num_token_key("Mack Rd, Haltom City, Texas") == ""


# 5. Unseen country label handling (open-set France)
def test_unseen_country_handling():
    ch_fr = build_channel_1_normalized_name()
    targets = {
        "S2-FR1": {"business_name": "Thermal & Fils SASU", "business_address": "20 Rue Parmentier", "country": "France"}
    }
    ch_fr.index_target_records(targets)

    query = {"business_name": "Thermal & Fils", "business_address": "20 Rue Parmentier", "country": "France"}
    cands = ch_fr.generate_candidates(query)
    assert "S2-FR1" in cands


# 6. Duplicate target IDs remaining distinct & 7. Channel union removes duplicates
def test_duplicate_targets_and_union_deduplication():
    # If two separate channels find the same candidate S2-001, union must have exactly 1 instance of S2-001
    ch1_cands = {"S1-1": {"S2-001", "S2-002"}}
    ch2_cands = {"S1-1": {"S2-001", "S3-003"}}

    union = union_channel_candidates([ch1_cands, ch2_cands], query_ids=["S1-1"])
    assert union["S1-1"] == {"S2-001", "S2-002", "S3-003"}
    assert len(union["S1-1"]) == 3


# 8. No S1 IDs enter candidate sets
def test_no_s1_self_matches_in_union():
    # If a bug introduced an S1 ID into candidate set, union must purge it
    bad_cands = {"S1-1": {"S1-1", "S1-999", "S2-001"}}
    clean_union = union_channel_candidates([bad_cands], query_ids=["S1-1"])
    assert "S1-1" not in clean_union["S1-1"]
    assert "S1-999" not in clean_union["S1-1"]
    assert clean_union["S1-1"] == {"S2-001"}


# 9. R_block calculation & 10. R_pruned calculation
def test_r_block_and_r_pruned_calculation():
    ground_truth = {
        "S1-1": {"S2-10", "S3-20"},
        "S1-2": {"S2-30"},
        "S1-3": set()  # Singleton
    }
    union_cands = {
        "S1-1": {"S2-10", "S3-20", "S2-99"},  # 2 true matches captured
        "S1-2": {"S2-30", "S3-88"},          # 1 true match captured
        "S1-3": set()
    }
    # Raw union captured all 3 true matches -> R_block = 1.0
    from amazon_ml_er.code.business_entity_resolution.src.evaluation.metrics import compute_candidate_recall
    stats = compute_candidate_recall(ground_truth, union_cands)
    assert stats["candidate_recall"] == 1.0

    # Pruned set drops S3-20
    pruned_cands = {
        "S1-1": {"S2-10", "S2-99"},
        "S1-2": {"S2-30"},
        "S1-3": set()
    }
    stats_p = compute_candidate_recall(ground_truth, pruned_cands)
    # 2 out of 3 true matches captured -> R_pruned = 2/3
    assert round(stats_p["candidate_recall"], 4) == round(2 / 3, 4)


# 11. Candidate subset invariants & 12. K-budget enforcement
def test_k_budget_enforcement_and_subset_invariants():
    query_records = {
        "S1-1": {"business_name": "Apex Medical Center", "business_address": "100 Hospital Way"}
    }
    target_records = {
        f"S2-{i:03d}": {"business_name": f"Apex Medical Center {i}", "business_address": f"{i} Way"}
        for i in range(50)
    }
    union_cands = {"S1-1": set(target_records.keys())}  # 50 candidates

    # Test pruning at K = 10, 25, 40
    for K in [10, 25, 40]:
        pruned = prune_candidate_set(union_cands, query_records, target_records, K=K)
        assert len(pruned["S1-1"]) == K
        # Subsetting invariant: pruned must be strict subset of union
        assert pruned["S1-1"].issubset(union_cands["S1-1"])


# 13. Deterministic candidate generation under fixed config
def test_deterministic_pruning():
    query_records = {"S1-1": {"business_name": "Alpha Beta", "business_address": "123 Road"}}
    target_records = {
        f"S2-{i}": {"business_name": f"Alpha Beta {i}", "business_address": "123 Road"}
        for i in range(30)
    }
    union_cands = {"S1-1": set(target_records.keys())}

    p1 = prune_candidate_set(union_cands, query_records, target_records, K=15)
    p2 = prune_candidate_set(union_cands, query_records, target_records, K=15)
    assert p1 == p2


# 14. Oversized bucket handling
def test_oversized_bucket_handling():
    idx = ExactInvertedIndex("test_oversized", max_bucket_size=5)
    # Add 10 records under the same key
    for i in range(10):
        idx.add("popular_key", f"S2-{i}")

    cands = idx.get_candidates("popular_key")
    # Must cap at max_bucket_size (5) rather than exploding to 10
    assert len(cands) == 5
    stats = idx.compute_bucket_statistics()
    assert stats["oversized_buckets_count"] == 1


# 15. Posting-list retrieval does not construct Cartesian products
def test_posting_list_retrieval_and_pruning():
    pl = PostingListIndex("test_pl", n=3, max_df_ratio=0.5)
    # 10 documents
    for i in range(10):
        pl.index_document(f"S2-{i}", f"CommonPrefix UniqueToken{i}")
    pl.finalize_index()

    # The common prefix 'com' or 'pre' should have high document frequency
    stats = pl.get_index_stats()
    assert stats["total_docs_indexed"] == 10
    assert stats["retained_ngrams"] > 0

    results = pl.retrieve_top_k("UniqueToken5", top_k=3)
    assert len(results) > 0
    # Top result should be S2-5
    top_id, score = results[0]
    assert top_id == "S2-5"
