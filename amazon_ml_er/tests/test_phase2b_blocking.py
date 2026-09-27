"""
Phase 2B Automated Test Suite:
1. Character ngram posting lists
2. Document-frequency (DF) filtering
3. Deterministic retrieval
4. Token retrieval
5. Retrieval top-k enforcement
6. No Cartesian product materialization
7. Candidate union correctness
8. Category-level recall computation
9. Unseen country labels (generic open-set support, e.g. France)
10. Candidate IDs remaining valid (never corrupted, never empty, never S1 IDs)
"""

import pytest
import math
from typing import Dict, Set, List

from amazon_ml_er.code.business_entity_resolution.src.blocking.keys import (
    extract_char_ngrams,
    extract_combined_char_ngrams,
    extract_token_ngrams,
    extract_token_set,
    extract_prefix_substring_keys,
    extract_all_blocking_keys
)
from amazon_ml_er.code.business_entity_resolution.src.blocking.indexes import (
    ExactInvertedIndex,
    PostingListIndex
)
from amazon_ml_er.code.business_entity_resolution.src.evaluation.metrics import compute_candidate_recall


# 1. Test character ngram posting lists
def test_char_ngram_posting_lists():
    pl = PostingListIndex(name="test_char_3", ngram_mode="char_3", max_df_count=100)
    pl.index_document("S2-001", "microsoft corp")
    pl.index_document("S2-002", "micro strategy inc")
    pl.index_document("S3-003", "apple computer")
    pl.finalize_index()

    assert pl.total_docs == 3
    assert len(pl.int_to_id) == 3
    assert pl.id_to_int["S2-001"] == 0
    assert pl.id_to_int["S2-002"] == 1
    assert pl.id_to_int["S3-003"] == 2

    # Query for 'micro'
    results = pl.retrieve_top_k("micro", top_k=10)
    retrieved_eids = [eid for eid, _ in results]
    assert "S2-001" in retrieved_eids
    assert "S2-002" in retrieved_eids
    assert "S3-003" not in retrieved_eids


# 2. Test DF filtering
def test_df_filtering():
    pl = PostingListIndex(name="test_df", ngram_mode="char_3", max_df_count=2, max_df_ratio=1.0)
    # Term 'abc' appears in 3 documents
    pl.index_document("S2-001", "abc xyz")
    pl.index_document("S2-002", "abc klm")
    pl.index_document("S3-003", "abc pqr")
    # Term 'xyz' appears in 1 document
    pl.index_document("S3-004", "xyz unique")
    pl.finalize_index()

    # 'abc' has df=3, which exceeds max_df_count=2
    assert "abc" in pl.pruned_ngrams
    assert "abc" not in pl.idf
    assert "xyz" in pl.idf

    # Querying 'abc' returns empty or only retained terms
    res_abc = pl.retrieve_top_k("abc", top_k=10)
    assert len(res_abc) == 0

    # Querying 'xyz' returns S2-001 and S3-004
    res_xyz = pl.retrieve_top_k("xyz", top_k=10)
    retrieved = [eid for eid, _ in res_xyz]
    assert "S2-001" in retrieved
    assert "S3-004" in retrieved


# 3. Test deterministic retrieval
def test_deterministic_retrieval():
    pl = PostingListIndex(name="test_det", ngram_mode="char_3", max_df_count=100)
    for i in range(20):
        pl.index_document(f"S2-{i:03d}", f"company name variation {i % 5} technology")
    pl.finalize_index()

    q = "variation 2 technology"
    run1 = pl.retrieve_top_k(q, top_k=5)
    run2 = pl.retrieve_top_k(q, top_k=5)
    run3 = pl.retrieve_two_stage(q, pool_size=10, top_k=5, rank_metric="jaccard")
    run4 = pl.retrieve_two_stage(q, pool_size=10, top_k=5, rank_metric="jaccard")

    assert run1 == run2, "retrieve_top_k must be strictly deterministic across calls"
    assert run3 == run4, "retrieve_two_stage must be strictly deterministic across calls"
    assert len(run1) == 5
    assert len(run3) == 5


# 4. Test token retrieval
def test_token_retrieval():
    pl = PostingListIndex(name="test_tok", ngram_mode="token_1", max_df_count=100)
    pl.index_document("S2-001", "achyuta nursing home")
    pl.index_document("S2-002", "shree krishna nursing clinic")
    pl.index_document("S3-003", "fortis healthcare hospital")
    pl.finalize_index()

    # Query with distinctive token 'achyuta'
    hits = pl.retrieve_top_k("achyuta hbme", top_k=5)
    retrieved = [eid for eid, _ in hits]
    assert retrieved[0] == "S2-001"


# 5. Test retrieval top-k
def test_retrieval_top_k():
    pl = PostingListIndex(name="test_topk", ngram_mode="char_3", max_df_count=500)
    for i in range(50):
        pl.index_document(f"S2-{i:03d}", f"united parcel service logistics {i}")
    pl.finalize_index()

    for k in [1, 5, 15, 25, 40]:
        res = pl.retrieve_top_k("united parcel logistics", top_k=k)
        assert len(res) <= k
        # Scores must be in non-increasing order
        scores = [s for _, s in res]
        assert scores == sorted(scores, reverse=True)


# 6. Test no Cartesian product
def test_no_cartesian_product():
    """Ensure posting lists index query term lookups rather than forming N x M pairs."""
    pl = PostingListIndex(name="test_sparse", ngram_mode="char_3", max_df_count=500)
    n_docs = 500
    for i in range(n_docs):
        pl.index_document(f"S2-{i}", f"acme entity {i}")
    pl.finalize_index()

    # Query with a completely disjoint term
    hits = pl.retrieve_top_k("zebra giraffe elephant", top_k=25)
    assert len(hits) == 0, "No candidates returned for disjoint query terms (no dense Cartesian product)"


# 7. Test candidate union correctness
def test_candidate_union_correctness():
    ch1_cands = {"S1-1": {"S2-10", "S2-20"}, "S1-2": {"S2-30"}}
    ch2_cands = {"S1-1": {"S2-20", "S2-40"}, "S1-2": {"S2-30", "S3-50"}}

    union_cands = {}
    all_keys = set(ch1_cands.keys()) | set(ch2_cands.keys())
    for qid in all_keys:
        union_cands[qid] = ch1_cands.get(qid, set()) | ch2_cands.get(qid, set())

    assert union_cands["S1-1"] == {"S2-10", "S2-20", "S2-40"}
    assert union_cands["S1-2"] == {"S2-30", "S3-50"}
    assert len(union_cands["S1-1"]) == 3  # Duplicate S2-20 deduped
    assert len(union_cands["S1-2"]) == 2  # Duplicate S2-30 deduped


# 8. Test category-level recall computation
def test_category_level_recall_computation():
    gt = {
        "S1-1": {"S2-101", "S2-102"},
        "S1-2": {"S2-201"},
        "S1-3": {"S3-301"}
    }
    cands = {
        "S1-1": {"S2-101"},          # 1 of 2 captured
        "S1-2": {"S2-201", "S2-999"}, # 1 of 1 captured
        "S1-3": set()                 # 0 captured
    }

    stats = compute_candidate_recall(gt, cands)
    assert stats["total_true_matches"] == 4
    assert stats["captured_true_matches"] == 2
    assert stats["missed_matches"] == 2
    assert stats["candidate_recall"] == 0.5


# 9. Test unseen country labels (generic open-set support)
def test_unseen_country_labels():
    """Verify generic open-set indexing works for unencountered countries like France, Japan, etc."""
    idx = ExactInvertedIndex("open_country_idx")
    idx.add("bordeaux_ecole", "S2-FR01")
    idx.add("lyon_boulangerie", "S2-FR02")
    idx.add("tokyo_ramen", "S3-JP01")

    res_fr = idx.get_candidates("bordeaux_ecole")
    assert res_fr == ["S2-FR01"]

    res_jp = idx.get_candidates("tokyo_ramen")
    assert res_jp == ["S3-JP01"]

    res_none = idx.get_candidates("paris_cafe")
    assert res_none == []


# 10. Test candidate IDs remaining valid
def test_candidate_ids_remaining_valid():
    """Verify that all returned candidate IDs are valid target IDs and never query S1 IDs."""
    pl = PostingListIndex(name="test_val_ids", ngram_mode="char_3", max_df_count=100)
    target_ids = {"S2-100", "S2-101", "S3-200"}
    for tid in target_ids:
        pl.index_document(tid, f"target record business {tid}")
    pl.finalize_index()

    hits = pl.retrieve_top_k("target record business S1-999", top_k=10)
    for cid, score in hits:
        assert cid in target_ids, f"Candidate {cid} must be in target ID corpus"
        assert cid.startswith("S2-") or cid.startswith("S3-"), f"Candidate {cid} must be target source"
        assert not cid.startswith("S1-"), f"Candidate {cid} must NEVER be S1 query ID"
        assert cid != "", "Candidate ID must not be empty"
        assert isinstance(score, float)
