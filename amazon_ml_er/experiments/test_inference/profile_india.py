import time, csv
from pathlib import Path
from amazon_ml_er.experiments.test_inference.run_chunked_inference import (
    load_targets, build_index, BlockerConfig, generate_raw_candidates,
    RankerEntityProfile, extract_ranker_features, NUM_RANKER_FEATURES
)

print("Loading 10 India queries...")
queries = []
with open("dataset/test/country_targets/queries_India.tsv") as f:
    r = csv.reader(f, delimiter="\t")
    next(r)
    for i, row in enumerate(r):
        if i >= 10: break
        queries.append({"entity_id": row[0], "business_name": row[1], "business_address": row[2], "country": row[3]})

print("Loading India targets...")
t0 = time.time()
targets = load_targets("India")
print(f"Loaded {len(targets):,} targets in {time.time()-t0:.1f}s")

cfg = BlockerConfig(max_bucket_size=500, ch5_top_k=50, ch5_max_df=3000, ch6_tokens_to_query=2, ch6_max_df=3000)
t0 = time.time()
idx = build_index(targets, cfg)
print(f"Index built in {time.time()-t0:.1f}s")

for q in queries:
    t0 = time.time()
    raw = generate_raw_candidates(q, idx, cfg)
    addr_sub = q["business_address"][:30] if q["business_address"] else ""
    print(f"Query {q['entity_id']} ({q['business_name']} | {addr_sub}): raw={len(raw)} in {time.time()-t0:.3f}s")
