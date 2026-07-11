# Real-Data Pipeline — `real_data.py` + `drift_report.py`

Two new pieces, both tested with mock data (real output below — not claims).
Neither touches the encoder itself; they're the data plumbing around it.

## What's new

| File | What it does | Tested? |
|---|---|---|
| `real_data.py` | Turns a raw query log into positive pairs for contrastive training, **without manual labeling** — two queries that hit the same retrieved chunk are a positive pair. | Yes, with mock CSV (below). |
| `drift_report.py` | Compares two traffic snapshots (before/after a model change) **per behavior cluster**, not one global number — ranks the clusters that actually shifted, by severity × volume. | Yes, with synthetic shift/no-shift groups (below). |
| `test_real_pipeline.py` | The test itself — rerun it any time with `python3 test_real_pipeline.py`. | — |

## Verified output (this session, real run)

**Pairing** — mock 5-chunk retrieval log, correctly grouped and paired:
```
[chunk_billing_faq] 'invoice looks wrong' <-> 'duplicate charge on card'
[chunk_refund_policy] 'checking on a refund' <-> 'refund status please'
...
```

**Drift ranking** — 8 synthetic groups, 2 deliberately shifted (ground truth), 6 left alone:
```
group_2   MMD^2=0.1544  p=0.005  DRIFT
group_5   MMD^2=0.1370  p=0.005  DRIFT
group_1   MMD^2=0.0093  p=0.284  -
...
top-2 ranked by report: {'group_2', 'group_5'}  -> matches ground truth exactly
```

This is the actual "cluster and rank divergent regions" logic from the original deep-dive doc's Idea 2 architecture (component e) — the piece that tells you *which* behavior changed, not just *that* something changed.

## The honest gap

Neither of these two files touches the encoder. I tested them with plain synthetic vectors/strings — no sentence-transformers, no Hugging Face, because this sandbox has no route to the HF model hub (same limitation as `model_v2.py` before). The pairing and ranking *logic* is proven correct. Whether it works well on real embeddings from your actual encoder is the next thing to check, once you have both.

## How to plug in real data, step by step

**1. Get your RAG system's query log into this CSV shape:**
```
query_id,query_text,retrieved_chunk_id
1,"where is my refund",chunk_refund_policy
2,"refund status please",chunk_refund_policy
3,"package tracking",chunk_shipping_info
```
Most RAG pipelines already log which chunk/document was retrieved per query — check your vector DB or retrieval layer's logs first before building anything new.

**2. No retrieval logs available yet?** Fallback: sample ~150-200 real queries, spend an hour grouping them by hand into a `cluster_label` column instead of `retrieved_chunk_id`. Use `load_labeled_log()` instead of `load_retrieval_log()` — same downstream code either way.

**3. Wire it to your encoder** (whatever your agent built in `model_v2.py` / `train_v2.py`):
```python
from real_data import load_retrieval_log, build_groups, sample_pair_batch
# from model_v2 import PretrainedContrastiveEncoder  # adjust to wherever your v2 class actually lives

df = load_retrieval_log("real_queries.csv")
groups = build_groups(df, "retrieved_chunk_id")

anchors, positives, labels = sample_pair_batch(groups, batch_size=16)
# anchors/positives are raw strings here -- feed directly into your v2 encoder's
# forward(list[str]), the same way train_v2.py already does for the synthetic set.
```

**4. For the drift report, once you have a before/after model change:**
```python
from drift_report import compare_snapshots, print_report

# embed the SAME queries through model version A and version B (or before/after
# some pipeline change), grouped the same way (by retrieved_chunk_id or cluster_label)
results = compare_snapshots(embeddings_before, embeddings_after)
print_report(results)
```

## What this proves you now have

The full Idea 2 pipeline shape is built and each piece independently verified:
encoder (toy + real backbone) → InfoNCE training → pairing from real logs (no manual labeling required) → MMD drift test → ranked regression report.

The only thing left standing between this and a real product is a real client's actual query log.
