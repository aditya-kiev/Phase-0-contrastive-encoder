# Contrastive Encoder — Phase 0

Learning how contrastive learning actually works, before using it for real: a
drift-detection tool that flags when an LLM/RAG pipeline's behavior changes
between model versions.

## What's here

A small encoder trained with InfoNCE loss on a toy dataset (support-ticket
style sentences grouped by intent), plus an MMD two-sample test to check
whether two sets of embeddings come from different distributions. Two
versions of the encoder:

- **v1** (`model.py`) — word embeddings, mean-pooled, projection head.
  Trained from scratch, no pretrained weights. Bag-of-words — word order
  doesn't matter. Good enough to learn the loss/embedding mechanics, not
  good enough for real text.
- **v2** (`train_v2.py`) — same loss and training shape, but the backbone is
  a pretrained sentence-transformer (`all-MiniLM-L6-v2`), frozen, with only
  a small projection head trained on top. Dataset expanded to 12 intents.

## Files

| File | What it does |
|---|---|
| `data.py` | toy dataset (5 intents) + pair sampler for v1 |
| `model.py` | v1 encoder — embeddings, mean pool, projection head |
| `losses.py` | InfoNCE / NT-Xent loss |
| `train.py` / `validate.py` | v1 training + clustering/drift checks |
| `train_v2.py` / `validate_v2.py` | v2 training on 12 intents with pretrained backbone |
| `generalization_check.py` / `generalization_check_v2.py` | held-out sentence checks |
| `eval_drift.py` | MMD two-sample test |
| `real_data.py` | turns a real query log into training pairs — groups queries by which chunk they retrieved, no manual labeling needed |
| `drift_report.py` | ranks which behavior cluster actually shifted between two traffic snapshots, instead of one global number |
| `test_real_pipeline.py` | smoke test for the two files above, using mock data |

## Results so far

v1 (toy, 5 intents): same-intent cosine sim 0.95 vs. different-intent -0.23.
MMD test correctly flags two different intents as drifted (p=0.015) and
correctly passes a same-intent split (p=1.0, small sample though). 5/5
held-out sentences matched their correct cluster.

v2 (pretrained backbone, 12 intents): same-intent sim 0.87 vs. -0.07,
12/12 held-out sentences matched correctly, MMD test still correctly
separates drift from no-drift (p=0.005 / p=0.93).

Pairing + drift-ranking logic (`real_data.py`, `drift_report.py`): tested on
mock data, correctly identified 2/8 deliberately-shifted synthetic groups as
the top priority, ignoring the other 6.

## Run it

```bash
pip install -r requirements.txt
python3 validate.py                # v1: train + clustering + drift check
python3 generalization_check.py    # v1: held-out check
python3 validate_v2.py             # v2: same, with pretrained backbone
python3 generalization_check_v2.py
python3 test_real_pipeline.py      # pairing + ranked drift report, mock data
```

## What's next

Everything above runs on synthetic/toy data. The real next step is plugging
in an actual client's query log (`query_id, query_text, retrieved_chunk_id`
CSV — see `real_data.py`) and re-running the same clustering + drift checks
on real embeddings instead of made-up sentences.
