# Contrastive Encoder — Phase 0

Learning how contrastive learning actually works, before using it for real: a drift-detection tool that flags when an LLM/RAG pipeline's behavior changes between model versions.

## What's here

A small encoder trained with InfoNCE loss on a toy dataset (support-ticket style sentences grouped by intent), plus an MMD two-sample test to check whether two sets of embeddings come from different distributions. Two versions of the encoder:

- **v1 (`model.py`)** — word embeddings, mean-pooled, projection head. Trained from scratch, no pretrained weights. Bag-of-words — word order doesn't matter. Good enough to learn the loss/embedding mechanics, not good enough for real text.
- **v2 (`train_v2.py`)** — same loss and training shape, but the backbone is a pretrained sentence-transformer (`all-MiniLM-L6-v2`), frozen, with only a small projection head trained on top. Dataset expanded to 12 intents.

## Files

| File | What it does |
|---|---|
| `data.py` | toy dataset (5 intents) + pair sampler for v1 |
| `model.py` | v1 encoder — embeddings, mean pool, projection head |
| `losses.py` | InfoNCE / NT-Xent loss |
| `train.py` / `validate.py` | v1 training + clustering/drift checks |
| `train_v2.py` / `validate_v2.py` | v2 training on 12 intents with pretrained