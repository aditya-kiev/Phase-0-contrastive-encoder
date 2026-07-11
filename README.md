# Phase 0 — Contrastive Encoder, Built and Verified

This is the thing the deep-dive doc assumes you already had. You didn't, so we built
it from scratch and actually ran it — every number below is a real output from this
code, not a claim.

## What's in here

| File | What it does |
|---|---|
| `data.py` | Toy dataset: 5 support-ticket intents, 5 paraphrases each. Pair sampler for in-batch negatives. |
| `model.py` | `ContrastiveEncoder` — word embeddings, masked mean-pool, projection head, L2-normalized output. Trained from random init, not pretrained. |
| `losses.py` | `info_nce_loss` — real NT-Xent/InfoNCE, symmetric, in-batch negatives. This is the loss both Idea 1 and Idea 2 depend on. |
| `train.py` | Training loop. `python3 train.py` |
| `eval_drift.py` | `mmd2` + `permutation_test` — the MMD two-sample test that's the statistical core of Idea 2's drift detector. |
| `validate.py` | End-to-end: train, check clustering, run the MMD test on trained embeddings. |
| `generalization_check.py` | Tests 5 held-out sentences (never seen in training) to check the encoder learned something real, not just memorized. |

## Verified results (actual run, this session)

**Training** converged: loss went from 2.14 to 0.0002 over 300 steps.

**Clustering** (`validate.py`):
- avg same-intent cosine similarity: **0.952**
- avg different-intent cosine similarity: **-0.233**
- Same-intent paraphrases sit close together in embedding space; different intents sit far apart. This is the entire point of a contrastive encoder.

**MMD drift test** (`validate.py`):
- Different intents (refund vs. shipping): MMD²=0.110, **p=0.015 → correctly flagged as drift**
- Same intent split in half: MMD²=-0.076, **p=1.000 → correctly NOT flagged**
- (Caveat: the "same intent" split is only 2 vs 2 points — a toy sanity check, not a real power analysis. Real traffic will have far more samples per cluster.)

**Generalization** (`generalization_check.py`):
- 5 held-out sentences, built from the same vocab but never seen verbatim in training, all matched their correct intent cluster (5/5).
- This means the word embeddings learned real structure, not just memorized the 25 training strings.

## The honest limitation

`ContrastiveEncoder` mean-pools word embeddings — it's a **bag-of-words model**.
Word order carries zero information ("cancel my plan" and "plan my cancel" embed
identically). That's fine for Phase 0 — the goal was to understand InfoNCE loss and
embedding-space mechanics, which you now have working code and real numbers for.

It is NOT fine for the real Idea 2 build. Real support/RAG conversations need
actual sequence understanding. The fix is to swap `model.py`'s encoder for a
pretrained sentence encoder (e.g. `sentence-transformers/all-MiniLM-L6-v2`) and
fine-tune its output with the same `info_nce_loss` you already have. The loss
function, training loop shape, and MMD test in this repo carry over unchanged —
only the encoder's forward pass changes.

I couldn't install/test that swap in this sandbox (no access to Hugging Face's
model hub from here — only PyPI/GitHub are reachable). On your own machine or
Colab, it's:

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer
import torch.nn as nn

class PretrainedContrastiveEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = SentenceTransformer("all-MiniLM-L6-v2")
        self.proj = nn.Linear(384, proj_dim)  # MiniLM's native dim is 384

    def forward(self, sentences):  # list[str], not token ids
        with torch.no_grad():  # freeze backbone at first; unfreeze later to fine-tune
            base_emb = self.backbone.encode(sentences, convert_to_tensor=True)
        z = self.proj(base_emb)
        return nn.functional.normalize(z, dim=-1)
```

Then feed real anchor/positive pairs from your own client conversation logs
(two user messages resolved by the same underlying intent = positive pair)
through the exact same `info_nce_loss` and training loop shape you already have here.

## Run order

```bash
pip install torch numpy
python3 validate.py               # train + clustering check + MMD check
python3 generalization_check.py   # held-out generalization check
```

## Next steps toward the real Idea 2 track

1. Swap in the pretrained backbone above.
2. Replace the toy dataset with real (anonymized) client conversation pairs —
   your freelance RAG clients are the data source the original doc pointed at.
3. Re-run `validate.py`'s structure (clustering check + MMD check) on real data.
4. Only once that works on real traffic does the 16-week Idea 2 roadmap
   (OTel instrumentation, paired replay harness, dashboard) start making sense.
