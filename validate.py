"""
End-to-end validation, tying together everything above:

1. Train the contrastive encoder on the toy dataset (from random init).
2. Check it actually clusters same-intent paraphrases together in embedding space.
3. Run the MMD drift test on the trained embeddings:
   - two DIFFERENT intents        -> should be flagged as different (low p-value)
   - same intent split in half    -> should NOT be flagged (high p-value)

This is the exact test structure Idea 2's drift detector depends on -- run here
on toy data first, so you see the whole pipeline work before touching real traffic.

Run: python3 validate.py
"""

import numpy as np
import torch

from data import build_vocab, encode, INTENTS
from train import train
from eval_drift import permutation_test


def embed_all(model, word2idx):
    embeddings = {}
    for intent, sentences in INTENTS.items():
        ids = torch.tensor([encode(s, word2idx) for s in sentences], dtype=torch.long)
        with torch.no_grad():
            z = model(ids)
        embeddings[intent] = z.numpy()
    return embeddings


def check_clustering(embeddings):
    print("\n--- clustering check (cosine similarity) ---")
    intents = list(embeddings.keys())
    same_sims, diff_sims = [], []
    for i, a in enumerate(intents):
        Ea = embeddings[a]
        sim_same = Ea @ Ea.T
        iu = np.triu_indices(len(Ea), k=1)
        same_sims.extend(sim_same[iu])
        for b in intents[i + 1:]:
            Eb = embeddings[b]
            diff_sims.extend((Ea @ Eb.T).flatten())
    same_avg, diff_avg = float(np.mean(same_sims)), float(np.mean(diff_sims))
    print(f"avg same-intent cosine sim: {same_avg:.3f}")
    print(f"avg diff-intent cosine sim: {diff_avg:.3f}")
    print(f"separation (same - diff):  {same_avg - diff_avg:.3f}  (want this clearly > 0)")
    return same_avg, diff_avg


def check_mmd(embeddings):
    print("\n--- MMD drift test check ---")
    a = embeddings["refund_status"]
    b = embeddings["shipping_delay"]
    stat, p = permutation_test(a, b, n_permutations=200)
    verdict = "DRIFT DETECTED (correct)" if p < 0.05 else "no drift (MISS)"
    print(f"different intents (refund vs shipping): MMD^2={stat:.4f}  p={p:.3f}  -> {verdict}")

    c = embeddings["refund_status"]
    half1, half2 = c[:2], c[2:4]
    stat2, p2 = permutation_test(half1, half2, n_permutations=200)
    verdict2 = "DRIFT DETECTED (false alarm)" if p2 < 0.05 else "no drift (correct)"
    print(f"same intent, split in half            : MMD^2={stat2:.4f}  p={p2:.3f}  -> {verdict2}")
    print("(note: only 2 vs 2 points in the same-intent split -- small-sample toy check, "
          "not a rigorous power analysis. Real data will have far more samples per cluster.)")


if __name__ == "__main__":
    print("--- training ---")
    model, word2idx = train(epochs=300)
    embeddings = embed_all(model, word2idx)
    check_clustering(embeddings)
    check_mmd(embeddings)
