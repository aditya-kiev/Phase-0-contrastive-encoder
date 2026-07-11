"""
Phase 0 training loop.
Trains ContrastiveEncoder from random initialization on the toy paraphrase
dataset, using in-batch InfoNCE loss.

Run directly: python3 train.py
"""

import torch

from data import build_vocab, sample_batch
from model import ContrastiveEncoder
from losses import info_nce_loss


def train(epochs=300, lr=1e-2, embed_dim=32, proj_dim=16, temperature=0.1, seed=0, verbose=True):
    torch.manual_seed(seed)
    word2idx = build_vocab()
    model = ContrastiveEncoder(len(word2idx), embed_dim, proj_dim)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        anchors, positives, labels = sample_batch(word2idx, batch_size=5)
        z1 = model(anchors)
        z2 = model(positives)
        loss = info_nce_loss(z1, z2, temperature=temperature)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")

    return model, word2idx


if __name__ == "__main__":
    model, word2idx = train()
    torch.save({"model_state": model.state_dict(), "word2idx": word2idx}, "encoder_ckpt.pt")
    print("saved encoder_ckpt.pt")
