"""
v2 training loop.
Trains PretrainedContrastiveEncoder (sentence-transformer backbone, frozen)
on the toy paraphrase dataset using in-batch InfoNCE loss.

Run directly: python3 train_v2.py
"""

import torch

from data import sample_batch_v2
from model_v2 import PretrainedContrastiveEncoder
from losses import info_nce_loss


def train(epochs=300, lr=1e-2, proj_dim=128, temperature=0.1, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = PretrainedContrastiveEncoder(proj_dim)
    optim = torch.optim.Adam(model.proj.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        anchors, positives, labels = sample_batch_v2()
        z1 = model(anchors)
        z2 = model(positives)
        loss = info_nce_loss(z1, z2, temperature=temperature)

        optim.zero_grad()
        loss.backward()
        optim.step()

        if verbose and (epoch % 50 == 0 or epoch == 1):
            print(f"epoch {epoch:4d}  loss {loss.item():.4f}")

    return model


if __name__ == "__main__":
    model = train()
    torch.save(model.proj.state_dict(), "encoder_v2_ckpt.pt")
    print("saved encoder_v2_ckpt.pt")
