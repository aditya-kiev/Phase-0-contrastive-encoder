"""
InfoNCE / NT-Xent contrastive loss (SimCLR-style, in-batch negatives).

Given L2-normalized anchor embeddings z1 and matching positive embeddings z2
(z1[i] and z2[i] are a positive pair, same order), every OTHER item in the
batch is treated as an implicit negative for both z1[i] and z2[i].
"""

import torch
import torch.nn.functional as F


def info_nce_loss(z1, z2, temperature=0.1):
    batch_size = z1.shape[0]

    # z1, z2 are L2-normalized -> dot product IS cosine similarity
    logits = z1 @ z2.T / temperature                     # (batch, batch)
    labels = torch.arange(batch_size, device=z1.device)   # correct match = diagonal

    # symmetric: predict z2 from z1, and z1 from z2
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2
