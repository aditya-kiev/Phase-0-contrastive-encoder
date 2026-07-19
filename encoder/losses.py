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
    logits = z1 @ z2.T / temperature
    labels = torch.arange(batch_size, device=z1.device)
    loss_a = F.cross_entropy(logits, labels)
    loss_b = F.cross_entropy(logits.T, labels)
    return (loss_a + loss_b) / 2
