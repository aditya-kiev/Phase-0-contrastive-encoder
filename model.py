"""
Minimal trainable encoder: word embeddings, masked mean-pool, projection head.

This is deliberately NOT a pretrained transformer. The point of Phase 0 is to
train a real contrastive encoder end-to-end from random initialization, so you
understand what's actually happening in the loss/gradient/embedding-space
mechanics before you swap in a pretrained backbone (e.g. sentence-transformers)
for the real Idea 2 build, where you'll have real production text to fine-tune on.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, proj_dim=16):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size + 1, embed_dim, padding_idx=0)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, token_ids):
        # token_ids: (batch, seq_len), 0 = padding
        mask = (token_ids != 0).unsqueeze(-1).float()      # (batch, seq_len, 1)
        embedded = self.embedding(token_ids)                # (batch, seq_len, embed_dim)
        summed = (embedded * mask).sum(dim=1)                # (batch, embed_dim)
        counts = mask.sum(dim=1).clamp(min=1e-6)             # (batch, 1)
        pooled = summed / counts                             # masked mean pool

        z = self.proj(pooled)
        return F.normalize(z, dim=-1)                        # unit-norm -> dot product = cosine sim
