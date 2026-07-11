"""
Pretrained contrastive encoder: sentence-transformers backbone + projection head.

Swaps Phase 0's bag-of-words ContrastiveEncoder for a real pretrained sentence
transformer (all-MiniLM-L6-v2). The backbone is frozen (torch.no_grad()) for
this pass — only the linear projection head trains. Unfreezing on tiny datasets
causes catastrophic forgetting, so that comes later on real data with a low LR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer


class PretrainedContrastiveEncoder(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        self.backbone = SentenceTransformer("all-MiniLM-L6-v2")
        self.proj = nn.Linear(384, proj_dim)

    def forward(self, sentences):
        with torch.no_grad():
            base_emb = self.backbone.encode(sentences, convert_to_tensor=True).clone()
        z = self.proj(base_emb)
        return F.normalize(z, dim=-1)
