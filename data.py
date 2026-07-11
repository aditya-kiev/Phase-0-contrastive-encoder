"""
Phase 0 toy dataset: paraphrase pairs across 5 support-ticket intents.

This mirrors the kind of data you'll actually use later for Idea 2 -- support/RAG
conversation traffic, grouped by underlying intent. Two phrasings of the same
intent = a positive pair. Everything else in a batch = an implicit negative.
"""

import random
import torch

INTENTS = {
    "refund_status": [
        "where is my refund",
        "when will i get my refund",
        "refund has not arrived yet",
        "status of my refund request",
        "i want to know about my refund",
    ],
    "shipping_delay": [
        "my package is late",
        "shipping is delayed",
        "order has not arrived",
        "when will my package arrive",
        "delivery is taking too long",
    ],
    "password_reset": [
        "i forgot my password",
        "how do i reset my password",
        "cannot log into my account",
        "reset password link not working",
        "help me change my password",
    ],
    "billing_question": [
        "why was i charged twice",
        "explain this charge on my bill",
        "invoice seems wrong",
        "billing issue with my account",
        "unexpected charge on my card",
    ],
    "cancel_subscription": [
        "i want to cancel my subscription",
        "how do i cancel my plan",
        "stop my subscription please",
        "cancel my membership",
        "end my subscription now",
    ],
}


def build_vocab():
    words = set()
    for sentences in INTENTS.values():
        for s in sentences:
            words.update(s.split())
    # 0 is reserved for padding
    word2idx = {w: i + 1 for i, w in enumerate(sorted(words))}
    return word2idx


def encode(sentence, word2idx, max_len=8):
    ids = [word2idx[w] for w in sentence.split() if w in word2idx]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return ids


def sample_batch(word2idx, batch_size=5, max_len=8):
    """
    One anchor + one paraphrase (positive) per intent, one intent per slot.
    Other intents in the same batch act as in-batch negatives for InfoNCE.
    """
    intents = random.sample(list(INTENTS.keys()), k=min(batch_size, len(INTENTS)))
    anchors, positives, labels = [], [], []
    for intent in intents:
        a, p = random.sample(INTENTS[intent], 2)
        anchors.append(encode(a, word2idx, max_len))
        positives.append(encode(p, word2idx, max_len))
        labels.append(intent)
    return (
        torch.tensor(anchors, dtype=torch.long),
        torch.tensor(positives, dtype=torch.long),
        labels,
    )
