"""
Generalization check: the previous validate.py run only proves the encoder
memorized the 25 training sentences (it saw every one during training).
This tests held-out sentences it never saw, built from the same vocabulary,
to see whether the learned word embeddings actually combine sensibly -- or
whether it just memorized exact training pairs.

Also a deliberate architectural note: this encoder mean-pools word embeddings,
so it's a bag-of-words model -- word ORDER carries no information. That's a
real limitation vs. a pretrained sentence encoder (which you'll swap in for
the real Idea 2 build). This script tells you honestly how much that costs you.

Run: python3 generalization_check.py
"""

import numpy as np
import torch

from data import build_vocab, encode
from train import train

HELD_OUT = {
    "refund_status": "i want to know refund status",
    "shipping_delay": "package delivery is late",
    "password_reset": "cannot reset my password",
    "billing_question": "billing charge seems wrong",
    "cancel_subscription": "please cancel my plan now",
}


def main():
    model, word2idx = train(epochs=300, verbose=False)

    embeddings = {}
    for intent, sentence in HELD_OUT.items():
        ids = torch.tensor([encode(sentence, word2idx)], dtype=torch.long)
        with torch.no_grad():
            z = model(ids)
        embeddings[intent] = z.numpy()[0]

    intents = list(embeddings.keys())
    print("--- held-out sentence similarity matrix (never seen verbatim in training) ---")
    print(f"{'':20s}" + "".join(f"{i[:10]:>12s}" for i in intents))
    for a in intents:
        row = [float(embeddings[a] @ embeddings[b]) for b in intents]
        print(f"{a:20s}" + "".join(f"{v:12.3f}" for v in row))

    # does each held-out sentence match its OWN intent's trained cluster best?
    from data import INTENTS
    correct = 0
    for intent, sentence in HELD_OUT.items():
        held_vec = embeddings[intent]
        best_intent, best_sim = None, -2
        for other_intent in INTENTS:
            ids = torch.tensor([encode(s, word2idx) for s in INTENTS[other_intent]], dtype=torch.long)
            with torch.no_grad():
                z = model(ids)
            sim = float((z.numpy() @ held_vec).mean())
            if sim > best_sim:
                best_sim, best_intent = sim, other_intent
        match = "correct" if best_intent == intent else f"WRONG (matched {best_intent})"
        print(f"'{sentence}' -> nearest cluster: {best_intent}  ({match})")
        correct += (best_intent == intent)

    print(f"\n{correct}/{len(HELD_OUT)} held-out sentences matched their correct intent cluster")


if __name__ == "__main__":
    main()
