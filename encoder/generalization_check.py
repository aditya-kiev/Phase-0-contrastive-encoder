"""
Generalization check: the previous validate.py run only proves the encoder
memorized the 25 training sentences (it saw every one during training).
This tests held-out sentences it never saw, built from the same vocabulary,
to see whether the learned word embeddings actually combine sensibly -- or
whether it just memorized exact training pairs.

Run: python -m encoder.generalization_check
"""

import numpy as np
import torch

from data.dataset import INTENTS, build_vocab, encode
from encoder.train import train

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
