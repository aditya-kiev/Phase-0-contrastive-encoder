"""
v2 generalization check: tests held-out sentences with the pretrained encoder.

The pretrained backbone (all-MiniLM-L6-v2) already understands language from
pre-training. This checks whether the projection head's fine-tuning on the toy
dataset preserves or improves intent-level separation for unseen phrasings.

Run: python3 generalization_check_v2.py
"""

import numpy as np
import torch

from data import INTENTS
from train_v2 import train

HELD_OUT = {
    "refund_status": "i want to know refund status",
    "shipping_delay": "package delivery is late",
    "password_reset": "cannot reset my password",
    "billing_question": "billing charge seems wrong",
    "cancel_subscription": "please cancel my plan now",
    "order_tracking": "can you track my shipment",
    "payment_method": "i need to change my card",
    "account_deletion": "erase my account please",
    "product_inquiry": "tell me the specifications",
    "return_exchange": "i need a return label",
    "technical_support": "app is not working",
    "feedback_complaint": "i want to file a complaint",
}


def main():
    model = train(epochs=300, verbose=False)

    embeddings = {}
    for intent, sentence in HELD_OUT.items():
        with torch.no_grad():
            z = model([sentence])
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
            with torch.no_grad():
                z = model(INTENTS[other_intent])
            sim = float((z.numpy() @ held_vec).mean())
            if sim > best_sim:
                best_sim, best_intent = sim, other_intent
        match = "correct" if best_intent == intent else f"WRONG (matched {best_intent})"
        print(f"'{sentence}' -> nearest cluster: {best_intent}  ({match})")
        correct += (best_intent == intent)

    print(f"\n{correct}/{len(HELD_OUT)} held-out sentences matched their correct intent cluster")


if __name__ == "__main__":
    main()
