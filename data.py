"""
Phase 0 toy dataset: paraphrase pairs across support-ticket intents.

Phase 0 built a vocab and tokenized manually. Phase 1 (model_v2) passes raw
sentence strings — the pretrained backbone handles tokenization internally.
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
        "has my refund been processed",
        "why is my refund taking so long",
        "i need an update on my refund",
        "refund status check please",
        "when was my refund issued",
        "my refund is overdue",
        "can you track my refund",
    ],
    "shipping_delay": [
        "my package is late",
        "shipping is delayed",
        "order has not arrived",
        "when will my package arrive",
        "delivery is taking too long",
        "where is my order",
        "my shipment is stuck",
        "expected delivery date was yesterday",
        "package hasn't moved in days",
        "shipping status shows no update",
        "my order should have arrived by now",
        "delivery is past the promised date",
    ],
    "password_reset": [
        "i forgot my password",
        "how do i reset my password",
        "cannot log into my account",
        "reset password link not working",
        "help me change my password",
        "i need a new password",
        "password reset email never arrived",
        "my account is locked",
        "forgot my login credentials",
        "reset link expired",
        "can i get a password reset code",
        "trouble signing in",
    ],
    "billing_question": [
        "why was i charged twice",
        "explain this charge on my bill",
        "invoice seems wrong",
        "billing issue with my account",
        "unexpected charge on my card",
        "i was overcharged",
        "this charge is not mine",
        "billing statement has an error",
        "duplicate payment on my account",
        "please correct my invoice",
        "charge amount does not match",
        "refund not reflected in billing",
    ],
    "cancel_subscription": [
        "i want to cancel my subscription",
        "how do i cancel my plan",
        "stop my subscription please",
        "cancel my membership",
        "end my subscription now",
        "i no longer need this service",
        "turn off auto renewal",
        "please terminate my account",
        "unsubscribe me from this plan",
        "i want to stop the payments",
        "cancel before next billing cycle",
        "deactivate my subscription",
    ],
    "order_tracking": [
        "track my order",
        "where is my package now",
        "order status shows in transit",
        "latest tracking update please",
        "when will my order be delivered",
        "my tracking number is not working",
        "package location unknown",
        "carrier has not scanned my package",
        "what is the estimated delivery",
        "tracking has not updated in a week",
        "is my order still on schedule",
        "delivery window was missed",
    ],
    "payment_method": [
        "update my payment method",
        "add a new credit card",
        "change my billing address",
        "my card was declined",
        "why was my payment rejected",
        "can i pay with paypal",
        "switch payment method",
        "remove my old card on file",
        "payment failed try another card",
        "update my card expiration date",
        "i want to use a different card",
        "billing address does not match",
    ],
    "account_deletion": [
        "delete my account",
        "i want to close my account",
        "remove my personal data",
        "how do i delete my profile",
        "permanently delete my information",
        "i want my data erased",
        "close my account immediately",
        "request account termination",
        "delete all my records",
        "how to deactivate my account",
        "i no longer want an account",
        "erase my account and data",
    ],
    "product_inquiry": [
        "tell me about this product",
        "what are the specifications",
        "is this item in stock",
        "product dimensions please",
        "what colors are available",
        "how long is the warranty",
        "does this come with accessories",
        "compatibility with other devices",
        "what is the return policy",
        "are there any discounts available",
        "compare this model to the older one",
        "customer reviews for this product",
    ],
    "return_exchange": [
        "i want to return this item",
        "how do i start a return",
        "my product arrived damaged",
        "exchange for a different size",
        "return shipping label please",
        "where do i send the return",
        "refund for returned item not received",
        "return window has passed",
        "can i exchange instead of refund",
        "item does not match description",
        "return request was denied",
        "how long do returns take",
    ],
    "technical_support": [
        "the website is not loading",
        "app keeps crashing",
        "error code 500 on checkout",
        "my account was hacked",
        "two factor authentication not working",
        "page shows blank screen",
        "payment gateway timeout error",
        "can't upload my documents",
        "mobile app is freezing",
        "browser compatibility issue",
        "session keeps expiring",
        "download link is broken",
    ],
    "feedback_complaint": [
        "i want to speak to a manager",
        "this is terrible service",
        "i am very disappointed",
        "your support team was unhelpful",
        "file a complaint about my experience",
        "i want to give feedback",
        "please escalate my issue",
        "worst customer service ever",
        "i will take my business elsewhere",
        "nobody responded to my email",
        "your policy is unfair",
        "i demand a resolution",
    ],
}


def sample_batch_v2(batch_size=10):
    """
    Return raw sentence strings — one anchor + one positive per intent.
    Other intents in the batch act as in-batch negatives for InfoNCE.
    """
    intents = random.sample(list(INTENTS.keys()), k=min(batch_size, len(INTENTS)))
    anchors, positives, labels = [], [], []
    for intent in intents:
        a, p = random.sample(INTENTS[intent], 2)
        anchors.append(a)
        positives.append(p)
        labels.append(intent)
    return anchors, positives, labels


# --- Phase 0 legacy helpers (kept for backward compat, not used by Phase 1) ---

def build_vocab():
    words = set()
    for sentences in INTENTS.values():
        for s in sentences:
            words.update(s.split())
    word2idx = {w: i + 1 for i, w in enumerate(sorted(words))}
    return word2idx


def encode(sentence, word2idx, max_len=8):
    ids = [word2idx[w] for w in sentence.split() if w in word2idx]
    ids = ids[:max_len] + [0] * (max_len - len(ids))
    return ids


def sample_batch(word2idx, batch_size=5, max_len=8):
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
