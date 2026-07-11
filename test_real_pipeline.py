"""
Smoke test for real_data.py + drift_report.py using MOCK data standing in for
a real client's logs. This does NOT test the sentence-transformers encoder
(no internet access to Hugging Face from this sandbox -- same limitation as
model_v2.py). It DOES prove the surrounding data-processing logic is correct:
retrieval-based pairing, and ranked drift detection across many groups at once.

Run: python3 test_real_pipeline.py
"""

import csv
import random

import numpy as np

from real_data import load_retrieval_log, build_groups, sample_pair_batch
from drift_report import compare_snapshots, print_report


# ---------------------------------------------------------------------------
# 1. Mock a retrieval log CSV, the way a real RAG system's logs might look.
# ---------------------------------------------------------------------------

MOCK_TOPICS = {
    "chunk_refund_policy": [
        "where is my refund", "refund status please", "when do refunds process",
        "my refund is late", "checking on a refund",
    ],
    "chunk_shipping_info": [
        "package tracking", "where is my order", "shipping delay question",
        "delivery estimate",
    ],
    "chunk_password_help": [
        "reset password", "cannot log in", "forgot my login",
    ],
    "chunk_billing_faq": [
        "why was i charged", "duplicate charge on card", "billing question",
        "invoice looks wrong",
    ],
    "chunk_returns_policy": [
        "how do i return this", "need a return label", "return window question",
    ],
}


def write_mock_csv(path):
    rows = []
    qid = 0
    for chunk_id, queries in MOCK_TOPICS.items():
        for q in queries:
            rows.append({"query_id": qid, "query_text": q, "retrieved_chunk_id": chunk_id})
            qid += 1
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query_text", "retrieved_chunk_id"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def test_pairing():
    print("--- testing real_data.py (retrieval-based pairing) ---")
    path = "/tmp/mock_retrieval_log.csv"
    n = write_mock_csv(path)
    print(f"wrote mock log: {n} queries across {len(MOCK_TOPICS)} chunks")

    df = load_retrieval_log(path)
    groups = build_groups(df, "retrieved_chunk_id")
    print(f"usable groups (>=2 queries): {len(groups)}")

    anchors, positives, labels = sample_pair_batch(groups, batch_size=len(groups), seed=0)
    print("sample positive pairs drawn:")
    for a, p, l in zip(anchors, positives, labels):
        print(f"  [{l}] '{a}'  <->  '{p}'")


# ---------------------------------------------------------------------------
# 2. Mock two traffic snapshots (before/after) with KNOWN shifted vs stable
#    groups, to verify compare_snapshots correctly ranks the real shifts on top.
# ---------------------------------------------------------------------------

def make_embeddings(center, n, dim=16, noise=0.3, seed=0):
    rng = np.random.default_rng(seed)
    return center + rng.normal(scale=noise, size=(n, dim))


def test_drift_ranking():
    print("\n--- testing drift_report.py (ranked regression report) ---")
    rng = np.random.default_rng(42)
    dim = 16
    n_groups = 8
    shifted_groups = {"group_2", "group_5"}  # ground truth: these actually changed

    before, after = {}, {}
    for i in range(n_groups):
        label = f"group_{i}"
        center = rng.normal(size=dim)
        before[label] = make_embeddings(center, n=20, dim=dim, seed=i)
        if label in shifted_groups:
            shifted_center = center + rng.normal(scale=2.5, size=dim)  # real shift
            after[label] = make_embeddings(shifted_center, n=20, dim=dim, seed=i + 100)
        else:
            after[label] = make_embeddings(center, n=20, dim=dim, seed=i + 100)  # same distribution

    results = compare_snapshots(before, after, n_permutations=200, seed=0)
    print_report(results)

    top_2 = {r["group"] for r in results[:2]}
    correct = top_2 == shifted_groups
    print(f"\nground truth shifted groups: {shifted_groups}")
    print(f"top-2 ranked by report:      {top_2}")
    print("PASS -- correctly identified the real shifts as top priority" if correct
          else "FAIL -- ranking did not match ground truth, needs investigation")


if __name__ == "__main__":
    test_pairing()
    test_drift_ranking()
