"""
Loader + positive-pair builder for REAL production query logs, replacing the
synthetic INTENTS dataset in data.py once you have actual client traffic.

Two supported input formats:

A) Retrieval-based (no manual labeling needed -- the free option):
   CSV with columns: query_id, query_text, retrieved_chunk_id
   Two queries that retrieved the SAME chunk from your RAG system are treated
   as a positive pair -- they represent the same underlying information need,
   even if worded completely differently. If your RAG pipeline logs which
   chunk/document it retrieved per query (most do, or can easily), you
   already have this data with zero extra labeling work.

B) Manually labeled (fallback, ~1hr grouping ~100-200 real queries by hand):
   CSV with columns: query_id, query_text, cluster_label

Either format produces the same output: groups of query strings that belong
together, ready to sample positive pairs from -- the same shape data.py's
INTENTS dict provided for the synthetic dataset.
"""

import random
from collections import defaultdict

import pandas as pd


def load_retrieval_log(path):
    df = pd.read_csv(path)
    required = {"query_id", "query_text", "retrieved_chunk_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in retrieval log: {missing}")
    return df


def load_labeled_log(path):
    df = pd.read_csv(path)
    required = {"query_id", "query_text", "cluster_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in labeled log: {missing}")
    return df


def build_groups(df, group_col):
    """
    Groups query_text by whatever column defines "same underlying need"
    (retrieved_chunk_id, or cluster_label). Drops singleton groups --
    you can't sample a positive pair from a group of one.
    """
    groups = defaultdict(list)
    for _, row in df.iterrows():
        groups[row[group_col]].append(row["query_text"])
    usable = {k: v for k, v in groups.items() if len(v) >= 2}
    dropped = len(groups) - len(usable)
    if dropped:
        print(f"[real_data] dropped {dropped}/{len(groups)} groups with < 2 queries "
              f"(no positive pair possible from a singleton)")
    return usable


def sample_pair_batch(groups, batch_size=16, seed=None):
    """
    One positive pair per group, batch_size distinct groups per call --
    mirrors data.py's sample_batch: other groups in the same batch act as
    in-batch negatives for InfoNCE.
    """
    rng = random.Random(seed)
    group_ids = list(groups.keys())
    if len(group_ids) < batch_size:
        raise ValueError(
            f"Only {len(group_ids)} usable groups (>=2 queries each) available, "
            f"need at least {batch_size} for this batch size. "
            f"Lower batch_size, or collect more grouped real data first."
        )
    chosen = rng.sample(group_ids, batch_size)
    anchors, positives, labels = [], [], []
    for gid in chosen:
        a, p = rng.sample(groups[gid], 2)
        anchors.append(a)
        positives.append(p)
        labels.append(gid)
    return anchors, positives, labels
