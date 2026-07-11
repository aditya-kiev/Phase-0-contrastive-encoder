"""
Regression report generator -- implements the "cluster and rank divergent
regions" piece from the original doc's Idea 2 architecture (component e):
given two snapshots of embedded production traffic (e.g. the same queries
replayed through model version A vs version B, or before/after some pipeline
change), rank which behavior clusters shifted the most, instead of a single
global drift/no-drift number.

Input: dict[group_label -> np.ndarray of embeddings], one dict per snapshot,
with matching group_label keys (e.g. retrieved_chunk_id, or intent/cluster
label -- whatever grouping you used in real_data.py).
"""

import numpy as np

from eval_drift import permutation_test


def compare_snapshots(embeddings_before, embeddings_after, n_permutations=300, seed=0, min_group_size=2):
    """
    Returns a list of dicts, one per group that had enough samples in BOTH
    snapshots to test, sorted by priority_score (severity * traffic volume)
    descending -- the regressions most worth a human looking at, first.
    """
    shared_groups = set(embeddings_before) & set(embeddings_after)
    skipped = (set(embeddings_before) | set(embeddings_after)) - shared_groups
    if skipped:
        print(f"[drift_report] skipping {len(skipped)} group(s) not present in both snapshots")

    results = []
    for label in shared_groups:
        before = np.asarray(embeddings_before[label])
        after = np.asarray(embeddings_after[label])
        if len(before) < min_group_size or len(after) < min_group_size:
            continue  # not enough samples in this group to run a meaningful test

        stat, p = permutation_test(before, after, n_permutations=n_permutations, seed=seed)
        volume = len(before) + len(after)
        severity = max(stat, 0.0)  # MMD^2 can dip slightly negative under the null; floor it
        results.append({
            "group": label,
            "mmd2": stat,
            "p_value": p,
            "volume": volume,
            "flagged": p < 0.05,
            "priority_score": severity * volume,
        })

    results.sort(key=lambda r: r["priority_score"], reverse=True)
    return results


def print_report(results, top_n=15):
    print(f"{'group':<25s}{'MMD^2':>10s}{'p-value':>10s}{'volume':>8s}{'flagged':>10s}")
    print("-" * 63)
    for r in results[:top_n]:
        flag = "DRIFT" if r["flagged"] else "-"
        print(f"{str(r['group'])[:25]:<25s}{r['mmd2']:>10.4f}{r['p_value']:>10.3f}{r['volume']:>8d}{flag:>10s}")

    n_flagged = sum(r["flagged"] for r in results)
    print(f"\n{n_flagged}/{len(results)} clusters flagged as drifted (p < 0.05)")
