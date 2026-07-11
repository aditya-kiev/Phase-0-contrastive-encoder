"""
MMD (maximum mean discrepancy) two-sample test in embedding space.
This is the statistical core of Idea 2's drift detector: given two sets of
embeddings, test whether they come from the same distribution.
"""

import numpy as np


def rbf_kernel(X, Y, gamma=None):
    X = np.asarray(X)
    Y = np.asarray(Y)
    if gamma is None:
        # median-distance heuristic for kernel bandwidth
        combined = np.vstack([X, Y])
        dists = np.sum((combined[:, None, :] - combined[None, :, :]) ** 2, axis=-1)
        nonzero = dists[dists > 0]
        median_dist = np.median(nonzero) if nonzero.size else 1.0
        gamma = 1.0 / (median_dist + 1e-8)
    sq_dists = np.sum((X[:, None, :] - Y[None, :, :]) ** 2, axis=-1)
    return np.exp(-gamma * sq_dists)


def mmd2(X, Y, gamma=None):
    X = np.asarray(X)
    Y = np.asarray(Y)
    Kxx = rbf_kernel(X, X, gamma)
    Kyy = rbf_kernel(Y, Y, gamma)
    Kxy = rbf_kernel(X, Y, gamma)
    m, n = len(X), len(Y)
    # unbiased estimator: exclude diagonal for the within-sample terms
    term_xx = (Kxx.sum() - np.trace(Kxx)) / (m * (m - 1))
    term_yy = (Kyy.sum() - np.trace(Kyy)) / (n * (n - 1))
    term_xy = Kxy.sum() / (m * n)
    return term_xx + term_yy - 2 * term_xy


def permutation_test(X, Y, n_permutations=500, seed=0):
    """Returns (observed MMD^2, p-value) via label-permutation null distribution."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X)
    Y = np.asarray(Y)
    observed = mmd2(X, Y)

    combined = np.vstack([X, Y])
    m = len(X)
    n_total = len(combined)

    count = 0
    for _ in range(n_permutations):
        perm = rng.permutation(n_total)
        X_perm = combined[perm[:m]]
        Y_perm = combined[perm[m:]]
        stat = mmd2(X_perm, Y_perm)
        if stat >= observed:
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return observed, p_value
