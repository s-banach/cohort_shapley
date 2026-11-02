"""Shapley-kernel coalitions, the basis of the constrained regression, and the exact Shapley formula."""

from math import comb

import numpy as np

from cohort_shapley._arrays import BoolArray2D, FloatArray1D, FloatArray2D


def kernel_coalitions(n_draws: int, n_feature_groups: int, rng: np.random.Generator) -> BoolArray2D:
    """`n_draws` independent coalitions strictly between no feature groups and all feature groups, from the Shapley kernel: a size s with probability proportional to 1 / (s (n - s)), then s feature groups chosen uniformly by shuffling s True and n - s False entries."""
    sizes = np.arange(1, n_feature_groups)
    size_p = 1.0 / (sizes * (n_feature_groups - sizes))
    drawn = rng.choice(sizes, size=n_draws, p=size_p / size_p.sum())
    return rng.permuted(np.arange(n_feature_groups) < drawn[:, None], axis=1)


def zero_sum_basis(n: int) -> FloatArray2D:
    """An orthonormal basis (n, n - 1) of the vectors whose entries sum to zero: the first n - 1 columns of Q in the QR decomposition of the centering matrix, whose first n - 1 columns span those vectors."""
    q, _ = np.linalg.qr(np.eye(n) - 1 / n)
    return q[:, : n - 1]


def shapley_values(table: FloatArray1D) -> FloatArray1D:
    """contributions_j = sum over S without j of |S|! (n - |S| - 1)! / n! * (v(S + j) - v(S)).

    `table` holds v(S) for every coalition S of n feature groups, indexed by its code, whose bit j is feature group j.
    """
    n = len(table).bit_length() - 1
    codes = np.arange(len(table))
    sizes = np.bitwise_count(codes)
    size_weight = np.array([1 / (n * comb(n - 1, s)) for s in range(n)])
    contributions = np.empty(n)
    for j in range(n):
        without = codes[(codes >> j) & 1 == 0]
        contributions[j] = size_weight[sizes[without]] @ (table[without | 1 << j] - table[without])
    return contributions
