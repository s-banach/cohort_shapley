"""Brute-force pair attributions from permutation-weighted marginal contributions."""

from collections.abc import Callable
from itertools import combinations
from math import factorial

import numpy as np

from cohort_shapley.core import BoolArray2D, FloatArray1D, IndexArray2D


def shapley(value: Callable[[frozenset[int]], float], n_feature_groups: int) -> FloatArray1D:
    """Exact Shapley values from the permutation-weighted marginal contributions over all subsets."""
    feature_groups = range(n_feature_groups)
    v = {
        frozenset(s): value(frozenset(s))
        for size in range(n_feature_groups + 1)
        for s in combinations(feature_groups, size)
    }
    contributions = np.zeros(n_feature_groups)
    for j in feature_groups:
        for s, v_s in v.items():
            if j not in s:
                weight = (
                    factorial(len(s))
                    * factorial(n_feature_groups - len(s) - 1)
                    / factorial(n_feature_groups)
                )
                contributions[j] += weight * (v[s | {j}] - v_s)
    return contributions


def pair_contributions(
    evaluate: Callable[[IndexArray2D, BoolArray2D], FloatArray1D],
    pairs: IndexArray2D,
    weights: FloatArray1D,
    n_feature_groups: int,
) -> FloatArray1D:
    def value(coalition: frozenset[int]) -> float:
        selections = np.tile(
            np.array([j in coalition for j in range(n_feature_groups)]), (len(pairs), 1)
        )
        return float(weights @ evaluate(pairs, selections) / weights.sum())

    return shapley(value, n_feature_groups)
