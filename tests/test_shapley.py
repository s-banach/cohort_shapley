from itertools import combinations

import numpy as np

from cohort_shapley._shapley import kernel_coalitions, shapley_values, zero_sum_basis
from tests.oracle_core import shapley


def test_exact_formula_matches_the_permutation_definition() -> None:
    n_feature_groups = 4
    table = np.random.default_rng(0).normal(size=2**n_feature_groups)
    expected = shapley(lambda s: table[sum(1 << j for j in s)], n_feature_groups)
    np.testing.assert_allclose(shapley_values(table), expected, rtol=0, atol=1e-12)


def test_zero_sum_basis_is_orthonormal_and_sums_to_zero() -> None:
    basis = zero_sum_basis(5)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), rtol=0, atol=1e-12)
    np.testing.assert_allclose(basis.sum(axis=0), 0, rtol=0, atol=1e-12)


def test_kernel_coalitions_follow_the_shapley_kernel() -> None:
    """Each coalition of s of 4 feature groups has probability proportional to 1 / (C(4, s) s (4 - s)): 2 / 22 for each single and each triple, 1 / 22 for each pair."""
    masks = kernel_coalitions(220_000, 4, np.random.default_rng(0))
    codes = masks @ (1 << np.arange(4))
    counts = np.bincount(codes, minlength=16)
    assert counts[0] == counts[15] == 0
    expected = np.zeros(16)
    for size, share in ((1, 2), (2, 1), (3, 2)):
        for feature_groups in combinations(range(4), size):
            expected[sum(1 << j for j in feature_groups)] = 10_000 * share
    assert np.all(np.abs(counts - expected) <= 5 * np.sqrt(expected))
