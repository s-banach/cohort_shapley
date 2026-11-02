from itertools import combinations

import numpy as np

from cohort_shapley._arrays import BoolArray2D, FloatArray1D
from cohort_shapley._estimate import DrawChunk, fit_draws
from cohort_shapley._shapley import shapley_values


def chunk(coalitions: BoolArray2D, contrasts: FloatArray1D, totals: FloatArray1D) -> DrawChunk:
    return DrawChunk(np.packbits(coalitions, axis=1), contrasts, totals)


def test_draws_in_kernel_proportions_give_the_exact_shapley_values() -> None:
    """Over 4 feature groups each single and each triple appears twice and each pair once, the Shapley kernel's proportions, so the paired regression reproduces the exact Shapley values."""
    n = 4
    table = np.random.default_rng(0).normal(size=2**n)
    table[0] = 0.0
    full = 2**n - 1
    codes = [
        sum(1 << j for j in feature_groups)
        for size in range(1, n)
        for feature_groups in combinations(range(n), size)
        for _ in range(2 if size != 2 else 1)
    ]
    coalitions = (np.array(codes)[:, None] >> np.arange(n)) & 1 == 1
    contrasts = np.array([table[c] - table[full ^ c] for c in codes])
    totals = np.full(len(codes), table[full])
    fit = fit_draws([chunk(coalitions, contrasts, totals)], n, table[full], 0.0, block=5)
    np.testing.assert_allclose(fit.contributions, shapley_values(table), rtol=0, atol=1e-12)


def test_feature_groups_never_separated_share_equally_with_infinite_standard_errors() -> None:
    """Coalitions {0, 1} and {2} cannot tell feature groups 0 and 1 apart."""
    coalitions = np.array([[True, True, False], [False, False, True]] * 600)
    contrasts = (2 * coalitions.astype(np.float64) - 1) @ np.array([1.0, 2.0, 3.0])
    fit = fit_draws([chunk(coalitions, contrasts, np.full(1200, 6.0))], 3, 6.0, 0.0, block=1000)
    np.testing.assert_allclose(fit.contributions, [1.5, 1.5, 3.0], rtol=0, atol=1e-12)
    assert np.all(fit.stderr == np.inf)
    assert not fit.checks_pass


def test_totals_differing_by_rounding_leave_the_fit_unadjusted_and_fail_the_checks() -> None:
    """Every sampled total is 1 up to an ulp, while the full-data total is 1.001: the draws missed a contribution, which the control variate must not try to fit."""
    rng = np.random.default_rng(0)
    coalitions = rng.random((2000, 3)) < 0.5
    coalitions[:, 0] ^= coalitions.all(axis=1) | ~coalitions.any(axis=1)
    contrasts = (2 * coalitions.astype(np.float64) - 1) @ np.array([0.2, 0.3, 0.5])
    constant = np.ones(2000)
    rounded = np.nextafter(constant, 2.0 * rng.integers(0, 2, 2000))
    fits = [
        fit_draws([chunk(coalitions, contrasts, totals)], 3, 1.001, 0.0, block=512)
        for totals in (constant, rounded)
    ]
    np.testing.assert_array_equal(fits[0].contributions, fits[1].contributions)
    assert abs(fits[1].contributions.sum() - 1.001) <= 1e-15
    assert not fits[1].checks_pass


def _two_feature_group_draws(n: int, seed: int) -> tuple[BoolArray2D, FloatArray1D]:
    """Coalitions {0} or {1} of 2 feature groups, and their signs d_0 = -d_1 = +1 or -1."""
    signs = np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=n)
    return np.column_stack([signs > 0, signs < 0]), signs


def test_standard_errors_of_contributions_the_totals_nearly_explain_stay_positive() -> None:
    """Contrasts d_k (t_k + 2 e_k) with totals t_k and total 0: each influence is (t_k + 2 e_k) / 2 centered, and the control variate leaves e_k centered minus its projection on t, of order 1e-8 against totals of order 1.
    Subtracting moments of order 1 cancels it.
    """
    n = 1000
    coalitions, signs = _two_feature_group_draws(n, 2)
    rng = np.random.default_rng(4)
    t, e = rng.normal(size=n), rng.normal(size=n) * 1e-8
    fit = fit_draws([chunk(coalitions, signs * (t + 2 * e), t)], 2, 0.0, 0.0, block=256)
    t_centered, e_centered = t - t.mean(), e - e.mean()
    residual = e_centered - (e_centered @ t_centered) / (t_centered @ t_centered) * t_centered
    expected = np.hypot.reduce(residual) / np.sqrt(n * (n - 1))
    np.testing.assert_allclose(fit.stderr, expected, rtol=1e-6, atol=0)
