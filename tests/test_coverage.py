"""Coverage of the reported standard errors over many seeds, against full-data brute force."""

import numpy as np
import pytest

from tests.cohorts import cohorts, features, nonlinear
from tests.oracle import oracle_contributions
from tests.preparation import decompose_frames

SEEDS = range(200)


def _coverage(n_matched: int, n_exits: int, n_entrants: int, n_features: int, tol: float) -> float:
    """Share of errors within 2 total SEs, over feature groups and seeds."""
    d_1, d_2 = cohorts(
        0, n_matched=n_matched, n_exits=n_exits, n_entrants=n_entrants, n_features=n_features
    )
    expected = oracle_contributions(
        nonlinear, d_1, d_2, features(n_features), id_column="id", weight_column="w"
    ).sum(axis=1)
    covered = []
    for seed in SEEDS:
        result = decompose_frames(
            nonlinear,
            d_1,
            d_2,
            batch_rows=8192,
            id_column="id",
            weight_column="w",
            tol=tol,
            seed=seed,
        )
        assert np.all(result.stderr_total > 0)
        covered.append(
            np.abs(result.contributions.sum(axis=1) - expected) <= 2 * result.stderr_total
        )
    return float(np.mean(covered))


@pytest.mark.parametrize(
    ("n_matched", "n_exits", "n_entrants", "n_features", "tol"),
    [
        pytest.param(2000, 100, 100, 4, 0.01, id="three games of 4 feature groups"),
        pytest.param(20000, 0, 0, 10, 0.02, id="one game of 10 feature groups"),
        pytest.param(60, 20, 20, 10, 0.02, id="three small games of 10 feature groups"),
    ],
)
def test_two_standard_errors_cover_about_95_percent(
    n_matched: int, n_exits: int, n_entrants: int, n_features: int, tol: float
) -> None:
    assert 0.9 <= _coverage(n_matched, n_exits, n_entrants, n_features, tol) <= 0.99
