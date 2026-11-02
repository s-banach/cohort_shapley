"""Core contracts and numerical attribution, without dataframe imports."""

import subprocess
import sys
from dataclasses import asdict, dataclass, field
from itertools import product

import numpy as np
import pytest

from cohort_shapley import core
from cohort_shapley._accounting import matched_game
from cohort_shapley._estimate import _draw
from cohort_shapley._model import Model
from cohort_shapley.numpy import NumpyAdapter
from tests.oracle_core import pair_contributions


def predict(values: core.FloatArray2D) -> core.FloatArray1D:
    return values[:, 0] + values[:, 1] + values[:, 0] * values[:, 1]


def test_public_api_runs_without_polars() -> None:
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
sys.modules["polars"] = None
import numpy as np
from cohort_shapley import decompose_with_turnover, NumpyAdapter
adapter = NumpyAdapter(np.array([[0.0, 0.0]]), np.array([[1.0, 2.0]]), lambda rows: rows.sum(axis=1))
decompose_with_turnover(adapter, first_n_matched=1, tol=1e-3)
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@dataclass
class CheckedAdapter:
    adapter: NumpyAdapter[np.float64]
    batch_rows: int
    evaluations: int = 0
    endpoints: list[int] = field(default_factory=list)

    @property
    def n_rows(self) -> tuple[int, int]:
        return self.adapter.n_rows

    @property
    def n_feature_groups(self) -> int:
        return self.adapter.n_feature_groups

    def evaluate(self, pairs: core.IndexArray2D, selections: core.BoolArray2D) -> core.FloatArray1D:
        assert 0 < len(pairs) <= self.batch_rows
        assert not pairs.flags.writeable
        assert not selections.flags.writeable
        self.evaluations += len(pairs)
        self.endpoints.extend(pairs[pairs[:, 0] == pairs[:, 1], 0].tolist())
        return self.adapter.evaluate(pairs, selections)

    def differs(self, pairs: core.IndexArray2D) -> core.BoolArray2D:
        assert 0 < len(pairs) <= self.batch_rows
        assert not pairs.flags.writeable
        return self.adapter.differs(pairs)


@pytest.mark.parametrize("grouped", [False, True])
def test_three_operations_agree_with_pair_oracle(*, grouped: bool) -> None:
    values = np.random.default_rng(21).normal(size=(8, 4))
    values[:, 3] = 0
    feature_groups = [0, 0, 1, 2] if grouped else range(4)
    n_feature_groups = max(feature_groups) + 1
    adapter = NumpyAdapter(
        values[:4],
        values[4:],
        lambda x: x[:, 0] * x[:, 1] + x[:, 2] + x[:, 3],
        feature_group_of_column=feature_groups,
    )
    pair_weights = np.array([1.0, 3.0, 0.0, 0.0])
    matched = core.decompose_paired(
        adapter,
        batch_rows=64,
        weights=pair_weights,
        tol=0,
    )
    expected = pair_contributions(
        adapter.evaluate,
        np.column_stack((np.arange(4), np.arange(4, 8))),
        pair_weights,
        n_feature_groups,
    )
    np.testing.assert_allclose(matched.contributions, expected, atol=1e-14)
    w_1, w_2 = np.array([1.0, 3.0, 2.0, 4.0]), np.array([4.0, 2.0, 3.0, 1.0])
    independent = core.decompose_independent_pairs(
        adapter,
        batch_rows=64,
        weights=(w_1, w_2),
        tol=0,
    )
    independent_pairs = np.array(list(product(range(4), range(4, 8))))
    expected = pair_contributions(
        adapter.evaluate, independent_pairs, np.outer(w_1, w_2).ravel(), n_feature_groups
    )
    np.testing.assert_allclose(independent.contributions, expected, atol=1e-14)
    decomposed = core.decompose_with_turnover(
        adapter,
        batch_rows=64,
        weights=(w_1, w_2),
        first_n_matched=2,
        tol=0,
    )
    expected_matched = pair_contributions(
        adapter.evaluate,
        np.column_stack((np.arange(2), np.arange(4, 6))),
        (w_1[:2] / w_1[:2].sum() + w_2[:2] / w_2[:2].sum()) / 2,
        n_feature_groups,
    )
    entry_pairs = np.array(list(product(range(4, 6), range(6, 8))))
    expected_entry = (
        pair_contributions(
            adapter.evaluate, entry_pairs, np.outer(w_2[:2], w_2[2:]).ravel(), n_feature_groups
        )
        * w_2[2:].sum()
        / w_2.sum()
    )
    exit_pairs = np.array(list(product(range(2, 4), range(2))))
    expected_exit = (
        pair_contributions(
            adapter.evaluate, exit_pairs, np.outer(w_1[2:], w_1[:2]).ravel(), n_feature_groups
        )
        * w_1[2:].sum()
        / w_1.sum()
    )
    contributions = np.column_stack(
        (
            decomposed.matched.contributions,
            decomposed.entry.contributions,
            decomposed.exit.contributions,
        )
    )
    np.testing.assert_allclose(
        contributions,
        np.column_stack((expected_matched, expected_entry, expected_exit)),
        atol=1e-14,
    )
    assert decomposed.total == pytest.approx(
        w_2 @ adapter.predict(values[4:]) / w_2.sum()
        - w_1 @ adapter.predict(values[:4]) / w_1.sum()
    )
    np.testing.assert_array_equal(decomposed.stderr_total, 0)
    np.testing.assert_array_equal(contributions[-1], 0)


@pytest.mark.parametrize("batch_rows", [1, 3, 4, 64])
def test_only_used_endpoints_are_evaluated_and_calls_respect_batch_rows(batch_rows: int) -> None:
    values = np.arange(12.0).reshape(6, 2)
    adapter = CheckedAdapter(NumpyAdapter(values[:3], values[3:], predict), batch_rows)
    result = core.decompose_paired(
        adapter,
        weights=np.array([1.0, 0.0, 0.0]),
        tol=1e-3,
        batch_rows=batch_rows,
        max_evals=100,
    )
    assert adapter.endpoints == [0, 3]
    assert adapter.evaluations == result.n_evals <= 100
    independent_adapter = CheckedAdapter(adapter.adapter, batch_rows)
    independent = core.decompose_independent_pairs(
        independent_adapter,
        weights=(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        tol=1e-3,
        batch_rows=batch_rows,
        max_evals=100,
    )
    assert independent_adapter.endpoints == [0, 4]
    assert independent_adapter.evaluations == independent.n_evals <= 100
    decomposition_adapter = CheckedAdapter(adapter.adapter, batch_rows)
    decomposition = core.decompose_with_turnover(
        decomposition_adapter,
        weights=(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
        first_n_matched=2,
        tol=1e-3,
        batch_rows=batch_rows,
        max_evals=100,
    )
    assert decomposition_adapter.endpoints == [0, 1, 3, 4]
    assert decomposition_adapter.evaluations == decomposition.n_evals <= 100


def test_results_do_not_depend_on_batch_rows() -> None:
    """Enumeration chosen at one batch size is chosen at every batch size, and sampling draws the same pairs and coalitions."""

    def interacting(x: core.FloatArray2D) -> core.FloatArray1D:
        return x.sum(axis=1) + x[:, 0] * x[:, 1] * x[:, 2] + np.sin(x[:, 3] * x[:, 4] * x[:, 5])

    rng = np.random.default_rng(1)
    values = rng.normal(size=(10, 6))
    adapter = NumpyAdapter(values[:5], values[5:], interacting)
    exact = [core.decompose_paired(adapter, tol=1e-2, batch_rows=b) for b in (1, 32, 64)]
    for result in exact:
        assert result.n_evals == 10 + 5 * (2**6 - 2)
        np.testing.assert_array_equal(result.stderr, 0)
        np.testing.assert_allclose(result.contributions, exact[0].contributions, rtol=0, atol=1e-12)
    values = rng.normal(size=(400, 12))
    adapter = NumpyAdapter(values[:200], values[200:], interacting)
    sampled = [
        core.decompose_paired(adapter, tol=0, max_evals=40_000, batch_rows=b) for b in (64, 32768)
    ]
    assert np.all(sampled[0].stderr > 0)
    np.testing.assert_equal(asdict(sampled[0]), asdict(sampled[1]))


def test_overreported_differences_preserve_attributions() -> None:
    @dataclass(frozen=True)
    class ConservativeAdapter:
        adapter: core.Adapter

        @property
        def n_rows(self) -> tuple[int, int]:
            return self.adapter.n_rows

        @property
        def n_feature_groups(self) -> int:
            return self.adapter.n_feature_groups

        def evaluate(
            self, pairs: core.IndexArray2D, selections: core.BoolArray2D
        ) -> core.FloatArray1D:
            return self.adapter.evaluate(pairs, selections)

        def differs(self, pairs: core.IndexArray2D) -> core.BoolArray2D:
            return np.ones((len(pairs), self.n_feature_groups), dtype=bool)

    values = np.array([[0.0, 0.0, 5.0], [1.0, 2.0, 5.0], [0.0, 0.0, 5.0], [3.0, 4.0, 5.0]])
    adapter = NumpyAdapter(values[:2], values[2:], predict)
    expected = pair_contributions(
        adapter.evaluate, np.array([[0, 2], [1, 3]]), np.array([9.0, 1.0]), 3
    )

    result = core.decompose_paired(
        ConservativeAdapter(adapter),
        weights=np.array([9.0, 1.0]),
        batch_rows=16,
        tol=0,
    )
    np.testing.assert_allclose(result.contributions, expected, atol=1e-14)
    np.testing.assert_array_equal(result.stderr, 0)


@pytest.mark.parametrize("first_n_matched", [0, -1, 3, 1.0, True])
def test_invalid_first_n_matched_raises(first_n_matched: int) -> None:
    adapter = NumpyAdapter(np.ones((2, 2)), np.ones((3, 2)), predict)
    with pytest.raises(ValueError, match="first_n_matched"):
        core.decompose_with_turnover(adapter, first_n_matched=first_n_matched, tol=1e-3)


def test_paired_cohorts_require_equal_row_counts() -> None:
    adapter = NumpyAdapter(np.ones((1, 2)), np.ones((2, 2)), predict)
    with pytest.raises(ValueError, match="equal row counts"):
        core.decompose_paired(adapter, tol=1e-3)


@pytest.mark.parametrize("n_feature_groups", [2, 8, 25, 50])
def test_stored_draw_payload_obeys_evaluation_bound(n_feature_groups: int) -> None:
    values = np.random.default_rng(0).normal(size=(6, n_feature_groups))
    adapter = NumpyAdapter(
        values[:3],
        values[3:],
        lambda x: x.sum(axis=1),
        feature_group_of_column=range(n_feature_groups),
    )
    model = Model(adapter, 64)
    predictions = np.asarray(adapter.predict(values), dtype=np.float64)
    game = matched_game(model, predictions, np.arange(3), np.arange(3, 6), np.full(3, 1 / 3))
    max_evals = 1000
    draws = _draw(model, game, (max_evals - 6) // 2, 32, np.random.default_rng(1))
    assert sum(
        array.nbytes for chunk in draws for array in (chunk.packed, chunk.contrasts, chunk.totals)
    ) == (max_evals - 6) // 2 * (16 + (n_feature_groups + 7) // 8)


def test_a_non_finite_hybrid_prediction_raises() -> None:
    adapter = NumpyAdapter(
        np.array([[0.0, 0.0]]),
        np.array([[1.0, 1.0]]),
        lambda rows: np.where(rows[:, 0] != rows[:, 1], np.nan, 1.0),
    )

    with pytest.raises(ValueError, match="non-finite prediction"):
        core.decompose_independent_pairs(adapter, tol=1e-3)


@pytest.mark.parametrize(
    "weights",
    [
        np.array([-1.0, 2.0]),
        np.array([0.0, 0.0]),
        np.array([np.nan, 1.0]),
        np.array([np.inf, 1.0]),
        np.array([1.0]),
    ],
)
def test_invalid_pair_weights_raise(weights: core.FloatArray1D) -> None:
    adapter = NumpyAdapter(np.ones((2, 2)), np.ones((2, 2)), predict)
    with pytest.raises(ValueError, match="weight"):
        core.decompose_paired(adapter, weights=weights, tol=1e-3)


def test_enumeration_counts_previous_sampling_toward_its_cost() -> None:
    values = np.random.default_rng(3).normal(size=(2200, 2))
    adapter = NumpyAdapter(
        values[:1100],
        values[1100:],
        predict,
        feature_group_of_column=range(2),
    )
    pairs = (np.arange(1100), np.arange(1100, 2200))
    pilot = core.decompose_paired(
        adapter,
        tol=0,
        max_evals=4120,
    )
    result = core.decompose_paired(
        adapter,
        tol=0.99 * float(pilot.stderr.max()),
        max_evals=10_000,
    )
    assert result.n_evals == 2200 + 1920 + 2200
    assert result.converged
    np.testing.assert_array_equal(result.stderr, 0)
    np.testing.assert_allclose(
        result.contributions,
        pair_contributions(adapter.evaluate, np.column_stack(pairs), np.ones(1100), 2),
        atol=1e-14,
    )


@pytest.mark.parametrize(
    ("tol", "max_evals", "batch_rows", "message"),
    [
        (-1.0, 100, 8, "tol"),
        (float("nan"), 100, 8, "tol"),
        (1e-3, 0, 8, "max_evals"),
        (1e-3, 3, 8, "used endpoint rows"),
        (1e-3, 100, 0, "batch_rows"),
    ],
)
def test_invalid_estimation_options_raise(
    tol: float, max_evals: int, batch_rows: int, message: str
) -> None:
    adapter = NumpyAdapter(np.ones((2, 2)), np.ones((2, 2)), predict)
    with pytest.raises(ValueError, match=message):
        core.decompose_with_turnover(
            adapter,
            first_n_matched=1,
            batch_rows=batch_rows,
            tol=tol,
            max_evals=max_evals,
        )


@pytest.mark.parametrize(
    "weights",
    [
        (np.zeros(2), np.ones(2)),
        (np.ones(2), np.zeros(2)),
        (np.array([-1.0, 2.0]), np.ones(2)),
        (np.array([np.nan, 1.0]), np.ones(2)),
    ],
)
def test_invalid_cohort_weights_raise(weights: tuple[core.FloatArray1D, core.FloatArray1D]) -> None:
    adapter = NumpyAdapter(np.ones((2, 2)), np.ones((2, 2)), predict)
    with pytest.raises(ValueError, match="weight"):
        core.decompose_with_turnover(adapter, first_n_matched=1, weights=weights, tol=1e-3)


def test_matched_cohort_weights_need_positive_totals() -> None:
    adapter = NumpyAdapter(np.ones((2, 2)), np.ones((2, 2)), predict)
    with pytest.raises(ValueError, match="matched rows of cohort 1 have zero total weight"):
        core.decompose_with_turnover(
            adapter,
            first_n_matched=1,
            weights=(np.array([0.0, 1.0]), np.ones(2)),
            tol=1e-3,
        )
