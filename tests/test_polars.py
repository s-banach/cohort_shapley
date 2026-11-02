"""Polars hybrid values, equality and agreement with the NumPy adapter."""

from dataclasses import asdict

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from cohort_shapley import core
from cohort_shapley.numpy import NumpyAdapter
from cohort_shapley.polars import PolarsAdapter


@pytest.mark.parametrize("grouped", [False, True])
def test_adapters_agree_for_all_operations(*, grouped: bool) -> None:
    values = np.random.default_rng(7).normal(size=(30, 3))
    values[:, 2] = 1
    feature_groups = (0, 0, 1) if grouped else range(3)
    numpy_adapter = NumpyAdapter(
        values[:15],
        values[15:],
        lambda x: x[:, 0] * x[:, 1] + x[:, 2],
        feature_group_of_column=feature_groups,
    )
    polars_adapter = PolarsAdapter(
        pl.DataFrame(values[:15], schema=["x", "y", "z"]),
        pl.DataFrame(values[15:], schema=["x", "y", "z"]),
        lambda frame: numpy_adapter.predict(frame.to_numpy()),
        feature_group_of_column=feature_groups,
    )
    weights = (np.arange(1.0, 16.0), np.arange(1.0, 16.0))
    results = [
        (
            core.decompose_paired(
                adapter,
                weights=np.arange(1.0, 16.0),
                tol=1e-3,
                batch_rows=16,
                max_evals=1000,
            ),
            core.decompose_independent_pairs(
                adapter,
                weights=weights,
                tol=1e-3,
                batch_rows=16,
                max_evals=1000,
            ),
            core.decompose_with_turnover(
                adapter,
                first_n_matched=10,
                weights=weights,
                tol=1e-3,
                batch_rows=16,
                max_evals=1000,
            ),
        )
        for adapter in (numpy_adapter, polars_adapter)
    ]
    for numpy_result, polars_result in zip(*results, strict=True):
        np.testing.assert_equal(asdict(numpy_result), asdict(polars_result))


def test_mixed_hybrids_preserve_schema_nulls_and_feature_groups() -> None:
    frame = pl.DataFrame(
        {
            "number": [0.0, float("nan"), 2.0],
            "category": pl.Series(["a", None, "b"], dtype=pl.Enum(["a", "b"])),
            "integer": pl.Series([1, 2, None], dtype=pl.Int32),
        }
    )
    expected = pl.DataFrame(
        [
            frame["number"].gather([1, 2]),
            frame["category"].gather([1, 2]),
            frame["integer"].gather([0, 1]),
        ]
    )

    def predict(hybrid: pl.DataFrame) -> np.ndarray:
        assert_frame_equal(hybrid, expected)
        return np.array([1, 2], dtype=np.float32)

    cohort_1, cohort_2 = frame[:1], frame[1:]
    adapter = PolarsAdapter(cohort_1, cohort_2, predict, feature_group_of_column=[0, 0, 1])
    assert adapter.cohort_1 is cohort_1
    assert adapter.cohort_2 is cohort_2
    assert adapter.n_rows == (1, 2)
    result = adapter.evaluate(np.array([[0, 1], [1, 2]]), np.array([[True, False], [True, False]]))
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1, 2])


def test_unchanged_groups_need_only_endpoint_evaluations() -> None:
    values = np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]])
    adapters: tuple[core.Adapter, ...] = (
        NumpyAdapter(values[:1], values[1:], lambda rows: rows.sum(axis=1)),
        PolarsAdapter(
            pl.DataFrame(values[:1]),
            pl.DataFrame(values[1:]),
            lambda frame: frame.sum_horizontal().to_numpy(),
        ),
    )
    for adapter in adapters:
        paired = core.decompose_paired(adapter, tol=0, max_evals=2)
        independent = core.decompose_independent_pairs(adapter, tol=0, max_evals=2)
        turnover = core.decompose_with_turnover(adapter, first_n_matched=1, tol=0, max_evals=2)
        for attribution in (paired, independent, turnover.matched):
            np.testing.assert_array_equal(attribution.contributions, [4.0, 0.0, 0.0])
            np.testing.assert_array_equal(attribution.stderr, 0)
        for component in (turnover.entry, turnover.exit):
            np.testing.assert_array_equal(component.contributions, 0)
        for result in (paired, independent, turnover):
            assert result.n_evals == 2
            assert result.converged


def test_polars_equality_and_grouping() -> None:
    frame = pl.DataFrame(
        {
            "zero": [-0.0, 0.0, 1.0],
            "nan": [float("nan"), float("nan"), 0.0],
            "null": [None, None, "a"],
        }
    )
    adapter = PolarsAdapter(
        frame[:2], frame[2:], lambda x: np.zeros(x.height), feature_group_of_column=(0, 1, 1)
    )
    np.testing.assert_array_equal(
        adapter.differs(np.array([[0, 1], [0, 2]])), [[False, False], [True, True]]
    )


def test_object_columns_and_invalid_groups_raise() -> None:
    with pytest.raises(ValueError, match="Object"):
        PolarsAdapter(
            pl.DataFrame({"x": pl.Series([object()], dtype=pl.Object)}),
            pl.DataFrame({"x": [1.0]}),
            lambda x: np.zeros(x.height),
        )
    with pytest.raises(ValueError, match="feature_group_of_column"):
        PolarsAdapter(
            pl.DataFrame({"x": [1.0]}),
            pl.DataFrame({"x": [1.0]}),
            lambda x: np.zeros(x.height),
            feature_group_of_column=[1],
        )


@pytest.mark.parametrize(
    "cohort_2",
    [
        pl.DataFrame({"y": [1.0], "x": [2.0]}),
        pl.DataFrame({"x": [1], "y": [2.0]}),
        pl.DataFrame({"x": [1.0], "z": [2.0]}),
        pl.DataFrame({"x": [1.0]}),
    ],
)
def test_mismatched_schemas_raise(cohort_2: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="matching ordered schemas"):
        PolarsAdapter(
            pl.DataFrame({"x": [1.0], "y": [2.0]}), cohort_2, lambda x: np.zeros(x.height)
        )


def test_source_changes_and_same_cohort_gathers() -> None:
    cohort_1 = pl.DataFrame({"x": [0.0, 0.0]})
    cohort_2 = pl.DataFrame({"x": [0.0, 0.0]})
    adapter = PolarsAdapter(cohort_1, cohort_2, lambda x: x["x"].to_numpy())
    cohort_1[1, "x"] = 2.0
    cohort_2[0, "x"] = 3.0
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(adapter.differs(pairs), [[True], [True], [True]])
    np.testing.assert_array_equal(
        adapter.evaluate(pairs, np.array([[True], [True], [False]])), [2, 3, 3]
    )
