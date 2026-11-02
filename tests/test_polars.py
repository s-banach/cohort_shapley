"""Polars hybrid values, equality and agreement with the NumPy adapter."""

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
    feature_groups = np.array([0, 0, 1]) if grouped else np.arange(3)
    numpy_adapter = NumpyAdapter(
        values, lambda x: x[:, 0] * x[:, 1] + x[:, 2], feature_group_of_column=feature_groups
    )
    polars_adapter = PolarsAdapter(
        pl.DataFrame(values, schema=["x", "y", "z"]),
        lambda frame: numpy_adapter.predict(frame.to_numpy()),
        feature_group_of_column=feature_groups,
    )
    pairs = (np.arange(5, 15), np.arange(15, 25))
    weights = (np.arange(1.0, 16.0), np.arange(1.0, 16.0))
    results = [
        (
            core.decompose_supplied_pairs(
                adapter,
                n_rows=(15, 15),
                pairs=pairs,
                weights=np.arange(1.0, 11.0),
                batch_rows=16,
                max_evals=1000,
            ),
            core.decompose_independent_pairs(
                adapter,
                n_rows=(15, 15),
                weights=weights,
                batch_rows=16,
                max_evals=1000,
            ),
            core.decompose_with_turnover(
                adapter,
                n_rows=(15, 15),
                pairs=pairs,
                weights=weights,
                batch_rows=16,
                max_evals=1000,
            ),
        )
        for adapter in (numpy_adapter, polars_adapter)
    ]
    for numpy_result, polars_result in zip(*results, strict=True):
        np.testing.assert_array_equal(numpy_result.contributions, polars_result.contributions)
        np.testing.assert_array_equal(numpy_result.stderr, polars_result.stderr)
        assert numpy_result.n_evals == polars_result.n_evals


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

    adapter = PolarsAdapter(frame, predict, feature_group_of_column=np.array([0, 0, 1]))
    assert adapter.frame is frame
    result = adapter.evaluate(np.array([[0, 1], [1, 2]]), np.array([[True, False], [True, False]]))
    assert result.dtype == np.float64
    np.testing.assert_array_equal(result, [1, 2])


def test_unchanged_groups_need_only_endpoint_evaluations() -> None:
    values = np.array([[1.0, 2.0, 3.0], [5.0, 2.0, 3.0]])
    adapters: tuple[core.Adapter, ...] = (
        NumpyAdapter(values, lambda rows: rows.sum(axis=1)),
        PolarsAdapter(pl.DataFrame(values), lambda frame: frame.sum_horizontal().to_numpy()),
    )
    for adapter in adapters:
        supplied = core.decompose_supplied_pairs(
            adapter, n_rows=(1, 1), pairs=(np.array([0]), np.array([1])), batch_rows=8, max_evals=2
        )
        independent = core.decompose_independent_pairs(
            adapter, n_rows=(1, 1), batch_rows=8, max_evals=2
        )
        turnover = core.decompose_with_turnover(
            adapter, n_rows=(1, 1), pairs=(np.array([0]), np.array([1])), batch_rows=8, max_evals=2
        )
        for result in (supplied, independent):
            np.testing.assert_array_equal(result.contributions, [4.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            turnover.contributions, [[4.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        for result in (supplied, independent, turnover):
            np.testing.assert_array_equal(result.stderr, 0)
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
        frame, lambda x: np.zeros(x.height), feature_group_of_column=np.array([0, 1, 1])
    )
    np.testing.assert_array_equal(
        adapter.differs(np.array([[0, 1], [0, 2]])), [[False, False], [True, True]]
    )


def test_object_columns_and_invalid_groups_raise() -> None:
    with pytest.raises(ValueError, match="Object"):
        PolarsAdapter(
            pl.DataFrame({"x": pl.Series([object()], dtype=pl.Object)}),
            lambda x: np.zeros(x.height),
        )
    with pytest.raises(ValueError, match="feature_group_of_column"):
        PolarsAdapter(
            pl.DataFrame({"x": [1.0]}),
            lambda x: np.zeros(x.height),
            feature_group_of_column=np.array([1]),
        )
