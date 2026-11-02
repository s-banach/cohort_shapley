"""Prepare the frame fixtures used by the numerical tests."""

from collections.abc import Sequence

import numpy as np
import polars as pl

from cohort_shapley import core
from cohort_shapley.polars import PolarsAdapter, PredictFrame


def decompose_frames(
    predict: PredictFrame,
    d_1: pl.DataFrame,
    d_2: pl.DataFrame,
    *,
    id_column: str,
    weight_column: str | None = None,
    features: Sequence[str] | None = None,
    tol: float = 1e-3,
    max_evals: int = 50_000_000,
    seed: int | None = 0,
) -> core.Decomposition:
    """Decompose with turnover, matching rows with equal ids.

    Ids must be unique within each cohort.
    Matched rows move to the front of both cohorts in the row order of `d_1`.
    """
    position = (
        d_1.select(id_column)
        .join(d_2.select(id_column), on=id_column, how="semi", maintain_order="left")
        .with_row_index("position")
    )
    d_1, d_2 = (
        d.join(position, on=id_column, how="left", maintain_order="left")
        .sort("position", nulls_last=True, maintain_order=True)
        .drop("position")
        for d in (d_1, d_2)
    )
    columns = _feature_columns(d_1, (id_column, weight_column), features)
    return core.decompose_with_turnover(
        PolarsAdapter(d_1.select(columns), d_2.select(columns), predict),
        first_n_matched=position.height,
        weights=_weights(d_1, d_2, weight_column),
        tol=tol,
        max_evals=max_evals,
        seed=seed,
    )


def decompose_frames_independently(
    predict: PredictFrame,
    d_1: pl.DataFrame,
    d_2: pl.DataFrame,
    *,
    weight_column: str | None = None,
    features: Sequence[str] | None = None,
    tol: float = 1e-3,
    max_evals: int = 50_000_000,
) -> core.Attribution:
    columns = _feature_columns(d_1, (weight_column,), features)
    return core.decompose_independent_pairs(
        PolarsAdapter(d_1.select(columns), d_2.select(columns), predict),
        weights=_weights(d_1, d_2, weight_column),
        tol=tol,
        max_evals=max_evals,
    )


def components(result: core.Decomposition) -> tuple[core.Component, core.Component, core.Component]:
    """The matched, entry and exit components, in the order `oracle_contributions` returns them."""
    return result.matched, result.entry, result.exit


def _feature_columns(
    d: pl.DataFrame, excluded: Sequence[str | None], features: Sequence[str] | None
) -> list[str]:
    return list(features) if features is not None else [c for c in d.columns if c not in excluded]


def _weights(
    d_1: pl.DataFrame, d_2: pl.DataFrame, weight_column: str | None
) -> tuple[core.FloatArray1D, core.FloatArray1D] | None:
    if weight_column is None:
        return None
    return (
        d_1[weight_column].to_numpy().astype(np.float64),
        d_2[weight_column].to_numpy().astype(np.float64),
    )
