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
    batch_rows: int,
    id_column: str | None = None,
    weight_column: str | None = None,
    features: Sequence[str] | None = None,
    tol: float = 1e-3,
    max_evals: int = 50_000_000,
    seed: int | None = 0,
) -> core.Decomposition:
    columns = (
        list(features)
        if features is not None
        else [c for c in d_1.columns if c not in (id_column, weight_column)]
    )
    adapter = PolarsAdapter(pl.concat((d_1.select(columns), d_2.select(columns))), predict)
    pairs = None
    if id_column is not None:
        joined = (
            d_1.select(id_column)
            .with_row_index("rows_1")
            .join(
                d_2.select(id_column).with_row_index("rows_2", offset=d_1.height),
                on=id_column,
                how="inner",
                maintain_order="left",
            )
        )
        pairs = (
            joined["rows_1"].to_numpy().astype(np.intp),
            joined["rows_2"].to_numpy().astype(np.intp),
        )
    weights = (
        None
        if weight_column is None
        else (
            d_1[weight_column].to_numpy().astype(np.float64),
            d_2[weight_column].to_numpy().astype(np.float64),
        )
    )
    return core.decompose_with_turnover(
        adapter,
        n_rows=(d_1.height, d_2.height),
        batch_rows=batch_rows,
        pairs=pairs,
        weights=weights,
        tol=tol,
        max_evals=max_evals,
        seed=seed,
    )
