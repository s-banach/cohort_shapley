"""Brute-force Shapley values of the cohort games, computed from their definitions on the full data."""

from collections.abc import Callable, Sequence

import numpy as np
import polars as pl

from cohort_shapley._arrays import FloatArray1D, FloatArray2D
from cohort_shapley.polars import PredictFrame
from tests.oracle_core import shapley


def pair_game(
    predict: PredictFrame,
    x: pl.DataFrame,
    z: pl.DataFrame,
    weights: FloatArray1D,
    feature_groups: Sequence[str],
    scale: float,
) -> Callable[[frozenset[int]], float]:
    """v(S) = scale * weighted mean of f over rows taking feature groups in S from x and the rest from z."""

    def value(coalition: frozenset[int]) -> float:
        from_x = {feature_groups[j] for j in coalition}
        hybrid = pl.DataFrame([x[c] if c in from_x else z[c] for c in x.columns])
        return scale * float(weights @ predict(hybrid) / weights.sum())

    return value


def independent_pairs(
    x: pl.DataFrame, wx: FloatArray1D, z: pl.DataFrame, wz: FloatArray1D
) -> tuple[pl.DataFrame, pl.DataFrame, FloatArray1D]:
    """All (x, z) combinations with product weights."""
    xi = np.repeat(np.arange(x.height), z.height)
    zi = np.tile(np.arange(z.height), x.height)
    return x[xi], z[zi], np.outer(wx, wz).ravel()


def oracle_contributions(
    predict: PredictFrame,
    d_1: pl.DataFrame,
    d_2: pl.DataFrame,
    feature_groups: Sequence[str],
    *,
    id_column: str | None = None,
    weight_column: str | None = None,
) -> FloatArray2D:
    """Full-data Shapley values of the matched, entry and exit games, with one feature group per column in `feature_groups`; shape (feature groups, 3)."""
    order = [c for c in d_1.columns if c in feature_groups]
    w_1 = d_1[weight_column].cast(pl.Float64).to_numpy() if weight_column else np.ones(d_1.height)
    w_2 = d_2[weight_column].cast(pl.Float64).to_numpy() if weight_column else np.ones(d_2.height)
    x_1, x_2 = d_1.select(order), d_2.select(order)
    n = len(feature_groups)
    result = np.zeros((n, 3))
    if id_column is None:
        x, z, w = independent_pairs(x_2, w_2, x_1, w_1)
        result[:, 1] = shapley(pair_game(predict, x, z, w, feature_groups, 1.0), n)
        return result
    ids_1 = d_1[id_column].to_list()
    ids_2 = d_2[id_column].to_list()
    position_2 = {i: k for k, i in enumerate(ids_2) if i is not None}
    rows_1 = np.array([k for k, i in enumerate(ids_1) if i in position_2], dtype=np.int64)
    rows_2 = np.array([position_2[ids_1[k]] for k in rows_1], dtype=np.int64)
    pair_weights = (w_1[rows_1] / w_1[rows_1].sum() + w_2[rows_2] / w_2[rows_2].sum()) / 2
    matched = pair_game(predict, x_2[rows_2], x_1[rows_1], pair_weights, feature_groups, 1.0)
    result[:, 0] = shapley(matched, n)
    entrants = np.setdiff1d(np.arange(d_2.height), rows_2)
    exits = np.setdiff1d(np.arange(d_1.height), rows_1)
    if w_2[entrants].sum() > 0:
        x, z, w = independent_pairs(x_2[entrants], w_2[entrants], x_2, w_2)
        scale = w_2[entrants].sum() / w_2[rows_2].sum()
        result[:, 1] = shapley(pair_game(predict, x, z, w, feature_groups, scale), n)
    if w_1[exits].sum() > 0:
        x, z, w = independent_pairs(x_1, w_1, x_1[exits], w_1[exits])
        scale = w_1[exits].sum() / w_1[rows_1].sum()
        result[:, 2] = shapley(pair_game(predict, x, z, w, feature_groups, scale), n)
    return result
