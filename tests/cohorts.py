"""Synthetic cohorts and models for the tests."""

import numpy as np
import polars as pl

from cohort_shapley._arrays import FloatArray1D


def features(n_features: int) -> list[str]:
    return [f"x{j}" for j in range(n_features)]


def nonlinear(frame: pl.DataFrame) -> FloatArray1D:
    """A log-link model with an interaction between the first two features."""
    x = frame.to_numpy().astype(np.float64)
    coef = 0.3 / (1 + np.arange(x.shape[1]))
    return np.exp(x @ coef + 0.2 * x[:, 0] * x[:, 1])


def cohorts(
    seed: int,
    *,
    n_matched: int,
    n_exits: int,
    n_entrants: int,
    n_features: int,
    weighted: bool = True,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Two cohorts sharing `n_matched` ids; matched units drift between periods."""
    rng = np.random.default_rng(seed)
    names = features(n_features)
    n_1 = n_matched + n_exits
    x_1 = rng.normal(size=(n_1, n_features))
    drift = rng.normal(0.3, 0.5, size=(n_matched, n_features))
    x_2 = np.vstack([x_1[:n_matched] + drift, rng.normal(0.2, 1.0, size=(n_entrants, n_features))])
    ids_1 = np.arange(n_1)
    ids_2 = np.concatenate([np.arange(n_matched), n_1 + np.arange(n_entrants)])
    d_1 = pl.DataFrame({"id": ids_1, **dict(zip(names, x_1.T, strict=True))})
    d_2 = pl.DataFrame({"id": ids_2, **dict(zip(names, x_2.T, strict=True))})
    if weighted:
        d_1 = d_1.with_columns(w=rng.uniform(0.2, 1.0, n_1))
        d_2 = d_2.with_columns(w=rng.uniform(0.2, 1.0, len(ids_2)))
    return d_1, d_2
