"""Decompose weighted mean prediction changes over pairs of observations.

Each operation takes an `Adapter` that supplies cohort sizes, hybrid predictions, difference detection and the feature group count.
Adapter methods use shared row numbering: rows of cohort 1 precede rows of cohort 2.
`adapter.differs` identifies feature groups that cannot affect a comparison; those groups receive zero contribution and standard error.
See `Adapter` for the prediction and comparison contracts.

Weights must be finite and non-negative, with a positive total wherever a weighted mean needs one.
Each draw selects a pair and a set of feature groups, then evaluates replacements of that set and of the remaining groups.
Sampling continues until every feature group's estimated Monte Carlo standard error, combined across comparisons for `decompose_with_turnover`, is at most `tol`, in prediction units.
Convergence also requires at least 960 draws per sampled comparison, a full-rank regression, and agreement between the sampled and known totals.
Comparisons are enumerated exactly when their size, evaluation cost, and remaining budget permit it.
`stop_reason` is "tol" on convergence and "max_evals" when the remaining allowance cannot fund another step.
`seed` seeds the sampler; None draws fresh entropy.
`max_evals` bounds evaluated rows in total.
Retained draws take at most 16 + ceil(n_feature_groups / 8) bytes each, about 500 MB at the default allowance for 25 feature groups; other allocations add to this.
`max_evals` must cover the endpoint rows and one draw per sampled comparison; otherwise the operation raises ValueError.
`batch_rows` bounds the rows passed to either adapter method in one call.
Standard errors measure Monte Carlo uncertainty given the cohorts and model, not predictive uncertainty.
Standard errors are infinite when a sampled comparison has fewer than two draws or a rank-deficient regression.
A few pairs with large, offsetting contributions can understate standard errors.
"""

from dataclasses import dataclass

import numpy as np

from cohort_shapley._accounting import (
    ATTRIBUTED_COMPONENTS,
    Component,
    account,
    cohort_game,
    matched_game,
    normalised_weights,
)
from cohort_shapley._adapter import Adapter
from cohort_shapley._arrays import (
    BoolArray2D,
    FloatArray1D,
    FloatArray2D,
    IndexArray1D,
    IndexArray2D,
)
from cohort_shapley._estimate import Estimate, StopReason, estimate
from cohort_shapley._games import Game
from cohort_shapley._model import Model

N_COHORTS = 2

__all__ = [
    "Adapter",
    "Attribution",
    "BoolArray2D",
    "Decomposition",
    "FloatArray1D",
    "FloatArray2D",
    "IndexArray1D",
    "IndexArray2D",
    "decompose_independent_pairs",
    "decompose_paired",
    "decompose_with_turnover",
]


@dataclass(frozen=True)
class Attribution:
    """Contributions and Monte Carlo standard errors, one entry per feature group.

    `contributions` sums to `total`, the attributed change in weighted mean prediction.
    `n_evals` counts evaluated rows, endpoint rows included.
    `converged` is true when `stop_reason` is "tol".
    """

    contributions: FloatArray1D
    stderr: FloatArray1D
    total: float
    n_evals: int
    converged: bool
    stop_reason: StopReason


@dataclass(frozen=True)
class Decomposition:
    """Contributions and standard errors with columns matched, entry and exit.

    `stderr_total` combines the independently sampled comparisons' standard errors per feature group.
    `component_totals` sums to `total`, including `weight_change`, which has no feature allocation.
    `n_evals` counts evaluated rows, endpoint rows included.
    `converged` is true when `stop_reason` is "tol".
    """

    contributions: FloatArray2D
    stderr: FloatArray2D
    stderr_total: FloatArray1D
    total: float
    component_totals: dict[Component, float]
    n_evals: int
    converged: bool
    stop_reason: StopReason


def decompose_paired(
    adapter: Adapter,
    *,
    batch_rows: int,
    weights: FloatArray1D | None = None,
    tol: float = 1e-3,
    max_evals: int = 50_000_000,
    seed: int | None = 0,
) -> Attribution:
    """Decompose the weighted mean prediction change between aligned cohort rows.

    The cohorts must have equal lengths; row i of cohort 1 pairs with row i of cohort 2.
    `weights` gives one weight per pair; omitted weights give equal pair weights.
    Endpoint evaluations cover the rows of positive-weight pairs.
    """
    n_rows = adapter.n_rows
    _validate_options(n_rows, adapter.n_feature_groups, batch_rows, tol, max_evals)
    if n_rows[0] != n_rows[1]:
        raise ValueError("paired cohorts must have equal row counts")
    first_rows = np.arange(n_rows[0])
    second_rows = first_rows + n_rows[0]
    w = _weights(weights, len(first_rows), "pairs")
    used = np.concatenate((first_rows[w > 0], second_rows[w > 0]))
    model, predictions = _endpoints(
        adapter,
        batch_rows=batch_rows,
        n_rows=sum(n_rows),
        used=used,
        max_evals=max_evals,
    )
    game = matched_game(model, predictions, first_rows, second_rows, normalised_weights(w, "pairs"))
    fitted = _estimate([game], model, tol, max_evals - len(used), seed)
    return _attribution(game, fitted, len(used))


def decompose_independent_pairs(
    adapter: Adapter,
    *,
    batch_rows: int,
    weights: tuple[FloatArray1D, FloatArray1D] | None = None,
    tol: float = 1e-3,
    max_evals: int = 50_000_000,
    seed: int | None = 0,
) -> Attribution:
    """Decompose the weighted mean prediction of cohort 2 minus that of cohort 1.

    Each combination of a row of cohort 1 with a row of cohort 2 receives the product of their normalized cohort weights.
    `weights` gives one array per cohort; omitted weights give equal observation weights.
    Independence refers to selecting observations from the two cohorts.
    """
    n_rows = adapter.n_rows
    _validate_options(n_rows, adapter.n_feature_groups, batch_rows, tol, max_evals)
    w = _cohort_weights(weights, n_rows)
    used = np.flatnonzero(w > 0)
    model, predictions = _endpoints(
        adapter,
        batch_rows=batch_rows,
        n_rows=sum(n_rows),
        used=used,
        max_evals=max_evals,
    )
    game = cohort_game(model, predictions, w, n_rows[0])
    fitted = _estimate([game], model, tol, max_evals - len(used), seed)
    return _attribution(game, fitted, len(used))


def decompose_with_turnover(
    adapter: Adapter,
    *,
    batch_rows: int,
    weights: tuple[FloatArray1D, FloatArray1D] | None = None,
    pairs: tuple[IndexArray1D, IndexArray1D] | None = None,
    tol: float = 1e-3,
    max_evals: int = 50_000_000,
    seed: int | None = 0,
) -> Decomposition:
    """Split the weighted mean prediction change into matched, entry, exit and weight change.

    Cohort turnover means entry into and exit from the cohort.
    `pairs` contains row indices relative to cohort 1 and cohort 2, respectively.
    Pairs connect unique rows of cohort 1 to unique rows of cohort 2.
    Unpaired rows of cohort 1 are exits, and unpaired rows of cohort 2 are entrants.
    Matched pairs carry the mean of their two rows' weights, each normalized within the matched rows of its cohort.
    Entry compares matched rows of cohort 2 to entrants, scaled by the entrants' share of the weight of cohort 2.
    Exit compares exits to matched rows of cohort 1, scaled by the exits' share of the weight of cohort 1.
    Weight change measures changes in the matched rows' normalized weights and has no attribution to feature groups.
    The four component totals sum to `total`.
    Without pairs, the independent attribution occupies the entry column.
    """
    n_rows = adapter.n_rows
    _validate_options(n_rows, adapter.n_feature_groups, batch_rows, tol, max_evals)
    w = _cohort_weights(weights, n_rows)
    rows_1, rows_2 = _pairs(pairs, n_rows)
    rows_2 = rows_2 + n_rows[0]
    used_mask = w > 0
    positive = used_mask[rows_1] | used_mask[rows_2]
    used_mask[rows_1[positive]] = used_mask[rows_2[positive]] = True
    used = np.flatnonzero(used_mask)
    model, predictions = _endpoints(
        adapter,
        batch_rows=batch_rows,
        n_rows=sum(n_rows),
        used=used,
        max_evals=max_evals,
    )
    accounting = account(model, predictions, w, n_1=n_rows[0], rows_1=rows_1, rows_2=rows_2)
    fitted = _estimate(accounting.games, model, tol, max_evals - len(used), seed)
    contributions = np.zeros((adapter.n_feature_groups, 3))
    stderr = np.zeros_like(contributions)
    for game, fit in zip(accounting.games, fitted.fits, strict=True):
        column = ATTRIBUTED_COMPONENTS.index(game.component)
        contributions[:, column], stderr[:, column] = fit.contributions, fit.stderr
    return Decomposition(
        contributions=contributions,
        stderr=stderr,
        stderr_total=fitted.stderr_total,
        total=accounting.total,
        component_totals=accounting.component_totals,
        n_evals=len(used) + fitted.n_evals,
        converged=fitted.stop_reason == "tol",
        stop_reason=fitted.stop_reason,
    )


def _validate_options(
    n_rows: tuple[int, int], n_feature_groups: int, batch_rows: int, tol: float, max_evals: int
) -> None:
    if any(n < 0 for n in n_rows):
        raise ValueError("n_rows must contain non-negative row counts")
    for name, value in (
        ("n_feature_groups", n_feature_groups),
        ("batch_rows", batch_rows),
        ("max_evals", max_evals),
    ):
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be positive and integral")
    if not (np.isfinite(tol) and tol >= 0):
        raise ValueError("tol must be finite and non-negative")


def _weights(weights: FloatArray1D | None, n: int, name: str) -> FloatArray1D:
    values = np.ones(n) if weights is None else np.asarray(weights, dtype=np.float64)
    if values.shape != (n,):
        raise ValueError(f"{name} weights must have shape ({n},)")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{name} weights must be finite and non-negative")
    if not np.any(values > 0):
        raise ValueError(f"{name} has zero total weight")
    return values


def _cohort_weights(
    weights: tuple[FloatArray1D, FloatArray1D] | None, n_rows: tuple[int, int]
) -> FloatArray1D:
    if weights is not None and len(weights) != N_COHORTS:
        raise ValueError("weights must contain one array per cohort")
    return np.concatenate(
        (
            _weights(None if weights is None else weights[0], n_rows[0], "cohort 1"),
            _weights(None if weights is None else weights[1], n_rows[1], "cohort 2"),
        )
    )


def _pairs(
    pairs: tuple[IndexArray1D, IndexArray1D] | None, n_rows: tuple[int, int]
) -> tuple[IndexArray1D, IndexArray1D]:
    if pairs is None:
        return np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp)
    if len(pairs[0]) != len(pairs[1]):
        raise ValueError("pair arrays must have equal lengths")
    for cohort, (rows, n) in enumerate(zip(pairs, n_rows, strict=True), start=1):
        if np.any(rows < 0) or np.any(rows >= n):
            raise ValueError(f"pair indices must be within cohort {cohort}")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("each row may appear in at most one pair")
    return pairs


def _endpoints(
    adapter: Adapter,
    *,
    batch_rows: int,
    n_rows: int,
    used: IndexArray1D,
    max_evals: int,
) -> tuple[Model, FloatArray1D]:
    if max_evals < len(used):
        raise ValueError(f"max_evals must cover the {len(used)} used endpoint rows")
    model = Model(adapter, batch_rows)
    predictions = np.zeros(n_rows)
    predictions[used] = model.predict(used, used)
    return model, predictions


def _estimate(
    games: list[Game], model: Model, tol: float, budget: int, seed: int | None
) -> Estimate:
    return estimate(games, model, tol=tol, budget=budget, rng=np.random.default_rng(seed))


def _attribution(game: Game, fitted: Estimate, endpoints: int) -> Attribution:
    fit = fitted.fits[0]
    return Attribution(
        fit.contributions,
        fit.stderr,
        game.total,
        endpoints + fitted.n_evals,
        fitted.stop_reason == "tol",
        fitted.stop_reason,
    )
