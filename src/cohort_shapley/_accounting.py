"""Weight normalization, component accounting and numerical games."""

from dataclasses import dataclass
from math import fsum
from typing import Literal, Self

import numpy as np

from cohort_shapley._arrays import FloatArray1D, IndexArray1D
from cohort_shapley._games import (
    EPS,
    AttributedComponent,
    Game,
    IndependentPairs,
    MatchedPairs,
    defensive_proposal,
)
from cohort_shapley._model import Model

type Component = Literal["matched", "entry", "exit", "weight_change"]
ATTRIBUTED_COMPONENTS: tuple[AttributedComponent, ...] = ("matched", "entry", "exit")


@dataclass(frozen=True)
class Accounting:
    """Exact component totals and their numerical games.

    `total` sums the attributed game totals and weight change.
    """

    total: float
    component_totals: dict[Component, float]
    games: list[Game]


@dataclass(frozen=True)
class WeightedRows:
    """Shared row indices with positive weights."""

    rows: IndexArray1D
    weights: FloatArray1D

    @classmethod
    def from_weights(cls, rows: IndexArray1D, weights: FloatArray1D) -> Self:
        positive = weights > 0
        return cls(rows[positive], weights[positive])


def _share(part: WeightedRows, rest: WeightedRows) -> float:
    """Weight share of `part` in the union of two disjoint populations."""
    part_total = fsum(part.weights)
    return part_total / (part_total + fsum(rest.weights))


def normalised_weights(weights: FloatArray1D, name: str) -> FloatArray1D:
    """Divide non-negative finite weights by their sum; reject zero total weight."""
    total = fsum(weights)
    if total == 0:
        raise ValueError(f"{name} have zero total weight")
    return weights / total


def _measured_means(
    population: WeightedRows, predictions: FloatArray1D, reference: float
) -> tuple[float, float]:
    """Weighted means of predictions minus reference and of their absolute values."""
    total = fsum(population.weights)
    measured = predictions[population.rows] - reference
    mean = fsum(population.weights * measured) / total
    return mean, fsum(population.weights * np.abs(measured)) / total


def independent_game(
    model: Model,
    predictions: FloatArray1D,
    first_population: WeightedRows,
    second_population: WeightedRows,
    *,
    component: AttributedComponent,
    scale: float = 1.0,
) -> Game:
    """Pair independent populations, with total scale * (mean of second_population - mean of first_population).

    Both means are measured from the prediction of the heaviest row of first_population, so constant predictions give exactly zero.
    Correctly rounded sums of weight times prediction differences have rounding error bounded by 8 EPS times scale times the sum of their weighted absolute differences.
    Comparing every positive-weight row with one reference row identifies feature groups that cannot change any pair's prediction.
    """
    reference_row = first_population.rows[np.argmax(first_population.weights)]
    reference = predictions[reference_row]
    first_mean, first_magnitude = _measured_means(first_population, predictions, reference)
    second_mean, second_magnitude = _measured_means(second_population, predictions, reference)
    pairs = IndependentPairs(
        first_population.rows,
        second_population.rows,
        first_population.weights,
        second_population.weights,
    )
    rows = np.concatenate((first_population.rows, second_population.rows))
    _, active = model.changes(np.full(len(rows), reference_row, dtype=np.intp), rows)
    return Game(
        component,
        predictions,
        pairs,
        scale,
        scale * (second_mean - first_mean),
        8 * EPS * scale * (first_magnitude + second_magnitude),
        ~active,
    )


def matched_game(
    model: Model,
    predictions: FloatArray1D,
    first_rows: IndexArray1D,
    second_rows: IndexArray1D,
    weights: FloatArray1D,
) -> Game:
    """Attribute normalized pair weights, retaining unchanged pairs in their normalization.

    Each pair's gain is the prediction of its second row minus that of its first row.
    Pairs whose feature groups cannot change any hybrid prediction have zero gain for every coalition and need no draws.
    The remaining pairs use the defensive proposal, and their total weight supplies the game scale.
    The total's rounding error is bounded by 8 EPS times the weighted sum of absolute gains.
    """
    positive = weights > 0
    first_rows, second_rows, weights = (
        first_rows[positive],
        second_rows[positive],
        weights[positive],
    )
    changed, active = model.changes(first_rows, second_rows)
    first_rows, second_rows, weights = first_rows[changed], second_rows[changed], weights[changed]
    gains = predictions[second_rows] - predictions[first_rows]
    proposal = defensive_proposal(weights, gains) if len(weights) else np.empty(0)
    return Game(
        "matched",
        predictions,
        MatchedPairs(first_rows, second_rows, weights, proposal),
        fsum(weights),
        fsum(weights * gains),
        8 * EPS * fsum(weights * np.abs(gains)),
        ~active,
    )


def cohort_game(model: Model, predictions: FloatArray1D, weights: FloatArray1D, n_1: int) -> Game:
    return independent_game(
        model,
        predictions,
        WeightedRows.from_weights(np.arange(n_1), weights[:n_1]),
        WeightedRows.from_weights(np.arange(n_1, len(weights)), weights[n_1:]),
        component="entry",
    )


def account(
    model: Model,
    predictions: FloatArray1D,
    weights: FloatArray1D,
    *,
    n_1: int,
    rows_1: IndexArray1D,
    rows_2: IndexArray1D,
) -> Accounting:
    """Split the weighted mean change into matched, entry, exit and weight change.

    Matched pair weights average the two cohorts' normalized matched-row weights, omega_1 and omega_2.
    Weight change is sum((omega_2 - omega_1) * ((f_1 - r) + (f_2 - r))) / 2, with r the prediction of the heaviest pair's row of cohort 1.
    Measuring from r gives exactly zero weight change for constant matched predictions.
    Entry compares matched rows of cohort 2 to entrants, scaled by the entrants' share of the weight of cohort 2.
    Exit compares exits to matched rows of cohort 1, scaled by the exits' share of the weight of cohort 1.
    These comparisons preserve precision when entrants or exits dominate a cohort's weight.
    """
    if not len(rows_1):
        return _accounting([cohort_game(model, predictions, weights, n_1)], 0.0)
    omega_1 = normalised_weights(weights[rows_1], "matched rows of cohort 1")
    omega_2 = normalised_weights(weights[rows_2], "matched rows of cohort 2")
    pair_weights = (omega_1 + omega_2) / 2
    reference = predictions[rows_1[np.argmax(pair_weights)]]
    measured = (predictions[rows_1] - reference) + (predictions[rows_2] - reference)
    weight_change = fsum((omega_2 - omega_1) * measured) / 2
    games = [matched_game(model, predictions, rows_1, rows_2, pair_weights)]
    unmatched = weights > 0
    unmatched[rows_1] = unmatched[rows_2] = False
    exits_rows = np.flatnonzero(unmatched[:n_1])
    entrants_rows = np.flatnonzero(unmatched[n_1:]) + n_1
    if len(entrants_rows):
        matched_rows_2 = np.sort(rows_2)
        matched_2 = WeightedRows.from_weights(matched_rows_2, weights[matched_rows_2])
        entrants = WeightedRows.from_weights(entrants_rows, weights[entrants_rows])
        games.append(
            independent_game(
                model,
                predictions,
                matched_2,
                entrants,
                component="entry",
                scale=_share(entrants, matched_2),
            )
        )
    if len(exits_rows):
        matched_rows_1 = np.sort(rows_1)
        matched_1 = WeightedRows.from_weights(matched_rows_1, weights[matched_rows_1])
        exits = WeightedRows.from_weights(exits_rows, weights[exits_rows])
        games.append(
            independent_game(
                model,
                predictions,
                exits,
                matched_1,
                component="exit",
                scale=_share(exits, matched_1),
            )
        )
    return _accounting(games, weight_change)


def _accounting(games: list[Game], weight_change: float) -> Accounting:
    totals: dict[Component, float] = {
        **dict.fromkeys(ATTRIBUTED_COMPONENTS, 0.0),
        **{game.component: game.total for game in games},
        "weight_change": weight_change,
    }
    return Accounting(fsum(totals.values()), totals, games)
