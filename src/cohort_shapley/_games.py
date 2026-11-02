"""Cohort-level games and their pairs.

A game's gain for a coalition S is its scale times the weighted mean prediction change from each pair's first row to its hybrid.
Measuring each pair's prediction from its own first row makes the gain of a pair whose hybrids all equal its first row exactly zero.
"""

from dataclasses import dataclass

import numpy as np

from cohort_shapley._arrays import BoolArray1D, FloatArray1D, IndexArray1D

EPS = float(np.finfo(np.float64).eps)


@dataclass(frozen=True)
class MatchedPairs:
    """Pair k is (first_rows[k], second_rows[k]) with positive weight weights[k].

    Draws pick pair k with probability proposal[k], which is positive for every pair.
    """

    first_rows: IndexArray1D
    second_rows: IndexArray1D
    weights: FloatArray1D
    proposal: FloatArray1D

    @property
    def n_pairs(self) -> int:
        return len(self.weights)

    @property
    def weight_total(self) -> float:
        return float(self.weights.sum())

    def draw(
        self, n: int, rng: np.random.Generator
    ) -> tuple[IndexArray1D, IndexArray1D, FloatArray1D]:
        """`n` independent draws, with each drawn pair's likelihood ratio: its weight share divided by its proposal probability."""
        k = rng.choice(self.n_pairs, size=n, p=self.proposal)
        ratio = self.weights[k] / (self.weight_total * self.proposal[k])
        return self.first_rows[k], self.second_rows[k], ratio

    def chunk(self, start: int, stop: int) -> tuple[IndexArray1D, IndexArray1D, FloatArray1D]:
        """The first rows, second rows and weights of pairs `start` to `stop`."""
        return self.first_rows[start:stop], self.second_rows[start:stop], self.weights[start:stop]


def defensive_proposal(weights: FloatArray1D, gains: FloatArray1D) -> FloatArray1D:
    """Draw probabilities that put half the mass in proportion to weight and half in proportion to weight times |gain| (Hesterberg 1995, "Weighted average importance sampling and defensive mixture distributions").

    A few pairs with large gains among many with small ones are drawn often, and no pair falls below half its weight share, so no likelihood ratio exceeds 2.
    """
    by_weight = weights / weights.sum()
    by_gain = weights * np.abs(gains)
    if not by_gain.any():
        return by_weight
    return (by_weight + by_gain / by_gain.sum()) / 2


@dataclass(frozen=True)
class IndependentPairs:
    """Every combination of positive-weight rows, with product weights.

    `first_rows` and `second_rows` map weight positions to shared row indices.
    """

    first_rows: IndexArray1D
    second_rows: IndexArray1D
    first_weights: FloatArray1D
    second_weights: FloatArray1D

    @property
    def n_pairs(self) -> int:
        return len(self.second_rows) * len(self.first_rows)

    @property
    def weight_total(self) -> float:
        return float(self.second_weights.sum() * self.first_weights.sum())

    def draw(
        self, n: int, rng: np.random.Generator
    ) -> tuple[IndexArray1D, IndexArray1D, FloatArray1D]:
        """`n` independent draws in proportion to weight, each with likelihood ratio 1."""
        second_positions = rng.choice(
            len(self.second_weights), size=n, p=self.second_weights / self.second_weights.sum()
        )
        first_positions = rng.choice(
            len(self.first_weights),
            size=n,
            p=self.first_weights / self.first_weights.sum(),
        )
        return self.first_rows[first_positions], self.second_rows[second_positions], np.ones(n)

    def chunk(self, start: int, stop: int) -> tuple[IndexArray1D, IndexArray1D, FloatArray1D]:
        """The first rows, second rows and weights of pairs `start` to `stop`, with positions in `first_rows` varying fastest."""
        k = np.arange(start, stop)
        second_positions, first_positions = k // len(self.first_rows), k % len(self.first_rows)
        return (
            self.first_rows[first_positions],
            self.second_rows[second_positions],
            self.first_weights[first_positions] * self.second_weights[second_positions],
        )


@dataclass(frozen=True)
class Game:
    """Pairs and their numerical accounting.

    `predictions` uses shared row indices.
    `total` is the gain of all feature groups, and `total_rounding_bound` bounds its rounding error.
    Exchanging a null feature group within any pair cannot affect a prediction.
    """

    predictions: FloatArray1D
    pairs: MatchedPairs | IndependentPairs
    scale: float
    total: float
    total_rounding_bound: float
    null_feature_groups: BoolArray1D

    @property
    def active_feature_groups(self) -> IndexArray1D:
        return np.flatnonzero(~self.null_feature_groups)
