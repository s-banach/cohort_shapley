"""Batched evaluation and comparison of pairs in shared row numbering."""

from dataclasses import dataclass

import numpy as np

from cohort_shapley._adapter import Adapter
from cohort_shapley._arrays import BoolArray1D, BoolArray2D, FloatArray1D, IndexArray2D
from cohort_shapley._games import Game


def _read_only[A: np.ndarray](values: A) -> A:
    view = values.view()
    view.flags.writeable = False
    return view


@dataclass(frozen=True)
class Model:
    """Evaluate at most `batch_rows` hybrids per adapter call.

    Adapter methods receive read-only views of the caller's arrays.
    """

    adapter: Adapter
    batch_rows: int

    @property
    def n_feature_groups(self) -> int:
        return self.adapter.n_feature_groups

    def predict(self, pairs: IndexArray2D, selections: BoolArray2D) -> FloatArray1D:
        """The prediction of each pair's hybrid, which takes the feature groups selected in its row of `selections` from the pair's second row."""
        result = np.empty(len(pairs))
        for start in range(0, len(pairs), self.batch_rows):
            batch = slice(start, start + self.batch_rows)
            result[batch] = self.adapter.evaluate(
                _read_only(pairs[batch]), _read_only(selections[batch])
            )
        if not np.isfinite(result).all():
            raise ValueError("evaluate returned a non-finite prediction")
        return result

    def changes(self, pairs: IndexArray2D) -> tuple[BoolArray1D, BoolArray1D]:
        """Which pairs differ, and which feature groups differ within any pair."""
        changed = np.zeros(len(pairs), dtype=bool)
        active = np.zeros(self.n_feature_groups, dtype=bool)
        for start in range(0, len(pairs), self.batch_rows):
            batch = slice(start, start + self.batch_rows)
            differs = self.adapter.differs(_read_only(pairs[batch]))
            changed[batch] = differs.any(axis=1)
            active |= differs.any(axis=0)
        return changed, active


def contrasts(model: Model, pairs: IndexArray2D, masks: BoolArray2D) -> FloatArray1D:
    """Prediction differences between each coalition and its complement."""
    return model.predict(pairs, masks) - model.predict(pairs, ~masks)


def coalition_sums(
    model: Model,
    game: Game,
    pairs: IndexArray2D,
    *,
    coef: FloatArray1D,
    masks: BoolArray2D,
) -> FloatArray1D:
    """Weighted gains over pairs for every coalition, measured from the prediction of each pair's first row.

    `masks` has one column per active feature group of `game`.
    """
    sums = np.zeros(len(masks))
    per_call = max(1, model.batch_rows // len(pairs))
    for first in range(0, len(masks), per_call):
        block = masks[first : first + per_call]
        selections = np.zeros((len(block) * len(pairs), len(game.active)), dtype=bool)
        selections[:, game.active] = np.repeat(block, len(pairs), axis=0)
        f = model.predict(np.tile(pairs, (len(block), 1)), selections)
        sums[first : first + len(block)] = (
            f.reshape(len(block), len(pairs)) - game.predictions[pairs[:, 0]]
        ) @ coef
    return sums
