"""Batched evaluation and comparison using shared row indices."""

from dataclasses import dataclass

import numpy as np

from cohort_shapley._adapter import Adapter
from cohort_shapley._arrays import (
    BoolArray1D,
    BoolArray2D,
    FloatArray1D,
    IndexArray1D,
    IndexArray2D,
)
from cohort_shapley._games import Game


def _pairs_view(
    buffer: IndexArray2D, first_rows: IndexArray1D, second_rows: IndexArray1D
) -> IndexArray2D:
    pairs = buffer[: len(first_rows)].view()
    pairs[:, 0], pairs[:, 1] = first_rows, second_rows
    pairs.flags.writeable = False
    return pairs


@dataclass(frozen=True)
class Model:
    """Evaluate at most `batch_rows` hybrids per call.

    Each prediction or comparison call reuses local buffers across its batches.
    Adapter methods receive read-only views of those buffers.
    """

    adapter: Adapter
    batch_rows: int

    @property
    def n_feature_groups(self) -> int:
        return self.adapter.n_feature_groups

    def predict(
        self, first_rows: IndexArray1D, second_rows: IndexArray1D, masks: BoolArray2D | None = None
    ) -> FloatArray1D:
        result = np.empty(len(first_rows))
        capacity = min(len(first_rows), self.batch_rows)
        buffer = np.empty((capacity, 2), dtype=np.intp)
        selections_buffer = (
            np.zeros((capacity, self.n_feature_groups), dtype=bool) if masks is None else masks
        )
        for start in range(0, len(first_rows), self.batch_rows):
            batch = slice(start, start + self.batch_rows)
            pairs = _pairs_view(buffer, first_rows[batch], second_rows[batch])
            selections = selections_buffer[
                batch if masks is not None else slice(0, len(pairs))
            ].view()
            selections.flags.writeable = False
            predictions = self.adapter.evaluate(pairs, selections)
            if not np.isfinite(predictions).all():
                raise ValueError("evaluate returned a non-finite prediction")
            result[batch] = predictions
        return result

    def changes(
        self, first_rows: IndexArray1D, second_rows: IndexArray1D
    ) -> tuple[BoolArray1D, BoolArray1D]:
        """Which pairs differ, and which feature groups differ within any pair."""
        changed = np.zeros(len(first_rows), dtype=bool)
        active = np.zeros(self.n_feature_groups, dtype=bool)
        buffer = np.empty((min(len(first_rows), self.batch_rows), 2), dtype=np.intp)
        for start in range(0, len(first_rows), self.batch_rows):
            batch = slice(start, start + self.batch_rows)
            pairs = _pairs_view(buffer, first_rows[batch], second_rows[batch])
            differs = self.adapter.differs(pairs)
            changed[batch] = differs.any(axis=1)
            active |= differs.any(axis=0)
        return changed, active


def contrasts(
    model: Model, first_rows: IndexArray1D, second_rows: IndexArray1D, masks: BoolArray2D
) -> FloatArray1D:
    """Prediction differences between each coalition and its complement."""
    return model.predict(first_rows, second_rows, masks) - model.predict(
        first_rows, second_rows, ~masks
    )


def coalition_sums(
    model: Model,
    game: Game,
    first_rows: IndexArray1D,
    second_rows: IndexArray1D,
    *,
    coef: FloatArray1D,
    masks: BoolArray2D,
) -> FloatArray1D:
    """Weighted gains over pairs for every coalition, measured from the prediction of each pair's first row.

    `masks` has one column per active feature group of `game`.
    """
    sums = np.zeros(len(masks))
    per_call = max(1, model.batch_rows // len(first_rows))
    for first in range(0, len(masks), per_call):
        block = masks[first : first + per_call]
        selections = np.zeros((len(block) * len(first_rows), len(game.active)), dtype=bool)
        selections[:, game.active] = np.repeat(block, len(first_rows), axis=0)
        f = model.predict(
            np.tile(first_rows, len(block)),
            np.tile(second_rows, len(block)),
            selections,
        )
        sums[first : first + len(block)] = (
            f.reshape(len(block), len(first_rows)) - game.predictions[first_rows]
        ) @ coef
    return sums
