"""NumPy hybrid rows with byte-preserving selection and comparison."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from cohort_shapley._adapter import PredictionArray, column_feature_groups
from cohort_shapley._arrays import BoolArray2D, FloatArray1D, IndexArray1D, IndexArray2D

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type PredictArray[T: np.generic] = Callable[[Array2D[T]], PredictionArray]

FEATURE_ARRAY_NDIM = 2

__all__ = ["Array2D", "NumpyAdapter", "PredictArray", "PredictionArray"]


@dataclass(init=False)
class NumpyAdapter[T: np.generic]:
    """Hold feature rows in shared cohort numbering without copying their storage.

    `values` must have a fixed-width dtype without object references and at least one column.
    C-order sources make whole-row gathering efficient; F-order and strided sources also work.
    Keep the source unchanged while estimating.
    Each column is a feature group unless `feature_group_of_column` groups columns under consecutive indices starting at zero.
    The assignment is copied and made read-only.

    `predict` receives C-order hybrid rows with the source dtype and bytes preserved.
    It must depend only on those rows, return one finite float32 or float64 prediction per row, and leave the rows unchanged.
    Apply fitted transformations and interactions inside `predict` so they use the hybrid features.
    """

    values: Array2D[T]
    predict: PredictArray[T]
    feature_group_of_column: IndexArray1D

    def __init__(
        self,
        values: Array2D[T],
        predict: PredictArray[T],
        *,
        feature_group_of_column: IndexArray1D | None = None,
    ) -> None:
        if (
            values.ndim != FEATURE_ARRAY_NDIM
            or values.dtype.hasobject
            or values.dtype.itemsize == 0
        ):
            raise ValueError("values must be a 2D fixed-width array without object references")
        self.values = values
        self.predict = predict
        self.feature_group_of_column = column_feature_groups(
            values.shape[1], feature_group_of_column
        )

    @property
    def n_feature_groups(self) -> int:
        """Number of groups that can be selected independently in hybrid rows."""
        return int(self.feature_group_of_column.max()) + 1

    def evaluate(self, pairs: IndexArray2D, selections: BoolArray2D) -> FloatArray1D:
        """Select the features of each pair's second row for true feature group selections and of its first row otherwise."""
        storage = self.values.view(np.dtype(f"V{self.values.dtype.itemsize}"))
        hybrid = np.where(
            selections[:, self.feature_group_of_column], storage[pairs[:, 1]], storage[pairs[:, 0]]
        ).view(self.values.dtype)
        return np.asarray(self.predict(hybrid), dtype=np.float64)

    def differs(self, pairs: IndexArray2D) -> BoolArray2D:
        """Compare stored bytes, including signed zero, NaN payloads and dtype padding."""
        storage = self.values.view(np.dtype(f"V{self.values.dtype.itemsize}"))
        changed = storage[pairs[:, 0]] != storage[pairs[:, 1]]
        return np.column_stack(
            [
                changed[:, self.feature_group_of_column == feature_group].any(axis=1)
                for feature_group in range(self.n_feature_groups)
            ]
        )
