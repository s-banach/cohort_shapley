"""NumPy hybrid rows with byte-preserving selection and comparison."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from cohort_shapley._adapter import PredictionArray, column_feature_groups
from cohort_shapley._arrays import BoolArray2D, FloatArray1D, IndexArray1D, IndexArray2D

type Array2D[T: np.generic] = np.ndarray[tuple[int, int], np.dtype[T]]
type PredictArray[T: np.generic] = Callable[[Array2D[T]], PredictionArray]

FEATURE_ARRAY_NDIM = 2

__all__ = ["Array2D", "NumpyAdapter", "PredictArray"]


@dataclass(init=False)
class NumpyAdapter[T: np.generic]:
    """Hold two feature cohorts without copying their storage.

    Both cohorts must have the same fixed-width dtype without object references and the same positive number of columns.
    C-order sources make whole-row gathering efficient; F-order and strided sources also work.
    Keep both sources unchanged while estimating.
    Each column is a feature group unless `feature_group_of_column` groups columns under consecutive indices starting at zero.
    The assignment is copied and made read-only.

    `predict` receives C-order hybrid rows with the source dtype and bytes preserved.
    It must depend only on those rows, return one finite float32 or float64 prediction per row, and leave the rows unchanged.
    Apply fitted transformations and interactions inside `predict` so they use the hybrid features.
    """

    cohort_1: Array2D[T]
    cohort_2: Array2D[T]
    predict: PredictArray[T]
    feature_group_of_column: IndexArray1D
    n_feature_groups: int

    def __init__(
        self,
        cohort_1: Array2D[T],
        cohort_2: Array2D[T],
        predict: PredictArray[T],
        *,
        feature_group_of_column: Sequence[int] | None = None,
    ) -> None:
        for cohort in (cohort_1, cohort_2):
            if (
                cohort.ndim != FEATURE_ARRAY_NDIM
                or cohort.dtype.hasobject
                or cohort.dtype.itemsize == 0
            ):
                raise ValueError("cohorts must be 2D fixed-width arrays without object references")
        if cohort_1.dtype != cohort_2.dtype or cohort_1.shape[1] != cohort_2.shape[1]:
            raise ValueError("cohorts must have matching dtypes and column counts")
        self.cohort_1 = cohort_1
        self.cohort_2 = cohort_2
        self.predict = predict
        self.feature_group_of_column, self.n_feature_groups = column_feature_groups(
            cohort_1.shape[1], feature_group_of_column
        )

    @property
    def n_rows(self) -> tuple[int, int]:
        """Row counts in cohort order."""
        return self.cohort_1.shape[0], self.cohort_2.shape[0]

    def evaluate(self, pairs: IndexArray2D, selections: BoolArray2D) -> FloatArray1D:
        """Select the features of each pair's second row for true feature group selections and of its first row otherwise."""
        hybrid = np.where(
            selections[:, self.feature_group_of_column],
            self._gather(pairs[:, 1]),
            self._gather(pairs[:, 0]),
        ).view(self.cohort_1.dtype)
        return np.asarray(self.predict(hybrid), dtype=np.float64)

    def differs(self, pairs: IndexArray2D) -> BoolArray2D:
        """Compare stored bytes, including signed zero, NaN payloads and dtype padding."""
        changed = self._gather(pairs[:, 0]) != self._gather(pairs[:, 1])
        return np.column_stack(
            [
                changed[:, self.feature_group_of_column == feature_group].any(axis=1)
                for feature_group in range(self.n_feature_groups)
            ]
        )

    def _gather(self, rows: IndexArray1D) -> Array2D[np.void]:
        """Gather stored bytes using cohort 1 followed by cohort 2 row numbering."""
        dtype = np.dtype(f"V{self.cohort_1.dtype.itemsize}")
        gathered = np.empty((len(rows), self.cohort_1.shape[1]), dtype=dtype)
        from_cohort_1 = rows < self.cohort_1.shape[0]
        gathered[from_cohort_1] = self.cohort_1.view(dtype)[rows[from_cohort_1]]
        gathered[~from_cohort_1] = self.cohort_2.view(dtype)[
            rows[~from_cohort_1] - self.cohort_1.shape[0]
        ]
        return gathered
