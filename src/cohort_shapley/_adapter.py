"""Adapter protocol, prediction types and column-to-feature group assignments."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np

from cohort_shapley._arrays import BoolArray2D, FloatArray1D, IndexArray1D, IndexArray2D

type PredictionArray = np.ndarray[tuple[int], np.dtype[np.float32]] | FloatArray1D


class Adapter(Protocol):
    """Construct and compare hybrid observations in shared cohort row numbering.

    Rows of cohort 1 precede rows of cohort 2.
    Methods receive read-only arrays valid only during the call.
    Keep the observations and feature groups unchanged throughout decomposition.
    """

    @property
    def n_rows(self) -> tuple[int, int]:
        """Non-negative row counts for cohort 1 and cohort 2, respectively."""
        ...

    @property
    def n_feature_groups(self) -> int:
        """Positive number of groups that can be selected independently in hybrid rows."""
        ...

    def evaluate(self, pairs: IndexArray2D, selections: BoolArray2D) -> FloatArray1D:
        """Return one deterministic finite prediction per pair.

        `pairs` has shape (n, 2), and `selections` has shape (n, n_feature_groups).
        True selections take the group's features from the pair's second row; false selections take them from its first row.
        Predictions have shape (n,) and depend only on each hybrid observation's feature values.
        """
        ...

    def differs(self, pairs: IndexArray2D) -> BoolArray2D:
        """Return (n, n_feature_groups) flags for differences within each pair.

        A false entry promises that exchanging the group's values cannot change the prediction for any choice of the other groups' values.
        False positives cost computation; false negatives can produce incorrect contributions.
        Return all true when difference detection is unavailable.
        """
        ...


def column_feature_groups(
    n_columns: int, feature_group_of_column: Sequence[int] | None
) -> IndexArray1D:
    """Copy an assignment whose feature group indices cover consecutive integers starting at zero."""
    if n_columns == 0:
        raise ValueError("at least one feature column is required")
    feature_groups = (
        range(n_columns) if feature_group_of_column is None else feature_group_of_column
    )
    if len(feature_groups) != n_columns:
        raise ValueError("feature_group_of_column must contain one group index per column")
    unique = sorted(set(feature_groups))
    if unique != list(range(len(unique))):
        raise ValueError(
            "feature_group_of_column must cover consecutive feature groups starting at zero"
        )
    result = np.array(feature_groups, dtype=np.intp, copy=True)
    result.flags.writeable = False
    return result
