"""Hybrid selection preserves stored bytes across dtypes and source layouts."""

import numpy as np
import pytest

from cohort_shapley import core
from cohort_shapley.numpy import NumpyAdapter


@pytest.mark.parametrize("dtype", ["f4", "f8", ">f4", "i4", "u8", "S5", "U3", "V3", "M8[D]"])
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_hybrids_preserve_dtype_and_bytes(dtype: str, layout: str) -> None:
    item = np.dtype(dtype)
    raw = np.arange(8 * 6 * item.itemsize, dtype=np.uint8).reshape(8, 6 * item.itemsize)
    values = raw.view(item)
    values = np.asfortranarray(values) if layout == "F" else values
    values = values[::-2, ::2] if layout == "strided" else values
    pairs = np.array([[0, 1], [1, 2], [2, 0]])
    feature_groups = np.arange(values.shape[1]) // 2
    selections = np.array([[False, True, False], [True, False, True], [False, False, True]])[
        :, : int(feature_groups.max()) + 1
    ]
    expected = np.array(
        [
            [
                values.view(f"V{item.itemsize}")[
                    second_row if selections[k, feature_group] else first_row, column
                ].tobytes()
                for column, feature_group in enumerate(feature_groups)
            ]
            for k, (first_row, second_row) in enumerate(pairs)
        ],
        dtype=f"V{item.itemsize}",
    )

    def predict(hybrid: np.ndarray) -> np.ndarray:
        assert hybrid.dtype == values.dtype
        assert hybrid.flags.c_contiguous
        np.testing.assert_array_equal(hybrid.view(f"V{item.itemsize}"), expected)
        return np.arange(len(hybrid), dtype=np.float32)

    adapter = NumpyAdapter(values, predict, feature_group_of_column=feature_groups)
    assert adapter.values is values
    actual = adapter.evaluate(pairs, selections)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.arange(3))


def test_bit_patterns_control_differences_and_survive_selection() -> None:
    bits = np.array(
        [[0, 0x7FC00001, 0x7FC00002], [0x80000000, 0x7FC00001, 0x7FC00003]], dtype=np.uint32
    )
    values = bits.view(np.float32)
    adapter = NumpyAdapter(
        values,
        lambda x: x.view(np.uint32)[:, 0].astype(np.float64),
        feature_group_of_column=np.array([0, 1, 1]),
    )
    pairs = np.array([[0, 0], [0, 1]])
    np.testing.assert_array_equal(adapter.differs(pairs), [[False, False], [True, True]])
    np.testing.assert_array_equal(
        adapter.evaluate(pairs, np.ones((2, 2), dtype=bool)), [0, 0x80000000]
    )


@pytest.mark.parametrize(
    "feature_groups",
    [
        np.array([-1, 0]),
        np.array([0, 2]),
        np.array([1, 1]),
        np.array([0]),
        np.array([0.0, 1.0]),
        np.array([[0, 1]]),
    ],
)
def test_invalid_feature_group_assignment_raises(feature_groups: core.IndexArray1D) -> None:
    with pytest.raises(ValueError, match="feature_group_of_column"):
        NumpyAdapter(
            np.zeros((2, 2)), lambda x: x.sum(axis=1), feature_group_of_column=feature_groups
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.empty((2, 0)), "at least one feature column"),
        (np.empty(2), "2D fixed-width array"),
        (np.array([[object()]]), "2D fixed-width array"),
        (np.empty((2, 2), dtype="V0"), "2D fixed-width array"),
    ],
)
def test_invalid_storage_raises(values: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NumpyAdapter(values, lambda x: np.zeros(len(x)))


def test_assignment_is_owned() -> None:
    feature_groups = np.array([0, 1])
    adapter = NumpyAdapter(
        np.ones((2, 2)), lambda x: x.sum(axis=1), feature_group_of_column=feature_groups
    )
    feature_groups[:] = 0
    assert adapter.n_feature_groups == 2
