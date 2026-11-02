"""Hybrid selection preserves stored bytes across dtypes and source layouts."""

import numpy as np
import pytest

from cohort_shapley.numpy import NumpyAdapter


@pytest.mark.parametrize(
    "dtype",
    [
        "f4",
        "f8",
        ">f4",
        "i4",
        "u8",
        "S5",
        "U3",
        "V3",
        "M8[D]",
        np.dtype([("x", "u1"), ("y", ">u4")], align=True),
    ],
)
@pytest.mark.parametrize("layout", ["C", "F", "strided"])
def test_hybrids_preserve_dtype_and_bytes(dtype: str | np.dtype[np.void], layout: str) -> None:
    item = np.dtype(dtype)
    raw = np.arange(8 * 6 * item.itemsize, dtype=np.uint8).reshape(8, 6 * item.itemsize)
    values = raw.view(item)
    values = np.asfortranarray(values) if layout == "F" else values
    values = values[::-2, ::2] if layout == "strided" else values
    cohort_1, cohort_2 = values[:2], values[2:]
    pairs = np.array([[0, 1], [1, 2], [2, 0], [2, 3]])
    feature_groups = [column // 2 for column in range(values.shape[1])]
    selections = np.array(
        [[False, True, False], [True, False, True], [False, False, True], [True, True, True]]
    )[:, : max(feature_groups) + 1]
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

    adapter = NumpyAdapter(cohort_1, cohort_2, predict, feature_group_of_column=feature_groups)
    assert adapter.cohort_1 is cohort_1
    assert adapter.cohort_2 is cohort_2
    assert adapter.n_rows == (2, len(values) - 2)
    actual = adapter.evaluate(pairs, selections)
    assert actual.dtype == np.float64
    np.testing.assert_array_equal(actual, np.arange(4))


def test_bit_patterns_control_differences_and_survive_selection() -> None:
    bits = np.array(
        [[0, 0x7FC00001, 0x7FC00002], [0x80000000, 0x7FC00001, 0x7FC00003]], dtype=np.uint32
    )
    values = bits.view(np.float32)
    adapter = NumpyAdapter(
        values[:1],
        values[1:],
        lambda x: x.view(np.uint32)[:, 0].astype(np.float64),
        feature_group_of_column=(0, 1, 1),
    )
    pairs = np.array([[0, 0], [0, 1]])
    np.testing.assert_array_equal(adapter.differs(pairs), [[False, False], [True, True]])
    np.testing.assert_array_equal(
        adapter.evaluate(pairs, np.ones((2, 2), dtype=bool)), [0, 0x80000000]
    )


def test_dtype_padding_controls_differences_and_survives_selection() -> None:
    dtype = np.dtype([("x", "u1"), ("y", "u2")], align=True)
    storage = np.zeros((2, dtype.itemsize), dtype=np.uint8)
    storage[1, 1] = 1
    values = storage.view(dtype)
    adapter = NumpyAdapter(
        values[:1], values[1:], lambda x: x.view(np.uint8)[:, 1].astype(np.float64)
    )
    pairs = np.array([[0, 1]])
    np.testing.assert_array_equal(adapter.differs(pairs), [[True]])
    np.testing.assert_array_equal(adapter.evaluate(pairs, np.array([[True]])), [1])


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (np.array([[1 + 0j, 1 + 0j], [1 + 1j, 1 + 0j]]), [[True, False]]),
        (np.array([[b"aaa", b"aaa", b"aaa"], [b"aba", b"aaa", b"aaa"]]), [[True, False, False]]),
    ],
)
def test_differences_in_later_bytes_of_wide_items(
    values: np.ndarray, expected: list[list[bool]]
) -> None:
    adapter = NumpyAdapter(values[:1], values[1:], lambda x: np.zeros(len(x)))
    np.testing.assert_array_equal(adapter.differs(np.array([[0, 1]])), expected)


@pytest.mark.parametrize(
    "feature_groups",
    [
        [-1, 0],
        [0, 2],
        [1, 1],
        [0],
    ],
)
def test_invalid_feature_group_assignment_raises(feature_groups: list[int]) -> None:
    with pytest.raises(ValueError, match="feature_group_of_column"):
        NumpyAdapter(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            lambda x: x.sum(axis=1),
            feature_group_of_column=feature_groups,
        )


@pytest.mark.parametrize(
    ("values", "message"),
    [
        (np.empty((2, 0)), "at least one feature column"),
        (np.empty(2), "2D fixed-width arrays"),
        (np.array([[object()]]), "2D fixed-width arrays"),
        (np.empty((2, 2), dtype="V0"), "2D fixed-width arrays"),
    ],
)
def test_invalid_storage_raises(values: np.ndarray, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        NumpyAdapter(values, values, lambda x: np.zeros(len(x)))


@pytest.mark.parametrize(
    "cohort_2",
    [np.zeros((2, 3)), np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=">f8")],
)
def test_mismatched_cohorts_raise(cohort_2: np.ndarray) -> None:
    with pytest.raises(ValueError, match="matching dtypes and column counts"):
        NumpyAdapter(np.zeros((2, 2)), cohort_2, lambda x: x.sum(axis=1))


def test_source_changes_are_visible_before_evaluation() -> None:
    cohort_1 = np.zeros((2, 1))
    cohort_2 = np.zeros((2, 1))
    adapter = NumpyAdapter(cohort_1, cohort_2, lambda x: x[:, 0])
    cohort_1[1, 0] = 2
    cohort_2[0, 0] = 3
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    np.testing.assert_array_equal(adapter.differs(pairs), [[True], [True], [True]])
    np.testing.assert_array_equal(
        adapter.evaluate(pairs, np.array([[True], [True], [False]])), [2, 3, 3]
    )


def test_assignment_is_owned() -> None:
    feature_groups = [0, 1]
    adapter = NumpyAdapter(
        np.ones((2, 2)),
        np.ones((2, 2)),
        lambda x: x.sum(axis=1),
        feature_group_of_column=feature_groups,
    )
    feature_groups[1] = 0
    assert adapter.n_feature_groups == 2
