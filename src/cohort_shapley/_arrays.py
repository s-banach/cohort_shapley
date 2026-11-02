"""NumPy array types with a fixed number of dimensions."""

import numpy as np

type FloatArray1D = np.ndarray[tuple[int], np.dtype[np.float64]]
type FloatArray2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
type IndexArray1D = np.ndarray[tuple[int], np.dtype[np.intp]]
type IndexArray2D = np.ndarray[tuple[int, int], np.dtype[np.intp]]
type BoolArray1D = np.ndarray[tuple[int], np.dtype[np.bool_]]
type BoolArray2D = np.ndarray[tuple[int, int], np.dtype[np.bool_]]
type UInt8Array2D = np.ndarray[tuple[int, int], np.dtype[np.uint8]]
