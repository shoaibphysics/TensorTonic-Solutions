import numpy as np

def row_extremes(data: list) -> np.ndarray:
    """
    Returns float64 rows of maxima, maximum indices, minima, and minimum indices.
    """
    arr = np.array(data, dtype=np.float64)
    row_max = np.max(arr, axis = 1)
    row_argmax = np.argmax(arr, axis = 1).astype(np.float64)
    row_min = np.min(arr, axis = 1)
    row_argmin = np.argmin(arr, axis = 1).astype(np.float64)
    return np.stack([
        row_max,
        row_argmax,
        row_min,
        row_argmin
    ])
