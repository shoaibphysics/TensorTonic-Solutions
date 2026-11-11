import numpy as np

def original_and_clipped(data: list, row_idx: int, lo: float, hi: float) -> np.ndarray:
    """
    Returns a (2, n) float64 array: original row, then clipped row.
    """
    b = data[row_idx].copy()
    arr = np.array(b, dtype = np.float64)
    return np.stack((arr, np.clip(arr, lo, hi)), axis= 0)
