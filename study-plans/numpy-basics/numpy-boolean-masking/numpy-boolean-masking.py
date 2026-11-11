import numpy as np

def row_summary(data: list, threshold: float) -> np.ndarray:
    """
    Returns a float64 array of shape (3, m, n): mask, any-row, all-row.
    """
    arr = np.array(data, dtype=np.float64)
    mask = (arr > threshold).astype(np.float64)

    row_any = np.any(arr > threshold, axis =1)[:,None]
    row_any_filter = np.where(row_any, arr, 0.0)

    row_all = np.all(arr > threshold, axis =1)[:,None]
    row_all_filter = np.where(row_all, arr, 0.0)

    return np.stack((mask,row_any_filter, row_all_filter ))
