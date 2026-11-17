import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    """
    Returns the requested percentiles as a float64 array.
    """
    x = np.asarray(x, dtype=np.float64)
    return np.percentile(x,q, method="linear")
    