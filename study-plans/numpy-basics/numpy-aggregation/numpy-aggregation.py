import numpy as np

def summarize(data: list, axis: int) -> np.ndarray:
    """
    Returns float64 rows of mean, standard deviation, minimum, and maximum.
    """
    arr=np.array(data, dtype=np.float64)
    return np.stack([arr.mean(axis=axis),arr.std(axis=axis),arr.min(axis=axis),arr.max(axis=axis)])
