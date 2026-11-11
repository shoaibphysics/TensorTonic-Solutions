import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transpose as a float64 array.
    """
    return np.array(A, dtype=np.float64).T