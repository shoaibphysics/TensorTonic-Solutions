import numpy as np

def matrix_vector_multiply(A: list, x: list) -> np.ndarray:
    """
    Returns the matrix-vector product as a float64 array.
    """
    A_array = np.array(A, dtype = np.float64)
    x_array = np.array(x, dtype = np.float64)
    return A_array @ x_array