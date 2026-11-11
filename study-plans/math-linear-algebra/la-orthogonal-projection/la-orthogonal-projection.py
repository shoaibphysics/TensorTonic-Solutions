import numpy as np

def projection_matrix(A: list) -> np.ndarray:
    """
    Returns the float64 projector onto the column space of A.
    """
    a = np.array(A, dtype=np.float64)
    gram = a.T @ a
    coeff = np.linalg.solve(gram, a.T)
    return a@ coeff