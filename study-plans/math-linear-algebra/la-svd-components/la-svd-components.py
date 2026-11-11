import numpy as np

def svd(A: list) -> dict:
    """
    Returns a dictionary containing float64 arrays U, s, and Vt.
    """
    a = np.asarray(A, dtype=np.float64)
    U, s, Vt = np.linalg.svd(a, full_matrices = False)
    return {"U" : U, "s" : s, "Vt" :Vt}