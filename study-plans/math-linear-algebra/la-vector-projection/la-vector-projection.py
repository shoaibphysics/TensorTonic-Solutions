import numpy as np

def vector_projection(u: list, v: list) -> np.ndarray:
    """
    Returns the float64 projection of u onto v.
    """
    U = np.array(u, dtype= np.float64)
    V = np.array(v, dtype= np.float64)
    return (np.dot(U,V)/(np.dot(V,V)))*V