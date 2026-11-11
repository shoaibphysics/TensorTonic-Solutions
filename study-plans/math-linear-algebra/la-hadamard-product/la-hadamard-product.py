import numpy as np

def hadamard_product(A: list, B: list) -> np.ndarray:
    """
    Returns the element-wise product as a float64 array.
    """
    A,B=np.array(A,dtype=np.float64),np.array(B,dtype=np.float64)
    return A*B
    