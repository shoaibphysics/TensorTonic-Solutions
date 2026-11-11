import numpy as np

def linear_combination(vectors: list, coefficients: list) -> np.ndarray:
    """
    Returns the weighted sum as a float64 vector.
    """
    v=np.array(vectors , dtype=np.float64)
    c=np.array(coefficients , dtype=np.float64)[None,:]
    return (c@v).flatten()