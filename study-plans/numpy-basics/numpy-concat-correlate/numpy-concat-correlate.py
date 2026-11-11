import numpy as np

def compare_correlations(a: list, b: list) -> np.ndarray:
    """
    Returns float64 correlation matrices for a, b, and their combined rows.
    """
    A=np.array(a, dtype=np.float64)
    B=np.array(b, dtype=np.float64)
    C=np.concatenate([A,B],axis=0)
    return np.stack([
        np.corrcoef(A, rowvar=False),
        np.corrcoef(B, rowvar=False),
        np.corrcoef(C, rowvar=False)
    ])
