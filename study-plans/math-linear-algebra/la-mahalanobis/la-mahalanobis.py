import numpy as np

def mahalanobis_distance(x: list, mean: list, cov: list) -> float:
    """
    Returns the Mahalanobis distance as a Python float.
    """
    x=np.asarray(x, dtype= np.float64)
    mean=np.asarray(mean, dtype= np.float64)
    diff = x-mean
    cov=np.asarray(cov, dtype= np.float64)
    # Sigma_inverse * diff
    # Sigma * x = diff => x = Sigma_inverse * diff
    coeff = np.linalg.solve(cov, diff)
    return float(np.sqrt(diff.T@ coeff))