import numpy as np

def pearson_correlation(X: list) -> np.ndarray:
    """
    Returns the float64 feature-correlation matrix.
    """
    X = np.asarray(X, dtype=np.float64)
    X = X - np.mean(X, axis = 0, keepdims=True)
    sum = np.sqrt(np.sum(X**2, axis = 0))
    return (X.T @ X) / (sum[:,None]* sum[None,:])

    