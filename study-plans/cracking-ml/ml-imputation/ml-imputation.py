import numpy as np

def impute(X: list, method: str = "mean") -> list:
    """
    Returns a new matrix with missing entries filled.
    """
    X = np.asarray(X, dtype=np.float64)
    r, c = X.shape
    mask = np.isnan(X)
    num_nan = np.sum(mask, axis=0, keepdims=True) == r
    X = np.where (num_nan, 0, X)
    mask = np.isnan(X)
    

    if method == "mean":
        mean = np.nanmean(X, axis=0, keepdims=True)
        X = np.where(mask, mean, X)

    elif method == "median":
        median = np.nanmedian(X, axis=0, keepdims=True)
        X = np.where(mask, median, X)

    return X
