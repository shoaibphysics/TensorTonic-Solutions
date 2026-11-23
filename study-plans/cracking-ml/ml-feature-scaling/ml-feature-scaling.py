import numpy as np

def feature_scale(X: list, method: str = "minmax") -> list:
    """
    Returns the independently scaled feature columns.
    """
    X = np.asarray(X, dtype=np.float64)
    if method == "minmax":
        min = np.min(X, axis = 0, keepdims=True)
        max = np.max(X, axis = 0, keepdims=True)
        den = max-min
        den[den==0] = 1.0
        X = (X-min)/den
    elif method == "standard":
        mu = np.mean(X, axis=0, keepdims=True)
        std = np.std(X, axis=0, ddof=0, keepdims=True)
        std[std==0] = 1.0
        X = (X-mu)/std
    return X