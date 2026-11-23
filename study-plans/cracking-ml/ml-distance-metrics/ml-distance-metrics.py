import math
import numpy as np

def distance_metric(x: list, y: list, metric: str, p: float = 2) -> float:
    """
    Returns the selected distance as a rounded float.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if metric == "euclidean":
        return np.sqrt(np.sum((x-y)**2)).item()
    elif metric == "manhattan":
        return np.sum(np.abs(x-y)).item()
    elif metric == "cosine":
        n = np.sqrt(np.sum((x)**2))*np.sqrt(np.sum((y)**2))
        if n == 0:
            return 0
        return (1 -(np.sum(x*y))/n).item()
    elif metric == "chebyshev":
        return np.max(np.abs(x-y)).item()
    elif metric == "minkowski":
        return (np.sum(np.abs(x-y)**p)**(1/p)).item()
        
