import numpy as np

def euclidean_distance(x: list, y: list) -> float:
    """
    Returns the Euclidean distance as a float.
    """
    X,Y=np.array(x),np.array(y)
    return float(np.linalg.norm(X-Y))