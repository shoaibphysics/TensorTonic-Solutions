import numpy as np

def dot_product(x: list, y: list) -> float:
    """
    Returns the dot product as a float.
    """
    X, Y=np.array(x), np.array(y)
    return float(np.dot(x,y))
    