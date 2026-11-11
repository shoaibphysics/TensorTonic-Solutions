import numpy as np

def reshape_array(data: list, operation: str) -> np.ndarray:
    """
    Returns a float64 array with the shape selected by operation.
    """
    if operation == "flatten":
        return np.array(data, dtype = np.float64).flatten()
    elif operation == "transpose":
        return np.array(data, dtype = np.float64).T
    elif operation == "add_batch":
        return np.expand_dims(np.array(data, dtype = np.float64), axis = 0)
