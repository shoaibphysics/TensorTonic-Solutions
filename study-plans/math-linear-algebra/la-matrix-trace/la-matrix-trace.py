import numpy as np

def matrix_trace(A: list) -> float:
    """
    Returns the trace as a Python float.
    """
    a = np.array(A, dtype=np.float64)
    trace = 0.0
    for i in range(a.shape[0]):
        trace+=a[i,i]
    return float(trace)
    
    