import numpy as np

def matrix_transpose(A: list) -> np.ndarray:
    """
    Returns the transpose as a float64 array.
    """
    a = np.array(A,dtype=np.float64)
    r,c=a.shape
    o = np.empty((c,r))
    for i in range(r):
        for j in range(c):
            o[j,i]=a[i,j]
    return o