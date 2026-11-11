import numpy as np

def lu_decomposition(A: list) -> dict:
    """
    Returns a dictionary containing float64 arrays L and U.
    """
    a = np.array(A, dtype=np.float64)
    r = a.shape[0]
    L = np.eye(r)
    U = np.zeros_like(a)

    for i in range(r):
        U[i,i:] = a[i,i:]-L[i,:i]@U[:i,i:]
        L[i+1:,i] = (a[i+1:,i] - L[i+1:,:i]@U[:i,i])/U[i,i]

    return {"L":L,"U":U}
    
    