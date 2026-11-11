import numpy as np

def gram_schmidt(vectors: list) -> np.ndarray:
    """
    Returns a float64 array whose rows form the ordered orthonormal basis.
    """
    vec = np.array(vectors, dtype = np.float64)
    q = np.empty_like(vec)
    k = vec.shape[0]

    for j in range(1,k):
        q[j-1] = vec[j-1]/np.linalg.norm(vec[j-1])
        for i in range(j,k):
            vec[i] = vec[i] - (np.dot(q[j-1],vec[i]))*q[j-1]
        
    q[k-1] = vec[k-1]/np.linalg.norm(vec[k-1]) 
    return q

        
        