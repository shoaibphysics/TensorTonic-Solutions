import numpy as np

def rbf_kernel_matrix(X: list, gamma: float) -> np.ndarray:
    """
    Returns the float64 pairwise RBF kernel matrix.
    """
    x = np.array(X, dtype=np.float64) #[n, d]
    a = x[:,None,:]-x[None,:,:] #(n,n,d)
    norm_square = (a**2).sum(axis=-1)
    return np.exp(-gamma*norm_square)
    