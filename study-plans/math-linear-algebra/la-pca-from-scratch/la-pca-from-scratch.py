import numpy as np

def pca_project(X: list, n_components: int) -> np.ndarray:
    """
    Returns the projected samples as a float64 array.
    """
    x = np.asarray(X, dtype=np.float64)
    x = x-np.mean(x, axis=0, keepdims=True)
    n = x.shape[0]
    C = (x.T @ x)/(n-1)
    eigenvalues , eigenvectors = np.linalg.eig(C)
    idx = np.argsort(eigenvalues)[::-1]
    V = eigenvectors[:,idx[:n_components]]
    return x@V