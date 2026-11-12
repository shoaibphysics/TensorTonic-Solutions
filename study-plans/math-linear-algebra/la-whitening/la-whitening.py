import numpy as np

def whiten(X: list) -> np.ndarray:
    """
    Returns the whitened samples as a float64 array.
    """
    x = np.asarray(X, dtype=np.float64) #(n_sample, n_features)
    x = x - np.mean(x, axis=0)
    n = x.shape[0]
    C = (1/(n-1))*x.T @x
    eigenvalues, eigenvectors = np.linalg.eigh(C)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]

    m = eigenvectors.shape[1]
    for j in range(m):
        v=eigenvectors[:,j]
        idx = np.argmax(np.abs(v))
        if v[idx] <0 :
            eigenvectors[:,j] = (-1)* eigenvectors[:,j]
            
    mask = eigenvalues > 1e-10
    D = np.zeros_like(eigenvalues)
    D[mask] = np.sqrt(1/eigenvalues[mask])
    return x @ eigenvectors @ np.diag( D)
    