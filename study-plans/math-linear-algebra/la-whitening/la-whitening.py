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

    # x = np.asarray(X, dtype=np.float64)

    # if x.ndim != 2:
    #     raise ValueError("X must be a 2D array.")

    # n = x.shape[0]

    # if n < 2:
    #     raise ValueError("At least two samples are required.")

    # # Center each feature.
    # x = x - np.mean(x, axis=0)

    # # Compute sample covariance.
    # C = (x.T @ x) / (n - 1)

    # # Compute eigenvalues and eigenvectors.
    # eigenvalues, eigenvectors = np.linalg.eigh(C)

    # # Sort from largest eigenvalue to smallest.
    # order = np.argsort(eigenvalues)[::-1]
    # eigenvalues = eigenvalues[order]
    # eigenvectors = eigenvectors[:, order]

    # # Apply a consistent sign convention.
    # for j in range(eigenvectors.shape[1]):
    #     v = eigenvectors[:, j]
    #     pivot = np.argmax(np.abs(v))

    #     if v[pivot] < 0:
    #         eigenvectors[:, j] = -eigenvectors[:, j]

    # # Compute inverse standard deviations.
    # mask = eigenvalues > 1e-10
    # D = np.zeros_like(eigenvalues)
    # D[mask] = 1.0 / np.sqrt(eigenvalues[mask])

    # # Rotate into principal directions and scale.
    # return x @ eigenvectors @ np.diag(D)
    