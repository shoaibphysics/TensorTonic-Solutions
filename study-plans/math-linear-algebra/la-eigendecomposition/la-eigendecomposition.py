import numpy as np

def eigendecompose(A: list) -> dict:
    """
    Returns sorted float64 eigenvalues and eigenvectors in a dictionary.
    """
    a = np.asarray(A, dtype=np.float64)
    eigenvalues, eigenvectors = np.linalg.eig(a)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    norm = np.linalg.norm(eigenvectors, axis = 0, keepdims=True)
    eigenvectors = eigenvectors/norm
    return {"eigenvalues":eigenvalues,
           "eigenvectors":eigenvectors}