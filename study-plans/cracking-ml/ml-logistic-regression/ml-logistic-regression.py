import numpy as np

def logistic_regression(X: list, y: list, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64) # (n,d)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    w = np.zeros(d, dtype=np.float64) # (d)
    b = 0.0

    for _ in range(n_iters):
        z = np.clip(X@w+b, -500, 500)
        pred = 1/(1+np.exp(-z))
        err = pred - y
        w-=lr*(1/n)* (X.T @ err)
        b-=lr*(1/n)* np.sum(err)

    return np.round(w, 4).tolist(), round(b, 4)

    
    
