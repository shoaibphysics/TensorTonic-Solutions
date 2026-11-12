import numpy as np

def ridge_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64) # (n,d)
    y = np.asarray(y, dtype=np.float64) # (n)

    n, d = X.shape

    w = np.zeros(d, dtype= np.float64)
    b = 0.0

    for _ in range(epochs):
        err = X @ w + b - y
        w-= lr*((2/n)*(X.T @ err)+2*alpha*w)
        b-= lr*(2/n)* np.sum(err)

    return np.round(w,4).tolist(), round(b,4)
        
    
    
