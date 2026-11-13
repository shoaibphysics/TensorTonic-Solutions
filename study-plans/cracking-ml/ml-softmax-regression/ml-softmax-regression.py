import numpy as np

def softmax_regression(X: list, y: list, num_classes: int, lr: float, n_iters: int) -> tuple:
    """
    Returns the fitted weight matrix and bias vector.
    """
    X = np.asarray(X, dtype=np.float64) # (n,d)
    y = np.asarray(y, dtype=np.int64) # (n)

    n, d = X.shape

    Y = np.zeros((n, num_classes), dtype=np.float64)
    Y[np.arange(n),y] = 1

    w = np.zeros((d, num_classes), dtype=np.float64)
    b = np.zeros(num_classes, dtype=np.float64)

    for _ in range(n_iters):
        # d_w = (1/n)X.T(P-Y)
        # d_b = (1/n) (P-Y)
        Z = X @ w + b[None,:]
        Z = Z - np.max(Z, axis=-1, keepdims=True)
        exp = np.exp(Z)
        P = exp / np.sum(exp, axis = -1, keepdims= True)
        w-=lr*(1/n)*(X.T@(P-Y))
        b-=lr*np.mean(P-Y, axis=0)
    return np.round(w,4).tolist(), np.round(b,4).tolist()
        
        
    
    
