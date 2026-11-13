import numpy as np

def lasso_regression(X: list, y: list, alpha: float, lr: float, epochs: int) -> tuple:
    """
    Returns the fitted weight list and bias.
    """
    X = np.asarray(X, dtype=np.float64) # (n, d)
    y = np.asarray(y, dtype=np.float64) # (d)
    n, d = X.shape

    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    # L = (1/n)\sum{i=1}^n (X_i w+b-y_i)^2 +alpha* \sum_{j=1}^d |w_i|
    # d_wL =  (2/n)(X_i w+b -y_i)X_i +\alpha \sign(w_i)
    # d_bL = (2/n)\sum_i (X_i w +b-y_i)

    for _ in range(epochs):
        err = X@w+b-y
        w-=lr*((2/n)* (err @ X)+ alpha*np.sign(w))
        b-=lr*(2/n)* np.sum(err)

    return np.round(w, 4).tolist(), round(b, 4)
        
    

    
