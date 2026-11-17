import numpy as np

def skewness_kurtosis(data: list) -> dict:
    """
    Returns adjusted statistics and their interpretations in a dictionary.
    """
    x = np.array(data, dtype=np.float64)
    x = x-np.mean(x, keepdims=True)
    n = x.shape[-1]
    s = np.std(x, ddof=1)
    x = x/s
    g1 = n/((n-1)*(n-2))*np.sum(x**3)
    g1 = np.round(g1, 4)

    g2 = n*(n+1)/((n-1)*(n-2)*(n-3))*np.sum(x**4)
    g2 -= 3*((n-1)**2) /((n-2)*(n-3))
    g2 = np.round(g2, 4)

    if g1 > 0.5:
        a = "right-skewed"
    elif g1 < -0.5 :
        a = "left-skewed"
    else:
        a = "approximately symmetric"

    
    if g2 > 1:
        b = "leptokurtic"
    elif g2 < -1 :
        b = "platykurtic"
    else:
        b = "mesokurtic"
    

    return {"skewness": g1, "kurtosis" : g2 , "skew_interpretation" : a , "kurtosis_interpretation" : b}
    