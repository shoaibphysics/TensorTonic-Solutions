import numpy as np

def mean_median_mode(x: list) -> dict:
    """
    Returns mean, median, and mode as Python floats in a dictionary.
    """
    x = np.asarray(x, dtype=np.float64)
    mean = np.mean(x)
    median = np.median(x)

    candidate, freq = np.unique(x, return_counts=True)
    mode= candidate[np.argmax(freq)]
    return {"mean":mean,
           "median":median,
           "mode":mode}

    