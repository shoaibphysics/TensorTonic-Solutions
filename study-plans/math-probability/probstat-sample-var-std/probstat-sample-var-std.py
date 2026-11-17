import numpy as np

def sample_var_std(x: list) -> dict:
    """
    Returns sample variance and standard deviation as Python floats.
    """
    x=np.asarray(x, dtype=np.float64)
    x=x-np.mean(x, axis=-1, keepdims=True)
    n=x.shape[-1]
    s2=(1/(n-1))*np.sum(x**2)
    return { "variance": s2, "std_dev": np.sqrt(s2) }