import numpy as np

def quantize_and_frame(data: list, decimals: int, pad_width: int) -> np.ndarray:
    """
    Returns float64 slices of rounded, floored, and ceiling-rounded values with zero borders.
    """
    a = np.array(data, dtype = np.float64)
    round = np.pad(np.round(a, decimals), pad_width)
    floor = np.pad(np.floor(a), pad_width)
    ceiling =np.pad( np.ceil(a), pad_width)
    return np.stack([
        round,
        floor,
        ceiling
    ])
    
