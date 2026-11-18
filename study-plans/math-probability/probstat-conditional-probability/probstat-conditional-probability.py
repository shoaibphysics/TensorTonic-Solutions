def conditional_probability(p_a: float, p_b: float, p_a_and_b: float) -> list:
    """
    Returns both rounded conditional probabilities in the required order.
    """
    x1 = round(p_a_and_b/p_b,4)
    x2 = round(p_a_and_b/p_a,4)
    return [x1,x2]