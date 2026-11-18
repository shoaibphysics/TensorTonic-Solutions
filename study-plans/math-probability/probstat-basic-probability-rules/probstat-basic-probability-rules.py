def basic_probability(p_a: float, p_b: float, p_a_and_b: float) -> list:
    """
    Returns four rounded probability values in the required order.
    """
    x1 = round(p_a+p_b-p_a_and_b,4)
    x2 = round(1-p_a,4)
    x3 = round(1-p_b,4)
    x4 = round(p_a-p_a_and_b,4)
    return [x1,x2,x3,x4]