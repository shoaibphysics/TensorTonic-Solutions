def bayes_theorem(p_a: float, p_b_given_a: float, p_b_given_not_a: float) -> float:
    """
    Returns the posterior probability rounded to four decimals.
    """
    x1 = round(p_b_given_a*p_a+p_b_given_not_a * (1-p_a) ,4)
    x2 = round(p_b_given_a*p_a / x1,4)
    return x2