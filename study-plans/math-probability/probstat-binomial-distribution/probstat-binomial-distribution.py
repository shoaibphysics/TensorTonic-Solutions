from math import comb

def binomial_distribution(n: int, p: float, threshold: int) -> dict:
    """
    Returns the PMF, mean, variance, and probability at least threshold.
    """
    mean = round(n*p,4)
    variance = round(n*p*(1-p),4)
    pmf = [comb(n,k)*(p**k)*((1-p)**(n-k)) for k in range(n+1) ]
    prob_at_least = round(1-sum(pmf[:threshold]),4)
    return {"pmf": [round(p,4) for p in pmf],
           "mean":mean,
           "variance":variance,
           "prob_at_least":prob_at_least}