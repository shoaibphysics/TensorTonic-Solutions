def perms_and_combs(n: int, r: int) -> list:
    """
    Returns permutations, combinations, and n factorial as integers.
    """
    def fac(x: int):
        if x == 0 :
            return 1
        ans = x
        while x > 1:
            x -=1
            ans = ans*x
        return ans
    return [fac(n)//(fac(n-r)), fac(n)//(fac(n-r)*fac(r)), fac(n)]