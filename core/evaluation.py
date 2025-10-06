def aic(nll: float, k: int) -> float:
    return 2 * k + 2 * nll

def bic(nll: float, k: int, n: int) -> float:
    import math
    return math.log(n) * k + 2 * nll
