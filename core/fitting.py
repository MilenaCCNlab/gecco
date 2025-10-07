import numpy as np
from typing import Dict, Tuple, Callable, Optional
from dataclasses import dataclass

from .evaluation import aic as _aic, bic as _bic
from .data_structures import FitResult

try:
    from scipy.optimize import minimize
except Exception as e:  # pragma: no cover
    minimize = None
    _IMPORT_ERROR = e

@dataclass
class BoundsBox:
    names: list[str]
    lows: np.ndarray
    highs: np.ndarray

def _build_bounds_box(bounds: Dict[str, Tuple[float, float]], order: list[str]) -> BoundsBox:
    lows, highs = [], []
    for p in order:
        lo, hi = bounds[p]
        lows.append(lo)
        highs.append(hi)
    return BoundsBox(order, np.array(lows, float), np.array(highs, float))

def _random_in_bounds(bb: BoundsBox, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(bb.lows, bb.highs)

def _to_param_dict(x: np.ndarray, order: list[str]) -> Dict[str, float]:
    return {name: float(val) for name, val in zip(order, x)}

def fit_model(
    nll_func: Callable[[object, ...], float],             # callable(data=df, **params) -> nll
    param_order: list[str],
    bounds: Dict[str, Tuple[float, float]],
    data,
    n_trials: int,
    n_starts: int = 8,
    seed: Optional[int] = 123,
    method: str = "L-BFGS-B",
    compute_metrics: bool = True,
) -> FitResult:
    """
    Multi-start bounded minimize of the provided NLL callable.
    """
    if minimize is None:
        raise RuntimeError(f"scipy not available: {_IMPORT_ERROR}")

    rng = np.random.default_rng(seed)
    bb = _build_bounds_box(bounds, param_order)

    def obj(x: np.ndarray) -> float:
        params = _to_param_dict(x, param_order)
        return float(nll_func(data, **params))

    best = None
    best_fun = np.inf
    best_info = None

    # Start from mid + random jitter
    mids = (bb.lows + bb.highs) / 2.0
    starts = [mids] + [_random_in_bounds(bb, rng) for _ in range(max(0, n_starts - 1))]




    for x0 in starts:


        res = minimize(
            obj,
            x0=x0,
            method=method,
            bounds=list(zip(bb.lows, bb.highs)),
            options={"maxiter": 500}
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best = np.array(res.x, float)
            best_info = res

    param_dict = _to_param_dict(best, param_order)
    out = FitResult(
        params=param_dict,
        nll=best_fun,
        success=getattr(best_info, "success", True),
        info={"nit": getattr(best_info, "nit", None), "message": getattr(best_info, "message", "")},
    )

    if compute_metrics:
        k = len(param_order)
        out.aic = _aic(best_fun, k)
        out.bic = _bic(best_fun, k, n_trials)

    return out
