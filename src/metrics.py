import numpy as np


def total_return(equity: np.ndarray) -> float:
    """
    Total return as a fraction. Example: 0.25 = +25%
    """
    equity = equity.astype(float)
    if len(equity) < 2:
        return 0.0
    if equity[0] == 0:
        raise ValueError("Initial equity is zero; cannot compute return.")
    return (equity[-1] / equity[0]) - 1.0


def max_drawdown(equity: np.ndarray) -> float:
    """
    Max drawdown as a fraction (negative). Example: -0.20 = -20%
    """
    equity = equity.astype(float)
    if len(equity) == 0:
        return 0.0

    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity / running_max) - 1.0
    return float(np.min(drawdowns))


def sharpe_ratio_from_equity(equity: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Simple Sharpe ratio using daily returns derived from equity.
    Assumes risk-free rate ~0 for simplicity.
    """
    equity = equity.astype(float)
    if len(equity) < 3:
        return 0.0

    rets = np.empty_like(equity)
    rets[0] = np.nan
    rets[1:] = (equity[1:] - equity[:-1]) / equity[:-1]

    valid = ~np.isnan(rets)
    rets = rets[valid]

    if rets.size < 2:
        return 0.0

    mean = float(np.mean(rets))
    std = float(np.std(rets, ddof=1))
    if std == 0.0:
        return 0.0

    return (mean / std) * np.sqrt(periods_per_year)
