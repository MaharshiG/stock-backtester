import numpy as np


def ma_crossover_signals(
    prices: np.ndarray,
    fast_window: int = 10,
    slow_window: int = 50,
) -> np.ndarray:
    """
    Moving Average Crossover strategy.

    Returns:
        signals: np.ndarray of ints with values {-1, 0, +1}
                 +1 = buy on cross up
                 -1 = sell on cross down
                  0 = hold
    """
    if fast_window <= 0 or slow_window <= 0:
        raise ValueError("windows must be > 0")
    if fast_window >= slow_window:
        raise ValueError("fast_window must be < slow_window")

    prices = prices.astype(float)

    # --- SMA using convolution (same logic as indicators.sma, repeated here to keep this file standalone)
    def _sma(x: np.ndarray, window: int) -> np.ndarray:
        kernel = np.ones(window) / window
        v = np.convolve(x, kernel, mode="valid")
        pad = np.full(window - 1, np.nan)
        return np.concatenate([pad, v])

    fast = _sma(prices, fast_window)
    slow = _sma(prices, slow_window)

    # valid where both SMAs exist
    valid = ~np.isnan(fast) & ~np.isnan(slow)

    signals = np.zeros_like(prices, dtype=int)

    # Cross logic:
    # diff[t] = fast[t] - slow[t]
    diff = fast - slow

    # We detect sign changes in diff (from <=0 to >0 = cross up; from >=0 to <0 = cross down)
    prev = np.roll(diff, 1)
    prev[0] = np.nan

    cross_up = valid & (prev <= 0) & (diff > 0)
    cross_down = valid & (prev >= 0) & (diff < 0)

    signals[cross_up] = 1
    signals[cross_down] = -1

    return signals
