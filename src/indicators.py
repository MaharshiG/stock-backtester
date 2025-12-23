import numpy as np


def sma(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Simple Moving Average (SMA)
    Returns an array of same length as prices.
    First (window-1) values are NaN.
    """
    if window <= 0:
        raise ValueError("window must be > 0")

    prices = prices.astype(float)

    kernel = np.ones(window) / window
    sma_values = np.convolve(prices, kernel, mode="valid")

    # pad the beginning with NaN so length matches prices
    padding = np.full(window - 1, np.nan)
    return np.concatenate([padding, sma_values])


def daily_returns(prices: np.ndarray) -> np.ndarray:
    """
    Daily percentage returns.
    First value is NaN.
    """
    prices = prices.astype(float)

    returns = np.empty_like(prices)
    returns[0] = np.nan
    returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]

    return returns
def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Relative Strength Index (RSI) using Wilder's smoothing.
    Returns an array same length as prices.
    First `period` values are NaN (not enough history).
    """
    if period <= 0:
        raise ValueError("period must be > 0")

    prices = prices.astype(float)
    n = len(prices)
    out = np.full(n, np.nan, dtype=float)

    if n <= period:
        return out

    # Price changes
    delta = np.diff(prices)
    gains = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)

    # Initial average gain/loss (simple mean over first period)
    avg_gain = gains[:period].mean()
    avg_loss = losses[:period].mean()

    # First RSI value appears at index = period
    if avg_loss == 0:
        out[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        out[period] = 100.0 - (100.0 / (1.0 + rs))

    # Wilder smoothing for the rest
    for i in range(period + 1, n):
        g = gains[i - 1]
        l = losses[i - 1]
        avg_gain = (avg_gain * (period - 1) + g) / period
        avg_loss = (avg_loss * (period - 1) + l) / period

        if avg_loss == 0:
            out[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out

