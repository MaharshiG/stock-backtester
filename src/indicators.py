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
