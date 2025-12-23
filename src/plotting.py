from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_price_with_signals(
    dates: np.ndarray,
    close: np.ndarray,
    fast_sma: np.ndarray,
    slow_sma: np.ndarray,
    signals: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Saves a price chart with SMA lines + buy/sell markers.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(close))

    fig = plt.figure()
    plt.plot(x, close, label="Close")
    plt.plot(x, fast_sma, label="Fast SMA")
    plt.plot(x, slow_sma, label="Slow SMA")

    buy_idx = np.where(signals == 1)[0]
    sell_idx = np.where(signals == -1)[0]

    if buy_idx.size > 0:
        plt.scatter(buy_idx, close[buy_idx], marker="^", label="Buy")
    if sell_idx.size > 0:
        plt.scatter(sell_idx, close[sell_idx], marker="v", label="Sell")

    plt.title("Price + SMA Crossover Signals")
    plt.xlabel("Day Index")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_equity_curve(
    dates: np.ndarray,
    equity: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Saves an equity curve chart.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(equity))

    fig = plt.figure()
    plt.plot(x, equity, label="Equity")
    plt.title("Equity Curve")
    plt.xlabel("Day Index")
    plt.ylabel("Equity ($)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
