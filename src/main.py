from pathlib import Path

import numpy as np

from src.data_loader import load_prices
from src.indicators import sma, daily_returns


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "sample_prices.csv"

    dates, close = load_prices(csv_path)

    sma_3 = sma(close, window=3)
    rets = daily_returns(close)

    print("Hello Backtester")
    print(f"Loaded rows: {len(close)}")
    print(f"Date range: {dates[0]} -> {dates[-1]}")
    print(f"Close min/max: {close.min():.2f} / {close.max():.2f}")

    print("\nPreview (last 3 rows):")
    for i in range(-3, 0):
        print(
            f"{dates[i]} | Close={close[i]:.2f} | "
            f"SMA(3)={sma_3[i]:.2f} | Return={rets[i]:.4f}"
        )


if __name__ == "__main__":
    main()
