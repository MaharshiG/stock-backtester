from pathlib import Path

from src.data_loader import load_prices
from src.indicators import sma, daily_returns
from src.strategies import ma_crossover_signals


def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "sample_prices.csv"

    dates, close = load_prices(csv_path)

    # Indicators (Milestone 2)
    sma_3 = sma(close, window=3)
    rets = daily_returns(close)

    # Strategy (Milestone 3)
    signals = ma_crossover_signals(close, fast_window=2, slow_window=3)

    buy_count = int((signals == 1).sum())
    sell_count = int((signals == -1).sum())

    print("Hello Backtester")
    print(f"Loaded rows: {len(close)}")
    print(f"Date range: {dates[0]} -> {dates[-1]}")
    print(f"Close min/max: {close.min():.2f} / {close.max():.2f}")

    print("\nSignals summary:")
    print(f"BUY signals:  {buy_count}")
    print(f"SELL signals: {sell_count}")

    print("\nPreview (all rows):")
    for i in range(len(close)):
        print(
            f"{dates[i]} | Close={close[i]:.2f} | "
            f"SMA(3)={sma_3[i]:.2f} | Return={rets[i]:.4f} | Signal={signals[i]}"
        )


if __name__ == "__main__":
    main()
