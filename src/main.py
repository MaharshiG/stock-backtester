from pathlib import Path

from src.backtester import run_backtest
from src.data_loader import load_prices
from src.indicators import sma, daily_returns
from src.strategies import ma_crossover_signals
from src.metrics import total_return, max_drawdown, sharpe_ratio_from_equity



def main():
    project_root = Path(__file__).resolve().parents[1]
    csv_path = project_root / "data" / "sample_prices.csv"

    dates, close = load_prices(csv_path)

    # Indicators (Milestone 2)
    sma_3 = sma(close, window=3)
    rets = daily_returns(close)

    # Strategy (Milestone 3)
    signals = ma_crossover_signals(close, fast_window=2, slow_window=3)

    # Backtest (Milestone 4)
    cash_curve, shares_curve, equity_curve, trades = run_backtest(
        dates=dates,
        close=close,
        signals=signals,
        initial_cash=10_000.0,
    )

    print("Hello Backtester")
    print(f"Loaded rows: {len(close)}")
    print(f"Date range: {dates[0]} -> {dates[-1]}")
    print(f"Close min/max: {close.min():.2f} / {close.max():.2f}")

    print("\nSignals summary:")
    print(f"BUY signals:  {(signals == 1).sum()}")
    print(f"SELL signals: {(signals == -1).sum()}")

    print("\nBacktest summary:")
    print(f"Final equity: {equity_curve[-1]:.2f}")
    print(f"Trades: {len(trades)}")

    if trades:
        print("\nTrades:")
        for t in trades:
            print(f"{t.date} {t.side} price={t.price:.2f} shares={t.shares} cash_after={t.cash_after:.2f}")

    print("\nPreview (all rows):")
    for i in range(len(close)):
        print(
            f"{dates[i]} | Close={close[i]:.2f} | SMA(3)={sma_3[i]:.2f} | "
            f"Return={rets[i]:.4f} | Signal={signals[i]} | "
            f"Shares={shares_curve[i]} | Equity={equity_curve[i]:.2f}"
        )
    # Metrics (Milestone 5)
    tr = total_return(equity_curve)
    mdd = max_drawdown(equity_curve)
    sr = sharpe_ratio_from_equity(equity_curve)

    print("\nMetrics summary:")
    print(f"Total return: {tr * 100:.2f}%")
    print(f"Max drawdown: {mdd * 100:.2f}%")
    print(f"Sharpe ratio: {sr:.2f}")



if __name__ == "__main__":
    main()
