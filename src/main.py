from __future__ import annotations

import argparse
from pathlib import Path

from src.backtester import run_backtest
from src.data_loader import load_prices
from src.indicators import sma, daily_returns, rsi
from src.metrics import total_return, max_drawdown, sharpe_ratio_from_equity
from src.plotting import plot_price_with_signals, plot_equity_curve
from src.strategies import ma_crossover_signals, rsi_strategy_signals


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock Strategy Backtester (NumPy-only)")

    parser.add_argument(
        "--csv",
        type=str,
        default="data/sample_prices.csv",
        help="Path to CSV with columns: Date,Close",
    )

    parser.add_argument(
        "--strategy",
        type=str,
        choices=["ma", "rsi"],
        default="ma",
        help="Trading strategy to run",
    )

    # MA params
    parser.add_argument("--fast", type=int, default=10, help="Fast SMA window (ma strategy)")
    parser.add_argument("--slow", type=int, default=50, help="Slow SMA window (ma strategy)")

    # RSI params
    parser.add_argument("--period", type=int, default=14, help="RSI period (rsi strategy)")
    parser.add_argument("--buy-below", type=float, default=30.0, help="RSI buy threshold")
    parser.add_argument("--sell-above", type=float, default=70.0, help="RSI sell threshold")

    # Backtest params
    parser.add_argument("--initial-cash", type=float, default=10_000.0, help="Starting cash")

    # Output options
    parser.add_argument("--no-plots", action="store_true", help="Disable saving plots")

    return parser.parse_args()


def main():
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = project_root / csv_path

    dates, close = load_prices(csv_path)

    # Indicators (Milestone 2)
    sma_3 = sma(close, window=3)
    rets = daily_returns(close)

    # Strategy switch
    strategy = args.strategy

    if strategy == "ma":
        fast_w, slow_w = args.fast, args.slow
        signals = ma_crossover_signals(close, fast_window=fast_w, slow_window=slow_w)

        # Plot inputs
        fast_sma = sma(close, window=fast_w)
        slow_sma = sma(close, window=slow_w)

    elif strategy == "rsi":
        period = args.period
        signals = rsi_strategy_signals(
            close,
            period=period,
            buy_below=args.buy_below,
            sell_above=args.sell_above,
        )

        # Optional: compute RSI for printing
        r = rsi(close, period=period)

        # Plot inputs (still show SMAs, optional)
        fast_sma = sma(close, window=10 if len(close) >= 10 else 2)
        slow_sma = sma(close, window=50 if len(close) >= 50 else 3)

    else:
        raise ValueError("Unknown strategy. Use 'ma' or 'rsi'.")

    # Backtest
    cash_curve, shares_curve, equity_curve, trades = run_backtest(
        dates=dates,
        close=close,
        signals=signals,
        initial_cash=args.initial_cash,
    )

    # Metrics
    tr = total_return(equity_curve)
    mdd = max_drawdown(equity_curve)
    sr = sharpe_ratio_from_equity(equity_curve)

    # Prints
    print("Hello Backtester")
    print(f"CSV: {csv_path}")
    print(f"Strategy: {strategy}")
    if strategy == "ma":
        print(f"MA windows: fast={args.fast}, slow={args.slow}")
    else:
        print(f"RSI: period={args.period}, buy_below={args.buy_below}, sell_above={args.sell_above}")

    print(f"Loaded rows: {len(close)}")
    print(f"Date range: {dates[0]} -> {dates[-1]}")
    print(f"Close min/max: {close.min():.2f} / {close.max():.2f}")

    print("\nSignals summary:")
    print(f"BUY signals:  {(signals == 1).sum()}")
    print(f"SELL signals: {(signals == -1).sum()}")

    print("\nBacktest summary:")
    print(f"Initial cash: {args.initial_cash:.2f}")
    print(f"Final equity: {equity_curve[-1]:.2f}")
    print(f"Trades: {len(trades)}")

    if trades:
        print("\nTrades:")
        for t in trades:
            print(f"{t.date} {t.side} price={t.price:.2f} shares={t.shares} cash_after={t.cash_after:.2f}")

    print("\nMetrics summary:")
    print(f"Total return: {tr * 100:.2f}%")
    print(f"Max drawdown: {mdd * 100:.2f}%")
    print(f"Sharpe ratio: {sr:.2f}")

    print("\nPreview (last 5 rows):")
    start = max(0, len(close) - 5)
    for i in range(start, len(close)):
        line = (
            f"{dates[i]} | Close={close[i]:.2f} | SMA(3)={sma_3[i]:.2f} | "
            f"Return={rets[i]:.4f} | Signal={signals[i]} | "
            f"Shares={shares_curve[i]} | Equity={equity_curve[i]:.2f}"
        )
        if strategy == "rsi":
            line += f" | RSI={r[i]:.2f}"
        print(line)

    # Plots
    if not args.no_plots:
        outputs_dir = project_root / "outputs"
        plot_price_with_signals(
            dates=dates,
            close=close,
            fast_sma=fast_sma,
            slow_sma=slow_sma,
            signals=signals,
            output_path=outputs_dir / "price_signals.png",
        )
        plot_equity_curve(
            dates=dates,
            equity=equity_curve,
            output_path=outputs_dir / "equity_curve.png",
        )
        print("\nSaved plots to:")
        print(outputs_dir)


if __name__ == "__main__":
    main()
