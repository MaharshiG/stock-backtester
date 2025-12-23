from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Trade:
    index: int
    date: str
    side: str  # "BUY" or "SELL"
    price: float
    shares: int
    cash_after: float


def run_backtest(
    dates: np.ndarray,
    close: np.ndarray,
    signals: np.ndarray,
    initial_cash: float = 10_000.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Trade]]:
    """
    Simple all-in/all-out backtest.

    Rules:
      - BUY (+1): invest all cash into whole shares at today's close
      - SELL (-1): sell all shares at today's close
      - HOLD (0): do nothing

    Returns:
      cash_curve:   cash balance per day
      shares_curve: shares held per day
      equity_curve: cash + shares * close per day
      trades:       list of Trade records
    """
    if len(dates) != len(close) or len(close) != len(signals):
        raise ValueError("dates, close, and signals must have the same length")

    close = close.astype(float)
    signals = signals.astype(int)

    n = len(close)
    cash_curve = np.zeros(n, dtype=float)
    shares_curve = np.zeros(n, dtype=int)
    equity_curve = np.zeros(n, dtype=float)

    cash = float(initial_cash)
    shares = 0
    trades: list[Trade] = []

    for i in range(n):
        price = float(close[i])
        sig = int(signals[i])

        # Execute at close price of the same day
        if sig == 1 and shares == 0:
            # buy whole shares
            buy_shares = int(cash // price)
            if buy_shares > 0:
                cash = cash - buy_shares * price
                shares = buy_shares
                trades.append(
                    Trade(
                        index=i,
                        date=str(dates[i]),
                        side="BUY",
                        price=price,
                        shares=buy_shares,
                        cash_after=cash,
                    )
                )

        elif sig == -1 and shares > 0:
            cash = cash + shares * price
            trades.append(
                Trade(
                    index=i,
                    date=str(dates[i]),
                    side="SELL",
                    price=price,
                    shares=shares,
                    cash_after=cash,
                )
            )
            shares = 0

        cash_curve[i] = cash
        shares_curve[i] = shares
        equity_curve[i] = cash + shares * price

    return cash_curve, shares_curve, equity_curve, trades
