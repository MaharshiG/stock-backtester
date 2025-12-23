from __future__ import annotations

from pathlib import Path
import numpy as np


def load_prices(csv_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load prices from a CSV with columns: Date, Close

    Returns:
        dates: np.ndarray of dtype '<U..' (strings)
        close: np.ndarray of dtype float
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    # Read two columns (Date, Close), skip header row
    data = np.genfromtxt(
        csv_path,
        delimiter=",",
        dtype=None,           # infer types
        encoding="utf-8",
        names=True,           # use header row as names
        autostrip=True,
    )

    # Handle case: file has only 1 data row (genfromtxt returns scalar record)
    if data.shape == ():
        dates = np.array([data["Date"]], dtype=str)
        close = np.array([data["Close"]], dtype=float)
    else:
        dates = data["Date"].astype(str)
        close = data["Close"].astype(float)

    # Basic validation
    if dates.size == 0 or close.size == 0:
        raise ValueError("CSV appears empty or missing required columns Date,Close")

    if np.any(np.isnan(close)):
        raise ValueError("Close column contains NaN values. Please clean the CSV.")

    # Sort by date string (works for YYYY-MM-DD)
    order = np.argsort(dates)
    dates = dates[order]
    close = close[order]

    return dates, close
