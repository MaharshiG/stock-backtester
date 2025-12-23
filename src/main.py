from pathlib import Path
from src.data_loader import load_prices


def main():
    csv_path = Path("data") / "sample_prices.csv"
    dates, close = load_prices(csv_path)

    print("Hello Backtester")
    print(f"Loaded rows: {len(close)}")
    print(f"Date range: {dates[0]} -> {dates[-1]}")
    print(f"Close min/max: {close.min():.2f} / {close.max():.2f}")


if __name__ == "__main__":
    main()
