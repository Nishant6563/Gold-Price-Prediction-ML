"""Generate deterministic educational market data for a reproducible demo.

The data is synthetic and must not be interpreted as historical market data.
"""

import numpy as np
import pandas as pd

from src.config import DATA_PATH


def generate(rows: int = 720, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2015-01-01", periods=rows, freq="W")
    trend = np.linspace(0, 1, rows)
    sp500 = 2050 + 3200 * trend + rng.normal(0, 90, rows)
    oil = 58 + 20 * np.sin(np.arange(rows) / 30) + rng.normal(0, 5, rows)
    silver = 16 + 12 * trend + rng.normal(0, 1.1, rows)
    usd_inr = 62 + 25 * trend + rng.normal(0, 1.0, rows)
    interest = 7.5 - 2.4 * trend + rng.normal(0, 0.25, rows)
    inflation = 4.8 + 1.0 * np.sin(np.arange(rows) / 18) + rng.normal(0, 0.35, rows)
    noise = rng.normal(0, 35, rows)
    gold = (
        210
        + 0.12 * sp500
        - 2.1 * oil
        + 31 * silver
        + 7.4 * usd_inr
        - 32 * interest
        + 22 * inflation
        + noise
    )
    return pd.DataFrame(
        {
            "date": dates,
            "sp500": sp500.round(2),
            "oil_usd": oil.round(2),
            "silver_usd": silver.round(2),
            "usd_inr": usd_inr.round(2),
            "interest_rate": interest.round(2),
            "inflation": inflation.round(2),
            "gold_usd": gold.round(2),
        }
    )


def main() -> None:
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = generate()
    data.to_csv(DATA_PATH, index=False)
    print(f"Created {len(data)} rows at {DATA_PATH}")


if __name__ == "__main__":
    main()

