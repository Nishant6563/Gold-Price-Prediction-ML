from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "sample_gold_market.csv"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

FEATURES = ["sp500", "oil_usd", "silver_usd", "usd_inr", "interest_rate", "inflation"]
TARGET = "gold_usd"

