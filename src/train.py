import json

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

from src.config import ARTIFACT_DIR, DATA_PATH, FEATURES, METRICS_PATH, MODEL_PATH, TARGET


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        from src.generate_data import main as generate_data

        generate_data()
    frame = pd.read_csv(DATA_PATH, parse_dates=["date"]).sort_values("date")
    required = {"date", *FEATURES, TARGET}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    return frame


def candidates() -> dict[str, Pipeline]:
    return {
        "linear_regression": Pipeline(
            [("imputer", SimpleImputer()), ("scale", StandardScaler()), ("model", LinearRegression())]
        ),
        "decision_tree": Pipeline(
            [("imputer", SimpleImputer()), ("model", DecisionTreeRegressor(max_depth=8, random_state=42))]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=250, max_depth=12, random_state=42, n_jobs=-1
                    ),
                ),
            ]
        ),
    }


def train() -> dict:
    frame = load_data()
    split = int(len(frame) * 0.8)
    train_df, test_df = frame.iloc[:split], frame.iloc[split:]
    scores: dict[str, dict[str, float]] = {}
    fitted = {}

    for name, model in candidates().items():
        model.fit(train_df[FEATURES], train_df[TARGET])
        predictions = model.predict(test_df[FEATURES])
        scores[name] = {
            "mae": round(float(mean_absolute_error(test_df[TARGET], predictions)), 4),
            "rmse": round(float(mean_squared_error(test_df[TARGET], predictions) ** 0.5), 4),
            "r2": round(float(r2_score(test_df[TARGET], predictions)), 4),
        }
        fitted[name] = model

    best_name = min(scores, key=lambda name: scores[name]["rmse"])
    result = {
        "best_model": best_name,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "features": FEATURES,
        "models": scores,
        "data_notice": "Metrics use deterministic synthetic educational data.",
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[best_name], MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))

