# Gold Price Prediction ML

An interview-ready machine-learning project that compares three regression models and serves the best model through a validated FastAPI endpoint.

> **Important:** The included dataset is deterministic synthetic data created for reproducible software and ML demonstrations. Outputs are educational estimates, not historical claims or financial advice.

## What this demonstrates

- Time-ordered train/test split to reduce future-data leakage
- Linear Regression, Decision Tree, and Random Forest comparison
- MAE, RMSE, and R-squared evaluation
- Scikit-learn pipelines for preprocessing and inference consistency
- Input validation and REST inference with FastAPI
- Automated tests, linting, Docker, and GitHub Actions CI

## Architecture

```text
Synthetic market generator -> CSV -> time split -> model comparison
                                               -> best artifact
Client -> FastAPI validation -> model pipeline -> prediction response
```

## Project structure

```text
app/                 FastAPI service
src/                 data generation, configuration, and training
tests/               data and API tests
data/                generated educational dataset
artifacts/           local model and metrics (gitignored)
.github/workflows/   continuous integration
```

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements-dev.txt
python -m src.generate_data
python -m src.train
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

## Example request

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"sp500":5200,"oil_usd":78,"silver_usd":29,"usd_inr":86,"interest_rate":6.5,"inflation":5.2}'
```

## Test and lint

```bash
ruff check .
pytest -q
```

## Docker

```bash
docker build -t gold-price-ml .
docker run -p 8000:8000 gold-price-ml
```

## Responsible interpretation

Market prediction is affected by regime changes, geopolitical events, currency shifts, and many variables not represented here. A real deployment would require licensed historical data, walk-forward validation, monitoring for drift, prediction intervals, and financial-domain review.

## Author

[Nishant Dharra](https://www.linkedin.com/in/nishantdharra) · [GitHub](https://github.com/Nishant6563)
