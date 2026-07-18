from fastapi.testclient import TestClient

from app.main import app


def test_health_and_prediction():
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model": "ready"}
        response = client.post(
            "/predict",
            json={
                "sp500": 5200,
                "oil_usd": 78,
                "silver_usd": 29,
                "usd_inr": 86,
                "interest_rate": 6.5,
                "inflation": 5.2,
            },
        )
        assert response.status_code == 200
        assert response.json()["predicted_gold_usd"] > 0


def test_invalid_input_is_rejected():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "sp500": -1,
                "oil_usd": 78,
                "silver_usd": 29,
                "usd_inr": 86,
                "interest_rate": 6.5,
                "inflation": 5.2,
            },
        )
        assert response.status_code == 422

