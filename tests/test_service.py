from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_basic():
    r = client.post("/predict", json={"text": "Refund failed twice; card charged."})
    assert r.status_code == 200
    data = r.json()
    assert set(data.keys()) == {"topic", "sentiment", "probs", "latency_ms", "model_version"}
    assert data["topic"]["label"] in {"billing", "bug", "login", "feature", "shipping", "other"}
