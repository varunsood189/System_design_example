from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_predict_endpoint():
    response = client.post("/predict", json={
        "transaction_id": "txn1",
        "user_id": "user123",
        "amount": 500.0,
        "merchant": "Ebay",
        "transaction_type": "purchase",
        "timestamp": "2025-05-18T10:00:00Z",
        "location": "US"
    })
    assert response.status_code == 200
    assert "fraud_score" in response.json()
