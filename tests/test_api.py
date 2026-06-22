from fastapi.testclient import TestClient
import pytest
from src.devops_ml.app import app

# Create a TestClient instance to interact with the API
client = TestClient(app)

def test_health_check():
    """Test that the root GET endpoint returns a 200 status and healthy message."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "model": "ONNX Housing Model"}

def test_successful_prediction():
    """Test a valid 2-feature prediction request."""
    payload = {
        "num_rooms": 3.0,
        "square_feet": 1200.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "estimated_price" in response.json()
    assert isinstance(response.json()["estimated_price"], float)

def test_invalid_prediction_missing_field():
    """Test that a missing field triggers a 422 error."""
    payload = {
        "num_rooms": 3.0
        # square_feet is missing
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422