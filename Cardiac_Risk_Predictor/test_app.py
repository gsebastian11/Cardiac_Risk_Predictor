import json
import pytest
import requests
from app import app

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_health_endpoint(client):
    response = client.get('/health')
    data = json.loads(response.data.decode('utf-8'))

    assert response.status_code == 200
    assert data['status'] == 'healthy'

def test_predict_endpoint(client):
    payload = json.dumps([
      {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
      }
    ])
    headers = {'Content-Type': 'application/json'}

    response = client.post('/predict', data=payload, headers=headers)
    data = json.loads(response.data.decode('utf-8'))

    assert response.status_code == 200
    assert 'prediction_result' in data
    assert data['prediction_result'] == "[1]"