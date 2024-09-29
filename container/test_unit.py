import pytest
from api import api
import json

@pytest.fixture
def client():
    return api.test_client()

def test_ping(client):
    res = client.get("/")
    assert res.status_code == 200
    # assert res.json == "Home Page."

def test_predict(client):
    data = {
    "year": 2028,
    "km_driven": 20000,
    "mileage": 18.5,
    "engine": 1497,
    "max_power": 100,
    "seats": 5,
    "company_name": "Mercedes-AMG",
    "seller_type": "Individual",
    "fuel_type": "Petrol",
    "transmission_type": "Manual"}
    res = client.post("/predict", json = data)
    assert res.status_code == 200
    assert res.json == {"Car Price": 6.17126}