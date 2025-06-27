import sys
import os
import pytest
from fastapi.testclient import TestClient

# Make sure the project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import app

client = TestClient(app)


def test_generate_tags_success():
    payload = {"text": "How do I implement a stack using arrays in Python?"}
    response = client.post("/tags/generate-tags", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_tags" in data
    assert isinstance(data["predicted_tags"], list)
    assert len(data["predicted_tags"]) >= 1


def test_generate_tags_empty_text():
    payload = {"text": ""}
    response = client.post("/tags/generate-tags", json=payload)
    assert response.status_code == 200  # Your app may return 200 with fallback tags
    data = response.json()
    assert "predicted_tags" in data
    assert isinstance(data["predicted_tags"], list)


def test_generate_tags_missing_field():
    payload = {}  # no "text" key
    response = client.post("/tags/generate-tags", json=payload)
    assert response.status_code == 422  # Validation error from FastAPI


def test_generate_tags_special_characters():
    payload = {"text": "@#$%^&*()_+{}|:\"<>?"}
    response = client.post("/tags/generate-tags", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "predicted_tags" in data
    assert isinstance(data["predicted_tags"], list)
