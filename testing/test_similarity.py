import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_similarity_endpoint_success():
    payload = {
        "post_id": "test_1",
        "user_id": "user_123",
        "post_title": "How to implement a stack in Python?",
        "post_body": "I want to use array or list to implement a stack in Python. How do I do that?"
    }
    response = client.post("/similarity/check-duplicate", json=payload)
    assert response.status_code == 200
    assert "matched_post" in response.json()
    assert "similarity_score" in response.json()
    assert "is_duplicate" in response.json()

def test_similarity_empty_fields():
    payload = {
        "post_id": "test_2",
        "user_id": "user_123",
        "post_title": "",
        "post_body": ""
    }
    response = client.post("/similarity/check-duplicate", json=payload)
    assert response.status_code == 200
    assert isinstance(response.json().get("similarity_score"), float)

def test_similarity_missing_title():
    payload = {
        "post_id": "test_3",
        "user_id": "user_123",
        "post_body": "Only body is provided."
    }
    response = client.post("/similarity/check-duplicate", json=payload)
    assert response.status_code == 422

def test_similarity_special_characters():
    payload = {
        "post_id": "test_4",
        "user_id": "user_123",
        "post_title": "@#$%^&*()_+=-[]{}|;':,./<>?",
        "post_body": "@#$%^&*()_+=-[]{}|;':,./<>?"
    }
    response = client.post("/similarity/check-duplicate", json=payload)
    assert response.status_code == 200
    assert "matched_post" in response.json()
