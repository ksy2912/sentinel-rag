from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_documents_list():
    r = client.get("/documents")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_ask_requires_body():
    r = client.post("/ask", json={})
    assert r.status_code == 422


def test_retrieve_requires_body():
    r = client.post("/retrieve", json={})
    assert r.status_code == 422
