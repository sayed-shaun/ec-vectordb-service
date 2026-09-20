"""End-to-end HTTP tests against the FastAPI app with fakes injected."""

import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.deps import get_service


@pytest.fixture
def client(service):
    app.dependency_overrides[get_service] = lambda: service
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_health_reports_store_and_models(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["store"]["exists"] is True
    assert set(body["models"]) == {"e5", "qwen3", "bge_m3"}


@pytest.fixture
def ec_client(service):
    from src.api.deps import get_ec_service
    from src.ec import AnswerStore
    from src.ec_service import EcBotService
    from src.schemas import Document
    from tests.test_ec import ANSWERS, CORPUS

    service.ingest([Document(doc_id=q, text=t, metadata={"tag": g}) for q, t, g in CORPUS])
    ec = EcBotService(service, AnswerStore(ANSWERS))
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_ec_service] = lambda: ec
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_qa_endpoint_returns_an_answer(ec_client):
    res = ec_client.post("/qa", json={"question": "নতুন ভোটার হতে কি কি লাগে?", "min_cosine": 0.0})
    assert res.status_code == 200
    body = res.json()
    assert body["confident"] is True
    assert body["tag"] == "new_voter_docs"
    assert body["answer"]
    assert set(body["per_model_tag"]) == {"e5", "qwen3", "bge_m3"}


def test_qa_endpoint_rejects_empty_question(ec_client):
    assert ec_client.post("/qa", json={"question": ""}).status_code == 422


def test_public_route_set_is_exactly_the_four_supported_endpoints():
    """A guard, not a formality: /documents and /documents/delete were once
    unauthenticated write and delete routes that no caller used."""
    routes = {
        (method, r.path)
        for r in app.routes
        if getattr(r, "methods", None)
        for method in r.methods
        if method not in ("HEAD", "OPTIONS") and not r.path.startswith(("/docs", "/redoc", "/openapi"))
    }
    assert routes == {
        ("POST", "/qa"),
        ("POST", "/ec_bot/top_similar/"),
        ("GET", "/ec_bot/audit"),
        ("GET", "/health"),
    }


def test_removed_write_routes_are_gone(client):
    for path in ("/documents", "/documents/delete", "/query", "/warmup"):
        assert client.post(path, json={}).status_code == 404, path
