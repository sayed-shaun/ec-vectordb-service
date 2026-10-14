"""End-to-end HTTP tests against the FastAPI app with fakes injected."""

import pytest
from fastapi.testclient import TestClient

from main import app
from src.core.config import MODEL_SPECS
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
    assert set(body["models"]) == {s.name for s in MODEL_SPECS}


@pytest.fixture
def ec_client(service):
    from src.api.deps import get_ec_service
    from src.ec_service import EcBotService
    from src.schemas import Document
    from tests.test_ec import CORPUS

    service.ingest([Document(doc_id=q, text=t, metadata={"tag": g}) for q, t, g in CORPUS])
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_ec_service] = lambda: EcBotService(service)
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_top_similar_endpoint_returns_ranked_hits(ec_client):
    res = ec_client.post(
        "/ec_bot/top_similar/", json={"question": "নতুন ভোটার হতে কি কি লাগে?", "top_k": 3}
    )
    assert res.status_code == 200
    hits = res.json()["top_similar"]
    assert len(hits) == 3
    assert [h["rank"] for h in hits] == [1, 2, 3]
    assert hits[0]["tag"] == "new_voter_docs"


def test_top_similar_rejects_empty_question(ec_client):
    assert ec_client.post("/ec_bot/top_similar/", json={"question": ""}).status_code == 422


def test_top_similar_rejects_out_of_range_top_k(ec_client):
    for k in (0, 101):
        res = ec_client.post("/ec_bot/top_similar/", json={"question": "ভোটার", "top_k": k})
        assert res.status_code == 422, k


def test_top_similar_unknown_model_is_a_400(ec_client):
    res = ec_client.post(
        "/ec_bot/top_similar/", json={"question": "ভোটার", "models": ["nope"]}
    )
    assert res.status_code == 400


def test_public_route_set_is_exactly_the_supported_endpoints():
    """A guard, not a formality: /documents and /documents/delete were once
    unauthenticated write and delete routes that no caller used. Read from the
    OpenAPI schema, which is stable across FastAPI's internal route wrapping."""
    routes = {
        (method.upper(), path)
        for path, ops in app.openapi()["paths"].items()
        for method in ops
    }
    assert routes == {
        ("POST", "/ec_bot/top_similar/"),
        ("GET", "/health"),
    }


def test_removed_routes_are_gone(client):
    for path in ("/qa", "/documents", "/documents/delete", "/query", "/warmup"):
        assert client.post(path, json={}).status_code == 404, path
