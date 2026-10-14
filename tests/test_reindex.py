import pytest
from fastapi.testclient import TestClient

from main import app
from src.api.deps import get_reindexer, get_service
from src.core.config import settings
from src.ec import load_questions
from src.reindex import Reindexer
from src.schemas import Document


def csv_fetcher(rows):
    """A fetch stand-in that writes ``rows`` the way the real one would."""

    def fetch(dest):
        dest.write_text(
            "question,tag\n" + "".join(f"{q},{t}\n" for q, t in rows), encoding="utf-8-sig"
        )
        return len(rows)

    return fetch


ROWS = [("নতুন ভোটার হতে কি লাগে", "voter"), ("একাউন্ট লক হয়েছে", "locked")]


@pytest.fixture
def dest(tmp_path):
    return tmp_path / "question_tag.csv"


def test_a_run_indexes_every_fetched_row(service, dest):
    job = Reindexer(service, fetch=csv_fetcher(ROWS), dest=dest)
    job.run()

    status = job.status()
    assert status.status == "succeeded"
    assert status.total == status.indexed == len(ROWS)
    assert service.store.count() == len(ROWS)


def test_stale_points_are_pruned_after_indexing(service, dest):
    service.ingest([Document(doc_id="q999999", text="পুরনো প্রশ্ন", metadata={"tag": "old"})])
    job = Reindexer(service, fetch=csv_fetcher(ROWS), dest=dest)
    job.run()

    assert job.status().pruned == 1
    assert service.store.count() == len(ROWS)


def test_a_failed_fetch_leaves_the_index_untouched(service, dest):
    service.ingest([Document(doc_id="keep", text="বিদ্যমান", metadata={"tag": "t"})])

    def broken(_dest):
        raise RuntimeError("HTTP Error 401: Unauthorized")

    job = Reindexer(service, fetch=broken, dest=dest)
    job.run()

    status = job.status()
    assert status.status == "failed"
    assert "401" in status.error
    assert service.store.count() == 1


def test_an_empty_dataset_never_wipes_the_index(service, dest):
    service.ingest([Document(doc_id="keep", text="বিদ্যমান", metadata={"tag": "t"})])
    job = Reindexer(service, fetch=csv_fetcher([]), dest=dest)
    job.run()

    assert job.status().status == "failed"
    assert service.store.count() == 1


def test_begin_claims_the_job_once(service, dest):
    job = Reindexer(service, fetch=csv_fetcher(ROWS), dest=dest)
    assert job.begin() is True
    assert job.status().status == "running"
    assert job.begin() is False


@pytest.fixture
def api(service, dest):
    job = Reindexer(service, fetch=csv_fetcher(ROWS), dest=dest)
    app.dependency_overrides[get_service] = lambda: service
    app.dependency_overrides[get_reindexer] = lambda: job
    yield TestClient(app), job
    app.dependency_overrides.clear()


def test_post_returns_202_and_get_reports_progress(api):
    client, _ = api
    res = client.post("/reindex")
    assert res.status_code == 202
    assert res.json()["status"] == "running"
    body = client.get("/reindex").json()
    assert body["status"] == "succeeded"
    assert body["indexed"] == len(ROWS)


def test_post_while_running_is_a_409(api):
    client, job = api
    job.begin()
    assert client.post("/reindex").status_code == 409


def test_token_is_required_when_configured(api, monkeypatch):
    client, _ = api
    monkeypatch.setattr(settings, "REINDEX_TOKEN", "s3cret")
    assert client.post("/reindex").status_code == 401
    assert client.post("/reindex", headers={"X-Reindex-Token": "nope"}).status_code == 401
    assert client.get("/reindex").status_code == 401
    ok = client.post("/reindex", headers={"X-Reindex-Token": "s3cret"})
    assert ok.status_code == 202


def test_fetch_keeps_the_previous_copy_when_the_download_is_unusable(tmp_path, monkeypatch):
    import io
    import urllib.request

    from src.dataset import fetch_dataset

    dest = tmp_path / "question_tag.csv"
    dest.write_text("question,tag\nقديم,old\n", encoding="utf-8")

    class Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: Resp(b"question,tag\n"))
    with pytest.raises(ValueError):
        fetch_dataset(dest)
    assert [r.question for r in load_questions(dest)] == ["قديم"]
