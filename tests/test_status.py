import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.routers.clip import router as clip_router
from app.services import jobs


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(clip_router)
    return TestClient(app)


def setup_function():
    jobs.JOBS.clear()


def test_status_not_found(client):
    res = client.get("/segment/status/unknown")
    assert res.status_code == 200
    assert res.json()["status"] == "not_found"
    assert "result" not in res.json()


def test_status_running(client):
    jobs.JOBS["abc"] = {"state": "running", "progress": 2, "total": 10}
    res = client.get("/segment/status/abc")
    data = res.json()
    assert res.status_code == 200
    assert data == {
        "status": "running",
        "progress": 2,
        "total": 10,
    }


def test_status_error(client):
    jobs.JOBS["abc"] = {"state": "error", "message": "boom"}
    res = client.get("/segment/status/abc")
    data = res.json()
    assert res.status_code == 200
    assert data == {"status": "error", "message": "boom"}


def test_status_done(client):
    jobs.JOBS["abc"] = {
        "state": "done",
        "result": {
            "uid": "abc",
            "count": 1,
            "master_zip": "/api/download/zips/abc.zip",
            "zips": [
                {
                    "programme_id": "p1",
                    "path": "/api/download/zips/abc/p1.zip",
                }
            ],
        },
    }
    res = client.get("/segment/status/abc")
    data = res.json()
    assert res.status_code == 200
    assert data["status"] == "done"
    assert data["result"]["uid"] == "abc"
    assert data["result"]["count"] == 1
