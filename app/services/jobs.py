import asyncio, uuid
from pathlib import Path

_jobs: dict[str, dict] = {}  # {uid: {"status": "pending|running|done|error", "note": str, "result": dict}}

def new_job():
    uid = uuid.uuid4().hex[:8]
    _jobs[uid] = {"status": "pending", "note": "", "result": {}}
    return uid

def set_status(uid, status, note=""):
    if uid in _jobs: _jobs[uid].update({"status": status, "note": note})

def set_result(uid, payload):
    if uid in _jobs: _jobs[uid].update({"status": "done", "result": payload})

def get_job(uid):
    return _jobs.get(uid, {"status": "unknown", "note": "no such job"})
