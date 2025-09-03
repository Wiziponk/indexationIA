import uuid

JOBS: dict[str, dict] = {}


def new_job() -> str:
    """Create a new job entry with a running state."""

    uid = uuid.uuid4().hex[:8]
    JOBS[uid] = {"state": "running", "progress": 0, "total": 0}
    return uid


def set_status(
    uid: str,
    state: str,
    *,
    progress: int | None = None,
    total: int | None = None,
    message: str | None = None,
) -> None:
    """Update job state, overwriting previous values."""

    if uid not in JOBS:
        return
    data: dict[str, object] = {"state": state}
    if progress is not None:
        data["progress"] = progress
    if total is not None:
        data["total"] = total
    if message is not None:
        data["message"] = message
    JOBS[uid] = data


def set_result(uid: str, payload: dict) -> None:
    if uid in JOBS:
        JOBS[uid] = {"state": "done", "result": payload}


def get_job(uid: str) -> dict | None:
    return JOBS.get(uid)
