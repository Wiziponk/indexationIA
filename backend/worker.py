from rq import Connection, Worker
from redis import Redis
from .config import settings

if __name__ == "__main__":
    redis = Redis.from_url(settings.REDIS_URL)
    with Connection(redis):
        worker = Worker(["default"])
        worker.work(with_scheduler=False)
