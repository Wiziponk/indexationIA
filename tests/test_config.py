import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indexation import config


def test_get_config_dev(monkeypatch):
    monkeypatch.setenv("APP_ENV", "dev")
    cfg = config.get_config()
    assert cfg.DEBUG is True
    assert cfg.MAX_CONTENT_LENGTH == config.MAX_CONTENT_LENGTH
    assert "csv" in cfg.ALLOWED_EXTENSIONS
    assert cfg.DATA_DIR.exists()
    assert cfg.RESULTS_DIR.exists()


def test_get_config_prod(monkeypatch):
    monkeypatch.setenv("APP_ENV", "prod")
    cfg = config.get_config()
    assert cfg.DEBUG is False
