import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from indexation.services import preview_cache
from indexation.services.clustering import auto_kmeans


def test_auto_kmeans_detects_clusters():
    rng = np.random.default_rng(0)
    X1 = rng.normal(loc=0.0, scale=0.1, size=(10, 2))
    X2 = rng.normal(loc=5.0, scale=0.1, size=(10, 2))
    X = np.vstack([X1, X2])
    k, sil, labels = auto_kmeans(X, k_min=2, k_max=3)
    assert k == 2
    assert len(labels) == 20


def test_preview_cache_cleanup(tmp_path):
    token = "tok"
    f = tmp_path / "tmp.txt"
    f.write_text("hi")
    preview_cache.register_preview(token, f)
    assert token in preview_cache.PREVIEW_INDEX
    preview_cache.cleanup_previews(max_age=-1)
    assert token not in preview_cache.PREVIEW_INDEX
    assert not f.exists()
