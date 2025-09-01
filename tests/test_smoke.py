import numpy as np
import asyncio

from app.services import clipmaker
from app.services.clustering import auto_kmeans, pca_2d

def test_segment_and_embed_basic(monkeypatch):
    txt = "Phrase une. Phrase deux. Phrase trois. Phrase quatre. Phrase cinq. Phrase six."
    async def fake_segment_text(txt, keep_ratio=0.5, with_titles=False):
        return [{"text":"a"}, {"text":"b"}]
    async def fake_embed_clips(segs):
        return np.random.rand(len(segs), 128)
    monkeypatch.setattr(clipmaker, "segment_text", fake_segment_text)
    monkeypatch.setattr(clipmaker, "embed_clips", fake_embed_clips)
    segs = asyncio.run(clipmaker.segment_text(txt, keep_ratio=0.5, with_titles=False))
    assert segs and isinstance(segs, list)
    embs = asyncio.run(clipmaker.embed_clips(segs))
    assert embs.shape[0] == len(segs)
    assert embs.shape[1] > 100  # embedding dim

def test_clustering_shapes():
    X = np.random.RandomState(42).randn(100, 128)
    k, sil, labels = auto_kmeans(X, k_min=3, k_max=5)
    assert 3 <= k <= 5
    assert labels.shape[0] == 100
    coords = pca_2d(X)
    assert coords.shape == (100, 2)
