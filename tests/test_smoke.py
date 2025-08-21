import numpy as np

from app.services.clipmaker import segment_text, embed_clips
from app.services.clustering import auto_kmeans, pca_2d

def test_segment_and_embed_basic():
    txt = "Phrase une. Phrase deux. Phrase trois. Phrase quatre. Phrase cinq. Phrase six."
    segs = segment_text(txt, keep_ratio=0.5, with_titles=False)
    assert segs and isinstance(segs, list)
    embs = embed_clips(segs)
    assert embs.shape[0] == len(segs)
    assert embs.shape[1] > 100  # embedding dim

def test_clustering_shapes():
    X = np.random.RandomState(42).randn(100, 128)
    k, sil, labels = auto_kmeans(X, k_min=3, k_max=5)
    assert 3 <= k <= 5
    assert labels.shape[0] == 100
    coords = pca_2d(X)
    assert coords.shape == (100, 2)
