from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class GeneratePreviewRequest(BaseModel):
    base: Optional[str] = None
    primary_key: str
    fields: List[str]

class GenerateRequest(BaseModel):
    base: Optional[str] = None
    primary_key: str
    fields: List[str]
    excel_token: Optional[str] = None
    include_transcripts: bool = False

class GenerateResult(BaseModel):
    raw_path: str
    emb_path: str
    count: int

class ClusterRequest(BaseModel):
    raw_path: str
    emb_path: str
    algorithm: str = "kmeans"  # kmeans|dbscan
    k_min: int = 6
    k_max: int = 12
    eps: float = 0.5
    min_samples: int = 5

class ClusterResult(BaseModel):
    result_path: str
    silhouette: Optional[float]
    n_clusters: int
