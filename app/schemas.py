from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    detail: str


class FieldsResponse(BaseModel):
    fields: List[str]
    api_base: str
    source: str
    sample_size: int
    note: Optional[str] = None


class SegmentPrepareResponse(BaseModel):
    primary_key: str
    count_included: int
    count_excluded: int
    excluded_ids: List[str]
    sample_ids: List[str]


class ClipSegment(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    text: Optional[str] = None


class SegmentPreviewResponse(BaseModel):
    pk_value: str
    segments: List[ClipSegment]
    n_segments: int
    clip_dim: int
    has_program_embedding: bool


class NamedSegmentsResponse(BaseModel):
    segments: List[ClipSegment]


class BatchLaunchResponse(BaseModel):
    uid: str
    status: str


class ZipInfo(BaseModel):
    programme_id: str
    path: str


class BatchResult(BaseModel):
    uid: str
    count: int
    master_zip: str
    zips: List[ZipInfo]


class StatusResponse(BaseModel):
    status: Literal["running", "done", "error", "not_found"]
    message: Optional[str] = None
    progress: Optional[int] = None
    total: Optional[int] = None
    result: Optional[BatchResult] = None


class ClusterMeta(BaseModel):
    k: int
    silhouette: float
    points: Optional[int] = None
    algo: Optional[str] = None
    projection: Optional[str] = None
    cluster_names: Dict[str, Any]


class ClusterPoint(BaseModel):
    id: str
    title: str
    cluster: int
    x: float
    y: float


class ClusterDownload(BaseModel):
    parquet: str
    embeddings: str


class ClusterResponse(BaseModel):
    meta: ClusterMeta
    points: List[ClusterPoint]
    download: ClusterDownload


class ClusterClipPoint(BaseModel):
    pk: str
    clip_index: int
    title: Optional[str] = None
    cluster: int
    x: float
    y: float


class ClusterEmissionPoint(BaseModel):
    pk: str
    cluster: int
    x: float
    y: float


class ClusterZipResponse(BaseModel):
    points: List[Dict[str, Any]]
    meta: ClusterMeta
    download: Dict[str, Any]


class DatasetInfo(BaseModel):
    uid: str
    raw_path: str
    emb_path: str
    created_at: Optional[str] = None
    label: Optional[str] = None
    config: Dict[str, Any]


class ProjectInfo(BaseModel):
    id: int
    name: str
    created_at: str
    primary_key: str
    embed_fields: List[str]
    keep_ratio: float
    with_titles: bool
    brief: Optional[str]
    mode: str
    excel_id_col: Optional[str]
    programs: int


class ProjectsResponse(BaseModel):
    projects: List[ProjectInfo]


class ProgramSummary(BaseModel):
    id: int
    pk_value: str
    num_clips: int
    last_zip_path: Optional[str]


class ProgramsResponse(BaseModel):
    programs: List[ProgramSummary]


class ClipInfo(BaseModel):
    id: int
    idx: int
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    title: Optional[str] = None
    summary: Optional[str] = None
    text: Optional[str] = None


class ProgramDetail(BaseModel):
    id: int
    project_id: int
    pk_value: str
    fields_json: Dict[str, Any]
    transcript_name: Optional[str] = None
    num_clips: int
    last_zip_path: Optional[str] = None


class ProgramDetailResponse(BaseModel):
    program: ProgramDetail
    clips: List[ClipInfo]


class ClipUpdateResponse(BaseModel):
    ok: bool
    clip: Dict[str, Any]


class ProgramRerunResponse(BaseModel):
    ok: bool
    program_id: int
    num_clips: int
    zip: str
