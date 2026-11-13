from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

class ImportRequest(BaseModel):
    source_dir: str
    rescan_mode: str = "new_version"
class RetryRequest(BaseModel):
    scope: str = "failed_files"
class DefinitionParseRequest(BaseModel):
    pdf_path: str
    mode: str = "new_version"
class DefinitionOut(BaseModel):
    id: int; source_path: str; student_key: str; class_code: str; student_number: int; name: str; label: str; status: str; matched_person_id: Optional[int] = None; match_score: Optional[float] = None; warning: Optional[str] = None; disabled: bool = False
class ReassignRequest(BaseModel): person_id: Optional[int] = None; create_label: Optional[str] = None
class RenameRequest(BaseModel): label: str = Field(min_length=1, max_length=128)
class MergeRequest(BaseModel): target_person_id: int
class SplitRequest(BaseModel): face_ids: list[int] = Field(min_length=1)
class FaceOut(BaseModel):
    id: int; photo_id: int; person_id: Optional[int]; person_label: Optional[str]; bbox: list[float]
    detector_confidence: float; quality_score: float; cluster_probability: float; similarity_score: float; confidence: float
    assignment_state: str; photo_path: Optional[str] = None; photo_width: Optional[int] = None; photo_height: Optional[int] = None; photo_url: Optional[str] = None; pinned: bool; ignored: bool; crop_url: Optional[str] = None
class PersonOut(BaseModel): id: int; public_id: str; label: str; is_unknown: bool; face_count: int; photo_count: int; average_confidence: float; cover_url: Optional[str] = None
class PhotoOut(BaseModel):
    id: int; path: str; width: Optional[int] = None; height: Optional[int] = None; face_count: int; photo_url: str; thumb_url: Optional[str] = None; medium_url: Optional[str] = None

class JobEventOut(BaseModel): id: int; job_id: int; status: str; stage: str; started_at: datetime; ended_at: Optional[datetime] = None; duration_ms: Optional[int] = None; message: Optional[str] = None

class JobOut(BaseModel): id: int; source_dir: str; batch_key: str; version: int; rescan_mode: str; active: bool; status: str; stage: str; current_file: Optional[str]; failed_count: int; retry_of_id: Optional[int] = None; hidden: bool = False; attempts: int = 1; total: int; processed: int; faces_found: int; error: Optional[str]; created_at: datetime
