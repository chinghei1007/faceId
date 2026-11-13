from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from .config import settings

class Base(DeclarativeBase): pass

def utcnow(): return datetime.now(timezone.utc)

class Photo(Base):
    __tablename__ = "photos"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    path: Mapped[str] = mapped_column(Text, unique=True, index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    width: Mapped[int] = mapped_column(Integer, nullable=True)
    height: Mapped[int] = mapped_column(Integer, nullable=True)
    thumb_path: Mapped[str] = mapped_column(Text, nullable=True)
    medium_path: Mapped[str] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="pending")
    error: Mapped[str] = mapped_column(Text, nullable=True)
    caption: Mapped[str] = mapped_column(Text, nullable=True)
    caption_status: Mapped[str] = mapped_column(String(24), default="not_requested")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class Person(Base):
    __tablename__ = "people"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    public_id: Mapped[str] = mapped_column(String(32), unique=True, index=True)
    label: Mapped[str] = mapped_column(String(128))
    is_unknown: Mapped[bool] = mapped_column(Boolean, default=False)
    merged_into_id: Mapped[int] = mapped_column(ForeignKey("people.id"), nullable=True)
    prototype_json: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class Face(Base):
    __tablename__ = "faces"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    photo_id: Mapped[int] = mapped_column(ForeignKey("photos.id"), index=True)
    bbox_json: Mapped[str] = mapped_column(Text)
    crop_path: Mapped[str] = mapped_column(Text, nullable=True)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=True)
    detector_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    quality_score: Mapped[float] = mapped_column(Float, default=0.0)
    cluster_probability: Mapped[float] = mapped_column(Float, default=0.0)
    similarity_score: Mapped[float] = mapped_column(Float, default=0.0)
    assignment_state: Mapped[str] = mapped_column(String(16), default="unknown")
    ignored: Mapped[bool] = mapped_column(Boolean, default=False)
    person_id: Mapped[int] = mapped_column(ForeignKey("people.id"), index=True, nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    label_override: Mapped[str] = mapped_column(String(128), nullable=True)
    alternatives_json: Mapped[str] = mapped_column(Text, nullable=True)

class ProcessingJob(Base):
    __tablename__ = "processing_jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_dir: Mapped[str] = mapped_column(Text)
    batch_key: Mapped[str] = mapped_column(String(512), index=True, default="")
    version: Mapped[int] = mapped_column(Integer, default=1)
    rescan_mode: Mapped[str] = mapped_column(String(24), default="new_version")
    active: Mapped[bool] = mapped_column(Boolean, default=True)
    status: Mapped[str] = mapped_column(String(24), default="queued")
    stage: Mapped[str] = mapped_column(String(32), default="queued")
    current_file: Mapped[str] = mapped_column(Text, nullable=True)
    failed_count: Mapped[int] = mapped_column(Integer, default=0)
    failed_files_json: Mapped[str] = mapped_column(Text, default="[]")
    target_files_json: Mapped[str] = mapped_column(Text, default="[]")
    retry_of_id: Mapped[int] = mapped_column(Integer, nullable=True)
    hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    total: Mapped[int] = mapped_column(Integer, default=0)
    processed: Mapped[int] = mapped_column(Integer, default=0)
    faces_found: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

class StudentDefinition(Base):
    __tablename__ = "student_definitions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_path: Mapped[str] = mapped_column(Text)
    student_key: Mapped[str] = mapped_column(String(32), index=True)
    class_code: Mapped[str] = mapped_column(String(16))
    student_number: Mapped[int] = mapped_column(Integer)
    name: Mapped[str] = mapped_column(String(128))
    label: Mapped[str] = mapped_column(String(128))
    reference_path: Mapped[str] = mapped_column(Text, nullable=True)
    embedding_json: Mapped[str] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(24), default="unused")
    matched_person_id: Mapped[int] = mapped_column(ForeignKey("people.id"), nullable=True)
    match_score: Mapped[float] = mapped_column(Float, nullable=True)
    warning: Mapped[str] = mapped_column(Text, nullable=True)
    disabled: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

class JobEvent(Base):
    __tablename__ = "job_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    job_id: Mapped[int] = mapped_column(ForeignKey("processing_jobs.id"), index=True)
    status: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str] = mapped_column(String(32))
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    ended_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=True)

class AuditEvent(Base):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    action: Mapped[str] = mapped_column(String(64))
    face_id: Mapped[int] = mapped_column(Integer, nullable=True)
    person_id: Mapped[int] = mapped_column(Integer, nullable=True)
    payload_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

engine = create_engine(f"sqlite:///{(settings.data_dir / 'faceid.sqlite3').resolve()}", connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)

def init_db():
    Base.metadata.create_all(engine)
    # Lightweight local migration for databases created by earlier faceId builds.
    with engine.begin() as conn:
        cols={row[1] for row in conn.exec_driver_sql("PRAGMA table_info(processing_jobs)").fetchall()}
        additions={"batch_key":"VARCHAR(512) DEFAULT ''","version":"INTEGER DEFAULT 1","rescan_mode":"VARCHAR(24) DEFAULT 'new_version'","active":"BOOLEAN DEFAULT 1"}
        for name,definition in additions.items():
            if name not in cols: conn.exec_driver_sql(f"ALTER TABLE processing_jobs ADD COLUMN {name} {definition}")
        photo_cols={row[1] for row in conn.exec_driver_sql("PRAGMA table_info(photos)").fetchall()}
        for name,definition in {"thumb_path":"TEXT","medium_path":"TEXT"}.items():
            if name not in photo_cols: conn.exec_driver_sql(f"ALTER TABLE photos ADD COLUMN {name} {definition}")
        job_cols={row[1] for row in conn.exec_driver_sql("PRAGMA table_info(processing_jobs)").fetchall()}
        for name,definition in {"stage":"VARCHAR(32) DEFAULT 'queued'","current_file":"TEXT","failed_count":"INTEGER DEFAULT 0","failed_files_json":"TEXT DEFAULT '[]'","target_files_json":"TEXT DEFAULT '[]'","retry_of_id":"INTEGER","hidden":"BOOLEAN DEFAULT 0"}.items():
            if name not in job_cols: conn.exec_driver_sql(f"ALTER TABLE processing_jobs ADD COLUMN {name} {definition}")
        conn.exec_driver_sql("UPDATE processing_jobs SET stage='completed' WHERE status='completed' AND stage='queued'")
        definition_cols={row[1] for row in conn.exec_driver_sql("PRAGMA table_info(student_definitions)").fetchall()}
        if "disabled" not in definition_cols: conn.exec_driver_sql("ALTER TABLE student_definitions ADD COLUMN disabled BOOLEAN DEFAULT 0")
        people_cols={row[1] for row in conn.exec_driver_sql("PRAGMA table_info(people)").fetchall()}
        if "prototype_json" not in people_cols: conn.exec_driver_sql("ALTER TABLE people ADD COLUMN prototype_json TEXT")

