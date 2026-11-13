import json, threading
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import select, delete
from .config import settings
from .db import AuditEvent, Face, JobEvent, Photo, ProcessingJob, SessionLocal, utcnow
from .vision import get_provider, image_hash, load_image, safe_json
from .clustering import cluster_faces
SUPPORTED={".jpg",".jpeg",".png",".webp",".heic",".heif"}

def _save_face_crop(image, bbox, face_id):
    x1,y1,x2,y2=[int(max(0,x)) for x in bbox]
    x2=min(image.width,x2); y2=min(image.height,y2)
    if x2<=x1 or y2<=y1: return None
    crop=image.crop((x1,y1,x2,y2))
    target=settings.storage_dir/"faces"/f"{face_id}.jpg"; target.parent.mkdir(parents=True,exist_ok=True); crop.save(target,"JPEG",quality=88)
    return str(target)


def _save_derivatives(image, photo_id):
    base=settings.storage_dir/"photos"; base.mkdir(parents=True,exist_ok=True)
    paths={}
    for name,size,quality in (("thumb",320,76),("medium",1200,82)):
        derivative=image.copy(); derivative.thumbnail((size,size))
        target=base/f"{photo_id}_{name}.jpg"; derivative.save(target,"JPEG",quality=quality,optimize=True); paths[name]=str(target)
    return paths


def _elapsed_ms(start, end):
    if start.tzinfo is None: start=start.replace(tzinfo=timezone.utc)
    return max(0,int((end-start).total_seconds()*1000))

def _transition(session, job, stage, status=None, message=None):
    now=utcnow()
    previous=session.scalar(select(JobEvent).where(JobEvent.job_id==job.id,JobEvent.ended_at.is_(None)).order_by(JobEvent.id.desc()))
    if previous:
        previous.ended_at=now; previous.duration_ms=_elapsed_ms(previous.started_at,now)
    job.stage=stage
    if status: job.status=status
    session.add(JobEvent(job_id=job.id,status=status or job.status,stage=stage,started_at=now,message=message))
    session.commit()

def _close_event(session, job):
    now=utcnow(); current=session.scalar(select(JobEvent).where(JobEvent.job_id==job.id,JobEvent.ended_at.is_(None)).order_by(JobEvent.id.desc()))
    if current:
        current.ended_at=now; current.duration_ms=_elapsed_ms(current.started_at,now); session.commit()

def run_job(job_id: int):
    session=SessionLocal(); job=session.get(ProcessingJob, job_id)
    try:
        _transition(session,job,"scanning","running"); provider=get_provider()
        target_files=set(json.loads(job.target_files_json or "[]"))
        files=[p for p in Path(job.source_dir).rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED]
        if target_files: files=[p for p in files if str(p) in target_files]
        job.total=len(files); session.commit(); _transition(session,job,"processing"); seen=set()
        for path in files:
            try:
                digest=image_hash(path)
                if digest in seen: job.processed+=1; continue
                seen.add(digest); photo=session.scalar(select(Photo).where(Photo.sha256==digest))
                force_reprocess = job.version > 1
                if photo and force_reprocess:
                    session.execute(delete(Face).where(Face.photo_id==photo.id, Face.pinned.is_(False)))
                    photo.status="processing"
                job.current_file=str(path); session.commit()
                if not photo:
                    image=load_image(path); photo=Photo(path=str(path),sha256=digest,width=image.width,height=image.height,status="processing"); session.add(photo); session.flush()
                if photo.status != "processed" or force_reprocess:
                    image=load_image(path); derivatives=_save_derivatives(image,photo.id); photo.thumb_path=derivatives["thumb"]; photo.medium_path=derivatives["medium"]; session.commit()
                    for detected in provider.detect_and_embed(image):
                        face=Face(photo_id=photo.id,bbox_json=safe_json(detected.bbox),embedding_json=safe_json(detected.embedding),detector_confidence=detected.confidence,quality_score=detected.quality)
                        session.add(face); session.flush(); face.crop_path=_save_face_crop(image,detected.bbox,face.id); job.faces_found+=1
                    photo.status="processed"
                job.processed+=1; session.commit()
            except Exception as exc:
                job.processed+=1; session.rollback(); job=session.get(ProcessingJob, job_id); job.failed_count=(job.failed_count or 0)+1; failed=json.loads(job.failed_files_json or "[]"); failed.append({"path":str(path),"error":str(exc)}); job.failed_files_json=json.dumps(failed,ensure_ascii=False); job.error=f"{path}: {exc}"; session.commit(); _transition(session,job,"file_failed",message=f"{path}: {exc}"); _transition(session,job,"processing")
        job=session.get(ProcessingJob, job_id); _transition(session,job,"clustering"); cluster_faces(session); job=session.get(ProcessingJob, job_id); final_status="completed_with_errors" if job.failed_count else "completed"; _transition(session,job,"completed",final_status); job.current_file=None; session.commit(); _close_event(session,job)
    except Exception as exc:
        session.rollback(); job=session.get(ProcessingJob, job_id); job.error=str(exc); session.commit(); _transition(session,job,"failed","failed",str(exc)); _close_event(session,job)
    finally: session.close()

def start_job(source_dir: str, rescan_mode: str = "new_version"):
    session=SessionLocal(); batch_key=str(Path(source_dir).resolve())
    prior=session.scalar(select(ProcessingJob).where(ProcessingJob.batch_key==batch_key).order_by(ProcessingJob.version.desc()))
    version=(prior.version+1) if prior else 1
    if rescan_mode == "replace_active":
        for old in session.scalars(select(ProcessingJob).where(ProcessingJob.batch_key==batch_key, ProcessingJob.active.is_(True))): old.active=False
    job=ProcessingJob(source_dir=source_dir,batch_key=batch_key,version=version,rescan_mode=rescan_mode,active=True); session.add(job); session.commit(); session.add(JobEvent(job_id=job.id,status="queued",stage="queued",started_at=utcnow())); session.commit(); job_id=job.id; session.close()
    threading.Thread(target=run_job,args=(job_id,),daemon=True).start(); return job_id

def start_retry_job(job_id: int, scope: str = "failed_files"):
    session=SessionLocal(); source=session.get(ProcessingJob,job_id)
    if not source: raise ValueError("job not found")
    if scope not in {"failed_files","batch"}: raise ValueError("scope must be failed_files or batch")
    targets=[] if scope=="batch" else [item["path"] for item in json.loads(source.failed_files_json or "[]")]
    job=ProcessingJob(source_dir=source.source_dir,batch_key=source.batch_key,version=source.version,rescan_mode="retry_failed" if scope=="failed_files" else "retry_batch",active=True,target_files_json=json.dumps(targets,ensure_ascii=False),retry_of_id=source.id)
    session.add(job); session.commit(); session.add(JobEvent(job_id=job.id,status="queued",stage="queued",started_at=utcnow(),message=f"retry of job {source.id}")); session.commit(); job_id=job.id; session.close(); threading.Thread(target=run_job,args=(job_id,),daemon=True).start(); return job_id
