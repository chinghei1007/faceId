import json
from pathlib import Path
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from .config import settings
from .db import AuditEvent, Base, Face, JobEvent, Person, Photo, ProcessingJob, SessionLocal, StudentDefinition, init_db, utcnow
from .pipeline import start_job, start_retry_job, _save_derivatives
from .vision import load_image
from .definitions import import_student_definitions, parse_student_pdf
from .captioning import get_caption_provider
from .schemas import DefinitionOut, DefinitionParseRequest, FaceOut, ImportRequest, JobEventOut, JobOut, MergeRequest, PersonOut, PhotoOut, ReassignRequest, RenameRequest, RetryRequest, SplitRequest

app=FastAPI(title="faceId", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173","http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])

def db():
    s=SessionLocal()
    try: yield s
    finally: s.close()

@app.on_event("startup")
def startup(): init_db()
@app.get("/api/health")
def health(): return {"status":"ok","model_provider":settings.model_provider}
@app.post("/api/import", response_model=JobOut)
def import_folder(req: ImportRequest, session: Session=Depends(db)):
    p=Path(req.source_dir).expanduser()
    if not p.exists() or not p.is_dir(): raise HTTPException(400,"source_dir must be an existing folder")
    if req.rescan_mode not in {"new_version", "replace_active"}: raise HTTPException(400,"rescan_mode must be new_version or replace_active")
    job_id=start_job(str(p.resolve()), req.rescan_mode); return session.get(ProcessingJob, job_id)
@app.post("/api/import/upload", response_model=JobOut)
async def import_upload(files:list[UploadFile]=File(...), rescan_mode:str=Form("new_version"), session:Session=Depends(db)):
    if rescan_mode not in {"new_version","replace_active"}: raise HTTPException(400,"rescan_mode must be new_version or replace_active")
    if not files: raise HTTPException(400,"select at least one file")
    import uuid
    base=settings.data_dir/"uploads"/uuid.uuid4().hex; base.mkdir(parents=True,exist_ok=True)
    saved=0
    for upload in files:
        parts=[part for part in Path(upload.filename or "photo").parts if part not in ("",".","..")]
        if not parts: parts=[f"photo_{saved}"]
        target=base.joinpath(*parts); target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(await upload.read()); saved+=1
    job_id=start_job(str(base.resolve()),rescan_mode); return session.get(ProcessingJob,job_id)
@app.get("/api/jobs", response_model=list[JobOut])
def jobs(session: Session=Depends(db)):
    all_jobs=list(session.scalars(select(ProcessingJob).order_by(ProcessingJob.id.desc())))
    grouped={}
    for item in all_jobs: grouped.setdefault((item.batch_key,item.version),[]).append(item)
    output=[]
    for items in grouped.values():
        latest=items[0]
        if latest.hidden: continue
        data={k:getattr(latest,k) for k in ("id","source_dir","batch_key","version","rescan_mode","active","status","stage","current_file","failed_count","retry_of_id","hidden","total","processed","faces_found","error","created_at")}
        data["attempts"]=len(items); output.append(data)
    return output
@app.post("/api/jobs/{job_id}/retry", response_model=JobOut)
def retry_job(job_id:int, req:RetryRequest, session:Session=Depends(db)):
    try: new_id=start_retry_job(job_id,req.scope)
    except ValueError as exc: raise HTTPException(400,str(exc))
    return session.get(ProcessingJob,new_id)
@app.post("/api/jobs/{job_id}/disable")
def disable_job(job_id:int,session:Session=Depends(db)):
    item=session.get(ProcessingJob,job_id)
    if not item: raise HTTPException(404,"job not found")
    item.hidden=True; item.active=False; session.commit(); return {"id":item.id,"hidden":True}
@app.delete("/api/jobs/{job_id}")
def delete_job(job_id:int,hard:bool=False,session:Session=Depends(db)):
    item=session.get(ProcessingJob,job_id)
    if not item: raise HTTPException(404,"job not found")
    if hard:
        session.query(JobEvent).filter(JobEvent.job_id==item.id).delete(); session.delete(item)
    else: item.hidden=True; item.active=False
    session.commit(); return {"id":job_id,"deleted":hard,"hidden":not hard}
@app.post("/api/student-definitions/import")
def import_definition(req:DefinitionParseRequest, session:Session=Depends(db)):
    try: return import_student_definitions(req.pdf_path,session,settings.storage_dir,req.mode)
    except ValueError as exc: raise HTTPException(400,str(exc))
@app.post("/api/student-definitions/import-upload")
async def import_definition_upload(file:UploadFile=File(...), mode:str=Form("new_version"), session:Session=Depends(db)):
    if not (file.filename or "").lower().endswith(".pdf"): raise HTTPException(400,"select a PDF definition file")
    import uuid
    target=settings.data_dir/"definition_uploads"/f"{uuid.uuid4().hex}.pdf"; target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(await file.read())
    try: return import_student_definitions(str(target.resolve()),session,settings.storage_dir,mode)
    except ValueError as exc: raise HTTPException(400,str(exc))
@app.get("/api/student-definitions", response_model=list[DefinitionOut])
def definitions(status:str|None=None, session:Session=Depends(db)):
    query=select(StudentDefinition).order_by(StudentDefinition.id)
    if status: query=query.where(StudentDefinition.status==status)
    return list(session.scalars(query))
@app.post("/api/student-definitions/{definition_id}/disable")
def disable_definition(definition_id:int,session:Session=Depends(db)):
    item=session.get(StudentDefinition,definition_id)
    if not item: raise HTTPException(404,"definition not found")
    item.disabled=True; session.commit(); return {"id":item.id,"disabled":True}
@app.delete("/api/student-definitions/{definition_id}")
def delete_definition(definition_id:int,hard:bool=False,session:Session=Depends(db)):
    item=session.get(StudentDefinition,definition_id)
    if not item: raise HTTPException(404,"definition not found")
    if hard:
        if item.reference_path and Path(item.reference_path).exists(): Path(item.reference_path).unlink()
        session.delete(item)
    else: item.disabled=True
    session.commit(); return {"id":definition_id,"deleted":hard,"disabled":not hard}
@app.post("/api/student-definitions/parse")
def parse_definition(req:DefinitionParseRequest):
    try: return parse_student_pdf(req.pdf_path)
    except ValueError as exc: raise HTTPException(400,str(exc))
@app.get("/api/jobs/{job_id}/timeline", response_model=list[JobEventOut])
def job_timeline(job_id:int,session:Session=Depends(db)):
    job=session.get(ProcessingJob,job_id)
    if not job: raise HTTPException(404,"job not found")
    job_ids=[j.id for j in session.scalars(select(ProcessingJob).where(ProcessingJob.batch_key==job.batch_key,ProcessingJob.version==job.version))]
    events=list(session.scalars(select(JobEvent).where(JobEvent.job_id.in_(job_ids)).order_by(JobEvent.started_at,JobEvent.id)))
    now=utcnow(); output=[]
    for event in events:
        duration=event.duration_ms
        if duration is None: duration=max(0,int((now-event.started_at).total_seconds()*1000))
        output.append(JobEventOut(id=event.id,job_id=event.job_id,status=event.status,stage=event.stage,started_at=event.started_at,ended_at=event.ended_at,duration_ms=duration,message=event.message))
    return output
@app.get("/api/jobs/{job_id}", response_model=JobOut)
def job(job_id:int,session:Session=Depends(db)):
    result=session.get(ProcessingJob,job_id)
    if not result: raise HTTPException(404,"job not found")
    return result

def person_out(p, session):
    faces=list(session.scalars(select(Face).where(Face.person_id==p.id, Face.ignored.is_(False))))
    return PersonOut(id=p.id,public_id=p.public_id,label=p.label,is_unknown=p.is_unknown,face_count=len(faces),photo_count=len({f.photo_id for f in faces}),average_confidence=round(sum((f.similarity_score or f.cluster_probability or f.detector_confidence) for f in faces)/len(faces),3) if faces else 0,cover_url=f"/api/people/{p.id}/cover" if faces else None)
@app.get("/api/people", response_model=list[PersonOut])
def people(session:Session=Depends(db)):
    return [item for p in session.scalars(select(Person).where(Person.merged_into_id.is_(None),Person.is_unknown.is_(False)).order_by(Person.id)) if (item:=person_out(p,session)).face_count>0]
@app.get("/api/people/{person_id}", response_model=PersonOut)
def person(person_id:int,session:Session=Depends(db)):
    p=session.get(Person,person_id)
    if not p: raise HTTPException(404,"person not found")
    return person_out(p,session)
@app.patch("/api/people/{person_id}")
def rename_person(person_id:int,req:RenameRequest,session:Session=Depends(db)):
    p=session.get(Person,person_id)
    if not p: raise HTTPException(404,"person not found")
    p.label=req.label; session.add(AuditEvent(action="rename_person",person_id=p.id,payload_json=json.dumps({"label":req.label}))); session.commit(); return person_out(p,session)
@app.get("/api/people/{person_id}/faces", response_model=list[FaceOut])
def person_faces(person_id:int,session:Session=Depends(db)):
    return face_outs(list(session.scalars(select(Face).where(Face.person_id==person_id,Face.ignored.is_(False)))),session)
@app.get("/api/people/{person_id}/cover")
def person_cover(person_id:int,session:Session=Depends(db)):
    face=session.scalar(select(Face).where(Face.person_id==person_id,Face.crop_path.is_not(None)).order_by(Face.quality_score.desc()))
    if not face or not Path(face.crop_path).exists(): raise HTTPException(404,"cover not found")
    return FileResponse(face.crop_path)

def face_outs(faces,session):
    out=[]
    for f in faces:
        p=session.get(Person,f.person_id) if f.person_id else None
        photo=session.get(Photo,f.photo_id)
        out.append(FaceOut(id=f.id,photo_id=f.photo_id,person_id=f.person_id,person_label=("Needs review" if p and p.is_unknown else (p.label if p else None)),bbox=json.loads(f.bbox_json),detector_confidence=f.detector_confidence,quality_score=f.quality_score,cluster_probability=f.cluster_probability,similarity_score=f.similarity_score,confidence=round((0.35*f.detector_confidence+0.15*f.quality_score+0.20*f.cluster_probability+0.30*max(0,f.similarity_score))/1.0,3),assignment_state=f.assignment_state,photo_path=photo.path if photo else None,photo_width=photo.width if photo else None,photo_height=photo.height if photo else None,photo_url=f"/api/photos/{f.photo_id}" if photo else None,pinned=f.pinned,ignored=f.ignored,label_override=f.label_override,next_possible_person=(json.loads(f.alternatives_json)[0].get("label") if f.alternatives_json and json.loads(f.alternatives_json) else None),crop_url=f"/api/faces/{f.id}/crop" if f.crop_path else None))
    return out
@app.get("/api/faces/review",response_model=list[FaceOut])
def review_faces(session:Session=Depends(db)):
    faces=list(session.scalars(select(Face).where((Face.assignment_state.in_(["unknown","needs_review"]))|(Face.cluster_probability<0.55),Face.ignored.is_(False)).order_by(Face.quality_score.desc())))
    return face_outs(faces,session)
@app.get("/api/faces/{face_id}/crop")
def face_crop(face_id:int,session:Session=Depends(db)):
    f=session.get(Face,face_id)
    if not f or not f.crop_path or not Path(f.crop_path).exists(): raise HTTPException(404,"face crop not found")
    return FileResponse(f.crop_path)
@app.get("/api/faces/{face_id}",response_model=FaceOut)
def face(face_id:int,session:Session=Depends(db)):
    f=session.get(Face,face_id)
    if not f: raise HTTPException(404,"face not found")
    return face_outs([f],session)[0]
@app.post("/api/faces/{face_id}/reassign",response_model=FaceOut)
def reassign(face_id:int,req:ReassignRequest,session:Session=Depends(db)):
    f=session.get(Face,face_id)
    if not f: raise HTTPException(404,"face not found")
    if req.create_label:
        n=session.query(Person).count()+1; p=Person(public_id=f"person_{n:03d}",label=req.create_label); session.add(p); session.flush()
    else:
        p=session.get(Person,req.person_id) if req.person_id else None
        if not p: raise HTTPException(400,"person_id or create_label is required")
    f.person_id=p.id; f.assignment_state="manual"; f.pinned=True; session.add(AuditEvent(action="reassign_face",face_id=f.id,person_id=p.id)); session.commit(); return face_outs([f],session)[0]
@app.post("/api/people/{person_id}/merge",response_model=PersonOut)
def merge(person_id:int,req:MergeRequest,session:Session=Depends(db)):
    source=session.get(Person,person_id); target=session.get(Person,req.target_person_id)
    if not source or not target: raise HTTPException(404,"person not found")
    for f in session.scalars(select(Face).where(Face.person_id==source.id)): f.person_id=target.id; f.assignment_state="manual"; f.pinned=True
    source.merged_into_id=target.id; session.add(AuditEvent(action="merge_people",person_id=target.id,payload_json=json.dumps({"source":source.id}))); session.commit(); return person_out(target,session)
@app.post("/api/people/{person_id}/split",response_model=PersonOut)
def split(person_id:int,req:SplitRequest,session:Session=Depends(db)):
    source=session.get(Person,person_id)
    if not source: raise HTTPException(404,"person not found")
    n=session.query(Person).count()+1; target=Person(public_id=f"person_{n:03d}",label=f"Person {n:03d}"); session.add(target); session.flush()
    for fid in req.face_ids:
        f=session.get(Face,fid)
        if f and f.person_id==source.id: f.person_id=target.id; f.assignment_state="manual"; f.pinned=True
    session.add(AuditEvent(action="split_person",person_id=target.id,payload_json=json.dumps({"source":source.id,"faces":req.face_ids}))); session.commit(); return person_out(target,session)
@app.patch("/api/faces/{face_id}/label", response_model=FaceOut)
def label_face(face_id:int, req:RenameRequest, session:Session=Depends(db)):
    f=session.get(Face,face_id)
    if not f: raise HTTPException(404,"face not found")
    f.label_override=req.label; f.assignment_state="manual"; f.pinned=True
    session.add(AuditEvent(action="label_face",face_id=f.id,payload_json=json.dumps({"label":req.label}))); session.commit()
    return face_outs([f],session)[0]
@app.post("/api/faces/{face_id}/ignore",response_model=FaceOut)
def ignore(face_id:int,session:Session=Depends(db)):
    f=session.get(Face,face_id)
    if not f: raise HTTPException(404,"face not found")
    f.ignored=True; f.assignment_state="manual"; f.pinned=True; session.commit(); return face_outs([f],session)[0]
@app.post("/api/photos/{photo_id}/caption")
def caption_photo(photo_id:int, session:Session=Depends(db)):
    p=session.get(Photo,photo_id)
    if not p or not Path(p.path).exists(): raise HTTPException(404,"photo not found")
    provider=get_caption_provider()
    if not provider: raise HTTPException(503,"caption provider is disabled; set FACEID_CAPTION_PROVIDER=transformers and install transformers/torch")
    p.caption_status="processing"; session.commit()
    try:
        p.caption=provider.describe(p.path); p.caption_status="completed"
    except Exception as exc:
        p.caption_status="failed"; p.error=str(exc); session.commit(); raise HTTPException(500,"caption generation failed")
    session.commit(); return {"photo_id":p.id,"caption":p.caption,"status":p.caption_status}
@app.get("/api/photos", response_model=list[PhotoOut])
def photos(person_id:int|None=Query(default=None), session:Session=Depends(db)):
    rows=[]
    photo_query=select(Photo).order_by(Photo.id.desc())
    if person_id is not None: photo_query=select(Photo).join(Face,Face.photo_id==Photo.id).where(Face.person_id==person_id,Face.ignored.is_(False)).distinct().order_by(Photo.id.desc())
    for p in session.scalars(photo_query):
        if (not p.thumb_path or not p.medium_path) and Path(p.path).exists():
            try:
                derivatives=_save_derivatives(load_image(Path(p.path)),p.id); p.thumb_path=derivatives["thumb"]; p.medium_path=derivatives["medium"]; session.commit()
            except Exception:
                pass
        count=session.scalar(select(func.count(Face.id)).where(Face.photo_id==p.id,Face.ignored.is_(False))) or 0
        rows.append(PhotoOut(id=p.id,path=p.path,width=p.width,height=p.height,face_count=count,photo_url=f"/api/photos/{p.id}",thumb_url=f"/api/photos/{p.id}/thumb" if p.thumb_path else None,medium_url=f"/api/photos/{p.id}/medium" if p.medium_path else None))
    return rows
@app.get("/api/photos/{photo_id}/thumb")
def photo_thumb(photo_id:int,session:Session=Depends(db)):
    p=session.get(Photo,photo_id)
    if not p or not p.thumb_path or not Path(p.thumb_path).exists(): raise HTTPException(404,"thumbnail not found")
    return FileResponse(p.thumb_path)
@app.get("/api/photos/{photo_id}/medium")
def photo_medium(photo_id:int,session:Session=Depends(db)):
    p=session.get(Photo,photo_id)
    if not p or not p.medium_path or not Path(p.medium_path).exists(): raise HTTPException(404,"medium preview not found")
    return FileResponse(p.medium_path)
@app.get("/api/photos/{photo_id}/faces", response_model=list[FaceOut])
def photo_faces(photo_id:int,session:Session=Depends(db)):
    if not session.get(Photo,photo_id): raise HTTPException(404,"photo not found")
    return face_outs(list(session.scalars(select(Face).where(Face.photo_id==photo_id,Face.ignored.is_(False)).order_by(Face.id))),session)
@app.get("/api/photos/{photo_id}")
def photo(photo_id:int,session:Session=Depends(db)):
    p=session.get(Photo,photo_id)
    if not p or not Path(p.path).exists(): raise HTTPException(404,"photo not found")
    return FileResponse(p.path)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"

@app.get("/", include_in_schema=False)
def frontend_root():
    index = FRONTEND_DIR / "index.html"
    if index.exists(): return FileResponse(index)
    return HTMLResponse("<h1>faceId API is running</h1><p>Build the frontend with <code>npm run build:frontend</code>.</p>")

@app.get("/{asset_path:path}", include_in_schema=False)
def frontend_fallback(asset_path: str):
    if asset_path.startswith("api/"): raise HTTPException(404, "not found")
    requested = FRONTEND_DIR / asset_path
    if requested.is_file(): return FileResponse(requested)
    index = FRONTEND_DIR / "index.html"
    if index.exists(): return FileResponse(index)
    raise HTTPException(404, "not found")

