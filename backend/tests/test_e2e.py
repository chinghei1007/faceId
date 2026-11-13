import json
from pathlib import Path
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import Base, Face, JobEvent, Person, Photo, ProcessingJob
from app.vision import DetectedFace
from app import pipeline

def test_import_cluster_and_manual_corrections(tmp_path, monkeypatch):
    source=tmp_path/'photos'; source.mkdir()
    for i in range(6): Image.new('RGB',(240,240),(30+i*20,50,90)).save(source/f'{i}.jpg')
    engine=create_engine(f"sqlite:///{tmp_path/'e2e.db'}"); Base.metadata.create_all(engine); Session=sessionmaker(bind=engine)
    class FakeProvider:
        def detect_and_embed(self,image):
            color=image.getpixel((0,0))[0]
            vec=[1,0,0] if color<50 else ([1,0,0] if color<70 else [0,1,0])
            return [DetectedFace([20,20,180,180],.95,.9,vec)]
    monkeypatch.setattr(pipeline,'SessionLocal',Session); monkeypatch.setattr(pipeline,'get_provider',lambda:FakeProvider())
    with Session() as s:
        job=ProcessingJob(source_dir=str(source)); s.add(job); s.commit(); jid=job.id
    pipeline.run_job(jid)
    with Session() as s:
        job=s.get(ProcessingJob,jid); assert job.status=='completed' and job.faces_found==6; events=s.query(JobEvent).filter_by(job_id=jid).all(); assert {e.stage for e in events} >= {'scanning','processing','clustering','completed'}; assert all(e.duration_ms is not None for e in events)
        faces=s.query(Face).all(); people=s.query(Person).filter_by(is_unknown=False).all(); assert len(people)>=1
        f=faces[0]; f.pinned=True; first=f.person_id; s.commit()
        # The manual assignment remains attached to the same person after rerunning clustering.
        pipeline.cluster_faces(s); assert s.get(Face,f.id).person_id==first

def test_one_photo_can_store_many_face_occurrences_and_crops(tmp_path, monkeypatch):
    source=tmp_path/'many'; source.mkdir(); image_path=source/'group.jpg'; Image.new('RGB',(800,800),(90,80,70)).save(image_path)
    engine=create_engine(f"sqlite:///{tmp_path/'many.db'}"); Base.metadata.create_all(engine); Session=sessionmaker(bind=engine)
    class ManyFaceProvider:
        def detect_and_embed(self,image):
            out=[]
            for i in range(40):
                x=(i%8)*90+5; y=(i//8)*150+5
                out.append(DetectedFace([x,y,x+70,y+100],.55+i/200,.4+i/1000,[1,0,0]))
            return out
    monkeypatch.setattr(pipeline,'SessionLocal',Session); monkeypatch.setattr(pipeline,'get_provider',lambda:ManyFaceProvider()); monkeypatch.setattr(pipeline.settings,'storage_dir',tmp_path/'storage')
    with Session() as s:
        job=ProcessingJob(source_dir=str(source)); s.add(job); s.commit(); jid=job.id
    pipeline.run_job(jid)
    with Session() as s:
        faces=s.query(Face).all(); assert len(faces)==40; assert all(f.crop_path and Path(f.crop_path).exists() for f in faces); assert len({f.detector_confidence for f in faces})>1
