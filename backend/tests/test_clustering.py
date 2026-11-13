import json
from pathlib import Path
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.db import Base, Face, Person
from app.clustering import cluster_faces

def test_clustering_keeps_pinned_face(tmp_path, monkeypatch):
    engine=create_engine(f"sqlite:///{tmp_path/'t.db'}"); Base.metadata.create_all(engine); S=sessionmaker(bind=engine); s=S()
    a=[1.0,0,0]; b=[1.0,0,0]; c=[0,1.0,0]
    pinned=Person(public_id="person_001",label="Kept"); s.add(pinned); s.flush()
    s.add_all([Face(photo_id=1,bbox_json="[0,0,1,1]",embedding_json=json.dumps(a),pinned=True,person_id=pinned.id,assignment_state="manual"),Face(photo_id=2,bbox_json="[0,0,1,1]",embedding_json=json.dumps(b)),Face(photo_id=3,bbox_json="[0,0,1,1]",embedding_json=json.dumps(c))]); s.commit()
    cluster_faces(s); assert s.query(Face).filter_by(pinned=True).one().person_id==pinned.id
