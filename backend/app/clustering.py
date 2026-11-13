import json
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session
from .config import settings
from .db import Face, Person

def cosine_distance_matrix(embeddings):
    x=np.asarray(embeddings, dtype=np.float32); x/=np.linalg.norm(x,axis=1,keepdims=True)+1e-8
    return 1-(x@x.T)

def _centroid(faces):
    x=np.mean([np.asarray(json.loads(f.embedding_json), dtype=np.float32) for f in faces], axis=0)
    return x/(np.linalg.norm(x)+1e-8)

def _similarity(face, centroid):
    emb=np.asarray(json.loads(face.embedding_json), dtype=np.float32)
    emb/=np.linalg.norm(emb)+1e-8
    return float(np.clip(np.dot(emb, centroid), -1, 1))

def cluster_faces(session: Session):
    faces=list(session.scalars(select(Face).where(Face.embedding_json.is_not(None), Face.ignored.is_(False), Face.pinned.is_(False))))
    if not faces: return 0
    embeddings=[json.loads(f.embedding_json) for f in faces]
    probabilities=np.ones(len(faces), dtype=float)
    try:
        import hdbscan
        model=hdbscan.HDBSCAN(min_cluster_size=settings.cluster_min_size, metric="euclidean", cluster_selection_epsilon=settings.cluster_distance)
        labels=model.fit_predict(np.asarray(embeddings)); probabilities=getattr(model, "probabilities_", probabilities)
    except Exception:
        from sklearn.cluster import DBSCAN
        labels=DBSCAN(eps=settings.cluster_distance, min_samples=settings.cluster_min_size, metric="cosine").fit_predict(np.asarray(embeddings))
    existing=list(session.scalars(select(Person).where(Person.is_unknown.is_(False), Person.merged_into_id.is_(None))))
    used_person_ids=set()
    assigned_groups=[]
    for label in sorted(set(int(x) for x in labels if x >= 0)):
        members=[faces[i] for i,x in enumerate(labels) if int(x)==label]
        centroid=_centroid(members)
        person=None
        if existing:
            best=(-1,None)
            for p in existing:
                if p.id in used_person_ids: continue
                if p.prototype_json:
                    prior=np.asarray(json.loads(p.prototype_json), dtype=np.float32); prior/=np.linalg.norm(prior)+1e-8; score=float(np.dot(centroid,prior))
                else:
                    pf=[f for f in faces if f.person_id==p.id and f.embedding_json]
                    if not pf: continue
                    score=float(np.dot(centroid,_centroid(pf)))
                if score>best[0]: best=(score,p)
            if best[0] >= .72: person=best[1]
        if person is None:
            next_num=session.query(Person).count()+1
            person=Person(public_id=f"person_{next_num:03d}", label=f"Person {next_num:03d}")
            session.add(person); session.flush()
        used_person_ids.add(person.id); person.prototype_json=json.dumps(centroid.tolist()); assigned_groups.append((person,members,centroid))
        for i,x in enumerate(labels):
            if int(x)==label:
                face=faces[i]; face.person_id=person.id; face.assignment_state="automatic"; face.cluster_probability=float(probabilities[i]); face.similarity_score=_similarity(face,centroid)
    for i,x in enumerate(labels):
        if int(x) < 0 and faces[i].person_id is None:
            next_num=session.query(Person).count()+1
            person=Person(public_id=f"person_{next_num:03d}", label=f"Person {next_num:03d}", is_unknown=False)
            session.add(person); session.flush()
            faces[i].person_id=person.id; faces[i].assignment_state="needs_review"; faces[i].cluster_probability=0.0; faces[i].similarity_score=0.0
            person.prototype_json=faces[i].embedding_json
    # Store ranked alternatives for low-confidence review.
    all_people=[p for p in session.scalars(select(Person).where(Person.is_unknown.is_(False), Person.merged_into_id.is_(None)))]
    centroids={p.id:(p,_centroid([f for f in session.scalars(select(Face).where(Face.person_id==p.id,Face.embedding_json.is_not(None)))])) for p in all_people if session.scalar(select(Face).where(Face.person_id==p.id,Face.embedding_json.is_not(None)))}
    for face in faces:
        ranked=sorted(((float(_similarity(face,c)),person.label,pid) for pid,(person,c) in centroids.items() if pid!=face.person_id), reverse=True)[:3]
        face.alternatives_json=json.dumps([{"person_id":pid,"label":label,"similarity":round(score,4)} for score,label,pid in ranked])
    session.commit(); return len(set(int(x) for x in labels if x>=0))
