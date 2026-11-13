import re, unicodedata
from pathlib import Path

HAN=r"\u4e00-\u9fff"
ENTRY=re.compile(rf"(?P<class>\d[A-Z])\s*(?P<number>\d{{1,2}})\s*(?P<name>[{HAN}\s]{{2,12}}?)(?=\s+\d[A-Z]\s*\d{{1,2}}\s|$)")

def parse_student_pdf(pdf_path: str):
    path=Path(pdf_path).expanduser()
    if path.suffix.lower() != ".pdf" or not path.is_file(): raise ValueError("pdf_path must point to an existing PDF")
    try:
        from pypdf import PdfReader
        raw="\n".join(page.extract_text() or "" for page in PdfReader(str(path)).pages)
    except Exception as exc: raise ValueError(f"could not extract PDF text: {exc}") from exc
    text=unicodedata.normalize("NFKC",raw).replace("\u00a0"," ")
    # Remove recurring page headers and join whitespace-separated Han fragments.
    text=re.sub(r"香港聖公會何明華會督中學", " ", text)
    text=re.sub(r"2026\s*學生相簿\s*\([^)]*\)", " ", text)
    text=re.sub(r"\s+", " ", text).strip()
    records=[]; warnings=[]; seen=set()
    for match in ENTRY.finditer(text):
        class_code=match.group("class"); number=int(match.group("number")); name=re.sub(r"\s+", "", match.group("name"))
        key=f"{class_code} {number:02d}"
        if not 2 <= len(name) <= 4: warnings.append(f"{key} {name}: expected 2-4 Chinese characters, found {len(name)}")
        if key in seen: warnings.append(f"duplicate student key: {key}")
        seen.add(key); records.append({"class_code":class_code,"student_number":number,"student_key":key,"name":name,"label":name})
    if not records: raise ValueError("no student entries matched the expected class-number-name pattern")
    return {"source":str(path),"count":len(records),"records":records,"warnings":warnings}

def extract_reference_images(pdf_path: str, records, storage_dir):
    from pypdf import PdfReader
    target=Path(storage_dir)/"definitions"; target.mkdir(parents=True,exist_ok=True)
    images=[]
    for page in PdfReader(pdf_path).pages:
        images.extend(page.images)
    if len(images)!=len(records):
        raise ValueError(f"PDF has {len(images)} embedded portraits but {len(records)} parsed definitions")
    output=[]
    for record,image in zip(records,images):
        path=target/f"{record['student_key'].replace(' ','_')}.jpg"; path.write_bytes(image.data); output.append(str(path))
    return output

def import_student_definitions(pdf_path: str, session, storage_dir, mode: str = "new_version"):
    from .db import StudentDefinition, Person, Face
    from .vision import get_provider, load_image
    if mode not in {"new_version", "replace_source"}: raise ValueError("mode must be new_version or replace_source")
    import json, numpy as np
    parsed=parse_student_pdf(pdf_path)
    if mode == "replace_source":
        for old in session.query(StudentDefinition).filter(StudentDefinition.source_path==parsed["source"],StudentDefinition.disabled.is_(False)): old.disabled=True
    paths=extract_reference_images(pdf_path,parsed["records"],storage_dir); provider=get_provider()
    people=list(session.query(Person).filter(Person.is_unknown.is_(False),Person.merged_into_id.is_(None)))
    prototypes={p.id:np.asarray(json.loads(p.prototype_json),dtype=np.float32) for p in people if p.prototype_json}
    for p in prototypes.values(): p/=np.linalg.norm(p)+1e-8
    results=[]
    for record,path in zip(parsed["records"],paths):
        definition=StudentDefinition(source_path=parsed["source"],student_key=record["student_key"],class_code=record["class_code"],student_number=record["student_number"],name=record["name"],label=record["label"],reference_path=path)
        detected=provider.detect_and_embed(load_image(Path(path)))
        if not detected or not prototypes:
            definition.status="unused"; definition.warning="no reference embedding or no identity prototypes"
        else:
            emb=np.asarray(detected[0].embedding,dtype=np.float32); emb/=np.linalg.norm(emb)+1e-8
            ranked=sorted(((float(np.dot(emb,proto)),pid) for pid,proto in prototypes.items()),reverse=True)
            best_score,best_id=ranked[0]; margin=best_score-ranked[1][0] if len(ranked)>1 else best_score
            definition.embedding_json=json.dumps(emb.tolist())
            definition.match_score=best_score; definition.matched_person_id=best_id
            if best_score>=0.65 and margin>=0.05:
                definition.status="matched"
                person=session.get(Person,best_id)
                if person and person.label.startswith("Person "): person.label=record["label"]
            else: definition.status="ambiguous"; definition.warning=f"best similarity {best_score:.3f}, margin {margin:.3f}"
        session.add(definition); results.append(definition)
    session.commit()
    return {"source":parsed["source"],"count":len(results),"matched":sum(x.status=="matched" for x in results),"ambiguous":sum(x.status=="ambiguous" for x in results),"unused":sum(x.status=="unused" for x in results),"warnings":parsed["warnings"]}
