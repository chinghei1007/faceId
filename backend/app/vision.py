import hashlib, json, math
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageFilter
from .config import settings

@dataclass
class DetectedFace:
    bbox: list[float]
    confidence: float
    quality: float
    embedding: list[float]

class FaceProvider:
    def detect_and_embed(self, image: Image.Image) -> list[DetectedFace]: raise NotImplementedError

class MockProvider(FaceProvider):
    """Deterministic provider for UI/API development without model downloads."""
    def detect_and_embed(self, image):
        w, h = image.size
        if min(w, h) < settings.min_face_size * 2: return []
        digest = hashlib.sha256(image.tobytes()[:200000]).digest()
        vec = np.frombuffer((digest * 32)[:512], dtype=np.uint8).astype(np.float32)
        vec = (vec - vec.mean()) / (vec.std() + 1e-6)
        vec /= np.linalg.norm(vec)
        return [DetectedFace([w*.25,h*.18,w*.75,h*.82], .91, .82, vec.tolist())]

class InsightFaceProvider(FaceProvider):
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
    def detect_and_embed(self, image):
        faces = self.app.get(np.asarray(image.convert("RGB"))[:, :, ::-1])
        output=[]
        for face in faces:
            x1,y1,x2,y2 = [float(x) for x in face.bbox]
            crop = image.crop((max(0,x1),max(0,y1),min(image.width,x2),min(image.height,y2))).convert("L")
            sharp = np.asarray(crop.filter(ImageFilter.FIND_EDGES), dtype=np.float32).var()
            quality = float(max(0, min(1, math.log1p(sharp) / 8)))
            emb = np.asarray(face.embedding, dtype=np.float32); emb /= np.linalg.norm(emb) + 1e-8
            output.append(DetectedFace([x1,y1,x2,y2], float(face.det_score), quality, emb.tolist()))
        return output

def get_provider():
    if settings.model_provider.lower() == "insightface":
        try: return InsightFaceProvider()
        except Exception: return MockProvider()
    return MockProvider()

def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")

def image_hash(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024), b""): h.update(block)
    return h.hexdigest()

def safe_json(value): return json.dumps(value, separators=(",", ":"))
