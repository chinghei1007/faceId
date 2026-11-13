from pathlib import Path
from .config import settings

class CaptionProvider:
    def describe(self, image_path: str) -> str: raise NotImplementedError

class LocalTransformersCaptionProvider(CaptionProvider):
    """Optional local captioning; install transformers/torch and set provider=transformers."""
    def __init__(self):
        from transformers import pipeline
        self.pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device=-1)
    def describe(self, image_path):
        result=self.pipe(image_path, max_new_tokens=40)
        return result[0]["generated_text"].strip()

def get_caption_provider():
    if getattr(settings, "caption_provider", "none").lower() == "transformers":
        try: return LocalTransformersCaptionProvider()
        except Exception: return None
    return None
