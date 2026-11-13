from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    data_dir: Path = Path("./data")
    storage_dir: Path = Path("./storage")
    model_provider: str = "mock"
    min_face_size: int = 48
    cluster_min_size: int = 2
    cluster_distance: float = 0.42
    caption_provider: str = "none"
    model_config = SettingsConfigDict(env_prefix="FACEID_", env_file=".env", extra="ignore")

settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.storage_dir.mkdir(parents=True, exist_ok=True)
