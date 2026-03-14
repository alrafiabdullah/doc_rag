import os

from pathlib import Path
from dataclasses import dataclass

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


ROOT_DIR = Path(__file__).resolve().parents[1]
if load_dotenv is not None:
    load_dotenv(ROOT_DIR / ".env", override=False)


@dataclass(frozen=True)
class Settings:
    cors_allowed_origins: list[str]
    embedding_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    max_question_chars: int
    max_file_size_megabytes: float
    max_file_size_bytes: int
    rate_limit_window_seconds: int
    rate_limit_max_requests: int



def _split_csv(value: str | None) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


settings = Settings(
    cors_allowed_origins=_split_csv(os.getenv("CORS_ALLOWED_ORIGINS")),
    embedding_model=os.getenv("EMBEDDING_MODEL"),
    llm_model=os.getenv("LLM_MODEL"),
    chunk_size=int(os.getenv("CHUNK_SIZE")),
    chunk_overlap=int(os.getenv("CHUNK_OVERLAP")),
    top_k=int(os.getenv("NUM_RETRIEVED_DOCS")),
    max_question_chars=int(os.getenv("MAX_QUESTION_CHARS")),
    max_file_size_megabytes=float(os.getenv("MAX_FILE_SIZE_MB")),
    max_file_size_bytes=round(float(os.getenv("MAX_FILE_SIZE_MB")) * 1024 * 1024),
    rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS")),
    rate_limit_max_requests=int(os.getenv("RATE_LIMIT_MAX_REQUESTS")),
)
