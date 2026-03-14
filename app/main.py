from typing import Any

from fastapi import FastAPI, File, Form, Header, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .rag import run_rag_query
from .security import resolve_hf_token
from .rate_limit import InMemoryRateLimiter


app = FastAPI(title="RAG API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

rate_limiter = InMemoryRateLimiter(
    window_seconds=settings.rate_limit_window_seconds,
    max_requests=settings.rate_limit_max_requests,
)


@app.post("/rag/query")
async def rag_query(
    request: Request,
    file: UploadFile = File(...),
    question: str = Form(...),
    top_k: int = Form(settings.top_k),
    stream: bool = Form(False),
    hf_token: str = Form(""),
    authorization: str | None = Header(default=None),
    x_hf_token: str | None = Header(default=None),
) -> Any:
    rate_limiter.enforce(request)
    effective_hf_token = resolve_hf_token(hf_token, authorization, x_hf_token)

    return await run_rag_query(
        file=file,
        question=question,
        top_k=top_k,
        stream=stream,
        hf_token=effective_hf_token,
        settings=settings,
    )
