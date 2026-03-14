import os
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html

from .config import settings
from .rag import run_rag_query
from .security import resolve_hf_token
from .rate_limit import InMemoryRateLimiter


app = FastAPI(
    title="RAG API",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


def _require_docs_access(
    authorization: str | None = Header(default=None),
    x_docs_token: str | None = Header(default=None),
) -> None:
    configured_token = (os.getenv("DOCS_ACCESS_TOKEN") or "").strip()
    if not configured_token:
        raise HTTPException(status_code=404, detail="Not Found")

    bearer_token = (authorization or "").strip()
    if bearer_token.lower().startswith("bearer "):
        bearer_token = bearer_token[7:].strip()

    provided_token = (x_docs_token or "").strip() or bearer_token
    if provided_token != configured_token:
        raise HTTPException(status_code=401, detail="Unauthorized")


@app.get("/openapi.json", include_in_schema=False)
def private_openapi(
    authorization: str | None = Header(default=None),
    x_docs_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_docs_access(authorization, x_docs_token)
    return app.openapi()


@app.get("/docs", include_in_schema=False)
def private_docs(
    authorization: str | None = Header(default=None),
    x_docs_token: str | None = Header(default=None),
) -> Any:
    _require_docs_access(authorization, x_docs_token)
    return get_swagger_ui_html(openapi_url="/openapi.json", title=f"{app.title} - Docs")

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
