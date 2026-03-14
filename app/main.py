import os
import logging
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration
from sentry_sdk.integrations.logging import LoggingIntegration

from .config import settings
from .rag import run_rag_query
from .security import resolve_hf_token
from .rate_limit import InMemoryRateLimiter


logger = logging.getLogger(__name__)


def _init_sentry() -> None:
    if not settings.sentry_dsn:
        return

    sentry_logging = LoggingIntegration(
        level=logging.INFO,
        event_level=logging.ERROR,
    )
    sentry_sdk.init(
        dsn=settings.sentry_dsn,
        # Add data like request headers and IP for users,
        # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
        send_default_pii=True,
        # Enable sending logs to Sentry
        enable_logs=True,
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for tracing.
        traces_sample_rate=1.0,
        # Set profile_session_sample_rate to 1.0 to profile 100%
        # of profile sessions.
        profile_session_sample_rate=1.0,
        # Set profile_lifecycle to "trace" to automatically
        # run the profiler on when there is an active transaction
        profile_lifecycle="trace",
        integrations=[FastApiIntegration(), sentry_logging],
    )


_init_sentry()


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
    allow_methods=["*"],
    allow_headers=["*"],
)

rate_limiter = InMemoryRateLimiter(
    window_seconds=settings.rate_limit_window_seconds,
    max_requests=settings.rate_limit_max_requests,
)


@app.middleware("http")
async def sentry_error_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as exc:
        logger.exception("Unhandled server error", extra={"path": str(request.url.path)})
        if settings.sentry_dsn:
            sentry_sdk.capture_exception(exc)
        raise


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
