# Backend: FastAPI RAG API

This folder contains the FastAPI backend that powers document QA.

## Features

- Endpoint: `POST /rag/query`
- Accepts only `.txt` and `.pdf`
- Uploaded files are processed in-memory (no media persistence)
- Question limit: 1,000 characters
- Supports normal and streaming responses
- Returns generation metrics including tokens per second

## Run

From repository root:

```bash
uvicorn app.main:app --reload
```

## API Contract

`POST /rag/query` multipart form fields:
- `file` (required): `.txt` or `.pdf`
- `question` (required): max 1,000 chars
- `top_k` (optional): 1-10
- `stream` (optional): true/false
- `hf_token` (optional): Hugging Face token passed from frontend form (legacy fallback)

Recommended auth header:
- `Authorization: Bearer <hf_token>`
- or `X-HF-Token: <hf_token>`

If `hf_token` is omitted, backend tries `HUGGINGFACE_API_KEY`/`HF_TOKEN`/`HUGGINGFACEHUB_API_TOKEN` from environment.

## Public Repo Security

- Never commit real tokens in code or `.env`.
- Keep real secrets in deployment environment variables.
- Rotate the token immediately if it is ever exposed.

## Useful Endpoints

- `GET /docs` (private; requires `DOCS_ACCESS_TOKEN`)

For private docs access, send one of:
- `Authorization: Bearer <DOCS_ACCESS_TOKEN>`
- `X-Docs-Token: <DOCS_ACCESS_TOKEN>`

## Backend Config

Optional environment variables:
- `EMBEDDING_MODEL`
- `LLM_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `NUM_RETRIEVED_DOCS`
- `CORS_ALLOWED_ORIGINS`
- `MAX_FILE_SIZE_MB`
- `RATE_LIMIT_WINDOW_SECONDS`
- `RATE_LIMIT_MAX_REQUESTS`
- `DOCS_ACCESS_TOKEN` (required to enable `/docs` and `/openapi.json`)
