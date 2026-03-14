# Backend: FastAPI RAG API

This folder contains the FastAPI backend that powers document QA.

## Features

- Endpoint: `POST /rag/query`
- Accepts only `.txt` and `.pdf`
- Uploaded files are processed in-memory (no media persistence)
- Question limit: 10,000 characters
- Supports normal and streaming responses
- Returns generation metrics including tokens per second

## Run

From repository root:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Contract

`POST /rag/query` multipart form fields:
- `file` (required): `.txt` or `.pdf`
- `question` (required): max 10,000 chars
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

## CORS

Set allowed frontend origins:

```bash
export CORS_ALLOWED_ORIGINS="http://localhost:5173,http://127.0.0.1:5173"
```

## Useful Endpoints

- `GET /health`
- `GET /docs`

## Backend Config

Optional environment variables:
- `EMBEDDING_MODEL`
- `LLM_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `NUM_RETRIEVED_DOCS`
- `CORS_ALLOWED_ORIGINS`
- `MAX_FILE_SIZE_MB` (default: `5`)
- `RATE_LIMIT_WINDOW_SECONDS` (default: `60`)
- `RATE_LIMIT_MAX_REQUESTS` (default: `15`)
