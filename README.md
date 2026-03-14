# Document QA RAG (FastAPI + React + Notebook)

This repository contains a complete Retrieval-Augmented Generation (RAG) workflow:
- Backend API in `app/`
- Frontend client in `front/`
- Notebook prototype in `notebooks/`

The system accepts only `.txt` and `.pdf`, does not persist uploaded files, and supports streamed answers.

## Repository Structure

- `app/` FastAPI backend for RAG queries
- `front/` React + Vite + TypeScript frontend
- `notebooks/` Jupyter notebook prototype and local document workspace
- `requirements.txt` Python dependencies for backend/notebook

## Prerequisites

- Python 3.10+
- Node.js 24+ and Yarn
- A Hugging Face token

## 1) Clone and Install

```bash
git clone <your-repo-url>
cd nlp_projects
pip install -r requirements.txt
cd front
yarn install
cd ..
```

## 2) Run Backend (FastAPI)

```bash
uvicorn app.main:app --reload
```

Backend docs:
- Swagger UI: http://localhost:8000/docs

## 3) Run Frontend

```bash
cd front
yarn dev
```

Open the URL printed by Vite (typically http://localhost:5173).

## 4) Use the App

- Paste your Hugging Face token in the frontend (stored in browser localStorage)
- Upload a `.txt` or `.pdf`
- Enter a question (up to 1,000 characters)
- Toggle stream on/off and submit

## Environment Variables (Backend)

Optional backend variables:
- `CORS_ALLOWED_ORIGINS` (comma-separated)
- `EMBEDDING_MODEL`
- `LLM_MODEL`
- `CHUNK_SIZE`
- `CHUNK_OVERLAP`
- `NUM_RETRIEVED_DOCS`
- `HUGGINGFACE_API_KEY` (fallback token if frontend token is not provided)

## Notes

- Uploaded files are processed in-memory and are not stored by the API.
- Frontend build check:
```bash
cd front
yarn build
```

## Security for Public Repo

- Never commit `.env` or real API keys.
- Use `.env.example` as template and keep real values only in local env / deployment secret manager.
- Frontend sends token using `Authorization: Bearer <token>` at request time.
- If you suspect a token leak, rotate/revoke it immediately in Hugging Face settings.

If a secret was committed in git history, rewrite history before publishing:

```bash
git filter-repo --path .env --invert-paths
git push --force --all
git push --force --tags
```

## Folder Docs

- [app/README.md](app/README.md)
- [front/README.md](front/README.md)
- [notebooks/README.md](notebooks/README.md)
