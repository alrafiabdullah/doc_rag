# Frontend: React + Vite Client

This folder contains the mobile-friendly UI for the RAG system.

## What It Does

- Lets users provide Hugging Face token (stored in browser localStorage)
- Uploads `.txt`/`.pdf` and asks questions against backend RAG API
- Supports non-stream and stream modes
- Displays answer, sources, and generation metrics

## Setup

```bash
cd front
yarn install
```

## Run (Development)

```bash
yarn dev
```

## Build (Production)

```bash
yarn build
yarn preview
```

## Backend URL

The app uses `VITE_API_URL` if provided, otherwise defaults to:

`http://localhost:8000/rag/query`

To override:

```bash
VITE_API_URL="https://your-backend-domain/rag/query" yarn dev
```

## Notes

- Token is saved in browser localStorage key `hf_token`.
- Do not commit real tokens to repository files.
