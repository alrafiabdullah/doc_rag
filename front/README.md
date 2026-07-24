# Frontend: React + Vite Client

This folder contains the mobile-friendly UI for the RAG system.

The frontend uses Yarn 4.17.1 via Corepack, and the deploy script runs Wrangler from this directory.

## What It Does

- Lets users provide Hugging Face token (stored in browser localStorage)
- Uploads `.txt`/`.pdf` and asks questions against backend RAG API
- Supports non-stream and stream modes
- Displays answer, sources, and generation metrics

## Setup

```bash
cd front
yarn --version
yarn install
```

If Corepack is not already enabled, run `corepack enable` once in your environment before installing.

## Run (Development)

```bash
yarn dev
```

## Build (Production)

```bash
yarn build
yarn preview
```

`yarn deploy` runs the production build and then executes `wrangler deploy`.

## Backend URL

The app uses `VITE_API_URL` if provided, otherwise defaults to:

`http://localhost:8000/rag/query`

To override:

```bash
VITE_API_URL="https://your-backend-domain/rag/query" VITE_MAX_FILE_SIZE_MB=1 VITE_RATE_LIMIT_MAX_REQUESTS=10 yarn dev
```

The CI workflow uses `yarn install --immutable` and `yarn build` in `front/`.

## Notes

- Token is saved in browser localStorage key `hf_token`.
- Do not commit real tokens to repository files.
