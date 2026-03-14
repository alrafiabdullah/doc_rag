# Notebooks: RAG Prototype

This folder contains the notebook prototype used to iterate on the RAG flow.

## Files

- `document_qa_rag.ipynb` end-to-end prototype
- `uploaded_docs/` local place for docs during notebook experiments

## Setup

From repository root:

```bash
pip install -r requirements.txt
```

Open `notebooks/document_qa_rag.ipynb`.

## Recommended Run Order

Run sections in order:
1. Install/Imports/Config
2. Document loading from `uploaded_docs/`
3. Chunking
4. Embeddings + vectorstore
5. Q&A

## Notes

- Notebook is for experimentation; production serving should use `app/`.
- Keep sensitive tokens in environment variables where possible.
- `uploaded_docs/` is gitignored in root `.gitignore`.
