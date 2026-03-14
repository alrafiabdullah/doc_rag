import io
import json
import time

from pathlib import Path
from typing import Any, Iterable

from fastapi import HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from huggingface_hub import InferenceClient
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import Settings


QA_PROMPT = PromptTemplate.from_template(
    """You are an personal assistant who answers questions based on the provided context. Use the following pieces of context to answer the question at the end.
If you don't know the answer:
- say that you don't know.
- Never try to make up an answer.
- Don't use any information that is not included in the provided context.
- Stop after saying this does not fall under the provided document, don't add anything extra.

Exception:
- Who are you? shoud be answered with "I am a Retrieval AI Assistant instructed by Abdullah Al Rafi (abdullahalrafi.com) designed to answer questions based on the provided document." even if the context doesn't explicitly say this.

Context:
{context}

Question: {question}

Helpful Answer:"""
)


def _extract_text_from_upload(upload: UploadFile, content: bytes) -> str:
    ext = Path(upload.filename or "").suffix.lower()

    if ext == ".txt":
        try:
            return content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=400, detail="TXT file must be UTF-8 encoded") from exc

    if ext == ".pdf":
        try:
            reader = PdfReader(io.BytesIO(content))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages).strip()
            if not text:
                raise HTTPException(status_code=400, detail="Could not extract text from PDF")
            return text
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid PDF file: {exc}") from exc

    raise HTTPException(status_code=400, detail="Only .txt and .pdf files are allowed")


def _estimate_token_count(text: str) -> int:
    text = (text or "").strip()
    if not text:
        return 0
    return len(text.split())


def _stream_answer(client: InferenceClient, prompt: str) -> Iterable[str]:
    started_at = time.perf_counter()
    full_answer_parts: list[str] = []

    stream_response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
        stream=True,
    )

    for chunk in stream_response:
        content = ""
        if hasattr(chunk, "choices") and chunk.choices:
            delta = getattr(chunk.choices[0], "delta", None)
            content = getattr(delta, "content", "") if delta else ""
        if content:
            full_answer_parts.append(content)
            yield content

    full_answer = "".join(full_answer_parts)
    generated_tokens = _estimate_token_count(full_answer)
    generation_seconds = max(time.perf_counter() - started_at, 1e-6)
    tokens_per_second = generated_tokens / generation_seconds

    stream_meta = {
        "generated_tokens": generated_tokens,
        "generation_seconds": round(generation_seconds, 4),
        "tokens_per_second": round(tokens_per_second, 4),
    }
    yield f"\n\n[STREAM_META]{json.dumps(stream_meta)}"


async def run_rag_query(
    file: UploadFile,
    question: str,
    top_k: int,
    stream: bool,
    hf_token: str,
    settings: Settings,
) -> Any:
    clean_question = (question or "").strip()
    if not clean_question:
        raise HTTPException(status_code=400, detail="Question is required")
    if len(clean_question) > settings.max_question_chars:
        raise HTTPException(
            status_code=400,
            detail=f"Question exceeds limit of {settings.max_question_chars} characters",
        )

    ext = Path(file.filename or "").suffix.lower()
    if ext not in {".txt", ".pdf"}:
        raise HTTPException(status_code=400, detail="Only .txt and .pdf files are allowed")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(content) > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"File size exceeds {settings.max_file_size_megabytes}MB limit",
        )

    raw_text = _extract_text_from_upload(file, content)

    docs = [Document(page_content=raw_text, metadata={"source": file.filename or "uploaded_file"})]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text chunks could be created from file")

    embeddings = HuggingFaceEndpointEmbeddings(
        model=settings.embedding_model,
        huggingfacehub_api_token=hf_token,
    )
    vectorstore = InMemoryVectorStore.from_documents(chunks, embedding=embeddings)

    k = max(1, min(int(top_k), 10))
    retrieved_docs = vectorstore.similarity_search(clean_question, k=k)

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    prompt = QA_PROMPT.format(context=context, question=clean_question)

    client = InferenceClient(model=settings.llm_model, token=hf_token)

    if stream:
        return StreamingResponse(_stream_answer(client, prompt), media_type="text/plain; charset=utf-8")

    started_at = time.perf_counter()
    response: Any = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
        temperature=0,
    )
    generation_seconds = max(time.perf_counter() - started_at, 1e-6)
    answer = response.choices[0].message.content.strip()
    generated_tokens = _estimate_token_count(answer)
    tokens_per_second = generated_tokens / generation_seconds

    sources = []
    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source", "uploaded_file")
        sources.append(
            {
                "rank": idx,
                "source": source,
                "preview": doc.page_content[:200],
            }
        )

    return JSONResponse(
        {
            "answer": answer,
            "meta": {
                "retrieved_chunks": len(retrieved_docs),
                "question_chars": len(clean_question),
                "accepted_file_types": [".txt", ".pdf"],
                "file_persisted": False,
                "generated_tokens": generated_tokens,
                "generation_seconds": round(generation_seconds, 4),
                "tokens_per_second": round(tokens_per_second, 4),
            },
            "sources": sources,
        }
    )
