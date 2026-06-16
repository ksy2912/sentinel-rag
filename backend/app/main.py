from __future__ import annotations

import io
import os
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import run_rag_pipeline
from app.api.schemas import (
    AgentRunSummary,
    AgentSpan,
    AskRequest,
    AskResponse,
    DocumentInfo,
    RetrieveRequest,
    RetrieveResponse,
    UploadResponse,
)
from app.core.chunking import split_into_overlapping_chunks
from app.core.config import EMBEDDING_DIM
from app.core.db import create_document, init_db, insert_chunk_embeddings, list_documents, query_top_chunks_cosine
from app.embeddings.client import embed_text
from app.observability.tracer import get_recent_runs, get_run_spans

app = FastAPI(title="Intelligent RAG API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    rows = list_documents()
    return [
        DocumentInfo(
            id=r["id"],
            filename=r["filename"],
            mime_type=r["mime_type"],
            chunk_count=int(r["chunk_count"]),
            created_at=r["created_at"],
        )
        for r in rows
    ]


@app.get("/traces", response_model=list[AgentRunSummary])
def traces(limit: int = 20) -> list[AgentRunSummary]:
    runs = get_recent_runs(limit=limit)
    return [
        AgentRunSummary(
            id=r["id"],
            question=r["question"],
            answer=r["answer"],
            critic_passed=r["critic_passed"],
            retry_count=r["retry_count"] or 0,
            total_latency_ms=r["total_latency_ms"],
            created_at=r["created_at"],
        )
        for r in runs
    ]


@app.get("/traces/{run_id}", response_model=AgentRunSummary)
def trace_detail(run_id: uuid.UUID) -> AgentRunSummary:
    runs = get_recent_runs(limit=100)
    match = next((r for r in runs if r["id"] == run_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Run not found")

    spans = get_run_spans(run_id)
    return AgentRunSummary(
        id=match["id"],
        question=match["question"],
        answer=match["answer"],
        critic_passed=match["critic_passed"],
        retry_count=match["retry_count"] or 0,
        total_latency_ms=match["total_latency_ms"],
        created_at=match["created_at"],
        spans=[
            AgentSpan(
                id=s["id"],
                node_name=s["node_name"],
                status=s["status"],
                latency_ms=s["latency_ms"],
                details=s["details"],
                created_at=s["created_at"],
            )
            for s in spans
        ],
    )


def _guess_extension(filename: str) -> str:
    _, ext = os.path.splitext(filename.lower())
    return ext


async def _extract_text_from_upload(file: UploadFile) -> tuple[str, str | None]:
    raw = await file.read()
    ext = _guess_extension(file.filename or "")
    content_type = file.content_type

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail="pypdf not installed") from e

        reader = PdfReader(io.BytesIO(raw))
        parts: list[str] = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            if txt.strip():
                parts.append(txt)
        extracted = "\n\n".join(parts).strip()
        return extracted, content_type

    if ext == ".docx":
        try:
            import docx  # python-docx
        except ImportError as e:  # pragma: no cover
            raise HTTPException(status_code=500, detail="python-docx not installed") from e

        document = docx.Document(io.BytesIO(raw))
        parts = [p.text for p in document.paragraphs if p.text and p.text.strip()]
        extracted = "\n\n".join(parts).strip()
        return extracted, content_type

    if ext == ".txt" or content_type == "text/plain" or ext == "":
        try:
            extracted = raw.decode("utf-8", errors="ignore")
        except UnicodeDecodeError:
            extracted = raw.decode("latin-1", errors="ignore")
        return extracted.strip(), content_type

    raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type: filename extension '{ext}'. Upload PDF, DOCX, or TXT.",
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    return await run_rag_pipeline(
        question=req.question,
        model=req.model,
        top_k=req.top_k,
        document_ids=req.document_ids,
    )


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
) -> UploadResponse:
    if chunk_overlap >= chunk_size:
        raise HTTPException(status_code=400, detail="chunk_overlap must be < chunk_size")
    if chunk_size <= 200:
        raise HTTPException(status_code=400, detail="chunk_size too small; use >= 200")

    extracted_text, mime_type = await _extract_text_from_upload(file)
    if not extracted_text or not extracted_text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in the uploaded file.")

    chunks = split_into_overlapping_chunks(
        extracted_text,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    if not chunks:
        raise HTTPException(status_code=400, detail="Text chunking produced no chunks.")

    document_id = create_document(filename=file.filename or "uploaded", mime_type=mime_type)

    chunks_with_embeddings: list[tuple[int, str, list[float]]] = []
    for idx, chunk_text in enumerate(chunks):
        embedding = await embed_text(chunk_text)
        chunks_with_embeddings.append((idx, chunk_text, embedding))

    inserted = insert_chunk_embeddings(document_id=document_id, chunks=chunks_with_embeddings)

    preview_n = min(10, len(chunks))
    chunks_preview = [{"chunk_index": i, "text": chunks[i][:500]} for i in range(preview_n)]

    return UploadResponse(
        document_id=document_id,
        filename=file.filename or "uploaded",
        mime_type=mime_type,
        chunks_inserted=inserted,
        chunks_preview=chunks_preview,
        embedding_dim=EMBEDDING_DIM,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    top_chunks = query_top_chunks_cosine(
        query_embedding=await embed_text(req.query),
        top_k=req.top_k,
        document_ids=req.document_ids,
    )

    chunks = []
    for r in top_chunks:
        chunks.append(
            {
                "chunk_id": r["id"],
                "document_id": r["document_id"],
                "chunk_index": r["chunk_index"],
                "chunk_text": r["chunk_text"],
                "distance": float(r["distance"]),
                "similarity": float(r["similarity"]),
            }
        )

    return RetrieveResponse(
        query=req.query,
        top_k=req.top_k,
        embedding_dim=EMBEDDING_DIM,
        chunks=chunks,
    )
