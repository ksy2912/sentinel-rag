from __future__ import annotations

import asyncio
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.agents.graph import run_rag_pipeline
from app.ai.client import embed_text
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
from app.core.config import EMBEDDING_DIM
from app.core.db import create_document, init_db, insert_chunk_embeddings, list_documents, query_top_chunks_cosine
from app.core.text import extract_text, split_chunks
from app.observability.tracer import get_recent_runs, get_run_spans

app = FastAPI(title="Sentinel RAG API", version="0.4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    init_db()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentInfo])
def documents() -> list[DocumentInfo]:
    return [
        DocumentInfo(
            id=r["id"],
            filename=r["filename"],
            mime_type=r["mime_type"],
            chunk_count=int(r["chunk_count"]),
            created_at=r["created_at"],
        )
        for r in list_documents()
    ]


@app.get("/traces", response_model=list[AgentRunSummary])
def traces(limit: int = 20) -> list[AgentRunSummary]:
    return [
        AgentRunSummary(
            id=r["id"], question=r["question"], answer=r["answer"],
            critic_passed=r["critic_passed"], retry_count=r["retry_count"] or 0,
            total_latency_ms=r["total_latency_ms"], created_at=r["created_at"],
        )
        for r in get_recent_runs(limit)
    ]


@app.get("/traces/{run_id}", response_model=AgentRunSummary)
def trace_detail(run_id: uuid.UUID) -> AgentRunSummary:
    match = next((r for r in get_recent_runs(100) if r["id"] == run_id), None)
    if not match:
        raise HTTPException(status_code=404, detail="Run not found")
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
            for s in get_run_spans(run_id)
        ],
    )


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest) -> AskResponse:
    return await run_rag_pipeline(req.question, req.model, req.top_k, req.document_ids)


@app.post("/upload", response_model=UploadResponse)
async def upload(
    file: UploadFile = File(...),
    chunk_size: int = Form(1000),
    chunk_overlap: int = Form(150),
) -> UploadResponse:
    if chunk_overlap >= chunk_size or chunk_size <= 200:
        raise HTTPException(status_code=400, detail="Invalid chunk_size / chunk_overlap")

    text = extract_text(await file.read(), file.filename or "uploaded.txt")
    if not text:
        raise HTTPException(status_code=400, detail="No extractable text found.")
    chunks = split_chunks(text, chunk_size, chunk_overlap)
    if not chunks:
        raise HTTPException(status_code=400, detail="Chunking produced no chunks.")

    doc_id = create_document(file.filename or "uploaded", file.content_type)
    vectors = await asyncio.gather(*[embed_text(t) for t in chunks])
    inserted = insert_chunk_embeddings(doc_id, list(enumerate(zip(chunks, vectors))))

    return UploadResponse(
        document_id=doc_id,
        filename=file.filename or "uploaded",
        mime_type=file.content_type,
        chunks_inserted=inserted,
        chunks_preview=[{"chunk_index": i, "text": t[:500]} for i, t in enumerate(chunks[:10])],
        embedding_dim=EMBEDDING_DIM,
    )


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve(req: RetrieveRequest) -> RetrieveResponse:
    rows = query_top_chunks_cosine(await embed_text(req.query), req.top_k, req.document_ids)
    return RetrieveResponse(
        query=req.query,
        top_k=req.top_k,
        embedding_dim=EMBEDDING_DIM,
        chunks=[{
            "chunk_id": r["id"], "document_id": r["document_id"], "chunk_index": r["chunk_index"],
            "chunk_text": r["chunk_text"], "distance": float(r["distance"]), "similarity": float(r["similarity"]),
        } for r in rows],
    )
