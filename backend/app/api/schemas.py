from __future__ import annotations

import uuid
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    model: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=50)
    document_ids: Optional[list[uuid.UUID]] = None


class Citation(BaseModel):
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    chunk_id: uuid.UUID
    similarity: float


class RetrievedChunkDetail(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    chunk_text: str
    distance: float
    similarity: float


class QualityMetrics(BaseModel):
    grounding_score: float = Field(ge=0.0, le=1.0)
    retrieval_quality_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    critic_passed: bool
    retry_count: int = 0
    final_query: str


class AskResponse(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources: list[str] = []
    citations: list[Citation] = []
    retrieved_chunks: list[RetrievedChunkDetail] = []
    metrics: QualityMetrics
    run_id: uuid.UUID | None = None
    total_latency_ms: int | None = None


class DocumentInfo(BaseModel):
    id: uuid.UUID
    filename: str
    mime_type: Optional[str] = None
    chunk_count: int
    created_at: datetime


class AgentSpan(BaseModel):
    id: uuid.UUID
    node_name: str
    status: str
    latency_ms: int
    details: dict | None = None
    created_at: datetime


class AgentRunSummary(BaseModel):
    id: uuid.UUID
    question: str
    answer: Optional[str] = None
    critic_passed: Optional[bool] = None
    retry_count: int = 0
    total_latency_ms: Optional[int] = None
    created_at: datetime
    spans: list[AgentSpan] = []


class UploadChunkPreview(BaseModel):
    chunk_index: int
    text: str


class UploadResponse(BaseModel):
    document_id: uuid.UUID
    filename: str
    mime_type: Optional[str] = None
    chunks_inserted: int
    chunks_preview: list[UploadChunkPreview] = []
    embedding_dim: int


class RetrieveRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    document_ids: Optional[list[uuid.UUID]] = None


class RetrievedChunk(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    chunk_index: int
    chunk_text: str
    distance: float
    similarity: float


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    embedding_dim: int
    chunks: list[RetrievedChunk]
