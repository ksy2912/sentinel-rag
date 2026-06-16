from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, TypedDict


class RAGState(TypedDict, total=False):
    run_id: uuid.UUID
    question: str
    query: str
    model: str | None
    top_k: int
    document_ids: list[uuid.UUID] | None
    max_retries: int

    retrieved_chunks: list[dict[str, Any]]
    answer: str
    cited_sources: list[str]
    confidence: float

    grounding_score: float
    retrieval_quality_score: float
    critic_passed: bool
    critic_feedback: str

    retry_count: int
    trace: Annotated[list[str], operator.add]
