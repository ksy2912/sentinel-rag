from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.ai.client import critique_answer, embed_text, generate_rag_answer, rewrite_query
from app.api.schemas import AskResponse, Citation, QualityMetrics, RetrievedChunkDetail
from app.core.config import RAG_GROUNDING_THRESHOLD, RAG_RETRIEVAL_QUALITY_THRESHOLD, RAG_RETRY_BUDGET, RAG_TOP_K
from app.core.db import get_document_filenames, query_top_chunks_cosine
from app.observability.tracer import create_run, finalize_run, trace_node

_rag_graph = None


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


async def retriever_node(state: RAGState) -> dict:
    query = state.get("query") or state["question"]
    top_k = state.get("top_k", 5)
    fetch_k = max(top_k * 2, top_k + 3)
    rows = query_top_chunks_cosine(
        await embed_text(query), fetch_k, state.get("document_ids")
    )
    filenames = get_document_filenames(list({r["document_id"] for r in rows}))

    chunks = []
    for r in rows[:top_k]:
        name = filenames.get(r["document_id"], "unknown")
        idx = r["chunk_index"]
        chunks.append({
            "chunk_id": r["id"],
            "document_id": r["document_id"],
            "filename": name,
            "chunk_index": idx,
            "chunk_text": r["chunk_text"],
            "distance": float(r["distance"]),
            "similarity": float(r["similarity"]),
            "source_label": f"{name}#chunk_{idx}",
        })
    return {"retrieved_chunks": chunks}


async def generator_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {"answer": "No relevant information found.", "cited_sources": [], "confidence": 0.1}
    result = await generate_rag_answer(state["question"], chunks, state.get("model"))
    return {
        "answer": result.get("answer", ""),
        "cited_sources": result.get("cited_sources", []),
        "confidence": float(result.get("confidence", 0.5)),
    }


async def critic_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "grounding_score": 0.0,
            "retrieval_quality_score": 0.0,
            "critic_passed": False,
            "critic_feedback": "No chunks retrieved.",
        }
    result = await critique_answer(state["question"], state.get("answer", ""), chunks, state.get("model"))
    g = float(result.get("grounding_score", 0.0))
    r = float(result.get("retrieval_quality_score", 0.0))
    passed = bool(result.get("passed", False)) or (
        g >= RAG_GROUNDING_THRESHOLD and r >= RAG_RETRIEVAL_QUALITY_THRESHOLD
    )
    return {
        "grounding_score": g,
        "retrieval_quality_score": r,
        "confidence": float(result.get("confidence", state.get("confidence", 0.5))),
        "critic_passed": passed,
        "critic_feedback": result.get("feedback", ""),
    }


async def query_rewriter_node(state: RAGState) -> dict:
    result = await rewrite_query(
        state["question"],
        state.get("query") or state["question"],
        state.get("critic_feedback", "Improve retrieval."),
        state.get("model"),
    )
    return {
        "query": result.get("rewritten_query", state["question"]),
        "retry_count": state.get("retry_count", 0) + 1,
    }


def should_retry(state: RAGState) -> str:
    if state.get("critic_passed") or state.get("retry_count", 0) >= state.get("max_retries", 2):
        return "end"
    return "rewrite"


def _wrap(name: str, fn: Callable[[RAGState], Awaitable[dict]]):
    async def wrapped(state: RAGState) -> dict:
        run_id = state.get("run_id")
        if not run_id:
            return await fn(state)
        with trace_node(run_id, name) as details:
            result = await fn(state)
            if name == "critic":
                details.update(grounding_score=result.get("grounding_score"), critic_passed=result.get("critic_passed"))
            elif name == "retriever":
                details["chunk_count"] = len(result.get("retrieved_chunks", []))
            return result
    return wrapped


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        g = StateGraph(RAGState)
        g.add_node("retriever", _wrap("retriever", retriever_node))
        g.add_node("generator", _wrap("generator", generator_node))
        g.add_node("critic", _wrap("critic", critic_node))
        g.add_node("query_rewriter", _wrap("query_rewriter", query_rewriter_node))
        g.add_edge(START, "retriever")
        g.add_edge("retriever", "generator")
        g.add_edge("generator", "critic")
        g.add_conditional_edges("critic", should_retry, {"rewrite": "query_rewriter", "end": END})
        g.add_edge("query_rewriter", "retriever")
        _rag_graph = g.compile()
    return _rag_graph


def _to_response(final: RAGState, question: str, run_id: uuid.UUID, total_ms: int) -> AskResponse:
    chunks = final.get("retrieved_chunks", [])
    cited = set(final.get("cited_sources", []))
    citations = [
        Citation(
            document_id=c["document_id"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            chunk_id=c["chunk_id"],
            similarity=c["similarity"],
        )
        for c in chunks
        if c["source_label"] in cited
    ] or ([Citation(
        document_id=chunks[0]["document_id"],
        filename=chunks[0]["filename"],
        chunk_index=chunks[0]["chunk_index"],
        chunk_id=chunks[0]["chunk_id"],
        similarity=chunks[0]["similarity"],
    )] if chunks else [])

    metrics = QualityMetrics(
        grounding_score=float(final.get("grounding_score", 0.0)),
        retrieval_quality_score=float(final.get("retrieval_quality_score", 0.0)),
        confidence=float(final.get("confidence", 0.0)),
        critic_passed=bool(final.get("critic_passed", False)),
        retry_count=int(final.get("retry_count", 0)),
        final_query=final.get("query") or question,
    )
    return AskResponse(
        answer=final.get("answer", ""),
        confidence=metrics.confidence,
        sources=final.get("cited_sources") or [f"{c['filename']}#chunk_{c['chunk_index']}" for c in chunks[:3]],
        citations=citations,
        retrieved_chunks=[
            RetrievedChunkDetail(
                chunk_id=c["chunk_id"],
                document_id=c["document_id"],
                filename=c["filename"],
                chunk_index=c["chunk_index"],
                chunk_text=c["chunk_text"],
                distance=c["distance"],
                similarity=c["similarity"],
            )
            for c in chunks
        ],
        metrics=metrics,
        run_id=run_id,
        total_latency_ms=total_ms,
    )


async def run_rag_pipeline(
    question: str,
    model: str | None = None,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    max_retries: int | None = None,
) -> AskResponse:
    run_id = create_run(question)
    start = time.perf_counter()
    final = await get_rag_graph().ainvoke({
        "run_id": run_id,
        "question": question,
        "query": question,
        "model": model,
        "top_k": top_k or RAG_TOP_K,
        "document_ids": document_ids,
        "max_retries": max_retries if max_retries is not None else RAG_RETRY_BUDGET,
        "retry_count": 0,
    })
    total_ms = int((time.perf_counter() - start) * 1000)
    finalize_run(run_id, final.get("answer", ""), bool(final.get("critic_passed")), int(final.get("retry_count", 0)), total_ms)
    return _to_response(final, question, run_id, total_ms)
