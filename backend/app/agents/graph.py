from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from app.agents.nodes import (
    critic_node,
    generator_node,
    query_rewriter_node,
    retriever_node,
    should_retry,
)
from app.agents.state import RAGState
from app.api.schemas import (
    AskResponse,
    Citation,
    QualityMetrics,
    RetrievedChunkDetail,
)
from app.core.config import RAG_RETRY_BUDGET, RAG_TOP_K
from app.observability.tracer import create_run, finalize_run, trace_node


def _wrap_node(
    node_name: str,
    fn: Callable[[RAGState], Awaitable[dict]],
) -> Callable[[RAGState], Awaitable[dict]]:
    async def wrapped(state: RAGState) -> dict:
        run_id = state.get("run_id")
        if not run_id:
            return await fn(state)

        with trace_node(run_id, node_name) as details:
            result = await fn(state)
            if node_name == "critic":
                details["grounding_score"] = result.get("grounding_score")
                details["critic_passed"] = result.get("critic_passed")
            elif node_name == "retriever":
                details["chunk_count"] = len(result.get("retrieved_chunks", []))
            elif node_name == "query_rewriter":
                details["new_query"] = result.get("query")
            return result

    return wrapped


def build_rag_graph():
    graph = StateGraph(RAGState)

    graph.add_node("retriever", _wrap_node("retriever", retriever_node))
    graph.add_node("generator", _wrap_node("generator", generator_node))
    graph.add_node("critic", _wrap_node("critic", critic_node))
    graph.add_node("query_rewriter", _wrap_node("query_rewriter", query_rewriter_node))

    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "generator")
    graph.add_edge("generator", "critic")
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {"rewrite": "query_rewriter", "end": END},
    )
    graph.add_edge("query_rewriter", "retriever")

    return graph.compile()


_rag_graph = None


def get_rag_graph():
    global _rag_graph
    if _rag_graph is None:
        _rag_graph = build_rag_graph()
    return _rag_graph


async def run_rag_pipeline(
    question: str,
    model: str | None = None,
    top_k: int | None = None,
    document_ids: list[uuid.UUID] | None = None,
    max_retries: int | None = None,
) -> AskResponse:
    graph = get_rag_graph()
    run_id = create_run(question)
    pipeline_start = time.perf_counter()

    initial: RAGState = {
        "run_id": run_id,
        "question": question,
        "query": question,
        "model": model,
        "top_k": top_k or RAG_TOP_K,
        "document_ids": document_ids,
        "max_retries": max_retries if max_retries is not None else RAG_RETRY_BUDGET,
        "retry_count": 0,
        "trace": [],
    }

    final_state = await graph.ainvoke(initial)

    total_ms = int((time.perf_counter() - pipeline_start) * 1000)
    finalize_run(
        run_id=run_id,
        answer=final_state.get("answer", ""),
        critic_passed=bool(final_state.get("critic_passed", False)),
        retry_count=int(final_state.get("retry_count", 0)),
        total_latency_ms=total_ms,
    )

    chunks = final_state.get("retrieved_chunks", [])
    retrieved_details = [
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
    ]

    citations = [
        Citation(
            document_id=c["document_id"],
            filename=c["filename"],
            chunk_index=c["chunk_index"],
            chunk_id=c["chunk_id"],
            similarity=c["similarity"],
        )
        for c in chunks
        if c["source_label"] in final_state.get("cited_sources", [])
    ]
    if not citations and chunks:
        citations = [
            Citation(
                document_id=chunks[0]["document_id"],
                filename=chunks[0]["filename"],
                chunk_index=chunks[0]["chunk_index"],
                chunk_id=chunks[0]["chunk_id"],
                similarity=chunks[0]["similarity"],
            )
        ]

    sources = final_state.get("cited_sources") or [
        f"{c['filename']}#chunk_{c['chunk_index']}" for c in chunks[:3]
    ]

    metrics = QualityMetrics(
        grounding_score=float(final_state.get("grounding_score", 0.0)),
        retrieval_quality_score=float(final_state.get("retrieval_quality_score", 0.0)),
        confidence=float(final_state.get("confidence", 0.0)),
        critic_passed=bool(final_state.get("critic_passed", False)),
        retry_count=int(final_state.get("retry_count", 0)),
        final_query=final_state.get("query") or question,
    )

    return AskResponse(
        answer=final_state.get("answer", ""),
        confidence=metrics.confidence,
        sources=sources,
        citations=citations,
        retrieved_chunks=retrieved_details,
        metrics=metrics,
        run_id=run_id,
        total_latency_ms=total_ms,
    )
