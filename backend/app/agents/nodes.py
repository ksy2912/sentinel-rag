from __future__ import annotations

from app.agents.state import RAGState
from app.core.config import (
    RAG_GROUNDING_THRESHOLD,
    RAG_RETRIEVAL_QUALITY_THRESHOLD,
)
from app.core.rerank import lexical_overlap_score
from app.core.db import get_document_filenames, query_top_chunks_cosine
from app.embeddings.client import embed_text
from app.llm.client import critique_answer, generate_rag_answer, rewrite_query


async def retriever_node(state: RAGState) -> dict:
    query = state.get("query") or state["question"]
    top_k = state.get("top_k", 5)
    document_ids = state.get("document_ids")

    embedding = await embed_text(query)
    # Fetch extra candidates, then rerank with generic lexical overlap.
    fetch_k = max(top_k * 2, top_k + 3)
    rows = query_top_chunks_cosine(
        query_embedding=embedding,
        top_k=fetch_k,
        document_ids=document_ids,
    )

    doc_ids = list({r["document_id"] for r in rows})
    filenames = get_document_filenames(doc_ids)

    chunks: list[dict] = []
    for r in rows:
        filename = filenames.get(r["document_id"], "unknown")
        chunk_index = r["chunk_index"]
        source_label = f"{filename}#chunk_{chunk_index}"
        chunks.append(
            {
                "chunk_id": r["id"],
                "document_id": r["document_id"],
                "filename": filename,
                "chunk_index": chunk_index,
                "chunk_text": r["chunk_text"],
                "distance": float(r["distance"]),
                "similarity": float(r["similarity"]),
                "source_label": source_label,
                "rerank_score": float(r["similarity"]) + lexical_overlap_score(query, r["chunk_text"]),
            }
        )

    chunks.sort(key=lambda c: c["rerank_score"], reverse=True)
    chunks = chunks[:top_k]

    avg_sim = sum(c["similarity"] for c in chunks) / len(chunks) if chunks else 0.0
    return {
        "retrieved_chunks": chunks,
        "trace": [f"retriever: fetched {len(chunks)} chunks (avg_sim={avg_sim:.3f}) for query='{query}'"],
    }


async def generator_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        return {
            "answer": "I could not find relevant information in the uploaded documents.",
            "cited_sources": [],
            "confidence": 0.1,
            "trace": ["generator: no chunks, returning fallback"],
        }

    result = await generate_rag_answer(
        question=state["question"],
        chunks=chunks,
        model=state.get("model"),
    )
    return {
        "answer": result.get("answer", ""),
        "cited_sources": result.get("cited_sources", []),
        "confidence": float(result.get("confidence", 0.5)),
        "trace": ["generator: produced grounded answer"],
    }


async def critic_node(state: RAGState) -> dict:
    chunks = state.get("retrieved_chunks", [])
    answer = state.get("answer", "")

    if not chunks:
        return {
            "grounding_score": 0.0,
            "retrieval_quality_score": 0.0,
            "critic_passed": False,
            "critic_feedback": "No relevant chunks retrieved.",
            "trace": ["critic: failed (no chunks)"],
        }

    result = await critique_answer(
        question=state["question"],
        answer=answer,
        chunks=chunks,
        model=state.get("model"),
    )

    grounding = float(result.get("grounding_score", 0.0))
    retrieval_q = float(result.get("retrieval_quality_score", 0.0))
    passed = bool(result.get("passed", False))
    if not passed:
        passed = (
            grounding >= RAG_GROUNDING_THRESHOLD
            and retrieval_q >= RAG_RETRIEVAL_QUALITY_THRESHOLD
        )

    return {
        "grounding_score": grounding,
        "retrieval_quality_score": retrieval_q,
        "confidence": float(result.get("confidence", state.get("confidence", 0.5))),
        "critic_passed": passed,
        "critic_feedback": result.get("feedback", ""),
        "trace": [
            f"critic: grounding={grounding:.2f} retrieval={retrieval_q:.2f} passed={passed}"
        ],
    }


async def query_rewriter_node(state: RAGState) -> dict:
    result = await rewrite_query(
        original_question=state["question"],
        current_query=state.get("query") or state["question"],
        feedback=state.get("critic_feedback", "Improve retrieval relevance."),
        model=state.get("model"),
    )
    new_query = result.get("rewritten_query", state["question"])
    retry_count = state.get("retry_count", 0) + 1
    return {
        "query": new_query,
        "retry_count": retry_count,
        "trace": [f"rewriter: attempt {retry_count}, new_query='{new_query}'"],
    }


def should_retry(state: RAGState) -> str:
    if state.get("critic_passed"):
        return "end"
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)
    if retry_count >= max_retries:
        return "end"
    return "rewrite"
