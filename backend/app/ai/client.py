from __future__ import annotations

from fastapi import HTTPException
from langchain_core.embeddings import FakeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import BaseModel, Field

from app.core.config import (
    ALLOW_MOCK_LLM,
    EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
    OPENAI_EMBED_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_MODEL,
)


class RAGAnswer(BaseModel):
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    cited_sources: list[str]


class CriticResult(BaseModel):
    grounding_score: float = Field(ge=0.0, le=1.0)
    retrieval_quality_score: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    passed: bool
    feedback: str


class RewrittenQuery(BaseModel):
    rewritten_query: str


def _embeddings():
    if ALLOW_MOCK_LLM:
        return FakeEmbeddings(size=EMBEDDING_DIM)
    if OPENAI_API_KEY:
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL, dimensions=EMBEDDING_DIM)
    return OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model=OLLAMA_EMBED_MODEL)


def _chat(model: str | None = None) -> ChatOpenAI:
    if OPENROUTER_API_KEY:
        headers = {k: v for k, v in {
            "HTTP-Referer": OPENROUTER_HTTP_REFERER,
            "X-Title": OPENROUTER_APP_TITLE,
        }.items() if v}
        return ChatOpenAI(
            model=model or OPENROUTER_MODEL,
            api_key=OPENROUTER_API_KEY,
            base_url=OPENROUTER_BASE_URL,
            temperature=0.2,
            default_headers=headers or None,
        )
    if OPENAI_API_KEY:
        return ChatOpenAI(model=model or OPENAI_CHAT_MODEL, api_key=OPENAI_API_KEY, temperature=0.2)
    raise HTTPException(status_code=503, detail="No LLM API key configured.")


async def embed_text(text: str) -> list[float]:
    vec = await _embeddings().aembed_query(text.strip() or " ")
    if len(vec) != EMBEDDING_DIM:
        raise HTTPException(status_code=500, detail=f"Embedding dim mismatch: {len(vec)} != {EMBEDDING_DIM}")
    return [float(x) for x in vec]


def _context(chunks: list[dict]) -> str:
    return "\n\n---\n\n".join(
        f"[{c.get('source_label', 'chunk')}]\n{c['chunk_text']}" for c in chunks
    )


async def _structured(system: str, user: str, schema: type[BaseModel], model: str | None) -> dict:
    chain = (
        ChatPromptTemplate.from_messages([("system", "{system}"), ("user", "{user}")])
        | _chat(model).with_structured_output(schema)
    )
    return (await chain.ainvoke({"system": system, "user": user})).model_dump()


async def generate_rag_answer(question: str, chunks: list[dict], model: str | None = None) -> dict:
    if ALLOW_MOCK_LLM:
        return {
            "answer": f"(mock) Answer from {len(chunks)} chunks: {question[:100]}",
            "confidence": 0.7,
            "cited_sources": [c.get("source_label", "unknown") for c in chunks[:2]],
        }
    return await _structured(
        "Grounded RAG assistant. Use ONLY context. List cited [source_label] tags used.",
        f"Context:\n{_context(chunks)}\n\nQuestion: {question}",
        RAGAnswer,
        model,
    )


async def critique_answer(
    question: str, answer: str, chunks: list[dict], model: str | None = None
) -> dict:
    if ALLOW_MOCK_LLM:
        avg = sum(c.get("similarity", 0.5) for c in chunks) / max(len(chunks), 1)
        return {
            "grounding_score": min(0.85, avg + 0.1),
            "retrieval_quality_score": avg,
            "confidence": 0.7,
            "passed": avg >= 0.3,
            "feedback": "mock",
        }
    return await _structured(
        "RAG critic. PASS honest partial answers. FAIL hallucinations only.",
        f"Question: {question}\n\nContext:\n{_context(chunks)}\n\nAnswer:\n{answer}",
        CriticResult,
        model,
    )


async def rewrite_query(
    original_question: str, current_query: str, feedback: str, model: str | None = None
) -> dict:
    if ALLOW_MOCK_LLM:
        return {"rewritten_query": f"{original_question} (rewritten)"}
    return await _structured(
        "Rewrite the search query for better retrieval. Keep it concise.",
        f"Original: {original_question}\nCurrent: {current_query}\nFeedback: {feedback}",
        RewrittenQuery,
        model,
    )
