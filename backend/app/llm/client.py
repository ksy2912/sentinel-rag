from __future__ import annotations

import json
import hashlib

import httpx
from fastapi import HTTPException

from app.core.config import (
    ALLOW_MOCK_LLM,
    LLM_RETRY_LIMIT,
    OLLAMA_BASE_URL,
    OLLAMA_LLM_MODEL,
    OPENROUTER_API_KEY,
    OPENROUTER_APP_TITLE,
    OPENROUTER_BASE_URL,
    OPENROUTER_HTTP_REFERER,
    OPENROUTER_MODEL,
    OPENAI_API_KEY,
    OPENAI_CHAT_MODEL,
)


def _extract_first_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM output")
    candidate = text[start : end + 1]
    return json.loads(candidate)


async def _chat_completion(system: str, user: str, model: str | None = None) -> str:
    if ALLOW_MOCK_LLM:
        return json.dumps({"mock": True, "user": user[:200]})

    last_err: Exception | None = None
    for _attempt in range(LLM_RETRY_LIMIT + 1):
        try:
            if OPENROUTER_API_KEY:
                return await _openrouter_chat(system, user, model)
            if OPENAI_API_KEY:
                return await _openai_chat(system, user, model)
            return await _ollama_chat(system, user, model)
        except Exception as e:  # noqa: BLE001
            last_err = e

    raise HTTPException(
        status_code=503,
        detail=f"LLM call failed after retries: {type(last_err).__name__}: {last_err}",
    )


async def _openrouter_chat(system: str, user: str, model: str | None) -> str:
    chosen_model = model or OPENROUTER_MODEL
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}"}
    if OPENROUTER_HTTP_REFERER:
        headers["HTTP-Referer"] = OPENROUTER_HTTP_REFERER
    if OPENROUTER_APP_TITLE:
        headers["X-Title"] = OPENROUTER_APP_TITLE

    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(f"{OPENROUTER_BASE_URL}/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _openai_chat(system: str, user: str, model: str | None) -> str:
    chosen_model = model or OPENAI_CHAT_MODEL
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {
        "model": chosen_model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]


async def _ollama_chat(system: str, user: str, model: str | None) -> str:
    chosen_model = model or OLLAMA_LLM_MODEL
    prompt = f"System: {system}\n\nUser: {user}"
    async with httpx.AsyncClient(timeout=90) as client:
        r = await client.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": chosen_model, "prompt": prompt, "stream": False},
        )
        r.raise_for_status()
        return r.json().get("response", "")


def _format_context(chunks: list[dict]) -> str:
    parts: list[str] = []
    for i, c in enumerate(chunks, start=1):
        source = c.get("source_label", f"chunk_{i}")
        parts.append(f"[{source}]\n{c['chunk_text']}")
    return "\n\n---\n\n".join(parts)


async def generate_rag_answer(
    question: str,
    chunks: list[dict],
    model: str | None = None,
) -> dict:
    if ALLOW_MOCK_LLM:
        h = hashlib.sha256(question.encode()).hexdigest()
        return {
            "answer": f"(mock RAG) Based on {len(chunks)} chunks: {question[:120]}",
            "confidence": (int(h[:6], 16) % 1000) / 1000.0,
            "cited_sources": [c.get("source_label", "unknown") for c in chunks[:2]],
        }

    context = _format_context(chunks)
    system = (
        "You are a grounded RAG assistant. Answer using ONLY the provided context.\n"
        "Rules:\n"
        "- If the question has multiple parts, answer each part you can support from context.\n"
        "- If some details are missing but others are present, state what you found and "
        "what is not mentioned in the context.\n"
        "- Do not say you don't know when you can answer part of the question.\n"
        "- Use moderate confidence (0.5-0.9) for partial answers; low confidence only when "
        "nothing in the context is relevant.\n"
        "Return ONLY valid JSON:\n"
        '{ "answer": string, "confidence": number 0-1, "cited_sources": string[] }\n'
        "cited_sources must list the [source_label] tags you used. No markdown."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}"
    content = await _chat_completion(system, user, model)
    return _extract_first_json_object(content)


async def critique_answer(
    question: str,
    answer: str,
    chunks: list[dict],
    model: str | None = None,
) -> dict:
    if ALLOW_MOCK_LLM:
        avg_sim = sum(c.get("similarity", 0.5) for c in chunks) / max(len(chunks), 1)
        return {
            "grounding_score": min(0.85, avg_sim + 0.1),
            "retrieval_quality_score": avg_sim,
            "confidence": 0.7,
            "passed": avg_sim >= 0.3,
            "feedback": "mock critique",
        }

    context = _format_context(chunks)
    system = (
        "You are a RAG quality critic. Score the answer against the retrieved evidence.\n"
        "Rules:\n"
        "- PASS if the answer correctly uses context, including honest partial answers.\n"
        "- FAIL only if the answer ignores available context or invents facts not in context.\n"
        "- Use nuanced scores (not only 0 or 1).\n"
        "Return ONLY valid JSON:\n"
        "{\n"
        '  "grounding_score": number 0-1,\n'
        '  "retrieval_quality_score": number 0-1,\n'
        '  "confidence": number 0-1,\n'
        '  "passed": boolean,\n'
        '  "feedback": string\n'
        "}\n"
        "No markdown."
    )
    user = (
        f"Question: {question}\n\n"
        f"Retrieved context:\n{context}\n\n"
        f"Answer to evaluate:\n{answer}"
    )
    content = await _chat_completion(system, user, model)
    return _extract_first_json_object(content)


async def rewrite_query(
    original_question: str,
    current_query: str,
    feedback: str,
    model: str | None = None,
) -> dict:
    if ALLOW_MOCK_LLM:
        return {"rewritten_query": f"{original_question} (rewritten)"}

    system = (
        "You rewrite search queries to improve document retrieval. "
        "Return ONLY valid JSON: { \"rewritten_query\": string }\n"
        "Expand the query with relevant keywords from the original question. "
        "Keep it concise. No markdown."
    )
    user = (
        f"Original question: {original_question}\n"
        f"Current query: {current_query}\n"
        f"Critic feedback: {feedback}\n"
        "Rewrite the search query:"
    )
    content = await _chat_completion(system, user, model)
    return _extract_first_json_object(content)
