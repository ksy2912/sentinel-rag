from __future__ import annotations

import hashlib
from typing import List

import httpx
from fastapi import HTTPException

from app.core.config import (
    ALLOW_MOCK_LLM,
    EMBEDDING_DIM,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL,
    OPENAI_API_KEY,
    OPENAI_EMBED_MODEL,
)


def _mock_embedding(text: str, dim: int) -> List[float]:
    # Deterministic embedding for dev/test when no external embedding backend exists.
    h = hashlib.sha256(text.encode("utf-8")).digest()
    # Expand digest bytes into floats in [0,1)
    floats: list[float] = []
    x = 0
    while len(floats) < dim:
        floats.append((h[x % len(h)] + 0.5) / 256.0)
        x += 1
    return floats[:dim]


async def embed_text(text: str) -> list[float]:
    clean = text.strip()
    if not clean:
        # Avoid empty inputs.
        return _mock_embedding("", EMBEDDING_DIM)

    if ALLOW_MOCK_LLM:
        return _mock_embedding(clean, EMBEDDING_DIM)

    async with httpx.AsyncClient(timeout=60) as client:
        if OPENAI_API_KEY:
            return await _embed_openai(client, clean)
        return await _embed_ollama(client, clean)


async def _embed_openai(client: httpx.AsyncClient, text: str) -> list[float]:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    payload = {"model": OPENAI_EMBED_MODEL, "input": text, "dimensions": EMBEDDING_DIM}
    r = await client.post("https://api.openai.com/v1/embeddings", headers=headers, json=payload)
    r.raise_for_status()
    data = r.json()
    emb = data["data"][0]["embedding"]
    if len(emb) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"OpenAI embedding dim mismatch: got {len(emb)} expected {EMBEDDING_DIM}",
        )
    return [float(x) for x in emb]


async def _embed_ollama(client: httpx.AsyncClient, text: str) -> list[float]:
    payload = {"model": OLLAMA_EMBED_MODEL, "prompt": text}
    r = await client.post(f"{OLLAMA_BASE_URL}/api/embeddings", json=payload)
    r.raise_for_status()
    data = r.json()
    emb = data.get("embedding")
    if not isinstance(emb, list) or not emb:
        raise HTTPException(status_code=500, detail="Ollama embeddings response missing embedding[]")
    if len(emb) != EMBEDDING_DIM:
        raise HTTPException(
            status_code=500,
            detail=f"Ollama embedding dim mismatch: got {len(emb)} expected {EMBEDDING_DIM}",
        )
    return [float(x) for x in emb]

