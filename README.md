# Sentinel RAG

A production-style **Retrieval-Augmented Generation (RAG)** system built for Block 1 (Days 1–15). Upload documents, ask questions, and get **grounded answers with citations** — backed by a **multi-agent LangGraph pipeline** that detects weak answers and **self-heals** via query rewriting.

**Chat UI:** http://localhost:3000  
**API docs:** http://localhost:8001/docs

---

## What this project does

| Capability | Implementation |
|------------|----------------|
| Document ingestion | PDF, DOCX, TXT → text extraction → overlapping chunks |
| Vector search | Ollama embeddings + Postgres **pgvector** (HNSW, cosine) |
| Grounded generation | OpenRouter LLM with context-only prompts |
| Multi-agent workflow | LangGraph: Retriever → Generator → Critic → Query Rewriter |
| Self-healing | Conditional retry loop with bounded budget |
| Hallucination checks | Critic scores grounding + retrieval quality |
| Explainability | Citations, source labels, retrieved chunks, confidence metrics |
| Observability | `agent_runs` / `agent_spans` in Postgres (+ optional Langfuse) |
| Chat UI | React + Vite — upload, ask, view scores and sources |

---

## Architecture

```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│  React UI   │────▶│  FastAPI API                                     │
│  :3000      │     │  /upload  /ask  /retrieve  /documents  /traces   │
└─────────────┘     └────────────┬─────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │  LangGraph RAG Pipeline │
                    │                         │
                    │  START → retriever      │
                    │       → generator       │
                    │       → critic          │
                    │       ↓ (if failed)     │
                    │       query_rewriter ───┘ (retry, max 2)
                    │       → END             │
                    └────────────┬────────────┘
                                 │
              ┌──────────────────┼──────────────────┐
              ▼                  ▼                  ▼
        ┌──────────┐      ┌──────────┐      ┌─────────────┐
        │ Postgres │      │  Ollama  │      │ OpenRouter  │
        │ pgvector │      │ embed    │      │ LLM (chat)  │
        └──────────┘      └──────────┘      └─────────────┘
```

### Agent pipeline

1. **Retriever** — embeds the query, fetches `2× top_k` candidates from pgvector, reranks with generic lexical overlap, returns top-k chunks.
2. **Generator** — stuffs context into the prompt, returns JSON answer + cited sources.
3. **Critic** — LLM-as-judge: grounding score, retrieval quality, pass/fail.
4. **Query Rewriter** — if critic fails and retries remain, rewrites the search query and loops back to retriever.

---

## Tech stack

| Layer | Technology |
|-------|------------|
| API | FastAPI, Pydantic, Uvicorn |
| Agents | LangGraph |
| Database | PostgreSQL 16 + pgvector |
| Embeddings | Ollama (`nomic-embed-text`, 768-dim) |
| LLM | OpenRouter (default: `openai/gpt-4o-mini`) |
| Frontend | React, Vite, TypeScript |
| Containers | Docker Compose |

---

## Quick start

### Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- [OpenRouter API key](https://openrouter.ai/) (for `/ask` generation and critique)

### 1. Clone and configure

```powershell
cd intelligent-rag
```

Create a `.env` file in the project root:

```env
ALLOW_MOCK_LLM=false
OPENROUTER_API_KEY=your_key_here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_HTTP_REFERER=http://localhost
OPENROUTER_APP_TITLE=intelligent-rag
EMBEDDING_DIM=768
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_EMBED_MODEL=nomic-embed-text
```

Optional — Langfuse tracing:

```env
LANGFUSE_PUBLIC_KEY=pk-...
LANGFUSE_SECRET_KEY=sk-...
LANGFUSE_HOST=https://cloud.langfuse.com
```

Optional — tune RAG behaviour:

```env
RAG_TOP_K=8
RAG_RETRY_BUDGET=2
RAG_GROUNDING_THRESHOLD=0.5
RAG_RETRIEVAL_QUALITY_THRESHOLD=0.25
```

### 2. Start the stack

```powershell
docker compose up -d --build
```

### 3. Pull the embedding model (first time only)

```powershell
docker compose exec -T ollama ollama pull nomic-embed-text
```

### 4. Verify health

Wait ~10 seconds after startup, then:

```powershell
Invoke-RestMethod http://localhost:8001/health
```

Expected: `{ "status": "ok" }`

### 5. Open the app

| Service | URL |
|---------|-----|
| **Chat UI** | http://localhost:3000 |
| **API (Swagger)** | http://localhost:8001/docs |
| **Health** | http://localhost:8001/health |
| **Traces** | http://localhost:8001/traces |

---

## Usage

### Chat UI

1. Open http://localhost:3000
2. Upload one or more PDF / DOCX / TXT files
3. Ask a question about your documents
4. Review the answer, confidence scores, grounding metrics, and citations

### API examples (PowerShell)

**Upload a document**

```powershell
curl.exe -X POST "http://localhost:8001/upload" `
  -F "file=@backend\sample_docs\sample1.txt" `
  -F "chunk_size=800" `
  -F "chunk_overlap=100"
```

**Ask a question (full RAG pipeline)**

```powershell
$body = '{"question":"What endpoints does FastAPI expose?","top_k":8}'
Invoke-RestMethod -Method Post http://localhost:8001/ask `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

**Semantic search only (no LLM)**

```powershell
$body = '{"query":"FastAPI health endpoint","top_k":5}'
Invoke-RestMethod -Method Post http://localhost:8001/retrieve `
  -ContentType "application/json" -Body $body | ConvertTo-Json -Depth 10
```

**List documents**

```powershell
Invoke-RestMethod http://localhost:8001/documents
```

**View agent traces**

```powershell
Invoke-RestMethod http://localhost:8001/traces
```

### `/ask` response shape

```json
{
  "answer": "...",
  "confidence": 0.85,
  "sources": ["file.pdf#chunk_2"],
  "citations": [
    {
      "document_id": "...",
      "filename": "file.pdf",
      "chunk_index": 2,
      "chunk_id": "...",
      "similarity": 0.72
    }
  ],
  "retrieved_chunks": [ "..." ],
  "metrics": {
    "grounding_score": 0.8,
    "retrieval_quality_score": 0.65,
    "confidence": 0.85,
    "critic_passed": true,
    "retry_count": 0,
    "final_query": "original question"
  },
  "run_id": "...",
  "total_latency_ms": 3200
}
```

---

## API reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | Upload PDF / DOCX / TXT, chunk, embed, store |
| `POST` | `/ask` | Full RAG pipeline — grounded answer + citations |
| `POST` | `/retrieve` | Vector search only (no generation) |
| `GET` | `/documents` | List uploaded documents with chunk counts |
| `GET` | `/traces` | Recent agent runs (observability) |
| `GET` | `/traces/{run_id}` | Run detail with per-node spans |

### Upload parameters

| Field | Default | Description |
|-------|---------|-------------|
| `file` | — | PDF, DOCX, or TXT |
| `chunk_size` | `1000` | Max characters per chunk (min 200) |
| `chunk_overlap` | `150` | Overlap between consecutive chunks |

### Ask parameters

| Field | Default | Description |
|-------|---------|-------------|
| `question` | — | User question (required) |
| `top_k` | `5` | Chunks to retrieve (UI sends `8`) |
| `model` | OpenRouter default | Override LLM model |
| `document_ids` | all docs | Restrict search to specific documents |

---

## How retrieval works

Chunking is **document-agnostic** — no domain-specific rules:

- Split on paragraph breaks (`\n\n`)
- For dense PDF blocks (single newlines), split by lines
- For very long lines, use a sliding character window
- Overlap between chunks preserves context at boundaries

Retrieval combines:

1. **Semantic search** — cosine similarity via pgvector HNSW index
2. **Lexical rerank** — generic token overlap boost (no keyword lists)
3. **Candidate pool** — fetch `2× top_k` then rerank to final `top_k`

Re-upload documents after changing chunking logic so the database reflects new chunks.

---

## Run tests

Inside Docker (recommended):

```powershell
docker compose exec -T api pytest -v
```

Locally:

```powershell
cd backend
pip install -r requirements.txt
$env:ALLOW_MOCK_LLM="true"
pytest -v
```

Current suite: API health, validation, chunking, and lexical rerank tests.

---

## Project structure

```
intelligent-rag/
├── backend/
│   ├── app/
│   │   ├── agents/          # LangGraph graph, nodes, state
│   │   ├── api/             # Pydantic request/response schemas
│   │   ├── core/            # chunking, db, config, rerank
│   │   ├── embeddings/      # Ollama embedding client
│   │   ├── llm/             # OpenRouter / Ollama chat client
│   │   ├── observability/   # Postgres + Langfuse tracing
│   │   └── main.py          # FastAPI routes
│   ├── tests/
│   ├── sample_docs/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/                # React chat UI
├── db/init.sql              # Postgres schema + pgvector index
├── docs/HARDEST_PROBLEMS.md # Engineering notes (5 hardest problems)
└── docker-compose.yml       # One-command stack
```

---

## Configuration reference

| Variable | Default | Purpose |
|----------|---------|---------|
| `DATABASE_URL` | set by compose | Postgres connection |
| `OPENROUTER_API_KEY` | — | LLM API key (required for real `/ask`) |
| `OPENROUTER_MODEL` | `openai/gpt-4o-mini` | Chat model |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Embedding service |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBEDDING_DIM` | `768` | Must match model + DB column |
| `ALLOW_MOCK_LLM` | `false` | Mock LLM for offline tests |
| `RAG_TOP_K` | `8` | Default retrieval count |
| `RAG_RETRY_BUDGET` | `2` | Max query-rewrite retries |
| `RAG_GROUNDING_THRESHOLD` | `0.5` | Critic pass threshold |
| `RAG_RETRIEVAL_QUALITY_THRESHOLD` | `0.25` | Retrieval pass threshold |
| `LANGFUSE_PUBLIC_KEY` | — | Optional Langfuse tracing |
| `LANGFUSE_SECRET_KEY` | — | Optional Langfuse tracing |

---

## Ports

| Port | Service |
|------|---------|
| `3000` | React UI (nginx) |
| `8001` | FastAPI API |
| `5434` | Postgres (host access) |
| `11434` | Ollama |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `/ask` fails immediately after `docker compose up` | Wait for `/health` to return 200 (~10s) |
| Upload works but retrieval is empty | Run `ollama pull nomic-embed-text` inside the ollama container |
| `/ask` returns 503 | Check `OPENROUTER_API_KEY` in `.env` and restart API |
| Stale or wrong answers after code changes | Re-upload documents (old chunks stay in DB) |
| Embedding dimension errors | Ensure `EMBEDDING_DIM=768` matches `nomic-embed-text` |

**Stop the stack:**

```powershell
docker compose down
```

**Reset database (deletes all documents):**

```powershell
docker compose down -v
docker compose up -d --build
```

---

## Block 1 milestones (Days 1–15)

| Days | Milestone | Status |
|------|-----------|--------|
| 1 | Docker Compose + Postgres/pgvector + FastAPI `/health` | Done |
| 2 | LLM integration (OpenRouter, structured JSON) | Done |
| 3 | Document upload + chunking | Done |
| 4 | Embeddings stored in pgvector with HNSW index | Done |
| 5 | Semantic retrieval (`POST /retrieve`) | Done |
| 6 | End-to-end RAG with citations (`POST /ask`) | Done |
| 7 | LangGraph agent pipeline | Done |
| 8 | Critic agent (grounding + quality scores) | Done |
| 9 | Self-healing retry loop + query rewriter | Done |
| 10 | Explainable responses (chunks, sources, metrics) | Done |
| 11 | Observability (Postgres traces + optional Langfuse) | Done |
| 12 | React chat UI | Done |
| 13 | Multi-document support + unit tests | Done |
| 14 | One-command Docker run | Done |
| 15 | README + architecture + hardest problems doc | Done |

See [docs/HARDEST_PROBLEMS.md](docs/HARDEST_PROBLEMS.md) for detailed engineering notes on the five hardest problems encountered while building this system.

---

## What you can explain after Block 1

- How **embeddings** and **vector similarity search** work
- How to build a **multi-agent LangGraph** workflow with conditional retry
- How to **detect hallucinations** with grounding and retrieval-quality checks
- How to **instrument** an AI pipeline with runs and spans
- How to **ship** a RAG product with a chat UI and one-command Docker setup

---

## License

MIT (or your choice — add a `LICENSE` file when publishing to GitHub).
