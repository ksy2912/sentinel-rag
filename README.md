# Sentinel RAG

Multi-agent **Retrieval-Augmented Generation** system: upload documents, ask questions, get **grounded answers with citations**. Built with FastAPI, LangGraph, pgvector, and React.

| | |
|--|--|
| **Live demo** | https://sentinel-rag-web.onrender.com |
| **API docs** | https://sentinel-rag-api.onrender.com/docs |
| **GitHub** | https://github.com/ksy2912/sentinel-rag |

---

## Highlights (for interviews)

- **RAG pipeline** — ingest → chunk → embed → retrieve → generate with citations
- **Multi-agent LangGraph** — Retriever → Generator → Critic → Query Rewriter (retry loop)
- **Self-healing** — critic fails → rewrite query → re-retrieve (max 2 retries)
- **Hallucination checks** — grounding score + retrieval quality score per answer
- **Explainability** — sources, citations, retrieved chunks, confidence metrics
- **Observability** — `agent_runs` / `agent_spans` in Postgres, `GET /traces`

---

## Architecture

```
React UI (:3000)
       │
       ▼
FastAPI  —  /upload  /ask  /retrieve  /documents  /traces
       │
       ▼
LangGraph:  retriever → generator → critic → [query_rewriter ↺]
       │
       ├── Postgres + pgvector (HNSW, cosine)
       ├── Ollama (embeddings, local)
       └── OpenRouter (LLM chat + critique)
```

---

## Tech stack

FastAPI · LangGraph · LangChain · PostgreSQL 16 + pgvector · Ollama · OpenRouter · React · Docker

---

## Run locally

**Prerequisites:** Docker Desktop, OpenRouter API key

```powershell
git clone https://github.com/ksy2912/sentinel-rag.git
cd sentinel-rag
```

Create `.env` (see `.env.example`), then:

```powershell
docker compose up -d --build
docker compose exec -T ollama ollama pull nomic-embed-text
```

| Service | URL |
|---------|-----|
| Chat UI | http://localhost:3000 |
| API | http://localhost:8001/docs |

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/upload` | PDF / DOCX / TXT → chunk + embed |
| `POST` | `/ask` | Full RAG (answer + citations + metrics) |
| `POST` | `/retrieve` | Vector search only |
| `GET` | `/documents` | List uploaded documents |
| `GET` | `/traces` | Agent run observability |

---

## Project structure

```
sentinel-rag/
├── backend/app/
│   ├── ai/client.py      # LangChain chat + embeddings
│   ├── agents/graph.py   # LangGraph pipeline (all nodes)
│   ├── core/text.py      # LangChain loaders + splitter
│   ├── core/db.py        # pgvector storage
│   └── main.py
├── frontend/
├── db/init.sql
├── docs/
└── docker-compose.yml
```

---

## Engineering notes

**Five hardest problems** while building this system → [docs/HARDEST_PROBLEMS.md](docs/HARDEST_PROBLEMS.md)

---

## Tests

```powershell
docker compose exec -T api pytest -v
```
