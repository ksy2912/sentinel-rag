# Five Hardest Problems — Sentinel RAG

Notes from building this system (Block 1, Days 1–15).

---

## 1. Splitting LLM vs embeddings providers

**Problem:** OpenRouter handles chat well, but embeddings need a separate provider (Ollama or OpenAI). Setting `ALLOW_MOCK_LLM=false` for real `/ask` calls also disabled mock embeddings, breaking `/upload` and `/retrieve`.

**Solution:** Run Ollama as a Docker service for `nomic-embed-text` embeddings while OpenRouter powers generation and critique. Two providers, one compose stack.

---

## 2. API readiness vs container "Started"

**Problem:** Docker marks the API container as started before Uvicorn finishes booting (especially after adding LangGraph). Immediate `/ask` calls failed with "connection closed unexpectedly."

**Solution:** Wait for `/health` after `docker compose up`, add an API healthcheck in compose, and document a 5–10s startup buffer.

---

## 3. Grounding the critic without infinite retries

**Problem:** The critic can fail on vague questions or weak retrieval. Unbounded retries waste tokens and latency.

**Solution:** LangGraph conditional edge with a fixed retry budget (`RAG_RETRY_BUDGET=2`), query rewriter node, and explicit thresholds for grounding (≥0.6) and retrieval quality (≥0.3). Return the best attempt when the budget is exhausted.

---

## 4. Explainability without overwhelming the client

**Problem:** Raw RAG responses can include many chunks, scores, and trace data — too much for a simple API consumer or chat UI.

**Solution:** Structured `AskResponse` with layered fields: `answer` + `sources` (human-readable), `citations` (structured), `retrieved_chunks` (full evidence), `metrics` (scores), and optional `run_id` for observability drill-down.

---

## 5. Observability across async agent steps

**Problem:** A single `/ask` runs multiple async nodes (retriever, generator, critic, maybe rewriter) — hard to debug latency or failures without per-step traces.

**Solution:** Postgres `agent_runs` + `agent_spans` tables with a `trace_node` context manager wrapping each LangGraph node. Optional Langfuse export when keys are set. `GET /traces` for analytics without opening the DB.

---

## Bonus: pgvector dimension consistency

Embedding model output must match the DB column (`vector(768)` for `nomic-embed-text`). Mismatched dimensions cause silent insert failures or bad retrieval — enforce `EMBEDDING_DIM` in config and validate in the embedding client.
