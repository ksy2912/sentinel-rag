# Five Hardest Problems — Sentinel RAG

Interview talking points from building this system.

---

## 1. Splitting LLM vs embeddings providers

**Problem:** OpenRouter handles chat well, but embeddings need a separate provider (Ollama or OpenAI). One mock flag for “LLM off” accidentally broke upload/retrieve too.

**Solution:** Ollama for `nomic-embed-text` embeddings locally; OpenRouter for generation and critique. Two providers, one Docker Compose stack. Cloud deploy uses OpenAI embeddings because Ollama isn’t available on free hosting.

---

## 2. API readiness vs container “Started”

**Problem:** Docker marks the API as started before Uvicorn is ready. Early `/ask` calls failed with connection errors.

**Solution:** API healthcheck in `docker-compose.yml`, wait for `/health` before testing, 5–10s startup buffer documented.

---

## 3. Grounding the critic without infinite retries

**Problem:** The critic fails on vague questions or weak retrieval. Unbounded retries waste tokens and latency.

**Solution:** LangGraph conditional edge, fixed retry budget (`RAG_RETRY_BUDGET=2`), query rewriter node, configurable grounding/retrieval thresholds. Return the best attempt when budget is exhausted.

---

## 4. Explainability without overwhelming the client

**Problem:** Raw RAG output can include many chunks, scores, and trace data — too much for a chat UI or API consumer.

**Solution:** Layered `AskResponse`: `answer` + `sources`, structured `citations`, full `retrieved_chunks` for debugging, `metrics` for scores, optional `run_id` for trace drill-down.

---

## 5. Observability across async agent steps

**Problem:** One `/ask` runs retriever → generator → critic → (maybe) rewriter — hard to debug latency or failures without per-step traces.

**Solution:** Postgres `agent_runs` + `agent_spans`, `trace_node` context manager on each LangGraph node, optional Langfuse export, `GET /traces` for analytics.
