from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from typing import Any, Generator

from app.core.config import LANGFUSE_HOST, LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY
from app.core.db import connect

_langfuse_client = None


def _get_langfuse():
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client
    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        return None
    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
        )
        return _langfuse_client
    except Exception:  # noqa: BLE001
        return None


def create_run(question: str) -> uuid.UUID:
    run_id = uuid.uuid4()
    with connect() as conn:
        conn.execute(
            "INSERT INTO agent_runs (id, question) VALUES (%s, %s);",
            (run_id, question),
        )
    return run_id


def finalize_run(
    run_id: uuid.UUID,
    answer: str,
    critic_passed: bool,
    retry_count: int,
    total_latency_ms: int,
) -> None:
    with connect() as conn:
        conn.execute(
            """
            UPDATE agent_runs
            SET answer = %s, critic_passed = %s, retry_count = %s, total_latency_ms = %s
            WHERE id = %s;
            """,
            (answer, critic_passed, retry_count, total_latency_ms, run_id),
        )


def log_span(
    run_id: uuid.UUID,
    node_name: str,
    status: str,
    latency_ms: int,
    details: dict[str, Any] | None = None,
) -> None:
    span_id = uuid.uuid4()
    details_json = json.dumps(details or {})
    with connect() as conn:
        conn.execute(
            """
            INSERT INTO agent_spans (id, run_id, node_name, status, latency_ms, details)
            VALUES (%s, %s, %s, %s, %s, %s::jsonb);
            """,
            (span_id, run_id, node_name, status, latency_ms, details_json),
        )

    lf = _get_langfuse()
    if lf:
        try:
            lf.trace(
                id=str(run_id),
                name="rag_pipeline",
            ).span(
                name=node_name,
                metadata={"status": status, "latency_ms": latency_ms, **(details or {})},
            )
        except Exception:  # noqa: BLE001
            pass


@contextmanager
def trace_node(
    run_id: uuid.UUID,
    node_name: str,
) -> Generator[dict[str, Any], None, None]:
    details: dict[str, Any] = {}
    start = time.perf_counter()
    status = "pass"
    try:
        yield details
    except Exception as exc:
        status = "error"
        details["error"] = str(exc)
        raise
    finally:
        latency_ms = int((time.perf_counter() - start) * 1000)
        log_span(run_id, node_name, status, latency_ms, details)


def get_recent_runs(limit: int = 20) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, question, answer, critic_passed, retry_count, total_latency_ms, created_at
            FROM agent_runs
            ORDER BY created_at DESC
            LIMIT %s;
            """,
            (limit,),
        ).fetchall()
    return list(rows)


def get_run_spans(run_id: uuid.UUID) -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT id, node_name, status, latency_ms, details, created_at
            FROM agent_spans
            WHERE run_id = %s
            ORDER BY created_at ASC;
            """,
            (run_id,),
        ).fetchall()
    return list(rows)
