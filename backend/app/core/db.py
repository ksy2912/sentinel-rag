from __future__ import annotations

import uuid
from typing import Iterable

import psycopg
from psycopg.rows import dict_row

from app.core.config import DATABASE_URL, EMBEDDING_DIM


def _normalize_database_url(database_url: str) -> str:
    # docker-compose uses SQLAlchemy-style URL; psycopg expects plain libpq-ish scheme.
    if database_url.startswith("postgresql+psycopg://"):
        return database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    if database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql://", 1)
    return database_url


def connect():
    return psycopg.connect(
        _normalize_database_url(DATABASE_URL),
        row_factory=dict_row,
    )


def embedding_to_vector_literal(embedding: Iterable[float], dim: int | None = None) -> str:
    vals = [float(x) for x in embedding]
    if dim is not None and len(vals) != dim:
        raise ValueError(f"Embedding dim mismatch: got {len(vals)} expected {dim}")

    # pgvector accepts text like: [0.1,0.2,...]
    return "[" + ",".join(f"{x:.6f}" for x in vals) + "]"


def init_db() -> None:
    # We keep this idempotent so local/dev runs work even if init.sql wasn't applied.
    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              id UUID PRIMARY KEY,
              filename TEXT NOT NULL,
              mime_type TEXT,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_chunks (
              id UUID PRIMARY KEY,
              document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
              chunk_index INTEGER NOT NULL,
              chunk_text TEXT NOT NULL,
              embedding vector(768),
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
              UNIQUE(document_id, chunk_index)
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS document_chunks_document_id_idx
              ON document_chunks (document_id);
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS document_chunks_embedding_hnsw_idx
              ON document_chunks
              USING hnsw (embedding vector_cosine_ops);
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_runs (
              id UUID PRIMARY KEY,
              question TEXT NOT NULL,
              answer TEXT,
              critic_passed BOOLEAN,
              retry_count INTEGER DEFAULT 0,
              total_latency_ms INTEGER,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS agent_spans (
              id UUID PRIMARY KEY,
              run_id UUID NOT NULL REFERENCES agent_runs(id) ON DELETE CASCADE,
              node_name TEXT NOT NULL,
              status TEXT NOT NULL,
              latency_ms INTEGER NOT NULL,
              details JSONB,
              created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS agent_spans_run_id_idx ON agent_spans (run_id);
            """
        )


def create_document(filename: str, mime_type: str | None) -> uuid.UUID:
    doc_id = uuid.uuid4()
    with connect() as conn:
        conn.execute(
            "INSERT INTO documents (id, filename, mime_type) VALUES (%s,%s,%s);",
            (doc_id, filename, mime_type),
        )
    return doc_id


def insert_chunk_embeddings(
    document_id: uuid.UUID,
    chunks: list[tuple[int, tuple[str, list[float]]]],
) -> int:
    if not chunks:
        return 0

    with connect() as conn:
        for chunk_index, (chunk_text, embedding) in chunks:
            chunk_id = uuid.uuid4()
            vec_literal = embedding_to_vector_literal(embedding, dim=EMBEDDING_DIM)
            conn.execute(
                """
                INSERT INTO document_chunks (id, document_id, chunk_index, chunk_text, embedding)
                VALUES (%s,%s,%s,%s,(%s)::vector);
                """,
                (chunk_id, document_id, chunk_index, chunk_text, vec_literal),
            )
    return len(chunks)


def query_top_chunks_cosine(
    query_embedding: list[float],
    top_k: int,
    document_ids: list[uuid.UUID] | None = None,
) -> list[dict]:
    if top_k <= 0:
        return []

    vec_literal = embedding_to_vector_literal(query_embedding, dim=EMBEDDING_DIM)

    where_sql = ""
    params: list = [vec_literal, vec_literal, vec_literal, top_k]
    if document_ids:
        where_sql = " AND document_id = ANY(%s)"
        params = [vec_literal, vec_literal, vec_literal, document_ids, top_k]

    sql = f"""
      SELECT
        id,
        document_id,
        chunk_index,
        chunk_text,
        (embedding <=> (%s)::vector) AS distance,
        (1 - (embedding <=> (%s)::vector)) AS similarity
      FROM document_chunks
      WHERE 1=1
      {where_sql}
      ORDER BY embedding <=> (%s)::vector ASC
      LIMIT %s;
    """

    with connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return list(rows)


def get_document_filenames(document_ids: list[uuid.UUID]) -> dict[uuid.UUID, str]:
    if not document_ids:
        return {}
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, filename FROM documents WHERE id = ANY(%s);",
            (document_ids,),
        ).fetchall()
    return {row["id"]: row["filename"] for row in rows}


def list_documents() -> list[dict]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.filename, d.mime_type, d.created_at,
                   COUNT(c.id) AS chunk_count
            FROM documents d
            LEFT JOIN document_chunks c ON c.document_id = d.id
            GROUP BY d.id, d.filename, d.mime_type, d.created_at
            ORDER BY d.created_at DESC;
            """
        ).fetchall()
    return list(rows)

