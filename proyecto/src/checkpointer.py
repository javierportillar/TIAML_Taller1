from __future__ import annotations

import atexit
from contextlib import AbstractContextManager
from functools import lru_cache
import os
from typing import Any

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver


DEFAULT_DB_URL = "postgresql://qbano:qbano_dev@localhost:5432/qbano_agent"

_checkpointer_cm: AbstractContextManager[PostgresSaver] | None = None
_checkpointer: PostgresSaver | None = None


def get_db_url() -> str:
    return os.getenv("LANGGRAPH_DB_URL", DEFAULT_DB_URL).strip()


def get_checkpointer() -> PostgresSaver:
    """Devuelve un PostgresSaver singleton y asegura sus tablas internas."""

    global _checkpointer_cm, _checkpointer
    if _checkpointer is None:
        _checkpointer_cm = PostgresSaver.from_conn_string(get_db_url())
        _checkpointer = _checkpointer_cm.__enter__()
        _checkpointer.setup()
    return _checkpointer


def close_checkpointer() -> None:
    global _checkpointer_cm, _checkpointer
    if _checkpointer_cm is not None:
        _checkpointer_cm.__exit__(None, None, None)
    _checkpointer_cm = None
    _checkpointer = None


atexit.register(close_checkpointer)


def setup_conversation_store() -> None:
    with psycopg.connect(get_db_url()) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id BIGSERIAL PRIMARY KEY,
                thread_id TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                route TEXT,
                context_mode TEXT,
                llm_provider TEXT,
                llm_model TEXT,
                created_at TIMESTAMPTZ NOT NULL DEFAULT now()
            )
            """
        )
        # Migracion idempotente: si la tabla ya existia sin las columnas nuevas, las agregamos.
        conn.execute("ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS llm_provider TEXT")
        conn.execute("ALTER TABLE conversation_messages ADD COLUMN IF NOT EXISTS llm_model TEXT")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_conversation_messages_thread_created
            ON conversation_messages (thread_id, created_at, id)
            """
        )


@lru_cache(maxsize=1)
def ensure_persistence_ready() -> bool:
    get_checkpointer()
    setup_conversation_store()
    return True


def load_thread_messages(thread_id: str, limit: int = 12) -> list[dict[str, str]]:
    if not thread_id:
        return []
    ensure_persistence_ready()
    with psycopg.connect(get_db_url()) as conn:
        rows = conn.execute(
            """
            SELECT role, content
            FROM conversation_messages
            WHERE thread_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
            """,
            (thread_id, limit),
        ).fetchall()
    return [
        {"role": str(role), "content": str(content)}
        for role, content in reversed(rows)
    ]


def save_thread_turn(
    *,
    thread_id: str,
    question: str,
    answer: str,
    route: str,
    context_mode: str,
    llm_provider: str | None = None,
    llm_model: str | None = None,
) -> None:
    if not thread_id:
        return
    ensure_persistence_ready()
    with psycopg.connect(get_db_url()) as conn:
        conn.execute(
            """
            INSERT INTO conversation_messages (thread_id, role, content, llm_provider, llm_model)
            VALUES (%s, 'user', %s, %s, %s)
            """,
            (thread_id, question, llm_provider, llm_model),
        )
        conn.execute(
            """
            INSERT INTO conversation_messages
                (thread_id, role, content, route, context_mode, llm_provider, llm_model)
            VALUES (%s, 'assistant', %s, %s, %s, %s, %s)
            """,
            (thread_id, answer, route, context_mode, llm_provider, llm_model),
        )


def clear_thread_messages(thread_id: str) -> None:
    if not thread_id:
        return
    ensure_persistence_ready()
    with psycopg.connect(get_db_url()) as conn:
        conn.execute(
            "DELETE FROM conversation_messages WHERE thread_id = %s",
            (thread_id,),
        )


def persistence_table_counts() -> dict[str, int]:
    ensure_persistence_ready()
    tables = [
        "checkpoints",
        "checkpoint_writes",
        "checkpoint_blobs",
        "checkpoint_migrations",
        "conversation_messages",
    ]
    counts: dict[str, int] = {}
    with psycopg.connect(get_db_url()) as conn:
        for table in tables:
            value = conn.execute(f"SELECT count(*) FROM {table}").fetchone()
            counts[table] = int(value[0]) if value else 0
    return counts
