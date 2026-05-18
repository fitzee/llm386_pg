"""Smoke tests for the Postgres backend dispatch.

These run only when `LLM386_PG_TEST_URL` is set (mirrors the
TEST_DATABASE_URL convention used by `cargo test -p llm386-store-pg`).
Each test pins a unique schema so concurrent runs don't collide.
"""

from __future__ import annotations

import os
import secrets

import pytest

from llm386 import LLM386Error, Store

PG_URL = os.environ.get("LLM386_PG_TEST_URL")

pytestmark = pytest.mark.skipif(
    PG_URL is None,
    reason="set LLM386_PG_TEST_URL=postgres://... to enable",
)


def _schema() -> str:
    return f"llm386_pytest_{os.getpid()}_{secrets.token_hex(4)}"


@pytest.fixture
def pg_store(tmp_path):
    """Open a PG-backed Store via TOML, scoped to a private schema."""
    schema = _schema()
    cfg = tmp_path / "llm386.toml"
    cfg.write_text(
        f'[store]\n'
        f'backend = "pg"\n'
        f'url = "{PG_URL}"\n'
        f'schema = "{schema}"\n'
    )
    return Store(profiles=str(cfg))


def test_url_kwarg_opens_pg_backend(tmp_path):
    """Bare `url=` kwarg path: no TOML, no positional, just a URL."""
    # Use TOML for the schema to keep this test isolated; the kwarg
    # path itself doesn't take a schema (defaults to public).
    schema = _schema()
    cfg = tmp_path / "schema.toml"
    cfg.write_text(
        f'[store]\nbackend = "pg"\nschema = "{schema}"\n'
    )
    store = Store(url=PG_URL, profiles=str(cfg))
    block_id = store.put(session=1, kind="user-message", body="hello pg")
    assert len(block_id) == 32
    block = store.show(block_id)
    assert block.body == b"hello pg"


def test_toml_only_opens_pg_backend(pg_store):
    """`[store]` TOML section with no kwargs needed."""
    block_id = pg_store.put(session=7, kind="fact", body="pg from toml")
    assert pg_store.show(block_id).body == b"pg from toml"


def test_pg_pack_round_trips(pg_store):
    pg_store.put(session=1, kind="system", body="be concise")
    pg_store.put(session=1, kind="user-message", body="2+2?")
    result = pg_store.pack(session=1, model="gpt-4o", task="answer", chat=True)
    assert result.messages is not None
    roles = [m.role for m in result.messages]
    assert "system" in roles
    assert "user" in roles


def test_pg_delete_and_purge(pg_store):
    a = pg_store.put(session=1, kind="fact", body="a")
    b = pg_store.put(session=1, kind="fact", body="b")
    assert pg_store.delete(a) is True
    with pytest.raises(LLM386Error):
        pg_store.show(a)
    purged = pg_store.purge_session(1)
    assert purged == 1  # b survived the per-id delete


def test_pg_summarize_persists_summary(pg_store):
    for i in range(3):
        pg_store.put(session=9, kind="fact", body=f"fact-{i}")
    before = len(pg_store.list_sessions())
    pg_store.summarize(session=9, summarizer="truncating", store_summary=True)
    assert len(pg_store.list_sessions()) == before


def test_path_and_url_together_is_error(tmp_path):
    with pytest.raises(LLM386Error):
        Store(str(tmp_path / "lmdb"), url=PG_URL)


def test_no_backend_is_error():
    with pytest.raises(LLM386Error):
        Store()
