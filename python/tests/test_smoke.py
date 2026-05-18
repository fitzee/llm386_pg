"""End-to-end smoke tests for the PyO3-backed `llm386` package."""

from __future__ import annotations

import pytest

from llm386 import LLM386Error, Store, list_models


def test_put_returns_hex_id(store):
    block_id = store.put(session=1, kind="user-message", body="hello world")
    assert isinstance(block_id, str)
    assert len(block_id) == 32
    int(block_id, 16)  # must parse as hex


def test_put_accepts_str_or_bytes(store):
    a = store.put(session=1, kind="fact", body="hello")
    b = store.put(session=1, kind="fact", body=b"hello")
    # Same content → same id (content-hash dedup).
    assert a == b


def test_show_roundtrips_block(store):
    block_id = store.put(session=1, kind="fact", body="Canberra is the capital.")
    block = store.show(block_id)
    assert block.id == block_id
    assert block.kind == "Fact"
    assert block.body == b"Canberra is the capital."
    assert block.priority == 0.0
    assert isinstance(block.created_at, int)
    assert len(block.hash) == 64  # 32 bytes as hex


def test_put_dedup_returns_same_id(store):
    a = store.put(session=1, kind="fact", body="dup")
    b = store.put(session=1, kind="fact", body="dup")
    assert a == b


def test_list_sessions_enumerates_distinct_sessions(store):
    store.put(session=1, kind="fact", body="x")
    store.put(session=2, kind="fact", body="y")
    store.put(session=2, kind="fact", body="z")
    sessions = store.list_sessions()
    assert "00000000000000000000000000000001" in sessions
    assert "00000000000000000000000000000002" in sessions


def test_page_returns_plan_with_selected_blocks(store):
    store.put(session=1, kind="user-message", body="hi")
    store.put(session=1, kind="fact", body="paris is the capital of france")
    plan = store.page(session=1, model="gpt-4o", task="answer")
    assert len(plan.selected) >= 1
    assert plan.estimated_tokens > 0
    for sid in plan.selected:
        assert len(sid) == 32


def test_pack_prompt_only_returns_rendered_string(store):
    store.put(session=1, kind="user-message", body="say hi")
    result = store.pack(session=1, model="gpt-4o", task="reply briefly")
    assert result.rendered is not None
    assert result.messages is None
    assert "say hi" in result.rendered


def test_pack_chat_returns_message_list(store):
    store.put(session=1, kind="system", body="be concise")
    store.put(session=1, kind="user-message", body="2+2?")
    result = store.pack(session=1, model="gpt-4o", task="answer", chat=True)
    assert result.messages is not None
    assert result.rendered is None
    roles = [m.role for m in result.messages]
    assert "system" in roles
    assert "user" in roles


def test_pack_with_plan_reuses_selection_for_a_different_model(store):
    """Cascade-routing path: page once, render twice for two models.

    Same selected blocks → both prompts contain the same content.
    Different model profiles → distinct cache_boundary values are
    valid; we only assert the rendering itself succeeds and the
    prompts overlap.
    """
    store.put(session=1, kind="system", body="be concise")
    store.put(session=1, kind="fact", body="canberra is the capital of australia")
    store.put(session=1, kind="user-message", body="capital question")

    plan = store.page(session=1, model="gpt-4o", task="answer")
    selected_before = list(plan.selected)

    cheap = store.pack_with_plan(plan, session=1, model="gpt-4o", task="answer", chat=True)
    expensive = store.pack_with_plan(
        plan, session=1, model="claude-opus-4-7", task="answer", chat=True,
    )

    assert cheap.messages is not None
    assert expensive.messages is not None
    # Same canberra fact lands in both renderings.
    cheap_blob = "\n".join(m.content for m in cheap.messages)
    exp_blob = "\n".join(m.content for m in expensive.messages)
    assert "canberra" in cheap_blob.lower()
    assert "canberra" in exp_blob.lower()
    # The plan is unchanged after re-use.
    assert list(plan.selected) == selected_before


def test_pack_with_trace_records_id(store, tmp_path):
    store.put(session=1, kind="user-message", body="x")
    trace_dir = str(tmp_path / "traces")
    result = store.pack(
        session=1, model="gpt-4o", task="reply", chat=True, trace=trace_dir
    )
    assert result.trace_id is not None
    assert len(result.trace_id) == 32


def test_trace_show_roundtrips_record(store, tmp_path):
    from llm386 import Trace

    store.put(session=42, kind="user-message", body="x")
    trace_dir = str(tmp_path / "traces")
    pack_result = store.pack(
        session=42, model="gpt-4o", task="reply", chat=True, trace=trace_dir
    )
    trace = Trace(trace_dir)
    record = trace.show(pack_result.trace_id)
    assert record.call_id == pack_result.trace_id
    assert record.session.endswith("2a")  # 42 in hex
    assert record.model == "gpt-4o"
    assert record.prompt_tokens > 0
    assert len(record.prompt_hash) == 64
    assert isinstance(record.started_at, int)


def test_trace_show_unknown_call_raises(tmp_path):
    from llm386 import LLM386Error, Trace

    trace = Trace(str(tmp_path / "empty-traces"))
    with pytest.raises(LLM386Error):
        trace.show("0" * 32)


def test_store_with_profiles_loads_custom_model(tmp_path):
    from llm386 import Store

    config_path = tmp_path / "llm386.toml"
    config_path.write_text(
        '[[profile]]\n'
        'name = "my-tiny"\n'
        'max_context_tokens = 4096\n'
        'reserved_output_tokens = 1024\n'
        'tokenizer = "cl100k_base"\n'
    )
    store = Store(str(tmp_path / "store"), profiles=str(config_path))
    store.put(session=1, kind="user-message", body="hi")
    plan = store.page(session=1, model="my-tiny", task="reply")
    assert len(plan.selected) >= 1


def test_store_with_profiles_applies_retriever_stack(tmp_path):
    from llm386 import Store

    config_path = tmp_path / "llm386.toml"
    config_path.write_text(
        '[[retriever]]\n'
        'kind = "bm25"\n'
        'k1 = 1.5\n'
        '\n'
        '[[retriever]]\n'
        'kind = "recency"\n'
        'half_life_secs = 60.0\n'
    )
    store = Store(str(tmp_path / "store"), profiles=str(config_path))
    store.put(session=1, kind="fact", body="paris is the capital of france")
    plan = store.page(session=1, model="gpt-4o", task="what is the capital of france")
    assert len(plan.selected) >= 1


def test_store_with_invalid_retriever_kind_raises(tmp_path):
    from llm386 import LLM386Error, Store

    config_path = tmp_path / "llm386.toml"
    config_path.write_text('[[retriever]]\nkind = "bogus-kind"\n')
    store = Store(str(tmp_path / "store"), profiles=str(config_path))
    store.put(session=1, kind="fact", body="x")
    with pytest.raises(LLM386Error):
        store.page(session=1, model="gpt-4o", task="x")


def test_python_retriever_runs_inside_pager(store):
    """A user-defined Python retriever can surface custom blocks
    into the pager's candidate set."""

    favored_id = store.put(session=1, kind="fact", body="favored block")
    store.put(session=1, kind="fact", body="other block")

    class FavoritesRetriever:
        name = "favorites"

        def retrieve(self, session, task, limit):
            # Always boost the favored block to the top.
            return [(favored_id, 1.0)]

    store.add_python_retriever(FavoritesRetriever())
    plan = store.page(session=1, model="gpt-4o", task="anything")
    assert favored_id in plan.selected
    # Highest-scored block is ours.
    assert plan.selected[0] == favored_id


def test_python_retriever_missing_name_raises(store):
    class BadRetriever:
        def retrieve(self, session, task, limit):
            return []

    with pytest.raises(AttributeError):
        store.add_python_retriever(BadRetriever())


def test_python_retriever_returning_bad_shape_raises(store):
    store.put(session=1, kind="fact", body="x")

    class BrokenRetriever:
        name = "broken"

        def retrieve(self, session, task, limit):
            # Wrong shape: list of strings instead of tuples.
            return ["not a tuple"]

    store.add_python_retriever(BrokenRetriever())
    with pytest.raises(LLM386Error):
        store.page(session=1, model="gpt-4o", task="x")


def test_clear_python_retrievers_removes_them(store):
    favored_id = store.put(session=1, kind="fact", body="favored")
    store.put(session=1, kind="fact", body="other")

    class FavoritesRetriever:
        name = "favorites"

        def retrieve(self, session, task, limit):
            return [(favored_id, 1.0)]

    store.add_python_retriever(FavoritesRetriever())
    store.clear_python_retrievers()
    plan = store.page(session=1, model="gpt-4o", task="x")
    # Without the booster, ordering reverts to the default
    # RecencyRetriever — favored_id was inserted first so it ranks
    # lowest, not highest.
    assert plan.selected[0] != favored_id


def test_summarize_truncating_returns_text(store):
    for i in range(3):
        store.put(session=1, kind="fact", body=f"fact number {i}")
    out = store.summarize(session=1, summarizer="truncating", max_chars=50)
    assert "fact" in out


def test_summarize_store_summary_persists_block(store):
    for i in range(3):
        store.put(session=1, kind="fact", body=f"fact number {i}")
    before = len(store.list_sessions())
    store.summarize(session=1, summarizer="truncating", store_summary=True)
    # The summary block lands in the same session.
    assert len(store.list_sessions()) == before


def test_list_models_includes_built_ins():
    models = list_models()
    names = {m.name for m in models}
    assert "gpt-4o" in names
    assert "claude-opus-4-7" in names
    for m in models:
        assert m.max_context_tokens > 0


def test_unknown_model_falls_back_without_raising(store):
    """Unknown model names resolve to the default fallback (currently
    `gpt-4o`) instead of raising. Pins the data-driven registry
    contract from `crates/llm386-core/data/models.toml`. See also the
    Rust resolver tests in `crates/llm386-core/src/model.rs`."""
    store.put(session=1, kind="fact", body="something")
    # Bare unknown name → default fallback.
    store.page(session=1, model="bogus-model-name", task="x")
    # Provider-prefixed unknown name → same default fallback.
    store.page(session=1, model="openrouter/google/gemini-2.5", task="x")
    # Provider-prefixed *known* name → exact match via prefix strip.
    store.page(session=1, model="anthropic/claude-sonnet-4-6", task="x")
    # Unknown version within a known family → family fallback.
    store.page(session=1, model="anthropic/claude-sonnet-4-9", task="x")


def test_show_unknown_block_raises_llm386_error(store):
    with pytest.raises(LLM386Error):
        store.show("0" * 32)


def test_delete_removes_block(store):
    block_id = store.put(session=1, kind="fact", body="to be deleted")
    assert store.delete(block_id) is True
    with pytest.raises(LLM386Error):
        store.show(block_id)


def test_delete_returns_false_for_unknown(store):
    assert store.delete("0" * 32) is False


def test_purge_session_removes_session_blocks(store):
    for i in range(4):
        store.put(session=1, kind="fact", body=f"fact {i}")
    sessions_before = store.list_sessions()
    assert "00000000000000000000000000000001" in sessions_before
    purged = store.purge_session(1)
    assert purged == 4
    sessions_after = store.list_sessions()
    assert "00000000000000000000000000000001" not in sessions_after


def test_purge_session_keeps_blocks_shared_with_other_sessions(store):
    a = store.put(session=1, kind="fact", body="shared")
    b = store.put(session=2, kind="fact", body="shared")
    assert a == b
    store.purge_session(1)
    # The block survives in session 2.
    block = store.show(b)
    assert block.body == b"shared"
