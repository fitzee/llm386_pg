"""Edge-aware paging through the Python binding.

Exercises every typed edge kind end-to-end: `add_edge` persistence for
all five kinds, the expansion pass (Supports co-retrieval driven through
`page()` with a tight budget), and the reconciliation pass (Contradicts
flagging and DerivedFrom suppression), including the `[edges]` config
overrides that let a consumer pick and choose per kind.
"""

from __future__ import annotations

from llm386 import Store


def _store_with_config(tmp_path, toml: str) -> Store:
    cfg = tmp_path / "profiles.toml"
    cfg.write_text(toml)
    return Store(str(tmp_path / "store"), profiles=str(cfg))


def test_add_edge_roundtrips_every_kind(store):
    a = store.put(session=1, kind="fact", body="A")
    b = store.put(session=1, kind="fact", body="B")
    for kind in ("parent", "derived-from", "supports", "contradicts", "tool-invocation"):
        store.add_edge(a, b, kind)
    out = dict(store.edges_from(a))
    assert out[b] in {"parent", "derived-from", "supports", "contradicts", "tool-invocation"}
    # edges_from returns one tuple per kind; all five point at b.
    kinds = {k for (_to, k) in store.edges_from(a)}
    assert kinds == {"parent", "derived-from", "supports", "contradicts", "tool-invocation"}
    # Mirror index: b sees a as the incoming endpoint for each kind.
    froms = {f for (f, _k) in store.edges_to(b)}
    assert froms == {a}


def test_supports_edge_co_retrieves_evidence_through_page(tmp_path):
    # Tiny budget + zero Retrieved allocation means a Fact is normally
    # omitted; a Supports edge from a selected claim should still pull
    # it in via the expansion pass, while an unlinked Fact stays out.
    toml = """
[[profile]]
name = "tiny"
max_context_tokens = 80
reserved_output_tokens = 0
tokenizer = "cl100k_base"

[section_budgets]
recent = 0.9
retrieved = 0.0
"""
    s = _store_with_config(tmp_path, toml)
    claim = s.put(session=1, kind="assistant-message", body="we should deploy on friday")
    evidence = s.put(session=1, kind="fact", body="the deploy checklist passed all gates")
    distractor = s.put(session=1, kind="fact", body="unrelated trivia about penguins")
    s.add_edge(claim, evidence, "supports")

    plan = s.page(session=1, model="tiny", task="q")
    assert claim in plan.selected
    assert evidence in plan.selected, "Supports edge should co-retrieve the evidence"
    assert distractor not in plan.selected, "unlinked Fact stays omitted (Retrieved budget 0)"
    # The pulled evidence is tagged as a dependency.
    ev_sel = next(sel for sel in plan.selections if sel.block_id == evidence)
    assert ev_sel.reason == "dependency"


def test_contradicts_flag_annotates_older_block_by_default(store):
    # Default edge policy flags (keeps both) on contradiction. Priority
    # disambiguates the winner so the test is deterministic regardless
    # of put() timing.
    older = store.put(session=1, kind="fact", body="the capital is sydney", priority=0.0)
    newer = store.put(session=1, kind="fact", body="the capital is canberra", priority=1.0)
    store.add_edge(newer, older, "contradicts")

    plan = store.page(session=1, model="gpt-4o", task="what is the capital")
    assert older in plan.selected and newer in plan.selected, "Flag mode keeps both"
    older_sel = next(s for s in plan.selections if s.block_id == older)
    newer_sel = next(s for s in plan.selections if s.block_id == newer)
    assert older_sel.note is not None
    assert "contradicted by newer block" in older_sel.note
    assert newer_sel.note is None

    # The flag is rendered into the packed chat output.
    result = store.pack(session=1, model="gpt-4o", task="what is the capital", chat=True)
    assert any("contradicted by newer block" in m.content for m in result.messages)


def test_contradicts_can_be_disabled_via_config(tmp_path):
    toml = "[edges]\nenabled = false\n"
    s = _store_with_config(tmp_path, toml)
    older = s.put(session=1, kind="fact", body="the capital is sydney", priority=0.0)
    newer = s.put(session=1, kind="fact", body="the capital is canberra", priority=1.0)
    s.add_edge(newer, older, "contradicts")

    plan = s.page(session=1, model="gpt-4o", task="what is the capital")
    older_sel = next(s for s in plan.selections if s.block_id == older)
    assert older_sel.note is None, "edges disabled → no contradiction flag"


def test_derived_from_suppress_source_via_config(tmp_path):
    toml = '[edges]\nderived_from = "suppress-source"\n'
    s = _store_with_config(tmp_path, toml)
    source = s.put(session=1, kind="fact", body="the long original passage with detail")
    derived = s.put(session=1, kind="summary", body="short summary of the passage")
    s.add_edge(derived, source, "derived-from")

    plan = s.page(session=1, model="gpt-4o", task="recap")
    assert derived in plan.selected
    assert source not in plan.selected, "source is superseded by the selected summary"
    dropped = next(o for o in plan.omitted if o.block_id == source)
    assert dropped.reason == "SupersededByDerived"


def test_derived_from_co_retrieves_source_by_default(store):
    # Default DerivedFrom mode keeps the source rather than suppressing.
    source = store.put(session=1, kind="fact", body="the long original passage with detail")
    derived = store.put(session=1, kind="summary", body="short summary of the passage")
    store.add_edge(derived, source, "derived-from")

    plan = store.page(session=1, model="gpt-4o", task="recap")
    assert derived in plan.selected
    assert source in plan.selected, "default CoRetrieveSource keeps the source available"
