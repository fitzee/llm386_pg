# llm386 (Python)

Python bindings for the [LLM386](../README.md) context virtualization runtime, built with PyO3 + maturin. The whole runtime ships as a native extension; no separate binary or daemon is needed.

## Install

```
pip install llm386
```

## Build from source

```
pip install maturin
cd python
maturin develop
```

## Status

Alpha (`1.0.0a0`). The package is a PyO3 native extension; the surface mirrors the earlier CLI-shelling pure-Python wrapper so existing code keeps working.

Custom retrievers written in Python work today (see "Custom Python retrievers" below). Embedder and Summarizer Python adapters follow the same pattern and land next.

## Quick start

```python
from llm386 import Store, list_models

# Open or initialize an LMDB store at ./store. Idempotent.
store = Store("./store")

block_id = store.put(session=1, kind="user-message", body="What is the capital of Australia?")
store.put(session=1, kind="assistant-message", body="Canberra.")

plan = store.page(session=1, model="gpt-4o", task="explain Australia's history")
print(plan.selected, plan.estimated_tokens)

result = store.pack(session=1, model="gpt-4o", task="explain Australia's history", chat=True)
for msg in result.messages:
    print(f"[{msg.role}] {msg.content}")
```

## Backend selection

`Store` supports two persistent backends. Pick by argument or by config:

```python
from llm386 import Store

# LMDB (positional path) — the default, embedded, single-process.
store = Store("./store")

# Postgres (url kwarg) — multi-process, multi-node, ACID across writers.
store = Store(url="postgres://user@host/db")

# Backend pinned in a TOML config — same schema the CLI reads.
store = Store(profiles="./llm386.toml")
```

`profiles` can carry a `[store]` section that pins the backend without code changes; positional `path` and the `url` kwarg override the matching TOML field if both are given. Passing both `path` and `url` raises `LLM386Error`.

For the decision of *which* to pick — and what you give up either way — see [FAQ → Should I use LMDB or Postgres?](../FAQ.md#should-i-use-lmdb-or-postgres-for-the-block-store-what-am-i-giving-up). Short version: LMDB unless you need multi-process writes, no shared filesystem, or you already operate Postgres.

Once opened, every `Store` method (`put`, `page`, `pack`, `summarize`, edges, traces, custom retrievers) works identically against either backend.

### TLS for Postgres

The default Postgres connection is plaintext (`tls = "disable"`). **Any non-localhost deployment should opt in to TLS** — set it in the `[store]` section of your TOML config:

```toml
[store]
backend = "pg"
url     = "postgres://user@host/db"
tls     = "require"               # or "require-custom-ca"
# tls_ca_path = "/etc/ssl/pg-ca.pem"   # required when tls = "require-custom-ca"
```

TLS support is feature-gated on the Rust side — build the extension with the feature on:

```bash
maturin develop --release -F tls-native-tls
# or for a wheel:
maturin build --release -F tls-native-tls
```

Without the feature, `tls = "require"` raises `LLM386Error: TLS mode … requires the tls-native-tls feature` at `Store(...)` time. There is **no silent fall-through to plaintext**. `tls = "require"` also forces `sslmode=require` on the underlying connection, so the postgres client refuses to fall back to plaintext if the server doesn't offer TLS. Full background in the [README → TLS section](../README.md#tls).

## Using as a memory layer in an agent loop

```python
from llm386 import Store
from openai import OpenAI

store = Store("./store")
client = OpenAI()

def turn(session_id: int, user_input: str) -> str:
    store.put(session_id, kind="user-message", body=user_input)

    result = store.pack(session=session_id, model="gpt-4o",
                         task="answer the user", chat=True)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": m.role, "content": m.content} for m in result.messages],
    )
    reply = response.choices[0].message.content

    asst_id = store.put(session_id, kind="assistant-message", body=reply)

    # If the model called tools, store each result with the assistant
    # message as a parent so the pager keeps them paired.
    # for tool_result in tool_results:
    #     store.put(session_id, kind="tool-result", body=tool_result,
    #               parents=[asst_id])

    return reply
```

## Trace + replay

```python
from llm386 import Store, Trace

store = Store("./store")

result = store.pack(session=1, model="gpt-4o", task="...",
                    chat=True, trace="./traces")

if result.trace_id:
    record = Trace("./traces").show(result.trace_id)
    print(f"{record.model} call took {record.duration_ms} ms, "
          f"{record.prompt_tokens} prompt tokens, "
          f"{len(record.plan.selected)} blocks selected")
```

`TraceRecord` exposes the full record: `call_id`, `session`, `model`, `plan` (a `PagePlan`), `prompt_tokens`, `prompt_hash`, `started_at` (ms since epoch), `duration_ms`, plus `model_version`, `tokenizer_version`, `output` (`Optional[str]`), and `output_tokens` (`Optional[int]`). The output fields are `None` until you patch them in after the model returns:

```python
trace_store = Trace("./traces")
trace_store.update_output(call_id, reply, usage.completion_tokens)
```

Doing this gives you a replay-complete trace: the exact prompt, the exact model build, and the exact response.

## Custom profiles, tokenizers, retrievers

Pass a TOML config path via `profiles=`. Same schema the CLI uses:

```python
store = Store("./store", profiles="./llm386.toml")
```

```toml
# llm386.toml

[store]
backend = "pg"
url     = "postgres://user@host/db"
schema  = "llm386"          # optional, defaults to public
pool_size = 8               # optional, defaults to 8
# Or:  backend = "lmdb", path = "./store"

[[profile]]
name = "my-tiny"
max_context_tokens = 4096
reserved_output_tokens = 1024
tokenizer = "cl100k_base"

[[hf_tokenizer]]
name = "llama-3"
path = "/path/to/llama-3-tokenizer.json"

[[retriever]]
kind = "bm25"
k1 = 1.5

[[retriever]]
kind = "recency"
half_life_secs = 60.0
```

`[store]` pins the backend — overridden by the positional `path` or `url=` kwarg if both are given. `[[profile]]` adds model profiles on top of the built-ins. `[[hf_tokenizer]]` registers a HuggingFace tokenizer.json for non-OpenAI models. `[[retriever]]` replaces the default RecencyRetriever stack with whatever you configure.

## Summarization

```python
# Pure (no API call):
print(store.summarize(session=1, summarizer="truncating", max_chars=80))

# Via Anthropic Claude (set ANTHROPIC_API_KEY):
print(store.summarize(session=1, summarizer="anthropic", store_summary=True))
```

## Typed edges

Beyond `Provenance.parents` (lineage), you can persist typed directed edges between blocks. Edge-aware paging follows them when assembling a working set so dependencies travel together.

```python
store.add_edge(claim_id, evidence_id, "supports")
store.add_edge(assistant_msg_id, tool_result_id, "tool-invocation")

# Inspect:
for to_id, kind in store.edges_from(claim_id):
    print(kind, to_id)
for from_id, kind in store.edges_to(evidence_id):
    print(from_id, kind)
```

Kinds: `"parent"`, `"derived-from"`, `"supports"`, `"contradicts"`, `"tool-invocation"`. Re-adding the same triple is a no-op. Deleting or purging a block removes every edge that touches it.

## Custom Python retrievers

Write a class with a `name` attribute and a `retrieve(session, task, limit)` method that returns a list of `(block_id_hex, score)` tuples. Register it on the Store, and the Rust pager calls back into your code as part of every `page()` / `pack()`.

```python
from llm386 import Store

class FavoritesRetriever:
    name = "favorites"

    def __init__(self, favored_ids: list[str]):
        self.favored_ids = favored_ids

    def retrieve(self, session: int, task: str, limit: int):
        return [(bid, 1.0) for bid in self.favored_ids[:limit]]

store = Store("./store")
store.add_python_retriever(FavoritesRetriever(["019abc..."]))
plan = store.page(session=1, model="gpt-4o", task="anything")
```

Python retrievers compose alongside any TOML-configured retrievers and the default `RecencyRetriever` fallback. Scores are clamped to `[0, 1]` and merged by `BlockId` (max wins).

`store.clear_python_retrievers()` drops everything previously registered.

For Pinecone, Weaviate, or any other vector DB, this is the integration point: implement `retrieve` against your client.

## API surface

```python
from llm386 import (
    Store,           # main entry point
    Trace,           # trace store reader
    list_models,     # discover available model profiles

    # Result types
    ChatMessage, ContextBlock, ModelProfile,
    OmittedBlock, PackResult, PagePlan, Provenance, Selection,

    LLM386Error,     # raised when the CLI invocation fails
)
```

## License

Apache-2.0.
