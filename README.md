# LLM386 — Postgres backend fork

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org)
[![Edition](https://img.shields.io/badge/edition-2024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](#status)

> This is a fork of [LLM386](https://github.com/fitzee/llm386) that adds a **PostgreSQL-backed `BlockStore`** alongside the upstream LMDB one. The trait surface in `llm386-core` is unchanged; backends are selected at construction time. See [Postgres backend](#postgres-backend) for the rationale, schema, and a head-to-head perf comparison against LMDB.

A Rust runtime that manages the external state needed to feed an LLM. It treats the model as a stateless inference function and handles the rest: persistent block storage, retrieval, paging into a model-specific token budget, and deterministic prompt assembly.

**Core invariant.** LLM386 does not extend the model's context window. It constructs a bounded working set for each call by selecting, ordering, and rendering blocks from a larger persistent memory. Every model invocation is independent; continuity is achieved by recomputing the working set each time.

The name is a nod to EMM386, the DOS-era memory manager that paged a larger external memory space into a smaller active working set. Same idea, applied to LLM context windows.

## Problem

The model is a pure function:

```
f(context) → output
```

It has no memory, no persistence, and no cross-call state. All continuity must be reconstructed per call.

An LLM call has three properties that make this hard to use directly from application code:

1. The model itself holds no state. Every call re-derives everything from the prompt.
2. The prompt has a hard token budget that varies by model.
3. The data you want to feed in (conversation history, retrieved documents, tool results) almost always exceeds that budget.

Most projects end up reinventing the same pieces per agent or chatbot:

- A storage layer for messages and documents.
- Some way to retrieve relevant content (often just "last N messages").
- Token counting per model family.
- A function that stitches everything into a prompt under the budget.
- Some way to inspect what got included and why when things go wrong.

That code tends to be ad-hoc, model-specific, and hard to test.

## What

Context is not history. Context is a *computed working set* derived from the current task, the active state, retrieval results, and the model's token budget. The pager defines this working set; the packer renders it. The model sees only the rendered output.

LLM386 is the runtime under that surface. The pieces:

- A persistent block store (LMDB-backed) that holds every input the model has seen or might see, keyed and deduplicated by content hash.
- A typed-edge graph between blocks (`Parent`, `DerivedFrom`, `Supports`, `Contradicts`, `ToolInvocation`) that the pager can follow when assembling a working set, so dependent blocks travel together.
- A model registry that knows context windows, output reservations, tokenizers, and capability flags per model.
- A pager that picks which blocks fit the current call, applying per-section budgets and pluggable retrievers (recency, lexical, BM25, embedding ANN, pinned ids). Each selection records *why* it was included (`Pinned`, `HighRelevance`, `Recency`, `Dependency`, `GlobalFact`, `ToolResult`).
- A packer that turns the pager's plan into a deterministic prompt string or a list of role-tagged chat messages. Only the rendered prompt text or chat messages are sent to the model. The model never sees block ids, hashes, provenance, edges, or retrieval scores. Store state ≠ model state.
- A tracer that records every page+pack call (with model build, tokenizer version, and the patched-in model output) so you can replay, audit, or diff it later.
- A diff layer that computes structured deltas between any two trace records: which blocks were added or removed, which kept blocks changed inclusion reason, and the input-token delta.
- A reducer trait for turning model output into committed state and event blocks. Reference impls cover the no-op case, raw output append, and a JSON-envelope parser.
- A summarizer trait with a pure truncating implementation, plus an Anthropic-backed implementation in a separate crate for LLM-driven summaries.
- A CLI that exposes the whole pipeline.

It is a library first. The CLI is a thin shell over the library.

## Postgres backend

This fork adds `llm386-store-pg`, a `BlockStore` implementation backed by PostgreSQL, sitting alongside the upstream `llm386-store-lmdb`. The trait surface in `llm386-core` is unchanged — every other crate (pager, packer, retriever, trace, reduce) is backend-agnostic.

### Why a Postgres backend

LMDB is a memory-mapped, single-writer, single-host store. That is the right answer for many deployments — embedded, latency-sensitive, single-process — and the LMDB backend wins on read paths by a wide margin (see the perf table below). It is the wrong answer for a few specific scenarios:

- **Multiple processes write to the same store.** LMDB serializes writers across the entire env; only one process holds the write lock at a time. Pod-replicated runtimes hit this immediately.
- **No shared filesystem.** Orchestrators that schedule processes across nodes have no portable story for a writable LMDB file — NFS/EFS-mounted LMDB is undefined behavior.
- **Operational machinery already exists for Postgres.** Connection pooling, schema migrations, snapshot backups, point-in-time recovery, observability, multi-tenant ACLs — every cloud-native runtime already runs this for its primary database. Reusing it for the LLM context store has no marginal ops cost.
- **Native ACID across processes.** `put` writes the block, the hash index row, and the session-membership row in a single transaction. Concurrent writers don't see partial state.
- **Future ANN co-location.** Adding `pgvector` lets `llm386-retrieve-ann` resolve against the same table that holds the blocks — schema metadata, conversation turns, tool results, and their embeddings all live in one table, queried by a single JOIN with an HNSW index. No separate vector database to operate.

The PG backend is not "better" than LMDB. It targets a different operational model.

### Schema

Four tables, created on first connect via idempotent `CREATE TABLE IF NOT EXISTS` and re-checked on every open via a `schema_version` row in `llm386_meta`:

```sql
CREATE TABLE llm386_blocks (
    id              BYTEA PRIMARY KEY,          -- 16-byte big-endian u128, time-ordered
    kind            TEXT NOT NULL,
    bytes           BYTEA NOT NULL,
    token_counts    JSONB NOT NULL DEFAULT '{}'::jsonb,
    priority        REAL NOT NULL DEFAULT 0.0,
    created_at      BIGINT NOT NULL,            -- unix ms
    updated_at      BIGINT NOT NULL,
    provenance      JSONB NOT NULL DEFAULT '{}'::jsonb,
    hash            BYTEA NOT NULL              -- blake3, 32 bytes
);
CREATE UNIQUE INDEX llm386_blocks_hash_idx ON llm386_blocks (hash);

CREATE TABLE llm386_session_blocks (
    session_id  BYTEA NOT NULL,
    block_id    BYTEA NOT NULL,
    PRIMARY KEY (session_id, block_id)
);

CREATE TABLE llm386_edges (
    from_id     BYTEA NOT NULL,
    to_id       BYTEA NOT NULL,
    kind        TEXT NOT NULL,
    PRIMARY KEY (from_id, kind, to_id)
);
CREATE INDEX llm386_edges_to ON llm386_edges (to_id, kind);
```

A few design notes:

- **`BYTEA(16)` for ids, not `UUID`.** `BlockId`'s chronological ordering comes from the raw bit layout (48-bit timestamp in the high bits). Postgres `UUID` compares lexicographically, which would scramble that ordering. Big-endian `BYTEA` preserves the natural `Ord` semantics exactly — `ORDER BY block_id ASC` returns blocks chronologically without a separate time column.
- **Dedup via `ON CONFLICT (hash) DO NOTHING`.** A single statement satisfies the content-hash dedup invariant; a follow-up `SELECT id FROM llm386_blocks WHERE hash = $1` recovers the existing id when the insert collided.
- **`JSONB` for structured fields.** `token_counts` and `provenance` round-trip through JSONB, with `BlockId` lineage encoded as hex strings (Postgres JSON can't represent u128 numerically). `provenance->>'labels'` queries work without schema changes.

### When to use which

- **LMDB**: embedded library, single process, read-heavy, low-latency. The default for the bundled CLI and Python SDK.
- **Postgres**: multi-process or multi-node deployments, shared operational infrastructure, ACID across writers, future ANN co-location with `pgvector`.

Full decision guide, including what you give up either way: [FAQ → Should I use LMDB or Postgres?](./FAQ.md#should-i-use-lmdb-or-postgres-for-the-block-store-what-am-i-giving-up).

### Selecting the backend

The same CLI binary and Python package handle both backends. Selection is driven by a `[store]` section in the shared `--profiles` TOML, with command-line / kwarg overrides.

**TOML config** (`llm386.toml`):

```toml
[store]
backend = "lmdb"
path    = "./store"
```

or:

```toml
[store]
backend = "pg"
url     = "postgres://user@host/db"
schema  = "llm386"          # optional, defaults to public
pool_size = 8               # optional, defaults to 8
```

**CLI** — `--store` for LMDB, `--pg-url` for Postgres. Either flag pinned in TOML, overridden by the matching flag, or supplied entirely on the command line:

```bash
# LMDB via flag (no config file needed):
llm386 --store ./store page --session 1 --model gpt-4o --task "..."

# Postgres via flag:
llm386 --pg-url postgres://user@host/db page --session 1 --model gpt-4o --task "..."

# Backend pinned in TOML; no flag needed:
llm386 --profiles ./llm386.toml page --session 1 --model gpt-4o --task "..."
```

**Python** — positional `path` opens LMDB, the `url=` kwarg opens Postgres, or load `[store]` from a config:

```python
from llm386 import Store

# LMDB:
store = Store("./store")

# Postgres:
store = Store(url="postgres://user@host/db")

# Backend from config:
store = Store(profiles="./llm386.toml")
```

**Library** — open whichever concrete store you need and hand it to the pager / packer:

```rust
use llm386_store_pg::{PgStore, PgStoreConfig};

let store = PgStore::open(
    "postgres://user:pass@host/db",
    &PgStoreConfig::default(),
)?;
```

The Postgres schema bootstraps on first connect; subsequent opens reuse the existing tables. Tests pass `schema: Some("my_test_schema".into())` to scope every pooled connection to an isolated schema via `SET search_path`.

`verify` / `repair` integrity tooling is intentionally not ported — Postgres has native equivalents (`pg_dump`, foreign-key checks, `REINDEX`, `VACUUM`). The CLI errors cleanly when you try (`verify is LMDB-only (active backend: pg)`).

### Async opt-in: `AsyncPgStore`

`PgStore` implements the synchronous `BlockStore` trait that the rest of the workspace (pager, packer, trace) consumes. That works fine for single-request-at-a-time call patterns — but it means a single Postgres query at a time per call site, with the calling thread blocked on the socket.

For service deployments handling many concurrent conversations, the `async` feature exposes a second store type backed by `tokio-postgres` + `deadpool-postgres`:

```toml
[dependencies]
llm386-store-pg = { version = "...", features = ["async"] }
```

```rust
use llm386_store_pg::{AsyncPgStore, PgStoreConfig};

let store = AsyncPgStore::open(
    "postgres://user:pass@host/db",
    &PgStoreConfig::default(),
).await?;

// Concurrent puts share a small pool, pipelined down each socket:
let (a, b, c) = tokio::try_join!(
    store.put(session, block_a),
    store.put(session, block_b),
    store.put(session, block_c),
)?;
```

`AsyncPgStore` **does not implement** the sync `BlockStore` trait — that's deliberate. Mixing async storage behind a sync trait would require `block_on` inside every call, defeating the point. Use `AsyncPgStore` directly from async call sites (axum handlers, tonic services, ingest pipelines) when you want real concurrency through a pool of connections. Keep using `PgStore` everywhere else, including as the storage backend handed to the pager / packer.

The two stores share the schema, the migration logic, and `PgStoreConfig` — they're interchangeable in terms of on-disk state. You can write through one and read through the other against the same database.

### Performance

Perf hammer in `crates/llm386-store-bench/`: 10,000 blocks of 1 KiB each, single thread, local Unix socket to PostgreSQL 18 on the same host (macOS 25.4, APFS, M-series).

```
backend  workload    samples        total    ops/sec        p50        p95        p99
----------------------------------------------------------------------------------
lmdb     put          10,000       37.70s        265      4.0ms      4.1ms      5.1ms
lmdb     get          10,000       24.0ms    416,165      2.0µs      4.9µs      6.5µs
lmdb     list            100        8.8ms     11,300     88.7µs     94.4µs    106.2µs
lmdb     dedup        10,000       33.70s        297      3.0ms      4.1ms      5.0ms
pg       put          10,000        3.10s      3,229    303.0µs    347.1µs    525.1µs
pg       get          10,000        1.24s      8,077    122.7µs    136.8µs    148.8µs
pg       list            100      126.1ms        793      1.3ms      1.3ms      1.3ms
pg       dedup        10,000        3.98s      2,515    392.5µs    427.5µs    469.2µs
```

What the numbers say:

| Op | Winner | Margin | Why |
|---|---|---|---|
| **put** | PG | ~12× | LMDB fsyncs the WAL on every commit (4 ms p50 on APFS). PG batches via its own WAL and the per-op commit cost amortizes better. |
| **dedup put** | PG | ~8× | Same story — LMDB's transaction commit dominates even when the dedup path skips two writes. |
| **get** | LMDB | ~50× | Mmap'd page lookup (~2 µs) vs. Postgres protocol roundtrip over a UNIX socket (~120 µs). The socket floor. |
| **list_session** | LMDB | ~14× | In-process BTree walk vs. an indexed range scan plus protocol serialization. |

**Caveat on the LMDB write number.** The current `LmdbStore::put` opens a fresh write transaction per block — that is what the `BlockStore` trait shape encourages, but it is the LMDB worst case. With batched commits (many puts per transaction) LMDB outpaces Postgres on writes by an order of magnitude. The numbers above reflect the trait as it stands, not LMDB's upper bound.

**Caveat on the PG read numbers.** Reads go over a local Unix socket here. Across a real network with a few ms RTT, PG `get` and `list` latencies grow accordingly. The relative ordering doesn't change, but the absolute floor moves.

The bench binary is parameterised; run your own:

```
cargo build --release -p llm386-store-bench
./target/release/store-bench \
    --pg-url postgres://user@host/db \
    --blocks 10000 --bytes 1024 \
    --workloads put,get,list,dedup \
    --backends lmdb,pg
```

## Why

A few design choices are worth calling out:

**The model never owns durable state.** Every byte the model sees comes from a block in the store, with provenance attached. Output gets parsed, validated, and committed back as new blocks. The system stays inspectable and recoverable.

**Prompt assembly is deterministic.** Same inputs produce a byte-identical rendered prompt. Determinism holds *only* when all four of the following hold:

- block ordering is stable;
- tokenizer version is identical;
- packer logic is unchanged;
- retriever outputs are deterministic.

Same inputs → byte-identical prompt → stable trace. The trace layer relies on this for replay.

**Budgeting is model-aware, not hand-rolled.** A `ModelProfile` carries `max_context_tokens`, `reserved_output_tokens`, `safety_margin_tokens`, and a `tokenizer` id. The pager and packer respect that contract regardless of which model you swap in.

**Sections, not just blocks.** The pager allocates the input budget across canonical sections (System, Task, State, Plan, Retrieved, Tools, Recent, Background) with a tunable `SectionBudgetTable`. This matches how a chat-style prompt is actually structured.

**Edge-aware paging.** Blocks rarely make sense in isolation. A tool result without the assistant message that called it is noise; a counter-claim without the claim it contradicts is misleading. LLM386 lets you persist typed directed edges (`add_edge from --to --kind`) and the pager follows them when assembling a working set. When `include_parents` is on, every selected block's outgoing edges are walked transitively and any unselected dependency that still fits the global budget is pulled in (tagged `SelectionReason::Dependency`). Existing `Provenance.parents` lineage is followed too, so old data keeps working.

**Pluggable retrieval.** The default `RecencyRetriever` is fine for chat-style use. Add `LexicalRetriever` or `Bm25Retriever` for keyword search. Add `LinearAnnRetriever` or `HnswAnnRetriever` (with the bundled OpenAI embedder, or your own `Embedder` impl) when you need semantic recall. They compose: the pager fans out across all configured retrievers and merges results by max score per block.

All retriever scores must be normalized to `[0, 1]`. Mixing scoring systems (BM25, cosine, recency) without normalization will bias selection — the pager assumes scores are comparable and does not fix mismatched scales.

**Storage and serialization are explicit.** LMDB for persistence, postcard for block bodies, hand-rolled big-endian keys. No JSON in the hot path. Content-hash dedup means identical bytes get stored once even across sessions. Reads observe a consistent snapshot at transaction start; writes become visible atomically after commit. There is no partial visibility of a multi-key write.

**Observability is built in.** Each `pack` call can record a `TraceRecord` (CallId, session, model, plan, prompt hash, duration, model and tokenizer version). After the model returns, patch the response back in via `TraceSink::update_output` so the trace is replay-complete. Inspect a single record with `llm386 trace show`; compare two with `llm386 trace diff` to see which blocks moved in or out and what the input-token delta was.

**Explicit state via reducers.** Reducers define the only path from model output to persistent state. Model output is never trusted directly: every state update must be parsed, validated, and committed as blocks and edges. A `Reducer` (`identity`, `append-output`, `json-events`, or your own impl) turns the response into a `Reduction { next_state, new_blocks, new_edges }` that the agent commits. Reducers are pure on `(state, output)` so a recorded trace plus its reducer is enough to reconstruct what changed.

LLM386 is a working-set manager for LLMs.

## Token savings

The runtime does not shrink any individual block. It controls *whether* a block is included, *how* it's ordered, and whether a cheaper substitute fits in its place. The levers:

**Section budgets are a hard ceiling.** Working sets drift upward over time as new retrievers and tools get bolted on. `SectionBudgetTable` caps each section; anything that doesn't fit is omitted with a reason recorded in the trace. Prompts cannot silently grow.

**Content-hash dedupe.** Identical bytes are stored once. The same fact arriving from two retrievers is one block in the store and one block in the prompt.

**Summary substitution.** With `llm386 summarize ... --store-summary` (truncating or Anthropic-backed), the pager can prefer a short `Summary` block over a long transcript block under budget pressure. The runtime provides the slot; the savings only materialize if you actually run summarization.

**Cache-friendly determinism + an explicit cache boundary.** Anthropic and OpenAI prompt caching key off stable prefixes. The `pack_chat` output emits one message per section, ordered with `stable_sections` (default: `system` + `background`) at the front, and returns a `cache_boundary` index pointing at the last message of the stable prefix. A provider adapter can set Anthropic `cache_control` on `messages[cache_boundary]` or slice `messages[0..=cache_boundary]` into a Gemini `CachedContent` — turn N and turn N+1 share that prefix and the cached portion is billed at a fraction of full price. OpenAI auto-caches and ignores the boundary. Configurable per project via `[cache] stable_sections` in the TOML config. Full breakdown — per-provider pricing, TTLs, when it doesn't pay off — in the [FAQ entry on cost and prompt caching](./FAQ.md#does-this-reduce-token-cost-how-do-i-get-prompt-caching-to-actually-hit).

**Pre-tokenized counts.** Token counts are cached per tokenizer and per block. The prompt cost is known before the call, not discovered at the API.

Realistic expectation: against a "concat everything into the prompt" baseline in a long-running session, the combined effect is large — often a 5–20× input-token reduction once budgets, summarization, and cache hits are all on. Against an already-tight RAG pipeline that hand-picks its context per call, the marginal win is smaller: primarily dedupe, cache hits, and the safety of a ceiling that cannot be exceeded.

## Failure modes

The runtime makes context assembly inspectable; it doesn't prevent you from feeding it nonsense. Common issues in production:

- **Context flooding.** Too many large blocks survive into the working set; the model gets a low-signal prompt and answers degrade.
- **Retriever dominance.** One retriever returns inflated scores and crowds out the others.
- **Stale facts.** Outdated blocks repeatedly retrieved and parroted as current.
- **Over-summarization.** Summary substitution drops a critical detail; the model has *less* useful information than if the original block had been omitted entirely.
- **Token fragmentation.** Many small low-value blocks clog the section budgets.

Mitigations:

- Normalize and weight retrievers (every retriever score in `[0, 1]`).
- Purge or downgrade stale blocks (drop priority toward `0.0`, or `purge` outright).
- Summarize cold data with `--store-summary` so the pager can substitute summaries for the original blocks under budget pressure.
- Enforce section budgets — defaults are starting points, not invariants.

`llm386 trace diff` between a healthy turn and a degraded turn is the fastest way to localize which of these is biting you. See the [Failure modes FAQ entry](./FAQ.md#failure-modes) for more.

## How

### Install

Requires Rust 1.95 or newer.

```
cargo build --release
```

The CLI binary is `target/release/llm386`. Full subcommand reference: [`docs/CLI.md`](./docs/CLI.md). For operational questions (latency, multi-tenancy, RAG, custom retrievers, failure modes), see the [`FAQ`](./FAQ.md).

### Quick start

```
llm386 init ./store

echo "You are a concise assistant." | llm386 --store ./store put --session 1 --kind system -
echo "What is the capital of Australia?" | llm386 --store ./store put --session 1 --kind user-message -
echo "Canberra." | llm386 --store ./store put --session 1 --kind assistant-message -
echo "It became the capital in 1908." | llm386 --store ./store put --session 1 --kind fact -

llm386 list-models
llm386 --store ./store page --session 1 --model gpt-4o --task "explain Australia's history"
llm386 --store ./store pack --session 1 --model gpt-4o --task "explain Australia's history"
```

`pack` prints the rendered prompt on stdout with a manifest header on stderr. Redirect with `> prompt.txt` to capture just the prompt.

To run the same commands against Postgres, swap `--store ./store` for `--pg-url postgres://user@host/db` — every subcommand below works identically. Or pin the backend once in a TOML config and drop the flag (see [Selecting the backend](#selecting-the-backend)).

### Variants

Render as JSON for a chat API:

```
llm386 --store ./store pack --session 1 --model gpt-4o --task "..." --chat
```

Record a trace and inspect it later:

```
llm386 --store ./store pack --session 1 --model gpt-4o --task "..." --trace ./traces
llm386 trace show --trace-store ./traces <call-id>
```

Diff two trace records to see what changed between turns:

```
llm386 trace diff --trace-store ./traces <prev-call-id> <next-call-id>
```

Output looks like `summary: +2 -1 ~1 (+184 tokens)` plus a per-block breakdown of additions, removals, and inclusion-reason changes.

Add typed edges between blocks and inspect them:

```
llm386 --store ./store add-edge --from <claim-id> --to <evidence-id> --kind supports
llm386 --store ./store edges <claim-id>
llm386 --store ./store edges <evidence-id> --incoming
```

Edge kinds: `parent`, `derived-from`, `supports`, `contradicts`, `tool-invocation`. Re-adding the same triple is a no-op; deleting a block scrubs every edge that touches it.

Inspect a single block:

```
llm386 --store ./store show <block-id>
llm386 --store ./store show <block-id> --json
```

List sessions in a store:

```
llm386 --store ./store list-sessions
```

Summarize a session:

```
llm386 --store ./store summarize --session 1 --summarizer truncating --max-chars 80
ANTHROPIC_API_KEY=... llm386 --store ./store summarize --session 1 --summarizer anthropic --store-summary
```

### Custom config

A TOML file (passed via `--profiles <path>` or the `LLM386_PROFILES` environment variable) carries seven optional sections:

```toml
[store]
backend = "lmdb"
path    = "./store"
# Or:  backend = "pg", url = "postgres://user@host/db", schema = "llm386"

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

[section_budgets]
state      = 0.10
plan       = 0.05
recent     = 0.20
retrieved  = 0.40
tools      = 0.15
background = 0.05
slack      = 0.05

[packer]
include_timestamps = true

[cache]
stable_sections = ["system", "background"]
```

`[store]` pins the block-store backend (LMDB path or Postgres URL); the matching CLI flag (`--store` / `--pg-url`) and the Python `path=` / `url=` kwargs override it. `[[profile]]` adds model profiles on top of the built-ins. `[[hf_tokenizer]]` registers a HuggingFace tokenizer.json (used by Llama, Qwen, Mistral, and similar). `[[retriever]]` replaces the default retriever stack. `[section_budgets]` overrides the per-section fractions of the variable budget — fractions sum to ≤ 1.0, anything routed to `slack` is reserved headroom that is never filled. `[packer]` toggles opt-in packer behavior — `include_timestamps = true` prepends each rendered block with its ISO 8601 UTC creation timestamp and emits a "Current time" anchor in the Task section so the model can reason about *when* things happened, not just *what* was said. `[cache]` declares which sections (`system`, `state`, `plan`, `retrieved`, `background`) are considered stable across turns; `pack_chat` emits stable sections first and returns a `cache_boundary` index pointing at the last stable message, for downstream adapters to set provider cache markers (Anthropic `cache_control`, Gemini `CachedContent`). Default `stable_sections = ["system", "background"]`. See [`examples/configs/`](./examples/configs/) for three worked profiles (focused Q&A, chat loop, RAG-heavy).

### Library

```rust
use std::sync::Arc;
use llm386_core::{PageRequest, SessionId, default_registry};
use llm386_pager::GreedyPager;
use llm386_packer::SimplePacker;
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::cl100k_base;

let store = Arc::new(LmdbStore::open("./store", StoreConfig::default())?);
let tokenizer = Arc::new(cl100k_base()?);
let model = default_registry().get("gpt-4o").unwrap().clone();

let pager = GreedyPager::new(store.clone(), tokenizer.clone());
let packer = SimplePacker::new(store, tokenizer);

let request = PageRequest {
    session_id: SessionId(1),
    task: "answer the user".into(),
    model,
    required_blocks: vec![],
};
let plan = pager.page(request.clone())?;
let prompt = packer.pack(&request, &plan)?;
println!("{}", prompt.rendered);
```

Every component is replaceable: `Pager`, `Packer`, `Retriever`, `Tokenizer`, `Embedder`, `Summarizer`, `BlockStore`, and `TraceSink` are all traits in `llm386-core`.

### Using as a memory layer

LLM386 is the memory and context-assembly layer behind an agent. The boundary must remain explicit:

- **LLM386 owns:** memory, retrieval, context construction.
- **The agent owns:** control flow, tool execution, model invocation.

LLM386 owns "what does the model see this turn?" and "what got produced?". The agent owns everything around that.

A single agent turn looks like this:

1. `put` the user input as a `UserMessage` block.
2. `pack` the session for the target model and task. You get back a rendered prompt (or, with `--chat`, a list of role-tagged chat messages ready to send to a chat-completion API). When `--trace` is set, the call is recorded with a `CallId` returned to you.
3. Send that to the model.
4. Run the response through a `Reducer` to produce a `Reduction { next_state, new_blocks, new_edges }`. Commit the new blocks and edges to the store. The simplest useful reducer (`AppendOutputReducer`) just stores the response as an `AssistantMessage` and links it to the prior state.
5. If the model called a tool, `put` each tool result as a `ToolResult` block with `provenance.parents = [assistant_block_id]` (or via `add_edge --kind tool-invocation`) so the pager keeps them paired on subsequent turns.
6. Patch the model output back into the trace with `TraceSink::update_output` so the record is replay-complete.
7. Repeat.

A Python sketch using the [`llm386` SDK](./python/) (in `python/`):

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
    # for tool_result in response.tool_results:
    #     store.put(session_id, kind="tool-result", body=tool_result,
    #               parents=[asst_id])
    return reply
```

The Python package is a PyO3-built native extension (no separate binary or daemon required). See [`python/README.md`](./python/README.md) for the full Python API.

### Framework hooks

Most Python agent frameworks expose a place to plug in custom memory. The pattern is the same in each case: the framework owns flow control and tool execution, LLM386 owns what the model sees.

**LangGraph:** in each node that calls the model, fetch context via `pack` and write the output back via `put`. Use the LangGraph thread id as the LLM386 session id so checkpoints and stored blocks line up.

**CrewAI:** subclass the framework's memory base class and route `save` to `put` and `search` to a `page` call. A `Bm25Retriever` plus `LinearAnnRetriever` (or `HnswAnnRetriever` for larger sessions) is a reasonable default retriever stack for this use.

**AutoGen:** wrap the agent's `generate_reply` so it draws context from `pack` instead of from the agent's local message list. The agent still emits its own messages; you just intercept ingestion and assembly.

`pack --trace ./traces` records each turn so you can later audit exactly what the model saw and why.

### Runnable demo

A working LangGraph integration ships under [`examples/langgraph-agent/`](./examples/langgraph-agent/). It's a small chatbot with two stub tools (a calculator and a fake user-profile lookup) using LLM386 as its memory layer. The whole thing runs in Docker — no Rust toolchain or local Python setup required — so you can be chatting in 5 minutes.

```
export ANTHROPIC_API_KEY=sk-ant-...
docker compose -f examples/langgraph-agent/docker-compose.yml run --rm agent
```

A sample session illustrating cross-turn recall (no LangGraph state is preserved between turns — the recall is entirely from LLM386):

```
you> what's 17 * 23?
[llm386] selected 1 blocks (54 est. tokens, 2 chat messages packed)
bot> 391.

you> look up user u-002
[llm386] selected 3 blocks (98 est. tokens, 4 chat messages packed)
bot> Diego, free tier, America/Bogota.

you> what was my arithmetic question's answer?
[llm386] selected 5 blocks (156 est. tokens, 6 chat messages packed)
bot> 391.
```

What the demo demonstrates concretely:

- **Memory-as-a-layer.** Every turn does `store.page() → store.pack(chat=True)`; LangGraph itself holds no chat history.
- **Tool result linkage via typed edges.** Tool outputs become `tool-result` blocks tied to the calling assistant via `add_edge(..., kind="tool-invocation")`, so the pager keeps call/result paired on later turns.
- **Pluggable retrievers from config.** A bundled `llm386.toml` switches in BM25 + recency, loaded by `Store(profiles=...)` with no code change.
- **Persistence across container restarts.** The store is a Docker volume; stop and restart the container and the agent picks up where it left off.
- **Same image carries the CLI.** `docker compose run --rm cli show --store /data/store <block-id>` works against the same volume — useful for poking at what got stored after a session.

The example's [README](./examples/langgraph-agent/README.md) has the full breakdown of what each turn does, how to inspect the store, how to reset it, and an honest list of what's deliberately *not* covered (real RAG ingest, MCP tool servers, multi-agent topologies — all of which are documented in the [FAQ](./FAQ.md)).

## Architecture

```
crates/
  llm386-core                 types and trait seams (incl. Edge, Selection, Reducer)
  llm386-store-lmdb           LMDB BlockStore impl, edges_from / edges_to indexes
  llm386-store-pg             PostgreSQL BlockStore impl (this fork)
  llm386-store-bench          perf hammer that runs identical workloads against both stores
  llm386-tokenizer            tiktoken + HuggingFace tokenizer adapters, registry, LRU cache
  llm386-pager                GreedyPager, SectionBudgetTable, retrievers, edge-aware inclusion
  llm386-packer               SimplePacker (string and chat-message rendering)
  llm386-trace                LMDB-backed TraceSink with update_output for post-call patching
  llm386-compress             pure summarizers (Noop, Truncating)
  llm386-compress-anthropic   Anthropic-backed Summarizer
  llm386-reduce               Reducer impls: Identity, AppendOutput, JsonEvents
  llm386-diff                 PromptDiff between two PagePlans / TraceRecords
  llm386-retrieve-ann         LinearAnnRetriever, HnswAnnRetriever, OpenAiEmbedder, EmbeddingCache
  llm386-cli                  the `llm386` binary
```

The dependency direction is one-way: every impl crate depends on `llm386-core` for traits and types, never on a sibling.

## Status

Early. The single-node embedded library, CLI, and Python SDK all work end to end against both LMDB and Postgres. The Postgres backend in this fork is feature-complete against the `BlockStore` trait (full parity with LMDB including `delete`, `purge_session`, `list_sessions`, edges in both directions) and is exercised by an integration test suite gated on `TEST_DATABASE_URL`. The bundled CLI and Python SDK pick a backend via the same `[store]` TOML section (with `--store` / `--pg-url` / `url=` overrides); see [Selecting the backend](#selecting-the-backend). Interfaces are stable enough for downstream consumers to build on, but expect breaking changes as new retrievers, summarizers, and storage backends land.

## Non-goals

- Hosting a chat UI.
- Hiding state inside prompts.
- Treating the model as the source of truth.
- A custom distributed storage layer (sharding, replication, consensus). The Postgres backend rides on Postgres's own primary-replica story; LMDB stays single-host.

## See also

- [`FAQ`](./FAQ.md) — operational reference: how context is exposed to the model, performance and sizing, data lifecycle, sessions and multi-tenancy, retrieval, MCP/tools integration, failure modes.
- [`docs/CLI.md`](./docs/CLI.md) — full `llm386` subcommand reference with worked examples.
- [`python/README.md`](./python/README.md) — Python SDK (PyO3 native extension), framework integration patterns, custom Python retrievers.
- [`examples/langgraph-agent/`](./examples/langgraph-agent/) — runnable Docker tutorial: a LangGraph chatbot with two tools, using LLM386 as its memory layer. `docker compose run --rm agent` and you're chatting in 5 minutes.
- [`examples/cascade-routing/`](./examples/cascade-routing/) — runnable Docker tutorial: cheap-first cascade routing (Haiku → escalate to Opus on low confidence), using `pack_with_plan` to re-render one selection for two models without re-paging. Demonstrates the `cache_boundary` field driving Anthropic `cache_control`.

## License

Apache-2.0.
