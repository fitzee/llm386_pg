# LLM386 — graph-backed working-set manager for LLMs

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.95%2B-orange.svg)](https://www.rust-lang.org)
[![Edition](https://img.shields.io/badge/edition-2024-orange.svg)](https://doc.rust-lang.org/edition-guide/rust-2024/index.html)
[![Status](https://img.shields.io/badge/status-alpha-yellow.svg)](#status)

> This is a fork of [LLM386](https://github.com/fitzee/llm386) that adds a **PostgreSQL-backed `BlockStore`** alongside the upstream LMDB one. The trait surface is unchanged; pick a backend at construction time. Backend rationale, schema, async opt-in, TLS, and benchmarks live in [POSTGRES.md](./POSTGRES.md).

**Store state ≠ model state.**

LLM386 is a Rust runtime that manages the external state needed to feed a language model. It treats the model as a stateless inference function and handles the rest: persistent block storage, **typed-edge graph retrieval**, and deterministic packing into a model-specific token budget. Continuity across calls is reconstructed each turn from the store — never preserved in the model.

The name is a nod to EMM386, the DOS-era memory manager that paged a larger external memory space into a smaller active working set. Same idea, applied to LLM context windows.

## What makes it different

Most "memory" libraries for LLMs give you messages, vectors, and retrieval. LLM386 adds:

- **A typed-edge graph between blocks.** `claim --supports--> evidence`, `summary --derived_from--> source`, `assistant --tool_invocation--> tool_result`, `claim --contradicts--> claim`. The pager treats edges as semantically meaningful — selecting a claim co-retrieves its supporting evidence, selecting a summary pulls in its source, contradictions get reconciled. Most memory systems treat blocks as independent; LLM386 treats them as a graph and pages over it.
- **Deterministic packing.** Same inputs produce a byte-identical rendered prompt. Trace, replay, and diff are first-class — `llm386 trace diff <prev> <next>` tells you exactly which blocks moved in or out between two turns.
- **Section-budgeted, model-aware paging.** The packer allocates the input budget across canonical sections (`System`, `State`, `Plan`, `Retrieved`, `Tools`, `Recent`, `Background`, `Task`) with a tunable `SectionBudgetTable`. Per-model context window, output reservation, and tokenizer are looked up from a `ModelProfile` registry.
- **Cache-friendly by construction.** `pack_chat` emits stable sections first and returns a `cache_boundary` index, so provider adapters can mark `cache_control` for Anthropic or slice into `CachedContent` for Gemini. Turn N and turn N+1 share that prefix at fractional cost.
- **Pluggable retrieval.** Recency, lexical, BM25, embedding ANN, custom Python — compose any. Scores normalized to `[0, 1]` and merged by max-per-block.
- **Two storage backends, identical trait.** Embedded (LMDB) or multi-process (Postgres) — pick at construction time; everything above is backend-agnostic. See [POSTGRES.md](./POSTGRES.md).
- **Library first.** No daemon. The CLI is a thin shell over the library; the Python SDK is a PyO3 native extension.

For the design rationale — why deterministic packing, what reducers are for, how edge-aware paging actually works inside the pager — see [DESIGN.md](./DESIGN.md). Operational answers (latency, multi-tenancy, RAG, custom retrievers, failure modes) are in the [FAQ](./FAQ.md).

## Who this is for

LLM386 is infrastructure. If your goal is `agent.chat(...)` and you don't want to think about memory, modern model vendors handle that for you. LLM386 is for people building:

- **Agent platforms and orchestration frameworks** that need explicit, inspectable working-set construction across many agents.
- **Enterprise AI infrastructure** that needs auditable, replayable prompts and deterministic context assembly across model upgrades.
- **Evaluation systems** that need byte-identical prompt reproduction between runs.
- **Long-running autonomous agents** whose value comes from accumulated reasoning over many turns and whose context can't be re-derived per call.

If you're building a chatbot and want a memory hook to drop into LangGraph or CrewAI, see [DESIGN.md → Framework hooks](./DESIGN.md#framework-hooks) — that path works and is documented, but it isn't where LLM386's differentiation lives.

## Quick start

Requires Rust 1.95 or newer.

```bash
cargo build --release
```

The CLI binary lands at `target/release/llm386`. Try it:

```bash
llm386 init ./store

echo "You are a concise assistant." | llm386 --store ./store put --session 1 --kind system -
echo "What is the capital of Australia?" | llm386 --store ./store put --session 1 --kind user-message -
echo "Canberra." | llm386 --store ./store put --session 1 --kind assistant-message -
echo "It became the capital in 1908." | llm386 --store ./store put --session 1 --kind fact -

llm386 --store ./store pack --session 1 --model gpt-4o --task "explain Australia's history"
```

`pack` prints the rendered prompt on stdout with a manifest header on stderr.

Add typed edges between blocks and the pager will co-retrieve them:

```bash
llm386 --store ./store add-edge --from <claim-id> --to <evidence-id> --kind supports
```

For Postgres instead of LMDB, swap `--store ./store` for `--pg-url postgres://user@host/db` — every subcommand works identically against either backend. See [POSTGRES.md](./POSTGRES.md) for the full backend story (when to use which, schema, async opt-in, TLS, benchmarks).

The Python SDK is in [`python/`](./python/) — a PyO3 native extension, `pip install llm386`. A runnable LangGraph + Docker demo (chat in 5 minutes, no toolchain needed) is in [`examples/langgraph-agent/`](./examples/langgraph-agent/) — full walkthrough in [DESIGN.md → Runnable demo](./DESIGN.md#runnable-demo).

Full CLI subcommand reference: [`docs/CLI.md`](./docs/CLI.md).

## When to use LLM386 — and when not to

LLM386's value-prop is **persistent, retrievable, edge-aware context that survives across turns and across sessions**. Whether that's a help or a hindrance for your agent depends on the agent's profile, not on LLM386's implementation quality. The fit question is worth answering up front: the cost of getting it wrong is paid every turn, in tokens and in a behavioral side-effect described below.

### The mechanism that matters

The model treats relevant retrieved context as **authoritative working memory**, not as reference material that might be stale or partial. The more on-point each retrieved block is, the harder it is for the model to choose "I should verify this with a tool call" over "I already know." Effective retrieval is the very thing that triggers this.

For some agents, that's exactly the design goal: don't re-derive what was already established. For others, it's an active failure mode: the agent's job is to verify current state, and skipping that step produces wrong answers.

### Agents this fits well

Persistent rich context is a strong fit when **the model's value comes from accumulated reasoning over expensive-to-rederive state**, and the state being reasoned about is stable enough that "cached belief" is a reasonable default:

- **Tool calls or retrievals are expensive** — slow APIs, rate-limited integrations, paid corpus access. Re-fetching has a real cost; remembering pays for itself.
- **The world being reasoned about changes slowly or not at all** — a code repository at a point in time, a research corpus, a document being drafted.
- **Reasoning IS the work product** — agents that build up plans, intermediate analyses, evidence chains, or synthesis across turns. The intermediate state is what gets delivered.

Concrete examples: code-review or pair-programming agents over a PR; long-form drafting assistants; multi-step planners and research/synthesis agents; agents over slow / rate-limited backends.

### Agents this fits poorly

Persistent rich context is the wrong default when **the agent's job is grounded verification of current state**, and the cost of a fabricated answer is high relative to the cost of an extra tool call:

- **Tool calls return current state of a mutable world** — workspace metadata, live query results, file contents under active edit.
- **Tool calls are cheap** — low-latency metadata reads, no LLM tokens, no quota cost.
- **A fabricated answer costs more than an extra tool call** — user trust, downstream actions on bad data, compliance / safety implications.

Concrete examples: workspace introspection agents, live-data assistants, operational agents that take actions, short-horizon Q&A. For these, a simpler `trim_messages`-style recency truncation is often accidentally well-calibrated — letting older context fall out of the prompt keeps the model honest about what it actually knows right now.

### In between?

If your agent's profile is mixed — long-running collaboration on partly-mutable state — there's a hybrid pattern (persistent context + per-tool freshness model + per-conversation grounding eval) documented in [DESIGN.md → Advanced pattern](./DESIGN.md#advanced-pattern-verification-heavy-agents-with-freshness-reasoning). It's more engineering than either extreme but can rescue verification-heavy agents from the over-confidence trap. Alternatively, expose retrieval as a **tool the model invokes on demand** rather than as auto-injected context — same data, different framing for the model's confidence.

### Evaluate before committing

Independent of which side of the line your agent sits on, the calibration cost is empirically non-negligible. Plan to evaluate both architectures (LLM386 active vs. simple recency-trim baseline) on a representative judge suite before committing. Specifically watch:

- **Provenance / grounding detectors** — does confidence calibration change when persistent context is active?
- **Tool-call rate per turn** — fewer can mean a win (less redundant work) or a loss (skipped verification), depending on the agent.
- **Per-conversation cost** — persistent context isn't automatically cheaper. The prompt-caching win has to exceed per-turn context-injection overhead.

The architecture is sound; the question is whether your agent's profile is one where richer working memory helps the model reason or fools it into skipping verification. Answer that empirically.

## Status

Early. The single-node embedded library, CLI, and Python SDK all work end-to-end against both LMDB and Postgres. The Postgres backend in this fork is feature-complete against the `BlockStore` trait (full parity with LMDB including `delete`, `purge_session`, `list_sessions`, edges in both directions) and is exercised by an integration test suite gated on `TEST_DATABASE_URL`. Interfaces are stable enough for downstream consumers to build on, but expect breaking changes as new retrievers, summarizers, and storage backends land.

## Where to go from here

- **[DESIGN.md](./DESIGN.md)** — design rationale, deterministic packing, edge-aware paging internals, token-savings mechanics, failure modes, custom TOML config, library API, framework hooks, runnable demo, advanced patterns (incl. the verification-heavy freshness-reasoning hybrid).
- **[POSTGRES.md](./POSTGRES.md)** — Postgres backend: when to pick it, schema, selection across CLI/library/Python, async opt-in, TLS, performance benchmarks (with honest framing on what they do and don't say).
- **[FAQ.md](./FAQ.md)** — operational answers: how context gets exposed to the model, performance and sizing, data lifecycle, sessions and multi-tenancy, retrieval / RAG, MCP and tools integration, failure modes, comparison to other approaches.
- **[`docs/CLI.md`](./docs/CLI.md)** — full `llm386` subcommand reference with worked examples.
- **[`python/README.md`](./python/README.md)** — Python SDK (PyO3 native extension), framework integration patterns, custom Python retrievers.
- **[`examples/langgraph-agent/`](./examples/langgraph-agent/)** — runnable Docker tutorial: a LangGraph chatbot with two tools, using LLM386 as its memory layer. `docker compose run --rm agent` and you're chatting in 5 minutes.
- **[`examples/cascade-routing/`](./examples/cascade-routing/)** — runnable Docker tutorial: cheap-first cascade routing (Haiku → escalate to Opus on low confidence), using `pack_with_plan` to re-render one selection for two models without re-paging.

## License

Apache-2.0.
