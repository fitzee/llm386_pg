# LLM386 — Design

This document explains the design decisions behind LLM386. For the elevator pitch and quick start, see the [README](./README.md). For the Postgres backend specifically, see [POSTGRES.md](./POSTGRES.md). For operational answers, see the [FAQ](./FAQ.md).

## The core invariant

**Store state ≠ model state.**

Everything else in LLM386's design follows from this. The model is a stateless function `f(context) → output`. Continuity across calls is *reconstructed*, not preserved. The store is the source of truth; what the model sees on any given call is a *computed working set* derived from that store, the current task, retrieval results, and the model's token budget.

Treating the store as authoritative — and the model's view as derived — is what makes the system inspectable, recoverable, replayable, and auditable. It also draws a clean line between "what's known about the conversation" (store) and "what's in front of the model right now" (working set), which most agent frameworks blur.

## Why these design choices

A few specific decisions are worth calling out:

### The model never owns durable state

Every byte the model sees comes from a block in the store, with provenance attached. Output gets parsed, validated, and committed back as new blocks via a `Reducer`. The system stays inspectable and recoverable.

### Prompt assembly is deterministic

Same inputs produce a byte-identical rendered prompt. Determinism holds *only* when all four of the following hold:

- block ordering is stable;
- tokenizer version is identical;
- packer logic is unchanged;
- retriever outputs are deterministic.

Same inputs → byte-identical prompt → stable trace. The trace layer relies on this for replay.

### Budgeting is model-aware, not hand-rolled

A `ModelProfile` carries `max_context_tokens`, `reserved_output_tokens`, `safety_margin_tokens`, and a `tokenizer` id. The pager and packer respect that contract regardless of which model you swap in.

### Sections, not just blocks

The pager allocates the input budget across canonical sections (System, Task, State, Plan, Retrieved, Tools, Recent, Background) with a tunable `SectionBudgetTable`. This matches how a chat-style prompt is actually structured.

### Edge-aware paging

Blocks rarely make sense in isolation. A tool result without the assistant message that called it is noise; a counter-claim without the claim it contradicts is misleading. LLM386 lets you persist typed directed edges (`add_edge from --to --kind`) and the pager acts on each kind's *meaning* when assembling a working set, in two bounded passes after the per-section fill:

- **Expansion** pulls dependent blocks in. `Supports`: selecting a claim co-retrieves its evidence. `ToolInvocation`: selecting an assistant turn drags along the tool result it consumed. `Parent`: selecting a child pulls its container (and, in `bidirectional` mode, a container pulls its children). `DerivedFrom` (`co-retrieve-source` mode): selecting a summary also pulls its source. Pulled blocks are tagged `SelectionReason::Dependency`.
- **Reconciliation** acts on the final set. `Contradicts`: the newer / higher-priority block wins; the other is either dropped (`prefer-newer`) or kept with an inline "contradicted by newer block …" flag (`flag`). `DerivedFrom` (`suppress-source` mode): a source the selected summary already covers is dropped as redundant.

Traversal is **bounded** — a depth cap (default one hop, hard ceiling five) and a per-kind fan-out cap (default eight) keep a long chain or a high-degree hub from blowing the budget or the latency floor. Every kind is **independently configurable** via an `[edges]` config block (see [docs/CLI.md](docs/CLI.md#edge-aware-paging)); set `enabled = false` to ignore edges entirely. Each kind defaults to its most useful mode. `Provenance.parents` lineage is followed only when `follow_provenance_parents` is set, since the typed `Parent`/`DerivedFrom` edges normally carry the same information.

### Pluggable retrieval

The default `RecencyRetriever` is fine for chat-style use. Add `LexicalRetriever` or `Bm25Retriever` for keyword search. Add `LinearAnnRetriever` or `HnswAnnRetriever` (with the bundled OpenAI embedder, or your own `Embedder` impl) when you need semantic recall. They compose: the pager fans out across all configured retrievers and merges results by max score per block.

All retriever scores must be normalized to `[0, 1]`. Mixing scoring systems (BM25, cosine, recency) without normalization will bias selection — the pager assumes scores are comparable and does not fix mismatched scales.

### Storage and serialization are explicit

LMDB for persistence, postcard for block bodies, hand-rolled big-endian keys. No JSON in the hot path. Content-hash dedup means identical bytes get stored once even across sessions. Reads observe a consistent snapshot at transaction start; writes become visible atomically after commit. There is no partial visibility of a multi-key write.

### Observability is built in

Each `pack` call can record a `TraceRecord` (CallId, session, model, plan, prompt hash, duration, model and tokenizer version). After the model returns, patch the response back in via `TraceSink::update_output` so the trace is replay-complete. Inspect a single record with `llm386 trace show`; compare two with `llm386 trace diff` to see which blocks moved in or out and what the input-token delta was.

### Explicit state via reducers

Reducers define the only path from model output to persistent state. Model output is never trusted directly: every state update must be parsed, validated, and committed as blocks and edges. A `Reducer` (`identity`, `append-output`, `json-events`, or your own impl) turns the response into a `Reduction { next_state, new_blocks, new_edges }` that the agent commits. Reducers are pure on `(state, output)` so a recorded trace plus its reducer is enough to reconstruct what changed.

## Token savings

LLM386 does not shrink any individual block. It controls *whether* a block is included, *how* it's ordered, and whether a cheaper substitute fits in its place. The levers:

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

## Custom TOML config

A TOML file (passed via `--profiles <path>` or the `LLM386_PROFILES` environment variable) carries seven optional sections:

```toml
[store]
backend = "lmdb"
path    = "./store"
# Or:  backend = "pg", url = "postgres://user@host/db", schema = "llm386"

[[profile]]
name = "my-tiny"
family = "my-tiny"          # optional; enables family fallback for "my-tiny-v2" etc.
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

## Library usage

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

## Using as a memory layer in an agent

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

## Advanced pattern: verification-heavy agents with freshness reasoning

The [README's "When to use" section](./README.md#when-to-use-llm386--and-when-not-to) frames the default fit question by agent archetype. Persistent rich context fits agents whose value comes from accumulated reasoning over expensive-to-rederive state; it misfits agents whose job is grounded verification of cheap, mutable, current state.

If your agent falls into the "fits poorly" bucket but you still want LLM386's benefits — cross-turn conversational continuity, prompt-cache friendliness, structured retrieval — there is a design space worth exploring. The over-confidence failure mode isn't strictly intrinsic to persistent context; it's the result of giving the model rich context **without** the meta-information needed to reason about whether to trust it. With the right coordination, persistent context and per-turn verification can coexist.

This pattern is more work than either the baseline "trust everything cached" or the simple "verify everything per turn" defaults, and it has only been partially validated in the wild. Treat it as a design hypothesis you'll evaluate, not a turnkey configuration.

The pattern has three coordinated parts. None is sufficient alone; the value comes from all three being in place.

### 1. Persist tool results with full provenance

LLM386's typed-edge graph and the `tool-invocation` edge kind already preserve assistant-message ↔ tool-result pairing. To make freshness reasoning possible, also surface in the rendered prompt:

- The `tool_call_id` (so the model sees a real assistant tool-call ↔ tool-result pair, not an unmoored "Tool result: ..." chunk).
- The tool name and arguments that produced the result.
- The wall-clock time the tool ran (typically the block's `created_at`, rendered as ISO 8601 in the prompt).

LLM386 stores all of this; persisting it with the block content (via a small wrapper encoding, or via per-block metadata fields if added to the schema) lets the consumer reconstruct properly-typed `AIMessage(tool_calls=[...])` + `ToolMessage(tool_call_id=...)` pairs on re-pack, and lets the model see *when* each cached tool call ran.

A representative prompt fragment after this is in place:

```
[assistant, 2026-05-31T10:42:00Z]
  tool_call: list_dashboards(owned_by_me=true) [id=tc_a8f3]
[tool, 2026-05-31T10:42:00Z, id=tc_a8f3]
  {"total_count": 7, "dashboards": [...]}
```

### 2. Teach the model to reason about freshness with a per-tool freshness model

Models are empirically poor at deriving "is this stale?" from a timestamp alone. The system prompt has to give them an explicit freshness model categorized by tool class. There is no generic answer — the right model is per-agent and per-tool-surface, authored by someone who knows what each tool returns and how mutable that data is. Indicative template:

```
Tool result freshness:
- Identity tools (whoami, get_user_info): valid for the session.
- Schema definition tools (list_schemas, describe_table): valid for hours;
  re-call only if the user mentions schema changes.
- Workspace metadata tools (list_dashboards, list_datasets): valid for
  ~5 minutes within an active conversation; re-call if the user implies
  recency ("just added", "what's new", "now").
- Query result tools (execute_sql, run_chart): valid only for the current
  turn — always re-call on follow-up questions about live data.

When using a prior tool result, cite when it was called
("as of 10:42 UTC, you had 7 dashboards"). If the data class above
says the result may have changed, re-call the tool instead of citing.
```

This is unusual prompt content but not radical. The model can follow rules about freshness *if* the rules are given; without them, it defaults to either "always trust the prompt" (over-confidence) or "always re-fetch" (cost + latency). The freshness model effectively becomes a fourth column in your tool catalog: name, args, description, **freshness class**.

### 3. Evaluate against a per-conversation grounding contract, not per-turn

Default LLM judge rubrics ask: *"Did this turn's claim come from this turn's tool calls?"* That contract penalizes any use of cached tool results, regardless of whether the use is correct. To evaluate this pattern fairly, the judge needs to reason about conversational grounding with freshness:

```
A factual claim is grounded if:
  (a) it comes from a tool call in the current turn, OR
  (b) it comes from a tool call in a previous turn in the same
      conversation, AND the data class is stable enough that the
      prior call is still valid, AND the assistant cited when the
      data was retrieved.

A claim is ungrounded if it relies on training-data general knowledge
with no supporting tool call (current or prior).

A claim is a stale-reference failure if it cites a prior tool result
for data that should have been re-verified.
```

This rubric is strictly harder to write well than per-turn grounding. The risk is that judges instructed to be lenient on prior-turn grounding also become lenient on real fabrications. The rubric has to be tight enough that the bar moves in the right direction without collapsing into "anything that looks plausible passes."

### Testing this pattern

A minimum-viable evaluation requires all three pieces in place, run against a representative judge suite:

1. LLM386 active with persisted tool results carrying provenance metadata as described above.
2. System prompt instrumented with a per-tool freshness model authored for the agent's actual tool surface.
3. Judge rubric updated to score per-conversation grounding with freshness rules.

Without all three, the eval will misattribute the result: with (1) but not (2), the model defaults to over-trust; with (1) and (2) but not (3), even correct multi-turn grounding gets penalized as "no tool call this turn."

Outcomes that meaningfully validate the pattern:

- **Quality reaches or beats a per-turn-verification baseline** on the grounding detectors, *at lower cost per conversation* than the always-verify baseline. This is the design win — persistent context + reasoned freshness is genuinely cheaper-and-better than per-turn verification.
- **Quality recovers partially**, but the recovery is concentrated on stable-data tool classes (identity, schema) while volatile classes (query results) still drag. This tells you which tool classes can move to the cached-with-freshness regime and which should stay always-verify.
- **Quality doesn't recover**: the model can't reason about freshness reliably enough at this scale, and the pattern doesn't rescue the agent — back to the "doesn't fit" bucket with a clearer understanding of why.

### When this pattern is worth the engineering

This is non-trivial coordination across the runtime, the agent's prompt, and the eval harness. It pays off when:

- The agent is large enough that the per-conversation cost difference matters (production volume, paid-per-call backend, or long-horizon sessions).
- The agent's tool surface has a meaningful spread of freshness classes — some genuinely stable data (worth caching) alongside volatile data (must re-verify). A surface where everything is one freshness class doesn't benefit much from the discrimination.
- There is operational appetite to maintain a freshness model for the tool catalog over time. Adding a tool means assigning its freshness class.

If those don't hold, the simpler per-turn-verification path is likely the right answer, and this pattern is over-engineering for the problem.

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

## Non-goals

- Hosting a chat UI.
- Hiding state inside prompts.
- Treating the model as the source of truth.
- A custom distributed storage layer (sharding, replication, consensus). The Postgres backend rides on Postgres's own primary-replica story; LMDB stays single-host.
