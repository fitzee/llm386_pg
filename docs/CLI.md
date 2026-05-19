# `llm386` CLI

Reference for every subcommand of the `llm386` binary. The CLI is a thin shell over the library; anything you can do here you can also do programmatically via the Rust crates or the Python SDK. The CLI exists for two audiences: the operator who wants to inspect / repair a store, and the developer who wants to iterate on context assembly without writing Rust.

If you only want to skim, the [Quick start](../README.md#quick-start) in the project README walks the happy path. This file documents *every* command.

## Conventions

- Block, session, call, and content-hash ids are 128-bit. They accept three forms: decimal (`42`), `0x`-prefixed hex (`0x7b73...`), or bare 32-char hex (the form printed by `llm386 show`, `page`, and `pack`).
- The block-store backend is selected by global flags (`--store` for LMDB, `--pg-url` for Postgres) or pinned in the `[store]` section of the `--profiles` TOML. The flag wins over the matching field in TOML; mixing kinds (e.g. `--pg-url` when TOML says `backend = "lmdb"`) errors. See [Backend selection](#backend-selection) below.
- `--profiles <path>` is a global flag that loads extra `[[profile]]`, `[[hf_tokenizer]]`, `[[retriever]]`, `[store]`, `[section_budgets]`, `[packer]`, and `[cache]` entries from a TOML file. Set once and use with any subcommand. Equivalent env var: `LLM386_PROFILES`.
- The trace store is a *separate* LMDB sink and uses its own `--trace-store <path>` flag on the trace subcommands. It is independent of the block-store backend choice.
- Long-running output (rendered prompts, JSON dumps) goes to stdout; status, manifest headers, and trace ids go to stderr. Redirect them separately.
- Logging is controlled by `RUST_LOG` (default `warn`). `RUST_LOG=info,llm386=debug llm386 pack ...` is useful when something looks wrong.

## Global flags

| Flag | Description |
| --- | --- |
| `--store <path>` | LMDB block-store directory. Created on first use. Overrides `path` in the TOML `[store]` section. Mutually exclusive with `--pg-url`. |
| `--pg-url <url>` | PostgreSQL connection URL for the block store. Schema bootstraps on first connect. Overrides `url` in the TOML `[store]` section. Mutually exclusive with `--store`. |
| `--profiles <path>` | TOML config with extra model profiles, HF tokenizers, a backend pin, a custom retriever stack, and packer/cache knobs. See [README → Custom config](../README.md#custom-config) for the full schema. |
| `--help`, `-h` | Help on any command. `llm386 trace diff --help` works too. |
| `--version` | Print the binary version. |

## Backend selection

The same `llm386` binary drives either LMDB or PostgreSQL. Resolution order, highest priority first:

1. **TOML `[store]` section** picks the backend *kind*. Within the chosen kind, an explicit field (e.g. `url`) is used unless overridden by the matching flag.
2. **CLI flag**: `--store` (LMDB) or `--pg-url` (PG). Wins over the matching field in TOML; with no `[store]` section, fully selects the backend.

Examples:

```bash
# LMDB, no config file:
llm386 --store ./store page --session 1 --model gpt-4o --task "..."

# Postgres, no config file:
llm386 --pg-url postgres://user@host/db page --session 1 --model gpt-4o --task "..."

# Backend pinned in TOML; no flag needed:
cat > llm386.toml <<EOF
[store]
backend = "pg"
url     = "postgres://user@host/db"
schema  = "llm386"
EOF
llm386 --profiles ./llm386.toml page --session 1 --model gpt-4o --task "..."
```

Subcommands that depend on backend-specific surface area (currently `verify` and `repair`, which walk LMDB internals) error cleanly when run against PG (`verify is LMDB-only (active backend: pg)`). Postgres has native equivalents — `pg_dump`, foreign-key checks, `REINDEX`, `VACUUM` — that are run outside this tool.

For the decision of *which* backend to pick, see [FAQ → Should I use LMDB or Postgres?](../FAQ.md#should-i-use-lmdb-or-postgres-for-the-block-store-what-am-i-giving-up).

### TLS for the Postgres backend

The Postgres URL alone says nothing about transport security — the CLI's default is plaintext (`tls = "disable"`). For any non-localhost deployment, set TLS in the `[store]` section of the `--profiles` TOML:

```toml
[store]
backend = "pg"
url     = "postgres://user@host/db"
tls     = "require"
```

The `llm386` binary needs to be built with the `tls-native-tls` feature for non-`disable` modes to work — install with:

```
cargo install --path crates/llm386-cli --features tls-native-tls
```

Without the feature, opening with `tls = "require"` fails loudly with `TlsUnsupported` rather than silently falling back to plaintext. Full TLS guide in the [README → TLS section](../README.md#tls), including `RequireCustomCa` for private CA bundles.

---

## Store lifecycle

### `init`

```
llm386 init <path>
```

Create (or open) an LMDB store directory. **LMDB-only**; the Postgres backend bootstraps its schema automatically on first connect, so there is no separate init step. `init` ignores the global `--store` / `--pg-url` flags.

**Use when:** bootstrapping a new LMDB project. After this every other command points `--store` at the same path (or pins it in TOML). Idempotent — running `init` against an existing store just confirms the schema version.

**Example:**

```
llm386 init ./store
```

---

### `put`

```
llm386 [--store <path> | --pg-url <url>] put --session <id> --kind <kind> [--priority <0..1>] <file|->
```

Insert a block from a file or stdin. Blocks are content-hash deduped: putting the same bytes twice returns the same `BlockId` and adds the new session as another reference. Works identically against either backend.

**Kinds:**

`system`, `user-message`, `assistant-message`, `tool-result`, `summary`, `fact`, `document-chunk`, `plan`, `state`, `trace`. Kind drives which canonical section the packer renders the block into.

**`--priority`:** floating point in `[0.0, 1.0]`. Higher priority gets weighted toward inclusion when section budgets are tight. Default `0.0`.

**Use when:** ingesting any input the model might see — user turns, retrieved documents, tool outputs, system prompts, persistent facts. This is the universal write path.

**Examples:**

```bash
# System prompt and a long-lived fact, both pinned to session 1.
echo "You are a concise assistant." \
  | llm386 --store ./store put --session 1 --kind system -

echo "User's tier is 'enterprise'." \
  | llm386 --store ./store put --session 1 --kind fact --priority 0.8 -

# Same call against a Postgres backend:
echo "User's tier is 'enterprise'." \
  | llm386 --pg-url postgres://user@host/db put --session 1 --kind fact --priority 0.8 -

# Ingest a markdown file as a document chunk (RAG-style).
llm386 --store ./store put --session 1 --kind document-chunk ./onboarding.md
```

---

### `list-sessions`

```
llm386 [--store <path> | --pg-url <url>] list-sessions
```

Print every distinct session id that owns at least one block.

**Use when:** auditing what's in a store, looking up the session id for a user you only know by name (cross-reference the listing with your own registry), or scripting per-session operations.

---

### `show`

```
llm386 [--store <path> | --pg-url <url>] show [--json] <block-id>
```

Print a single block: kind, hash, timestamps, priority, provenance, and the body. With `--json` you get a serialized `ContextBlock` suitable for piping into `jq`.

**Use when:** inspecting why a block was selected, verifying ingest worked, or grabbing the raw body of a stored document.

---

### `verify`

```
llm386 --store <store> verify
```

Read-only integrity check. Walks every block, recomputes its content hash, and verifies the `blocks_by_hash` and `blocks_by_session` indexes are consistent. Reports orphans and mismatches without touching anything.

**LMDB-only.** Postgres has native integrity primitives (foreign-key checks, `REINDEX`, `pg_dump --serializable-deferrable`) so this tool isn't ported. Running it with `--pg-url` errors with `verify is LMDB-only (active backend: pg)`.

**Use when:** after an ungraceful shutdown, before a backup, or when something feels off. It is safe to run on a busy store but it does walk every block.

---

### `repair`

```
llm386 --store <store> repair --yes
```

Rebuild derivable indexes (the hash index) from the primary block table and remove orphan session entries that point at missing blocks. **Destructive:** requires `--yes`. Hash *mismatches* (where the stored hash doesn't match the recomputed bytes) are quarantined and reported, not silently overwritten — a mismatch implies real corruption that warrants human review.

**LMDB-only**, same reasoning as `verify`. Errors cleanly under `--pg-url`.

**Use when:** `verify` reports problems that aren't quarantined hash mismatches. For mismatches, restore from backup.

---

### `purge`

```
llm386 [--store <path> | --pg-url <url>] purge --block <id>   --yes
llm386 [--store <path> | --pg-url <url>] purge --session <id> --yes
```

Delete a single block or every block in a session. `--block` and `--session` are mutually exclusive. Blocks still referenced by another session are kept (since the store is a multi-tenant blob keyed by content hash); blocks with no remaining references are removed entirely, including from the hash index and any edges that touched them. **Destructive:** requires `--yes`.

**Use when:** legal or security asks you to remove customer data ("right to be forgotten" workflows), pruning a stale demo session, or cleaning up after an experiment. For data-removal compliance, see the [data deletion FAQ entry](../FAQ.md#legalsecurity-asked-me-to-remove-customer-data-how-do-i-find-and-remove-it).

---

## Inspecting models

### `list-models`

```
llm386 list-models
```

List every built-in `ModelProfile`, plus any `[[profile]]` entries you added via `--profiles`. Columns: `name`, `max_context_tokens`, `reserved_output_tokens`, `tokenizer`, capability flags.

**Use when:** picking a `--model` value for `page` / `pack`, or sanity-checking that a custom profile in your TOML file actually loaded.

**On the `--model` value:** the CLI is permissive — `--model anthropic/claude-sonnet-4-6`, `--model openrouter/anthropic/claude-3.5-sonnet`, and `--model claude-sonnet-4-9` (a hypothetical future version) all resolve. Provider segments are stripped, exact matches win, then family-prefix matches, then a default fallback (currently `gpt-4o`). Fallbacks emit a single `WARN`-level log line per unknown name to stderr — set `RUST_LOG=warn` (default) to see them. Full resolver behavior in the [FAQ](../FAQ.md#what-happens-if-i-pass-a-model-name-llm386-doesnt-know-anthropicclaude-sonnet-45-openrouter).

---

## Context assembly

### `page`

```
llm386 [--store <path> | --pg-url <url>] page --session <id> --model <name> --task <text> [--json]
```

Run only the pager. Print the resulting `PagePlan`: which blocks were selected (with their selection reason), which were omitted (with why), and the estimated input-token total. Does not render a prompt.

**Use when:** debugging why a block did or didn't make it into the working set. Cheaper than `pack` because it skips rendering. The selection reason column tells you whether the block came in pinned, by relevance, by recency, by edge dependency, or as a global fact.

**Examples:**

```bash
# Human-readable plan.
llm386 --store ./store page --session 1 --model gpt-4o --task "explain Australia's history"

# Machine-readable for piping into jq, scripts, or a test harness.
llm386 --store ./store page --session 1 --model gpt-4o --task "..." --json \
  | jq '.selections | group_by(.reason) | map({reason: .[0].reason, count: length})'
```

---

### `pack`

```
llm386 [--store <path> | --pg-url <url>] pack --session <id> --model <name> --task <text>
       [--prompt-only] [--chat] [--trace <path>]
```

Run page + pack. By default prints a manifest header on stderr (model, input tokens, duration) and the rendered prompt on stdout.

- `--prompt-only` suppresses the manifest. Use this when piping into another tool: `llm386 ... pack --prompt-only > prompt.txt`.
- `--chat` renders as a JSON list of role-tagged messages instead of a single string. Suitable for chat-completion APIs (`{"role": "system", "content": "..."}`, etc.). Mutually exclusive with `--prompt-only`.
- `--trace <path>` records the call to an LMDB trace store. The `CallId` is printed on stderr. The trace store is always LMDB regardless of which block-store backend you're using — it's a separate sink. Replay-complete traces also need `update_output` from your agent loop after the model returns; see the [README → Using as a memory layer](../README.md#using-as-a-memory-layer) for the full turn shape.

**Use when:** the actual hot-path command for an agent that calls `llm386` from a shell or a non-Rust process. For programmatic use prefer the library API or the Python SDK so you can patch the trace's `output` field back in.

**Examples:**

```bash
# Render to a chat-API-shaped JSON list and pipe it.
llm386 --store ./store pack --session 1 --model gpt-4o --task "answer the user" --chat \
  | curl -s https://api.openai.com/v1/chat/completions \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d "$(jq '{model:"gpt-4o", messages: .}')"

# Record a trace for later inspection.
llm386 --store ./store pack --session 1 --model gpt-4o --task "..." \
  --trace ./traces > /dev/null
# call_id prints on stderr; copy it for `trace show` / `trace diff`.
```

---

## Edges

Typed directed edges add structure beyond `Provenance.parents`. The pager follows them when assembling a working set, so dependent blocks travel together. Five kinds: `parent`, `derived-from`, `supports`, `contradicts`, `tool-invocation`.

### `add-edge`

```
llm386 [--store <path> | --pg-url <url>] add-edge --from <id> --to <id> --kind <kind>
```

Persist a directed edge. Idempotent: re-adding the same `(from, to, kind)` triple is a no-op. Deleting either endpoint via `purge` scrubs the edge automatically.

**Use when:**

- Linking an assistant message to the tool result it consumed (`--kind tool-invocation`), so the pager pulls them in together.
- Citing an evidence fact for a derived claim (`--kind supports`).
- Marking a block as the structured derivative of an older one (`--kind derived-from`) for summary or transformation chains.
- Recording known contradictions explicitly (`--kind contradicts`) so a later evaluator or pager policy can reason about them.

**Examples:**

```bash
llm386 --store ./store add-edge --from 9c1d... --to 4f2a... --kind supports
llm386 --store ./store add-edge --from <assistant-msg> --to <tool-result> --kind tool-invocation
```

---

### `edges`

```
llm386 [--store <path> | --pg-url <url>] edges [--incoming] <block-id>
```

List edges incident to a block. Outgoing by default; `--incoming` flips the direction. Output format: `<from> --<Kind>--> <to>`.

**Use when:** auditing what's connected to a block, debugging why the pager pulled in an apparently-unrelated block (it likely arrived via an edge), or cleaning up after a buggy reducer that committed too many edges.

---

## Traces

Every `pack --trace ./traces` invocation writes a `TraceRecord`. Traces are an LMDB store of their own (always LMDB, regardless of which block-store backend you use). They give you full replay context for any past call.

### `trace show`

```
llm386 trace show --trace-store <path> <call-id>
```

Print a single trace record: model, model version, tokenizer version, prompt hash, started-at, duration, input tokens, output (if patched in), and the full page plan.

**Use when:** "why did the model see X on call Y?" Answers it with byte-level precision. Pair with `llm386 [--store ... | --pg-url ...] show <block-id>` to inspect each selected block.

---

### `trace diff`

```
llm386 trace diff --trace-store <path> <prev-call-id> <next-call-id>
```

Compute a structured diff between two trace records' page plans. Output looks like:

```
prev:    019abc...
next:    019def...
summary: +2 -1 ~1 (+184 tokens)
added (2):
  + 7c1e... (HighRelevance)
  + a420... (Dependency)
removed (1):
  - 33b1... (Recency)
reason changes (1):
  ~ 9f02... (HighRelevance -> Pinned)
```

**Use when:** the model gave a surprisingly different answer between two turns and you need to know what changed in its working set. Also useful in CI for catching unexpected drift in retrieval ranking. The diff is set-based on `BlockId`, ignoring section ordering — what you see is the actual change in *what the model saw*.

---

## Summarization

### `summarize`

```
llm386 [--store <path> | --pg-url <url>] summarize --session <id>
                                                    [--summarizer noop|truncating|anthropic]
                                                    [--max-chars <n>] [--last <n>]
                                                    [--store-summary]
                                                    [--anthropic-model <name>] [--anthropic-max-tokens <n>]
```

Summarize a session's blocks via the chosen summarizer. Default `truncating` (first N chars per block, no API calls). `anthropic` requires `ANTHROPIC_API_KEY` and uses Claude.

- `--max-chars` (truncating only): chars per block bullet. Default `80`.
- `--last <n>`: only summarize the most recent N blocks instead of the whole session.
- `--store-summary`: also persist the summary as a new `Summary` block whose `Provenance.parents` reference the originals. The pager's `summary_fallback` policy can then substitute the summary for the original blocks when the section budget is tight.
- `--anthropic-model`: model id (default `claude-haiku-4-5`).
- `--anthropic-max-tokens`: cap on the response (default `1024`).

**Use when:**

- One-shot rollup of a long-running conversation for a digest, email, or report.
- Pre-computing rolling summaries so the pager has something cheap to substitute when the original blocks blow the budget. Run periodically via cron with `--store-summary`.
- Dev-time inspection: `--summarizer truncating --max-chars 60` is a quick way to skim what's actually in a session without dumping every block.

**Examples:**

```bash
# Pure local summary, no API calls.
llm386 --store ./store summarize --session 1

# Persist a model-driven rollup for paging fallback.
ANTHROPIC_API_KEY=... llm386 --store ./store summarize --session 1 \
  --summarizer anthropic --store-summary
```

---

## Patterns

A few common things you'll do that span multiple commands:

**Set the backend once via env var.** With `LLM386_PROFILES` pointing at a TOML that pins `[store]`, you can drop the `--store` / `--pg-url` flag from every invocation:

```bash
cat > llm386.toml <<EOF
[store]
backend = "pg"
url     = "postgres://user@host/db"
schema  = "llm386"
EOF
export LLM386_PROFILES=$PWD/llm386.toml

# Now every command uses the configured backend automatically:
llm386 page --session 1 --model gpt-4o --task "..."
llm386 pack --session 1 --model gpt-4o --task "..." --chat
```

The CLI flags still win if you want to point at a different store for a single command.

**Bootstrap and ingest.**

```bash
llm386 init ./store
echo "You are a careful expert." | llm386 --store ./store put --session 1 --kind system -
for f in docs/*.md; do
  llm386 --store ./store put --session 1 --kind document-chunk "$f"
done
llm386 list-models
llm386 --store ./store page --session 1 --model gpt-4o --task "summarize the docs"
```

**Audit a model call after the fact.**

```bash
# 1. Was the call recorded?
llm386 trace show --trace-store ./traces 019abc...

# 2. What blocks did it see?
llm386 --store ./store show <block-id-from-the-plan> --json | jq

# 3. How is this turn different from the previous one?
llm386 trace diff --trace-store ./traces <prev-call-id> 019abc...
```

**Scrub customer data.**

```bash
# Find candidate blocks (script over `list-sessions` + `show`).
llm386 --store ./store list-sessions
# Once identified:
llm386 --store ./store purge --block <bad-id> --yes
# Or wholesale per session:
llm386 --store ./store purge --session 42 --yes
# Then verify the indexes are still consistent (LMDB-only).
llm386 --store ./store verify
```

**Wire up edge-aware paging for tool results.** When you `put` a tool result, also tie it to the assistant message that called it:

```bash
asst_id=$(echo "..." | llm386 --store ./store put --session 1 --kind assistant-message -)
tool_id=$(cat tool_output.json | llm386 --store ./store put --session 1 --kind tool-result -)
llm386 --store ./store add-edge --from "$asst_id" --to "$tool_id" --kind tool-invocation
```

The pager's `include_parents` policy (on by default in many configs) will then pull the assistant message back in whenever the tool result is selected, even on later turns.

---

## See also

- [README](../README.md) — architectural overview, library API, framework hooks.
- [FAQ](../FAQ.md) — operational questions: latency, sizing, multi-tenancy, RAG integration, custom retrievers.
- [Python SDK](../python/README.md) — same surface, programmatic, with `Trace.update_output` for replay-complete traces.
