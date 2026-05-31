# Postgres backend

This fork adds `llm386-store-pg`, a `BlockStore` implementation backed by PostgreSQL, sitting alongside the upstream `llm386-store-lmdb`. The trait surface in `llm386-core` is unchanged — every other crate (pager, packer, retriever, trace, reduce) is backend-agnostic.

## Picking a backend

- **LMDB** (default in the bundled CLI and Python SDK): embedded library, single process, read-heavy, low-latency. Right answer for single-host deployments.
- **Postgres**: multi-process or multi-node deployments, shared operational infrastructure (pooling, backups, observability already in place), ACID across writers, future ANN co-location with `pgvector`.

Neither is "better" than the other — they target different operational models. Full decision guide, including what you give up either way, in [FAQ → Should I use LMDB or Postgres?](./FAQ.md#should-i-use-lmdb-or-postgres-for-the-block-store-what-am-i-giving-up).

## Why a Postgres backend

LMDB is memory-mapped, single-writer, single-host. That is the right answer for many deployments and the LMDB backend wins on read paths by a wide margin (see [performance](#performance) below). It is the wrong answer for:

- **Multiple processes write to the same store.** LMDB serializes writers across the entire env; only one process holds the write lock at a time. Pod-replicated runtimes hit this immediately.
- **No shared filesystem.** Orchestrators that schedule processes across nodes have no portable story for a writable LMDB file — NFS/EFS-mounted LMDB is undefined behavior.
- **Operational machinery already exists for Postgres.** Connection pooling, schema migrations, snapshot backups, point-in-time recovery, observability, multi-tenant ACLs — every cloud-native runtime already runs this for its primary database. Reusing it for the LLM context store has no marginal ops cost.
- **Native ACID across processes.** `put` writes the block, the hash index row, and the session-membership row in a single transaction. Concurrent writers don't see partial state.
- **Future ANN co-location.** Adding `pgvector` lets `llm386-retrieve-ann` resolve against the same table that holds the blocks — schema metadata, conversation turns, tool results, and their embeddings all live in one table, queried by a single JOIN with an HNSW index. No separate vector database to operate.

## Schema

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

Design notes:

- **`BYTEA(16)` for ids, not `UUID`.** `BlockId`'s chronological ordering comes from the raw bit layout (48-bit timestamp in the high bits). Postgres `UUID` compares lexicographically, which would scramble that ordering. Big-endian `BYTEA` preserves the natural `Ord` semantics exactly — `ORDER BY block_id ASC` returns blocks chronologically without a separate time column.
- **Dedup via `ON CONFLICT (hash) DO NOTHING`.** A single statement satisfies the content-hash dedup invariant; a follow-up `SELECT id FROM llm386_blocks WHERE hash = $1` recovers the existing id when the insert collided.
- **`JSONB` for structured fields.** `token_counts` and `provenance` round-trip through JSONB, with `BlockId` lineage encoded as hex strings (Postgres JSON can't represent u128 numerically). `provenance->>'labels'` queries work without schema changes.

## Selecting the backend

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

## Async opt-in: `AsyncPgStore`

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

## TLS

The default is `TlsMode::Disable` (plaintext) for back-compat. **Any non-localhost deployment should opt in to TLS** — without it, the Postgres password and every block body travel the wire unencrypted.

TLS support lives behind the `tls-native-tls` feature on `llm386-store-pg` (and forwards through `llm386-config`, the CLI, and the Python wrapper):

```toml
# Cargo dependency for library users
llm386-store-pg = { version = "...", features = ["tls-native-tls"] }
```

```bash
# CLI
cargo install --path crates/llm386-cli --features tls-native-tls
```

```bash
# Python
maturin develop -F tls-native-tls   # (from python/)
```

Three modes:

| `TlsMode`                  | `[store] tls = "..."` | Behavior |
|----------------------------|-----------------------|----------|
| `Disable`                  | `"disable"` (default) | Plaintext. Sets `sslmode=disable` on the connection so it can't accidentally upgrade. |
| `Require`                  | `"require"`           | Mandate TLS. Verifies the server certificate against the system root CA store. Works out of the box against RDS, Cloud SQL, Supabase, Neon. |
| `RequireCustomCa { ca_path }` | `"require-custom-ca"` (with `tls_ca_path = "..."`) | Mandate TLS. Verifies against a private CA bundle (PEM file). Use this when your Postgres is behind a private CA not in the system store. |

`Require` and `RequireCustomCa` need the `tls-native-tls` feature. Without it, opening with either returns `StoreOpenError::TlsUnsupported` — **never** a silent fall-through to plaintext. And both modes force `sslmode=require` on the connection, so the postgres client genuinely refuses to fall back to plaintext if the server doesn't offer TLS (the default `sslmode=prefer` would silently downgrade).

Library:

```rust
use llm386_store_pg::{PgStore, PgStoreConfig, TlsMode};

let store = PgStore::open(
    "postgres://user:pass@host/db",
    &PgStoreConfig {
        tls: TlsMode::Require,
        ..Default::default()
    },
)?;
```

TOML config (consumed by CLI + Python via `--profiles`):

```toml
[store]
backend = "pg"
url     = "postgres://user@host/db"
tls     = "require"
```

Or with a private CA:

```toml
[store]
backend     = "pg"
url         = "postgres://user@host/db"
tls         = "require-custom-ca"
tls_ca_path = "/etc/ssl/private-pg-ca.pem"
```

## Performance

> **Important framing.** The benchmark below compares **the current `BlockStore` trait shape**, which opens a fresh write transaction per block. That is LMDB's worst case — every write incurs an fsync. With batched commits (many puts per transaction) LMDB outpaces Postgres on writes by an order of magnitude. The numbers below reflect the trait as it stands, not LMDB's upper bound.
>
> The honest one-line summary is: **"Postgres performs better than LMDB *when the BlockStore trait forces per-write commits*."** It is not a general "Postgres is faster than LMDB" claim.

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

What the numbers actually say (read with the caveat above in mind):

| Op | Winner | Margin | Why |
|---|---|---|---|
| **put** (trait-shape ceiling) | PG | ~12× | LMDB fsyncs the WAL on every commit (4 ms p50 on APFS) because the trait does one transaction per block. PG batches via its own WAL and the per-op commit cost amortizes better. **With batched LMDB writes, this number flips.** |
| **dedup put** (trait-shape ceiling) | PG | ~8× | Same story — LMDB's transaction commit dominates even when the dedup path skips two writes. |
| **get** | LMDB | ~50× | Mmap'd page lookup (~2 µs) vs. Postgres protocol roundtrip over a UNIX socket (~120 µs). The socket floor. This is real architectural difference, not trait shape. |
| **list_session** | LMDB | ~14× | In-process BTree walk vs. an indexed range scan plus protocol serialization. Real architectural difference. |

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

## Operational notes

- **Migrations.** Schema creation is idempotent (`CREATE TABLE IF NOT EXISTS`) and gated by a `schema_version` row in `llm386_meta`. Re-checked on every open. No external migration tool required.
- **Backups.** Standard Postgres tooling — `pg_dump`, snapshots, PITR. Nothing llm386-specific.
- **Multi-tenancy.** Run different agents against different `schema` values (`SET search_path` per pool), or different databases entirely. Sessions within a schema are isolated by `session_id` in `llm386_session_blocks`.
- **Connection pooling.** `PgStore` uses `r2d2-postgres` (sync). `AsyncPgStore` uses `deadpool-postgres`. `pool_size` defaults to 8; tune for concurrency profile.
- **`pgvector` co-location.** Future work — see [Why a Postgres backend](#why-a-postgres-backend). The block store and embedding index can live in the same table once `llm386-retrieve-ann` grows a `pgvector` adapter.
