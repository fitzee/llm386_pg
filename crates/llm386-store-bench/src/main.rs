//! Perf hammer for the LMDB and Postgres `BlockStore` backends.
//!
//! Runs four workloads against both stores back-to-back and prints a
//! comparison table:
//!   - put:   sustained writes of unique blocks
//!   - get:   random point reads after writes
//!   - list:  list_session over the populated session
//!   - dedup: same content written N times across distinct SessionIds
//!
//! Usage:
//!   store-bench --pg-url postgres://user@host/db [--blocks N] [--bytes B]

use std::time::{Duration, Instant};

use clap::Parser;
use llm386_core::{
    BlockId, BlockKind, BlockStore, ContentHash, ContextBlock, Provenance, SessionId, Timestamp,
    TokenCounts,
};
use llm386_store_lmdb::{LmdbStore, StoreConfig as LmdbConfig};
use llm386_store_pg::{PgStore, PgStoreConfig};
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

#[derive(Parser, Debug)]
#[command(version, about = "Compare LMDB vs PG BlockStore throughput and latency.")]
struct Args {
    /// Postgres connection URL.
    #[arg(long)]
    pg_url: String,

    /// Number of blocks to write in the put/get/list workloads.
    #[arg(long, default_value_t = 10_000)]
    blocks: usize,

    /// Bytes per block payload (the actual byte vec inside ContextBlock).
    #[arg(long, default_value_t = 1024)]
    bytes: usize,

    /// Comma-separated workload list: put,get,list,dedup. Default: all.
    #[arg(long, default_value = "put,get,list,dedup")]
    workloads: String,

    /// Comma-separated backends: lmdb,pg. Default: both.
    #[arg(long, default_value = "lmdb,pg")]
    backends: String,
}

fn main() {
    let args = Args::parse();
    let workloads: Vec<&str> = args.workloads.split(',').map(str::trim).collect();
    let backends: Vec<&str> = args.backends.split(',').map(str::trim).collect();

    println!(
        "config: blocks={} bytes={} workloads=[{}] backends=[{}]",
        args.blocks,
        args.bytes,
        workloads.join(","),
        backends.join(","),
    );

    let mut results: Vec<RowResult> = Vec::new();

    for backend in &backends {
        match *backend {
            "lmdb" => {
                let tmp = tempfile::TempDir::new().expect("tempdir");
                let store =
                    LmdbStore::open(tmp.path(), LmdbConfig::default()).expect("open lmdb");
                run_all(&workloads, &store, "lmdb", args.blocks, args.bytes, &mut results);
            }
            "pg" => {
                let schema = format!(
                    "llm386_bench_{}_{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_nanos(),
                );
                let store = PgStore::open(
                    &args.pg_url,
                    &PgStoreConfig {
                        max_pool_size: 4,
                        schema: Some(schema.clone()),
                        ..Default::default()
                    },
                )
                .expect("open pg");
                run_all(&workloads, &store, "pg", args.blocks, args.bytes, &mut results);
                // Schemas are best-effort cleanup; the pool customizer
                // already sets search_path so a separate cleanup
                // connection isn't trivial to wire. The bench schemas
                // are name-tagged so DROP SCHEMA from psql is easy:
                //   psql -c "DROP SCHEMA <name> CASCADE"
                eprintln!("note: bench schema `{schema}` left in place; drop manually if desired");
            }
            other => eprintln!("unknown backend: {other}"),
        }
    }

    println!();
    print_table(&results);
}

#[derive(Debug)]
struct RowResult {
    backend: &'static str,
    workload: &'static str,
    samples: usize,
    total: Duration,
    p50: Duration,
    p95: Duration,
    p99: Duration,
}

fn run_all<S: BlockStore>(
    workloads: &[&str],
    store: &S,
    backend: &'static str,
    n_blocks: usize,
    n_bytes: usize,
    out: &mut Vec<RowResult>,
) {
    eprintln!("running backend={backend}");
    let session = SessionId(1);

    let mut ids: Vec<BlockId> = Vec::new();
    if workloads.iter().any(|&w| w == "put" || w == "get" || w == "list") {
        let r = workload_put(store, session, n_blocks, n_bytes);
        out.push(RowResult { backend, workload: "put", ..r.0 });
        ids = r.1;
    }
    if workloads.contains(&"get") {
        let r = workload_get(store, &ids);
        out.push(RowResult { backend, workload: "get", ..r });
    }
    if workloads.contains(&"list") {
        let r = workload_list(store, session, n_blocks);
        out.push(RowResult { backend, workload: "list", ..r });
    }
    if workloads.contains(&"dedup") {
        let r = workload_dedup(store, n_blocks, n_bytes);
        out.push(RowResult { backend, workload: "dedup", ..r });
    }
}

fn workload_put<S: BlockStore>(
    store: &S,
    session: SessionId,
    n_blocks: usize,
    n_bytes: usize,
) -> (RowResult, Vec<BlockId>) {
    let mut latencies: Vec<Duration> = Vec::with_capacity(n_blocks);
    let mut ids: Vec<BlockId> = Vec::with_capacity(n_blocks);
    let total_start = Instant::now();
    for i in 0..n_blocks {
        let block = synth_block(i, n_bytes);
        let t = Instant::now();
        let id = store.put(session, block).expect("put");
        latencies.push(t.elapsed());
        ids.push(id);
    }
    let total = total_start.elapsed();
    (
        finalize("put", latencies, total),
        ids,
    )
}

fn workload_get<S: BlockStore>(store: &S, ids: &[BlockId]) -> RowResult {
    let mut order: Vec<BlockId> = ids.to_vec();
    let mut rng = StdRng::seed_from_u64(0xdead_beef);
    order.shuffle(&mut rng);
    let mut latencies: Vec<Duration> = Vec::with_capacity(order.len());
    let total_start = Instant::now();
    for id in &order {
        let t = Instant::now();
        let b = store.get(*id).expect("get");
        latencies.push(t.elapsed());
        assert!(b.is_some(), "missing id {id}");
    }
    let total = total_start.elapsed();
    finalize("get", latencies, total)
}

fn workload_list<S: BlockStore>(store: &S, session: SessionId, expected: usize) -> RowResult {
    // 100 iterations — list_session can be O(N), so we don't want to
    // dwarf the other workloads' wall-clock numbers.
    const ITERS: usize = 100;
    let mut latencies: Vec<Duration> = Vec::with_capacity(ITERS);
    let total_start = Instant::now();
    for _ in 0..ITERS {
        let t = Instant::now();
        let ids = store.list_session(session).expect("list");
        latencies.push(t.elapsed());
        assert_eq!(ids.len(), expected, "list_session returned wrong count");
    }
    let total = total_start.elapsed();
    finalize("list", latencies, total)
}

fn workload_dedup<S: BlockStore>(store: &S, n_blocks: usize, n_bytes: usize) -> RowResult {
    // Same content, written across N distinct sessions. First put
    // inserts the canonical row; subsequent N-1 puts hit the dedup
    // path (ON CONFLICT for PG; hash-index hit for LMDB) and just
    // record session membership.
    let payload = vec![0xab_u8; n_bytes.max(1)];
    let mut latencies: Vec<Duration> = Vec::with_capacity(n_blocks);
    let total_start = Instant::now();
    for i in 0..n_blocks {
        let block = ContextBlock {
            id: BlockId::from_parts(1, i as u128),
            kind: BlockKind::Fact,
            bytes: payload.clone(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: Timestamp(1),
            updated_at: Timestamp(1),
            provenance: Provenance::default(),
            hash: ContentHash::of(&payload),
        };
        let t = Instant::now();
        store
            .put(SessionId(0xdeadbeef_0000_0000_0000_0000_0000_0000 + i as u128), block)
            .expect("dedup put");
        latencies.push(t.elapsed());
    }
    let total = total_start.elapsed();
    finalize("dedup", latencies, total)
}

fn synth_block(i: usize, n_bytes: usize) -> ContextBlock {
    let mut bytes = format!("block-{i:020}").into_bytes();
    if bytes.len() < n_bytes {
        let pad = n_bytes - bytes.len();
        bytes.extend(std::iter::repeat_n(b'.', pad));
    } else {
        bytes.truncate(n_bytes);
    }
    let hash = ContentHash::of(&bytes);
    // 48-bit timestamp ceiling — keep the high bits below the cap so
    // BlockId ordering stays monotonic across `i`.
    let ts_ms = (i as u64).min((1u64 << 48) - 1);
    ContextBlock {
        id: BlockId::from_parts(ts_ms, i as u128),
        kind: BlockKind::UserMessage,
        bytes,
        token_counts: TokenCounts::new(),
        priority: 0.0,
        created_at: Timestamp(ts_ms),
        updated_at: Timestamp(ts_ms),
        provenance: Provenance::default(),
        hash,
    }
}

fn finalize(workload: &'static str, mut latencies: Vec<Duration>, total: Duration) -> RowResult {
    latencies.sort_unstable();
    let samples = latencies.len();
    let p50 = percentile(&latencies, 0.50);
    let p95 = percentile(&latencies, 0.95);
    let p99 = percentile(&latencies, 0.99);
    RowResult {
        backend: "",
        workload,
        samples,
        total,
        p50,
        p95,
        p99,
    }
}

fn percentile(sorted: &[Duration], p: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx]
}

fn print_table(rows: &[RowResult]) {
    println!(
        "{:<8} {:<8} {:>10} {:>12} {:>10} {:>10} {:>10} {:>10}",
        "backend", "workload", "samples", "total", "ops/sec", "p50", "p95", "p99",
    );
    println!("{:-<82}", "");
    for r in rows {
        let secs = r.total.as_secs_f64();
        let ops = if secs > 0.0 {
            r.samples as f64 / secs
        } else {
            f64::INFINITY
        };
        println!(
            "{:<8} {:<8} {:>10} {:>12} {:>10.0} {:>10} {:>10} {:>10}",
            r.backend,
            r.workload,
            r.samples,
            format_dur(r.total),
            ops,
            format_dur(r.p50),
            format_dur(r.p95),
            format_dur(r.p99),
        );
    }
}

fn format_dur(d: Duration) -> String {
    let ns = d.as_nanos();
    if ns < 1_000 {
        format!("{ns}ns")
    } else if ns < 1_000_000 {
        format!("{:.1}µs", ns as f64 / 1_000.0)
    } else if ns < 1_000_000_000 {
        format!("{:.1}ms", ns as f64 / 1_000_000.0)
    } else {
        format!("{:.2}s", d.as_secs_f64())
    }
}
