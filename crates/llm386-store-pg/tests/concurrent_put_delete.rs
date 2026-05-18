//! Concurrency test for `PgStore::put` dedup race with concurrent
//! `delete`.
//!
//! `put` does `INSERT ... ON CONFLICT (hash) DO NOTHING RETURNING id`
//! followed by a fallback `SELECT id FROM blocks WHERE hash = $1` when
//! the INSERT returned no row (because a block with that content
//! already exists). Under READ COMMITTED, a concurrent `delete()`
//! that commits between the two statements vaporizes the conflicting
//! row, so the fallback SELECT returns nothing — and the caller sees
//! a `Backend("dedup conflict but hash row missing")` error that's
//! indistinguishable from real corruption.
//!
//! Skips cleanly when `TEST_DATABASE_URL` is unset.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::thread;

use llm386_core::{
    BlockId, BlockKind, BlockStore, ContentHash, ContextBlock, Provenance, SessionId, Timestamp,
    TokenCounts,
};
use llm386_store_pg::{PgStore, PgStoreConfig};
use postgres::NoTls;

static SCHEMA_COUNTER: AtomicU32 = AtomicU32::new(0);

fn make_block(bytes: &[u8], kind: BlockKind, ts_ms: u64, rnd: u128) -> ContextBlock {
    ContextBlock {
        id: BlockId::from_parts(ts_ms, rnd),
        kind,
        bytes: bytes.to_vec(),
        token_counts: TokenCounts::new(),
        priority: 0.0,
        created_at: Timestamp(ts_ms),
        updated_at: Timestamp(ts_ms),
        provenance: Provenance::default(),
        hash: ContentHash::of(bytes),
    }
}

fn open_test() -> Option<(Arc<PgStore>, String, String)> {
    let url = std::env::var("TEST_DATABASE_URL").ok()?;
    let n = SCHEMA_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let schema = format!("llm386_putdel_{pid}_{n}");
    let store = PgStore::open(
        &url,
        &PgStoreConfig {
            max_pool_size: 4,
            schema: Some(schema.clone()),
        },
    )
    .expect("open PgStore");
    Some((Arc::new(store), url, schema))
}

fn format_err_chain<E: std::error::Error>(e: &E) -> String {
    let mut s = format!("{e}");
    let mut src: Option<&dyn std::error::Error> = e.source();
    while let Some(inner) = src {
        s.push_str(" | ");
        s.push_str(&inner.to_string());
        src = inner.source();
    }
    s
}

fn cleanup(url: &str, schema: &str) {
    let Ok(mut client) = postgres::Client::connect(url, NoTls) else {
        return;
    };
    let _ = client.batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE;"));
}

struct DropGuard<'a> {
    url: &'a str,
    schema: &'a str,
}
impl Drop for DropGuard<'_> {
    fn drop(&mut self) {
        cleanup(self.url, self.schema);
    }
}

/// Hammer the put/delete dedup race from N pairs of threads for a
/// fixed duration. Pairs share a body so they're all dedup-eligible.
/// Without the fix, the put's INSERT-then-SELECT window catches a
/// concurrent committed delete and surfaces `"dedup conflict but
/// hash row missing"` as a `Backend` error. The race window is
/// narrow (microseconds between INSERT and the fallback SELECT) so
/// the test relies on volume: dozens of pairs running tight loops
/// for several seconds reliably exposes it on a local Postgres.
#[test]
fn concurrent_put_and_delete_on_shared_content_never_errors() {
    let Some((store, url, schema)) = open_test() else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let _guard = DropGuard { url: &url, schema: &schema };

    use parking_lot::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU64};
    use std::time::{Duration, Instant};

    const PAIRS: usize = 8;
    const DURATION: Duration = Duration::from_millis(2_000);

    let put_session = SessionId(1);
    let seed_session = SessionId(2);
    let body = b"toctou-race-target".to_vec();

    // Seed once so the first put hits dedup.
    let seed = make_block(&body, BlockKind::Fact, 1, 1);
    let _seeded = store.put(seed_session, seed).expect("seed put");

    let stop = Arc::new(AtomicBool::new(false));
    let attempts = Arc::new(AtomicU64::new(0));
    let put_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let del_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    let mut handles = Vec::with_capacity(PAIRS * 2);
    for i in 0..PAIRS {
        let store_p = Arc::clone(&store);
        let store_d = Arc::clone(&store);
        let stop_p = Arc::clone(&stop);
        let stop_d = Arc::clone(&stop);
        let attempts_p = Arc::clone(&attempts);
        let attempts_d = Arc::clone(&attempts);
        let put_errors_p = Arc::clone(&put_errors);
        let del_errors_d = Arc::clone(&del_errors);
        let body_p = body.clone();
        let body_d = body.clone();

        handles.push(thread::spawn(move || {
            while !stop_p.load(Ordering::Relaxed) {
                attempts_p.fetch_add(1, Ordering::Relaxed);
                let block = make_block(
                    &body_p,
                    BlockKind::Fact,
                    100 + u64::try_from(i).unwrap(),
                    1,
                );
                if let Err(e) = store_p.put(put_session, block) {
                    put_errors_p.lock().push(format_err_chain(&e));
                    stop_p.store(true, Ordering::Relaxed);
                }
            }
        }));
        handles.push(thread::spawn(move || {
            while !stop_d.load(Ordering::Relaxed) {
                attempts_d.fetch_add(1, Ordering::Relaxed);
                match store_d.lookup_hash(ContentHash::of(&body_d)) {
                    Ok(Some(id)) => {
                        if let Err(e) = store_d.delete(id) {
                            del_errors_d.lock().push(format_err_chain(&e));
                            stop_d.store(true, Ordering::Relaxed);
                        }
                    }
                    Ok(None) => {}
                    Err(e) => {
                        del_errors_d.lock().push(format_err_chain(&e));
                        stop_d.store(true, Ordering::Relaxed);
                    }
                }
            }
        }));
    }

    let started = Instant::now();
    while started.elapsed() < DURATION && !stop.load(Ordering::Relaxed) {
        thread::sleep(Duration::from_millis(25));
    }
    stop.store(true, Ordering::Relaxed);
    for h in handles {
        h.join().expect("worker panicked");
    }

    let attempts_n = attempts.load(Ordering::Relaxed);
    let put_errs = put_errors.lock().clone();
    let del_errs = del_errors.lock().clone();
    assert!(
        put_errs.is_empty() && del_errs.is_empty(),
        "{} put error(s) + {} delete error(s) in {attempts_n} attempts \
         over {PAIRS} pairs (run for ~{:?})\n  put: {:?}\n  delete: {:?}",
        put_errs.len(),
        del_errs.len(),
        DURATION,
        put_errs,
        del_errs,
    );
}
