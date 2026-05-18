//! Concurrency test for `put` vs `purge_session` on content-deduped
//! blocks — the third race in the put/delete/purge family.
//!
//! Shape: block B exists in session S_seed. Two threads race:
//!   - P: `put(session=S_put, body)` with hash(body) == B's hash
//!   - G: `purge_session(S_seed)`
//!
//! Without `purge_session` taking `llm386_blocks.id FOR UPDATE`
//! before its orphan check, two failure modes are reachable:
//!
//!   1. Deadlock between `put`'s `INSERT INTO session_blocks` and
//!      `purge`'s `DELETE FROM llm386_blocks` — observable via
//!      `SQLSTATE 40P01` returned to the caller. Pre-fix runs of
//!      this test surface these deadlocks within a handful of
//!      attempts.
//!   2. Dangling `session_blocks` rows pointing at deleted blocks
//!      when purge's orphan check runs at a snapshot taken before
//!      put commits its session_blocks insert. This is a smaller
//!      timing window — the test exercises it but does not
//!      reliably trigger it on every PG / OS combination; the fix is
//!      a defensive lock-order alignment with `put` and `delete`
//!      (both take `blocks.id` first), which also eliminates the
//!      deadlock.
//!
//! Test asserts: no caller-visible errors, and zero dangling
//! `session_blocks` rows in the final state.
//!
//! Skips cleanly when `TEST_DATABASE_URL` is unset.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use llm386_core::{
    BlockId, BlockKind, BlockStore, ContentHash, ContextBlock, Provenance, SessionId, Timestamp,
    TokenCounts,
};
use llm386_store_pg::{PgStore, PgStoreConfig};
use parking_lot::Mutex;
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
    let schema = format!("llm386_putpurge_{pid}_{n}");
    let store = PgStore::open(
        &url,
        &PgStoreConfig {
            max_pool_size: 8,
            schema: Some(schema.clone()),
            ..Default::default()
        },
    )
    .expect("open PgStore");
    Some((Arc::new(store), url, schema))
}

fn cleanup(url: &str, schema: &str) {
    let Ok(mut client) = postgres::Client::connect(url, NoTls) else {
        return;
    };
    let _ = client.batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE;"));
}

/// Count dangling session_blocks rows: ones whose block_id has no
/// matching row in llm386_blocks. Connects fresh so it sees only
/// committed state.
fn count_dangling_session_refs(url: &str, schema: &str) -> u64 {
    let mut client =
        postgres::Client::connect(url, NoTls).expect("connect to count dangling refs");
    client
        .batch_execute(&format!("SET search_path TO \"{schema}\";"))
        .expect("set search_path");
    let row = client
        .query_one(
            "SELECT COUNT(*)::BIGINT FROM llm386_session_blocks sb
             WHERE NOT EXISTS (SELECT 1 FROM llm386_blocks b WHERE b.id = sb.block_id)",
            &[],
        )
        .expect("count dangling refs");
    let n: i64 = row.get(0);
    u64::try_from(n).unwrap_or(0)
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

const DURATION: Duration = Duration::from_secs(2);
const PUT_THREADS: usize = 4;

/// Hammer put + purge against the same content-deduped block for a
/// fixed duration. Pre-fix: purge's orphan check fires before put
/// commits, then deletes B after put's session_blocks insert
/// commits, leaving (S_put, B) dangling. Post-fix: zero dangling refs
/// regardless of interleaving.
#[test]
fn concurrent_put_and_purge_on_shared_content_leaves_no_dangling_ref() {
    let Some((store, url, schema)) = open_test() else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let _guard = DropGuard { url: &url, schema: &schema };

    let body = b"put-purge-target".to_vec();
    let seed_session = SessionId(0);

    let stop = Arc::new(AtomicBool::new(false));
    let attempts = Arc::new(AtomicU64::new(0));
    let put_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let purge_errors: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));

    // Several P threads, each using a unique session per iteration so
    // every put is a fresh session_blocks INSERT that needs to race
    // against G's orphan check — not an idempotent re-insert.
    let mut handles = Vec::with_capacity(PUT_THREADS + 1);
    for tid in 0..PUT_THREADS {
        let p_store = Arc::clone(&store);
        let p_stop = Arc::clone(&stop);
        let p_attempts = Arc::clone(&attempts);
        let p_errors = Arc::clone(&put_errors);
        let p_body = body.clone();
        handles.push(thread::spawn(move || {
            let mut i: u128 = 0;
            while !p_stop.load(Ordering::Relaxed) {
                p_attempts.fetch_add(1, Ordering::Relaxed);
                // Unique session id per iteration per thread → every
                // INSERT into session_blocks is a brand-new row, and
                // (session, block) hasn't been committed by any prior
                // iteration.
                let s = SessionId(
                    (u128::try_from(tid).unwrap() << 64) | (i + 1),
                );
                // Unique proposed BlockId per put. Dedup will collapse
                // to the existing hash anyway; the proposed id only
                // matters if the INSERT actually creates a new row
                // (which it does after a recent purge has deleted the
                // shared block).
                let block = make_block(
                    &p_body,
                    BlockKind::Fact,
                    1_000_000 + u64::try_from(i).unwrap_or(u64::MAX),
                    (u128::try_from(tid).unwrap() << 64) | (i + 1),
                );
                i += 1;
                if let Err(e) = p_store.put(s, block) {
                    p_errors.lock().push(format!("{e}"));
                    p_stop.store(true, Ordering::Relaxed);
                }
            }
        }));
    }

    // G thread: keep B reachable (re-seed before each purge) and
    // immediately purge the seed session to repeatedly trigger the
    // orphan-check codepath.
    let g_store = Arc::clone(&store);
    let g_stop = Arc::clone(&stop);
    let g_attempts = Arc::clone(&attempts);
    let g_errors = Arc::clone(&purge_errors);
    let g_body = body.clone();
    handles.push(thread::spawn(move || {
        let mut i: u128 = 0;
        while !g_stop.load(Ordering::Relaxed) {
            g_attempts.fetch_add(1, Ordering::Relaxed);
            // Unique proposed id per seed; dedup folds them all to
            // whatever current canonical row exists.
            let seed = make_block(
                &g_body,
                BlockKind::Fact,
                u64::try_from(i + 1).unwrap(),
                i + 1,
            );
            i += 1;
            if let Err(e) = g_store.put(seed_session, seed) {
                g_errors.lock().push(format!("seed put: {e}"));
                g_stop.store(true, Ordering::Relaxed);
                continue;
            }
            if let Err(e) = g_store.purge_session(seed_session) {
                g_errors.lock().push(format!("purge: {e}"));
                g_stop.store(true, Ordering::Relaxed);
            }
        }
    }));

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
    let purge_errs = purge_errors.lock().clone();
    assert!(
        put_errs.is_empty() && purge_errs.is_empty(),
        "errors during put/purge race (attempts={attempts_n}, dur={DURATION:?})\n  \
         put: {put_errs:?}\n  purge: {purge_errs:?}",
    );

    let dangling = count_dangling_session_refs(&url, &schema);
    assert_eq!(
        dangling, 0,
        "{dangling} session_blocks row(s) point at deleted blocks after \
         {attempts_n} put+purge attempts over ~{DURATION:?} — the orphan-check \
         window in purge_session let a concurrent put's session_blocks insert \
         survive a subsequent block delete",
    );
}
