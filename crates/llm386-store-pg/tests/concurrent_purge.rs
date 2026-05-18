//! Concurrency tests for `PgStore::purge_session`.
//!
//! The shape that matters: two sessions share a content-deduped block,
//! and both sessions are purged at the same instant. The store's
//! contract — matching LMDB's — is that a block with no remaining
//! session references must be deleted entirely (block row, hash entry,
//! any incident edges). A naive READ COMMITTED implementation can leak
//! the shared block as an orphan because each transaction's orphan
//! check sees the other's uncommitted session-row as still alive.
//!
//! Skips cleanly when `TEST_DATABASE_URL` is unset, like the rest of
//! the PG integration suite.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
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
    let schema = format!("llm386_concur_{pid}_{n}");
    // Pool size large enough that both purge threads can hold a
    // connection simultaneously.
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

/// Count orphan blocks: rows in `llm386_blocks` whose id appears in
/// no `llm386_session_blocks` row. Connects fresh so it sees only
/// committed state.
fn count_orphan_blocks(url: &str, schema: &str) -> u64 {
    let mut client =
        postgres::Client::connect(url, NoTls).expect("connect to count orphans");
    client
        .batch_execute(&format!("SET search_path TO \"{schema}\";"))
        .expect("set search_path");
    let row = client
        .query_one(
            "SELECT COUNT(*)::BIGINT FROM llm386_blocks b
             WHERE NOT EXISTS (SELECT 1 FROM llm386_session_blocks sb WHERE sb.block_id = b.id)",
            &[],
        )
        .expect("count orphans");
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

/// Hammer purge_session from two threads against the same shared
/// block, many times in a row. With the naive READ COMMITTED
/// implementation, at least one iteration will leave an orphan row in
/// `llm386_blocks`; with the FOR UPDATE fix, every iteration leaves
/// the table empty.
// Number of iterations is a balance: enough to reliably trigger the
// race on the buggy code path, but not so many that the test takes
// forever. 50 is comfortable on a local Postgres; the race typically
// fires within the first handful.
const RACE_ITERS: u32 = 50;

#[test]
fn concurrent_purge_of_shared_block_leaves_no_orphan() {
    let Some((store, url, schema)) = open_test() else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };

    let _guard = DropGuard { url: &url, schema: &schema };

    for iter in 0..RACE_ITERS {
        // Two distinct sessions, both referencing one content-deduped
        // block. The body varies per iteration so the dedup gives us
        // a different shared block each time.
        let s1 = SessionId(u128::from(iter) * 2 + 1);
        let s2 = SessionId(u128::from(iter) * 2 + 2);
        let body = format!("shared-{iter}").into_bytes();
        let block_for_s1 = make_block(&body, BlockKind::Fact, u64::from(iter) * 2 + 1, 1);
        let block_for_s2 = make_block(&body, BlockKind::Fact, u64::from(iter) * 2 + 2, 2);
        let id_a = store.put(s1, block_for_s1).expect("put s1");
        let id_b = store.put(s2, block_for_s2).expect("put s2");
        assert_eq!(id_a, id_b, "iter {iter}: content dedup should give same id");

        // Two purge threads, started as close together as possible.
        let store_a = Arc::clone(&store);
        let store_b = Arc::clone(&store);
        let (ready_tx, ready_rx) = mpsc::channel::<()>();
        let ready_tx2 = ready_tx.clone();
        let go = Arc::new(std::sync::Barrier::new(3));
        let go_a = Arc::clone(&go);
        let go_b = Arc::clone(&go);

        let h1 = thread::spawn(move || {
            ready_tx.send(()).unwrap();
            go_a.wait();
            store_a.purge_session(s1).expect("purge s1");
        });
        let h2 = thread::spawn(move || {
            ready_tx2.send(()).unwrap();
            go_b.wait();
            store_b.purge_session(s2).expect("purge s2");
        });
        ready_rx.recv().unwrap();
        ready_rx.recv().unwrap();
        go.wait();
        h1.join().expect("thread 1");
        h2.join().expect("thread 2");

        // Both sessions are now empty regardless of fix status.
        assert!(store.list_session(s1).expect("list s1").is_empty());
        assert!(store.list_session(s2).expect("list s2").is_empty());

        // The shared block must NOT survive — neither session
        // references it any more. Without the fix this leaks.
        let orphans = count_orphan_blocks(&url, &schema);
        assert_eq!(
            orphans, 0,
            "iter {iter}: orphan block(s) survived concurrent purge_session — \
             this is the READ COMMITTED race"
        );
    }
}

/// Single-threaded sanity check: the contract still holds when only
/// one of two shared-block sessions is purged.
#[test]
fn sequential_purge_keeps_shared_block_for_remaining_session() {
    let Some((store, url, schema)) = open_test() else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let _guard = DropGuard { url: &url, schema: &schema };

    let s1 = SessionId(1);
    let s2 = SessionId(2);
    let body = b"shared";
    let id = store
        .put(s1, make_block(body, BlockKind::Fact, 1, 1))
        .expect("put s1");
    let id_b = store
        .put(s2, make_block(body, BlockKind::Fact, 2, 2))
        .expect("put s2");
    assert_eq!(id, id_b);

    store.purge_session(s1).expect("purge s1");

    // The remaining session keeps the block.
    assert_eq!(store.list_session(s2).expect("list s2"), vec![id]);
    assert!(store.get(id).expect("get").is_some());

    // Now purge s2 too. No references remain anywhere — block is gone.
    store.purge_session(s2).expect("purge s2");
    assert!(store.get(id).expect("get").is_none());
    assert_eq!(count_orphan_blocks(&url, &schema), 0);
}
