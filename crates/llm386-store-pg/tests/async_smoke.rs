//! Async smoke tests for `AsyncPgStore`.
//!
//! These run only when the `async` feature is enabled (Cargo.toml gates
//! the test target on `required-features = ["async"]`). They skip
//! cleanly if `TEST_DATABASE_URL` is not set, mirroring the sync test
//! suite.

#![cfg(feature = "async")]

use std::sync::atomic::{AtomicU32, Ordering};

use llm386_core::{
    BlockId, BlockKind, ContentHash, ContextBlock, Edge, EdgeKind, Provenance, SessionId,
    Timestamp, TokenCounts,
};
use llm386_store_pg::{AsyncPgStore, PgStoreConfig};
use tokio_postgres::NoTls;

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

async fn open_test() -> Option<(AsyncPgStore, String, String)> {
    let url = std::env::var("TEST_DATABASE_URL").ok()?;
    let n = SCHEMA_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    let schema = format!("llm386_async_test_{pid}_{n}");
    let store = AsyncPgStore::open(
        &url,
        &PgStoreConfig {
            max_pool_size: 8,
            schema: Some(schema.clone()),
            ..Default::default()
        },
    )
    .await
    .expect("open AsyncPgStore");
    Some((store, url, schema))
}

async fn cleanup(url: &str, schema: &str) {
    let Ok((client, conn)) = tokio_postgres::connect(url, NoTls).await else {
        return;
    };
    let conn_handle = tokio::spawn(async move {
        let _ = conn.await;
    });
    let _ = client
        .batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE;"))
        .await;
    drop(client);
    let _ = conn_handle.await;
}

/// Drop guard that re-runs schema cleanup synchronously on its own
/// runtime — needed because async destructors aren't a thing and we
/// still want teardown to run on panic.
struct DropGuard {
    url: String,
    schema: String,
}
impl Drop for DropGuard {
    fn drop(&mut self) {
        // Cleanup synchronously by spawning a fresh runtime on a
        // dedicated thread. Cheap; only runs at test teardown.
        let url = self.url.clone();
        let schema = self.schema.clone();
        let handle = std::thread::spawn(move || {
            let rt = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("rt for cleanup");
            rt.block_on(cleanup(&url, &schema));
        });
        let _ = handle.join();
    }
}

async fn open_with_cleanup() -> Option<(AsyncPgStore, DropGuard)> {
    let (store, url, schema) = open_test().await?;
    Some((store, DropGuard { url, schema }))
}

#[tokio::test]
async fn put_then_get_roundtrips() {
    let Some((store, _g)) = open_with_cleanup().await else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let session = SessionId(1);
    let block = make_block(b"hello async", BlockKind::UserMessage, 1_000, 42);
    let id = store.put(session, block.clone()).await.unwrap();
    let fetched = store.get(id).await.unwrap().unwrap();
    assert_eq!(fetched.bytes, block.bytes);
    assert_eq!(fetched.hash, block.hash);
}

#[tokio::test]
async fn duplicate_content_returns_existing_id() {
    let Some((store, _g)) = open_with_cleanup().await else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let session = SessionId(1);
    let first = make_block(b"async dedup", BlockKind::UserMessage, 1_000, 42);
    let id1 = store.put(session, first).await.unwrap();
    let dup = make_block(b"async dedup", BlockKind::UserMessage, 2_000, 99);
    let id2 = store.put(session, dup).await.unwrap();
    assert_eq!(id1, id2);
}

#[tokio::test]
async fn list_session_chronological() {
    let Some((store, _g)) = open_with_cleanup().await else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let s = SessionId(7);
    let a = store.put(s, make_block(b"a", BlockKind::Fact, 1, 1)).await.unwrap();
    let b = store.put(s, make_block(b"b", BlockKind::Fact, 2, 2)).await.unwrap();
    let c = store.put(s, make_block(b"c", BlockKind::Fact, 3, 3)).await.unwrap();
    assert_eq!(store.list_session(s).await.unwrap(), vec![a, b, c]);
}

#[tokio::test]
async fn edges_roundtrip_and_delete_purges_them() {
    let Some((store, _g)) = open_with_cleanup().await else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let s = SessionId(1);
    let a = store.put(s, make_block(b"A", BlockKind::Fact, 1, 1)).await.unwrap();
    let b = store.put(s, make_block(b"B", BlockKind::Fact, 2, 2)).await.unwrap();
    store
        .put_edge(Edge { from: a, to: b, kind: EdgeKind::Supports })
        .await
        .unwrap();
    assert_eq!(store.edges_from(a).await.unwrap().len(), 1);
    assert_eq!(store.edges_to(b).await.unwrap().len(), 1);
    assert!(store.delete(a).await.unwrap());
    assert!(store.edges_from(a).await.unwrap().is_empty());
    assert!(store.edges_to(b).await.unwrap().is_empty());
}

/// The async store's whole reason for existing — fire many concurrent
/// puts through the pool and verify nothing serializes incorrectly,
/// they all succeed, and dedup still works.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_puts_dedup_correctly() {
    let Some((store, _g)) = open_with_cleanup().await else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    // 50 concurrent puts of unique blocks across 5 sessions.
    let mut handles = Vec::new();
    for i in 0..50_u32 {
        let store = store.clone();
        let session = SessionId(u128::from(i % 5));
        handles.push(tokio::spawn(async move {
            let bytes = format!("concurrent-{i}").into_bytes();
            let block = make_block(&bytes, BlockKind::Fact, u64::from(i), u128::from(i));
            store.put(session, block).await
        }));
    }
    let mut ids = Vec::new();
    for h in handles {
        ids.push(h.await.unwrap().unwrap());
    }
    assert_eq!(ids.len(), 50);

    // 50 concurrent puts of the SAME bytes across 50 sessions — every
    // result must collapse to one id (dedup). Each task proposes a
    // distinct BlockId (as a real caller would, generating fresh ids
    // from time+random); the content hash is identical, so they all
    // converge on whichever id won the race.
    let mut handles = Vec::new();
    for i in 0..50_u32 {
        let store = store.clone();
        let session = SessionId(u128::from(i) + 100);
        handles.push(tokio::spawn(async move {
            let block = make_block(
                b"shared-payload",
                BlockKind::Fact,
                u64::from(i) + 1_000,
                u128::from(i) + 1_000,
            );
            store.put(session, block).await
        }));
    }
    let mut dedup_ids = std::collections::HashSet::new();
    for h in handles {
        dedup_ids.insert(h.await.unwrap().unwrap());
    }
    assert_eq!(dedup_ids.len(), 1, "all dedup puts must converge on one id");
}
