//! TLS connectivity smoke tests, gated on `LLM386_PG_TLS_TEST_URL`.
//!
//! To run these locally:
//!
//!   docker run --rm -d --name pgssl -p 55432:5432 \
//!       -v $PWD/server.crt:/etc/ssl/server.crt:ro \
//!       -v $PWD/server.key:/etc/ssl/server.key:ro \
//!       -e POSTGRES_PASSWORD=test \
//!       postgres:18 -c ssl=on \
//!       -c ssl_cert_file=/etc/ssl/server.crt \
//!       -c ssl_key_file=/etc/ssl/server.key
//!
//!   LLM386_PG_TLS_TEST_URL='postgres://postgres:test@localhost:55432/postgres' \
//!     cargo test -p llm386-store-pg --features tls-native-tls --test tls_require
//!
//! Skips cleanly when the env var is unset. The negative case
//! ("Require correctly refuses non-TLS server") is covered by manual
//! smoke testing — it's covered in the FAQ + README — and doesn't
//! need a self-contained reproducer here because the failure mode
//! is a `Connect` error from the postgres driver, well-tested by
//! upstream.

#![cfg(feature = "tls-native-tls")]

use std::sync::atomic::{AtomicU32, Ordering};

use llm386_core::{
    BlockId, BlockKind, BlockStore, ContentHash, ContextBlock, Provenance, SessionId, Timestamp,
    TokenCounts,
};
use llm386_store_pg::{PgStore, PgStoreConfig, TlsMode};

static SCHEMA_COUNTER: AtomicU32 = AtomicU32::new(0);

fn fresh_schema() -> String {
    let n = SCHEMA_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    format!("llm386_tlstest_{pid}_{n}")
}

#[test]
fn tls_require_opens_and_round_trips_a_block() {
    let Ok(url) = std::env::var("LLM386_PG_TLS_TEST_URL") else {
        eprintln!("skipped: LLM386_PG_TLS_TEST_URL not set");
        return;
    };
    let schema = fresh_schema();
    // Use RequireCustomCa if LLM386_PG_TLS_TEST_CA is set (for
    // self-signed certs); otherwise Require trusts the system store.
    let tls = match std::env::var("LLM386_PG_TLS_TEST_CA") {
        Ok(path) => TlsMode::RequireCustomCa { ca_path: path.into() },
        Err(_) => TlsMode::Require,
    };

    let store = PgStore::open(
        &url,
        &PgStoreConfig {
            max_pool_size: 2,
            schema: Some(schema.clone()),
            tls,
            ..Default::default()
        },
    )
    .expect("PgStore::open with TLS");

    let session = SessionId(1);
    let bytes = b"tls round-trip";
    let block = ContextBlock {
        id: BlockId::from_parts(1, 1),
        kind: BlockKind::Fact,
        bytes: bytes.to_vec(),
        token_counts: TokenCounts::new(),
        priority: 0.0,
        created_at: Timestamp(1),
        updated_at: Timestamp(1),
        provenance: Provenance::default(),
        hash: ContentHash::of(bytes),
    };
    let id = store.put(session, block).expect("put over TLS");
    let fetched = store.get(id).expect("get over TLS").expect("block exists");
    assert_eq!(fetched.bytes, bytes);
}
