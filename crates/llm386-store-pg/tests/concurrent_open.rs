//! Concurrency tests for `PgStore::open` and `AsyncPgStore::open`.
//!
//! Server startup paths often fire several workers that each call
//! `open(url, cfg)` against the same database + schema. The schema
//! bootstrap + migration step issues DDL — `CREATE SCHEMA IF NOT
//! EXISTS`, `CREATE TABLE IF NOT EXISTS`, `CREATE INDEX IF NOT
//! EXISTS` — and despite the `IF NOT EXISTS` clauses, every one of
//! those is documented as non-transactional and races on
//! `pg_namespace_nspname_index` / `pg_type_typname_nsp_index`.
//! Concurrent openers crash on `SQLSTATE 23505`.
//!
//! The fix is a Postgres advisory lock shared across every opener;
//! these tests pin the contract by hammering both backends with N
//! concurrent opens against the same fresh schema.
//!
//! Skips cleanly when `TEST_DATABASE_URL` is unset.

use std::sync::atomic::{AtomicU32, Ordering};

use llm386_store_pg::{PgStore, PgStoreConfig};

static SCHEMA_COUNTER: AtomicU32 = AtomicU32::new(0);

fn fresh_schema(label: &str) -> String {
    let n = SCHEMA_COUNTER.fetch_add(1, Ordering::SeqCst);
    let pid = std::process::id();
    format!("llm386_{label}_{pid}_{n}")
}

fn cleanup_sync(url: &str, schema: &str) {
    use postgres::NoTls;
    let Ok(mut client) = postgres::Client::connect(url, NoTls) else {
        return;
    };
    let _ = client.batch_execute(&format!("DROP SCHEMA IF EXISTS \"{schema}\" CASCADE;"));
}

/// N concurrent `PgStore::open()` calls against the same fresh
/// schema. Without the advisory lock at least one races on the
/// migration DDL (`pg_type_typname_nsp_index`).
#[test]
fn sync_concurrent_open_against_same_schema_all_succeed() {
    let Ok(url) = std::env::var("TEST_DATABASE_URL") else {
        eprintln!("skipped: TEST_DATABASE_URL not set");
        return;
    };
    let schema = fresh_schema("sync_concuropen");

    let failures = std::thread::scope(|s| {
        const OPENERS: usize = 8;
        let mut handles = Vec::with_capacity(OPENERS);
        for _ in 0..OPENERS {
            let url = url.clone();
            let schema = schema.clone();
            handles.push(s.spawn(move || {
                PgStore::open(
                    &url,
                    &PgStoreConfig {
                        max_pool_size: 4,
                        schema: Some(schema),
                    },
                )
            }));
        }
        let mut failures = Vec::new();
        for (i, h) in handles.into_iter().enumerate() {
            match h.join().expect("thread panicked") {
                Ok(_store) => {}
                Err(e) => {
                    let mut chain = format!("{e}");
                    let mut src: Option<&dyn std::error::Error> =
                        std::error::Error::source(&e);
                    while let Some(s) = src {
                        use std::fmt::Write;
                        let _ = write!(chain, " | {s}");
                        src = s.source();
                    }
                    failures.push((i, chain));
                }
            }
        }
        failures
    });

    cleanup_sync(&url, &schema);

    assert!(
        failures.is_empty(),
        "{} concurrent PgStore::open() call(s) failed — bootstrap raced on Postgres catalogs: {:?}",
        failures.len(),
        failures,
    );
}

// Async sibling. Gated on the `async` feature so the file still
// compiles in the default build; the corresponding `[[test]]` entry
// in Cargo.toml also gates required-features.
#[cfg(feature = "async")]
mod async_open {
    use super::{fresh_schema, SCHEMA_COUNTER};
    use llm386_store_pg::{AsyncPgStore, PgStoreConfig};
    use tokio_postgres::NoTls;

    async fn cleanup_async(url: &str, schema: &str) {
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

    /// N concurrent `AsyncPgStore::open()` calls against the same
    /// fresh schema. Without the advisory lock the schema bootstrap
    /// races on `pg_namespace_nspname_index` and `pg_type_typname_nsp_index`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn async_concurrent_open_against_same_schema_all_succeed() {
        let Ok(url) = std::env::var("TEST_DATABASE_URL") else {
            eprintln!("skipped: TEST_DATABASE_URL not set");
            return;
        };
        let _ = SCHEMA_COUNTER; // silence unused-import in async-only builds
        let schema = fresh_schema("async_concuropen");

        let result = {
            let url = url.clone();
            let schema = schema.clone();
            async move {
                const OPENERS: usize = 8;
                let mut handles = Vec::with_capacity(OPENERS);
                for _ in 0..OPENERS {
                    let url = url.clone();
                    let schema = schema.clone();
                    handles.push(tokio::spawn(async move {
                        AsyncPgStore::open(
                            &url,
                            &PgStoreConfig {
                                max_pool_size: 4,
                                schema: Some(schema),
                            },
                        )
                        .await
                    }));
                }
                let mut failures = Vec::new();
                for (i, h) in handles.into_iter().enumerate() {
                    match h.await.expect("task panicked") {
                        Ok(_store) => {}
                        Err(e) => {
                            let mut chain = format!("{e}");
                            let mut src: Option<&dyn std::error::Error> =
                                std::error::Error::source(&e);
                            while let Some(s) = src {
                                use std::fmt::Write;
                        let _ = write!(chain, " | {s}");
                                src = s.source();
                            }
                            failures.push((i, chain));
                        }
                    }
                }
                failures
            }
            .await
        };

        cleanup_async(&url, &schema).await;

        assert!(
            result.is_empty(),
            "{} concurrent AsyncPgStore::open() call(s) failed — bootstrap raced on Postgres catalogs: {:?}",
            result.len(),
            result,
        );
    }
}
