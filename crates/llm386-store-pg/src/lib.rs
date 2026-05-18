//! `llm386-store-pg` — PostgreSQL-backed persistent block storage.
//!
//! This crate provides [`PgStore`], an implementation of the
//! `BlockStore` trait from `llm386-core` backed by PostgreSQL via the
//! synchronous `postgres` crate + an `r2d2_postgres` connection pool.
//! Blocks are content-hash deduplicated and indexed by id, hash, and
//! session.
//!
//! The schema is bootstrapped automatically on first connect — no
//! external migration tooling required. Existing stores are checked
//! against a schema version row in the `llm386_meta` table, mirroring
//! the behavior of the LMDB backend.
//!
//! `verify`/`repair` are intentionally not ported from the LMDB store
//! — Postgres has native integrity tools (`pg_dump`, foreign-key
//! checks, `REINDEX`, `VACUUM`) that supersede them.

#![doc(html_root_url = "https://docs.rs/llm386-store-pg/1.0.0-alpha")]

mod store;

pub use store::{PgStore, PgStoreConfig, StoreOpenError};
