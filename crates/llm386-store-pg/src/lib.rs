//! `llm386-store-pg` — PostgreSQL-backed persistent block storage.
//!
//! Provides two store types backed by the same schema:
//!
//! - [`PgStore`] (always available) — synchronous, implements the
//!   `BlockStore` trait from `llm386-core`. Uses the `postgres` crate
//!   with an `r2d2_postgres` connection pool. This is what the rest
//!   of the llm386 workspace (pager, packer, trace) consumes.
//!
//! - [`AsyncPgStore`] (behind the `async` feature) — pipelined async
//!   access on top of `tokio-postgres` + `deadpool-postgres`.
//!   `AsyncPgStore` does **not** implement the sync `BlockStore`
//!   trait — its inherent async methods are meant for direct use
//!   from async call sites (axum handlers, tonic services, ingest
//!   pipelines) where you want concurrent queries through a small
//!   pool of connections.
//!
//! The schema (`llm386_blocks`, `llm386_session_blocks`, `llm386_edges`,
//! `llm386_meta`) is bootstrapped on first connect by either store and
//! checked against a `schema_version` row on every open.
//!
//! `verify`/`repair` integrity tooling from the LMDB backend is
//! intentionally not ported — Postgres has native equivalents
//! (`pg_dump`, foreign-key checks, `REINDEX`, `VACUUM`).

#![doc(html_root_url = "https://docs.rs/llm386-store-pg/1.0.0-alpha")]

mod common;
mod store;

pub use common::{PgStoreConfig, StoreOpenError};
pub use store::PgStore;

#[cfg(feature = "async")]
mod async_store;
#[cfg(feature = "async")]
pub use async_store::AsyncPgStore;
