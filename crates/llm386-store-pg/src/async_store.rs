//! `AsyncPgStore` — pipelined async PostgreSQL access.
//!
//! Mirrors the sync [`crate::PgStore`] surface but exposes inherent
//! async methods instead of implementing the sync `BlockStore` trait.
//! Backed by `tokio-postgres` + `deadpool-postgres`: many concurrent
//! awaits share a small pool of connections, with the actual SQL
//! pipelined down each socket.
//!
//! `AsyncPgStore` does not implement `BlockStore` on purpose. The trait
//! is synchronous and the pager / packer consume it that way; mixing
//! async storage into a sync trait would require `block_on` calls
//! that defeat the point. Use this struct directly from async call
//! sites (axum handlers, tonic services, ingest pipelines) when you
//! want real concurrency against Postgres.

use deadpool_postgres::{Config, ManagerConfig, Pool, RecyclingMethod, Runtime, SslMode};
use llm386_core::{
    BlockId, ContentHash, ContextBlock, Edge, SessionId, StoreError,
};
use tokio_postgres::NoTls;
use tokio_postgres::types::ToSql;
use tracing::{debug, instrument};

use crate::common::{
    CURRENT_SCHEMA, MIGRATION_SQL, PgStoreConfig, StoreOpenError, block_kind_to_str,
    decode_block_id, edge_kind_from_str, edge_kind_to_str, id_bytes, is_valid_ident,
    provenance_to_json, row_to_block, session_bytes, token_counts_to_json, ts_to_i64,
};
use crate::tls::{TlsBackend, build_backend};

/// Async PostgreSQL-backed block store.
///
/// Cheap to clone (clones share the underlying deadpool `Pool`).
#[derive(Clone)]
pub struct AsyncPgStore {
    pool: Pool,
    /// Optional Postgres schema to scope every connection to. Applied
    /// on every checkout via `SET search_path`. `None` uses whatever
    /// search_path the server's role default supplies.
    schema: Option<String>,
}

impl std::fmt::Debug for AsyncPgStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AsyncPgStore")
            .field("schema_version", &CURRENT_SCHEMA)
            .field("search_path", &self.schema)
            .finish_non_exhaustive()
    }
}

impl AsyncPgStore {
    /// Connect to `url`, build the deadpool pool, run migrations.
    /// Idempotent: subsequent opens reuse the existing tables.
    ///
    /// `config.tls` controls transport security. Default is
    /// [`crate::TlsMode::Disable`] (plaintext) for back-compat;
    /// production deployments should set [`crate::TlsMode::Require`]
    /// or stronger. See [`crate::PgStore::open`] for the full
    /// feature-gate behavior.
    #[instrument(skip(config), fields(schema = ?config.schema, tls = ?config.tls))]
    pub async fn open(url: &str, config: &PgStoreConfig) -> Result<Self, StoreOpenError> {
        if let Some(schema) = &config.schema
            && !is_valid_ident(schema)
        {
            return Err(StoreOpenError::InvalidSchemaName(schema.clone()));
        }

        // Build the TLS connector once and reuse it for the pool
        // (and any side connection the bootstrap path needs).
        let tls = build_backend(&config.tls)?;

        let mut cfg = Config::new();
        cfg.url = Some(url.to_string());
        cfg.manager = Some(ManagerConfig {
            recycling_method: RecyclingMethod::Fast,
        });
        // Force sslmode to match the TLS backend so `TlsMode::Require`
        // genuinely requires TLS (the default `sslmode=prefer` would
        // silently fall back to plaintext if the server doesn't offer
        // TLS — a violation of the contract).
        cfg.ssl_mode = Some(match &tls {
            TlsBackend::Plain => SslMode::Disable,
            #[cfg(feature = "tls-native-tls")]
            TlsBackend::Native(_) => SslMode::Require,
        });

        // `Pool` itself isn't generic over the TLS connector —
        // deadpool-postgres erases it inside the Manager — so a single
        // Pool type holds either branch.
        let pool = match &tls {
            TlsBackend::Plain => cfg
                .builder(NoTls)
                .map_err(|e| StoreOpenError::Connect(format!("deadpool builder: {e}")))?
                .max_size(config.max_pool_size as usize)
                .runtime(Runtime::Tokio1)
                .build()
                .map_err(|e| StoreOpenError::Connect(format!("deadpool build: {e}")))?,
            #[cfg(feature = "tls-native-tls")]
            TlsBackend::Native(connector) => {
                let mts = postgres_native_tls::MakeTlsConnector::new(connector.clone());
                cfg.builder(mts)
                    .map_err(|e| StoreOpenError::Connect(format!("deadpool builder: {e}")))?
                    .max_size(config.max_pool_size as usize)
                    .runtime(Runtime::Tokio1)
                    .build()
                    .map_err(|e| StoreOpenError::Connect(format!("deadpool build: {e}")))?
            }
        };

        let store = Self {
            pool,
            schema: config.schema.clone(),
        };

        // Bootstrap: check out one connection, take a Postgres
        // advisory lock that's shared across every concurrent
        // `AsyncPgStore::open` (and every `PgStore::open`), then run
        // CREATE SCHEMA + SET search_path + the full migration set
        // under the lock. Without the lock, concurrent openers race
        // on `pg_namespace_nspname_index` (CREATE SCHEMA) and on
        // `pg_type_typname_nsp_index` (CREATE TABLE) — neither
        // `IF NOT EXISTS` clause makes those DDL statements safe
        // under concurrency, per the Postgres docs. The lock auto-
        // releases on commit. Subsequent per-checkout calls in
        // `Self::conn` only re-apply `SET search_path`, no DDL.
        let mut conn = store
            .pool
            .get()
            .await
            .map_err(|e| StoreOpenError::Connect(format!("pool checkout: {e}")))?;
        let tx = conn
            .transaction()
            .await
            .map_err(|e| StoreOpenError::Connect(format!("bootstrap begin: {e}")))?;
        tx.batch_execute(
            "SELECT pg_advisory_xact_lock(hashtextextended('llm386_bootstrap', 0));",
        )
        .await
        .map_err(|e| StoreOpenError::Connect(format!("advisory lock: {e}")))?;

        if let Some(schema) = &config.schema {
            tx.batch_execute(&format!(
                "CREATE SCHEMA IF NOT EXISTS \"{schema}\"; SET search_path TO \"{schema}\";",
            ))
            .await
            .map_err(|e| StoreOpenError::Connect(format!("CREATE SCHEMA: {e}")))?;
        }
        tx.batch_execute(MIGRATION_SQL).await?;
        tx.execute(
            "INSERT INTO llm386_meta (key, value)
             VALUES ('schema_version', $1)
             ON CONFLICT (key) DO NOTHING",
            &[&CURRENT_SCHEMA.to_be_bytes().to_vec()],
        )
        .await?;
        let row = tx
            .query_opt(
                "SELECT value FROM llm386_meta WHERE key = 'schema_version'",
                &[],
            )
            .await?
            .ok_or_else(|| StoreOpenError::CorruptMeta("schema_version row missing".into()))?;
        let bytes: &[u8] = row.get(0);
        if bytes.len() != 4 {
            return Err(StoreOpenError::CorruptMeta(format!(
                "schema_version width {}",
                bytes.len()
            )));
        }
        let arr: [u8; 4] = bytes.try_into().expect("checked width above");
        let found = i32::from_be_bytes(arr);
        if found != CURRENT_SCHEMA {
            return Err(StoreOpenError::SchemaMismatch {
                expected: CURRENT_SCHEMA,
                found,
            });
        }
        tx.commit()
            .await
            .map_err(|e| StoreOpenError::Connect(format!("bootstrap commit: {e}")))?;
        drop(conn);
        debug!(schema = CURRENT_SCHEMA, "AsyncPgStore ready");

        Ok(store)
    }

    /// Check a connection out of the pool and apply `SET search_path`
    /// if a schema was configured. The schema is guaranteed to exist
    /// because [`AsyncPgStore::open`] created it before returning,
    /// so we don't re-issue `CREATE SCHEMA` here — concurrent
    /// `CREATE SCHEMA IF NOT EXISTS` calls race against pg_namespace.
    async fn conn(&self) -> Result<deadpool_postgres::Object, StoreError> {
        let conn = self
            .pool
            .get()
            .await
            .map_err(|e| StoreError::Backend(format!("pool checkout: {e}")))?;
        if let Some(schema) = &self.schema {
            conn.batch_execute(&format!("SET search_path TO \"{schema}\";"))
                .await
                .map_err(|e| StoreError::Backend(format!("set search_path: {e}")))?;
        }
        Ok(conn)
    }

    #[instrument(skip(self, block), fields(id = %block.id, kind = ?block.kind))]
    pub async fn put(
        &self,
        session: SessionId,
        block: ContextBlock,
    ) -> Result<BlockId, StoreError> {
        let mut conn = self.conn().await?;
        let tx = conn
            .transaction()
            .await
            .map_err(|e| StoreError::Backend(format!("begin tx: {e}")))?;

        let proposed_id = id_bytes(block.id);
        let hash_bytes = block.hash.0.to_vec();
        let kind_str = block_kind_to_str(block.kind);
        let token_counts_json = token_counts_to_json(&block.token_counts);
        let provenance_json = provenance_to_json(&block.provenance);
        let created_at_i64 = ts_to_i64(block.created_at);
        let updated_at_i64 = ts_to_i64(block.updated_at);
        let priority_f32 = block.priority;

        let params: &[&(dyn ToSql + Sync)] = &[
            &proposed_id.as_slice(),
            &kind_str,
            &block.bytes,
            &token_counts_json,
            &priority_f32,
            &created_at_i64,
            &updated_at_i64,
            &provenance_json,
            &hash_bytes.as_slice(),
        ];

        // Single-statement UPSERT — see `PgStore::put` for the
        // detailed rationale. Closes the TOCTOU window between the
        // failed INSERT and a fallback SELECT under READ COMMITTED;
        // the no-op `SET hash = llm386_blocks.hash` acquires the
        // conflict row's lock without observable side effect.
        let row = tx
            .query_one(
                "INSERT INTO llm386_blocks
                    (id, kind, bytes, token_counts, priority,
                     created_at, updated_at, provenance, hash)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                 ON CONFLICT (hash) DO UPDATE SET hash = llm386_blocks.hash
                 RETURNING id",
                params,
            )
            .await
            .map_err(|e| StoreError::Backend(format!("insert block: {}", format_pg_err(&e))))?;
        let stored_id = {
            let bytes: &[u8] = row.get(0);
            let id = decode_block_id(bytes)?;
            if id != block.id {
                debug!(?id, "deduped on content hash");
            }
            id
        };

        tx.execute(
            "INSERT INTO llm386_session_blocks (session_id, block_id)
             VALUES ($1, $2)
             ON CONFLICT DO NOTHING",
            &[&session_bytes(session).as_slice(), &id_bytes(stored_id).as_slice()],
        )
        .await
        .map_err(|e| StoreError::Backend(format!("insert session_block: {e}")))?;

        tx.commit()
            .await
            .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;
        Ok(stored_id)
    }

    pub async fn get(&self, id: BlockId) -> Result<Option<ContextBlock>, StoreError> {
        let conn = self.conn().await?;
        let row = conn
            .query_opt(
                "SELECT id, kind, bytes, token_counts, priority,
                        created_at, updated_at, provenance, hash
                 FROM llm386_blocks WHERE id = $1",
                &[&id_bytes(id).as_slice()],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("get block: {e}")))?;
        match row {
            Some(r) => Ok(Some(row_to_block(&r)?)),
            None => Ok(None),
        }
    }

    pub async fn list_session(&self, session: SessionId) -> Result<Vec<BlockId>, StoreError> {
        let conn = self.conn().await?;
        let rows = conn
            .query(
                "SELECT block_id FROM llm386_session_blocks
                 WHERE session_id = $1 ORDER BY block_id ASC",
                &[&session_bytes(session).as_slice()],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("list_session: {e}")))?;
        let mut ids = Vec::with_capacity(rows.len());
        for row in &rows {
            let bytes: &[u8] = row.get(0);
            ids.push(decode_block_id(bytes)?);
        }
        Ok(ids)
    }

    pub async fn list_sessions(&self) -> Result<Vec<SessionId>, StoreError> {
        let conn = self.conn().await?;
        let rows = conn
            .query(
                "SELECT DISTINCT session_id FROM llm386_session_blocks
                 ORDER BY session_id ASC",
                &[],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("list_sessions: {e}")))?;
        let mut sessions = Vec::with_capacity(rows.len());
        for row in &rows {
            let bytes: &[u8] = row.get(0);
            let id = decode_block_id(bytes)?;
            sessions.push(SessionId(id.0));
        }
        Ok(sessions)
    }

    pub async fn lookup_hash(&self, hash: ContentHash) -> Result<Option<BlockId>, StoreError> {
        let conn = self.conn().await?;
        let row = conn
            .query_opt(
                "SELECT id FROM llm386_blocks WHERE hash = $1",
                &[&hash.0.as_slice()],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("lookup_hash: {e}")))?;
        match row {
            Some(r) => {
                let bytes: &[u8] = r.get(0);
                Ok(Some(decode_block_id(bytes)?))
            }
            None => Ok(None),
        }
    }

    #[instrument(skip(self), fields(id = %id))]
    pub async fn delete(&self, id: BlockId) -> Result<bool, StoreError> {
        let mut conn = self.conn().await?;
        let tx = conn
            .transaction()
            .await
            .map_err(|e| StoreError::Backend(format!("begin tx: {e}")))?;

        let id_b = id_bytes(id);
        let id_slice: &[u8] = id_b.as_slice();

        // Lock order matters — see the matching comment in
        // [`crate::PgStore::delete`]. `put`'s ON CONFLICT DO UPDATE
        // takes a row lock on `llm386_blocks.id` before writing
        // `llm386_session_blocks`; delete must do the same to avoid
        // deadlocking with concurrent puts.
        let block_deleted = tx
            .execute("DELETE FROM llm386_blocks WHERE id = $1", &[&id_slice])
            .await
            .map_err(|e| StoreError::Backend(format!("delete block: {e}")))?;
        let sessions_deleted = tx
            .execute(
                "DELETE FROM llm386_session_blocks WHERE block_id = $1",
                &[&id_slice],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("delete session refs: {e}")))?;
        tx.execute(
            "DELETE FROM llm386_edges WHERE from_id = $1 OR to_id = $1",
            &[&id_slice],
        )
        .await
        .map_err(|e| StoreError::Backend(format!("delete edges: {e}")))?;

        tx.commit()
            .await
            .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;

        let existed = block_deleted > 0 || sessions_deleted > 0;
        if existed {
            debug!(?id, sessions = sessions_deleted, "block deleted");
        }
        Ok(existed)
    }

    #[instrument(skip(self), fields(session = %session))]
    pub async fn purge_session(&self, session: SessionId) -> Result<usize, StoreError> {
        let mut conn = self.conn().await?;
        let tx = conn
            .transaction()
            .await
            .map_err(|e| StoreError::Backend(format!("begin tx: {e}")))?;

        let session_b = session_bytes(session);
        let session_slice: &[u8] = session_b.as_slice();

        // Step 1: collect every block id this session references.
        let rows = tx
            .query(
                "SELECT block_id FROM llm386_session_blocks WHERE session_id = $1",
                &[&session_slice],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("collect session blocks: {e}")))?;
        let block_ids: Vec<BlockId> = rows
            .iter()
            .map(|r| {
                let b: &[u8] = r.get(0);
                decode_block_id(b)
            })
            .collect::<Result<_, _>>()?;
        let count = block_ids.len();
        if count == 0 {
            tx.commit()
                .await
                .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;
            return Ok(0);
        }

        let id_byte_bufs: Vec<Vec<u8>> = block_ids
            .iter()
            .map(|id| id_bytes(*id).to_vec())
            .collect();
        let id_slices: Vec<&[u8]> = id_byte_bufs.iter().map(Vec::as_slice).collect();

        // Step 2: lock every block row we might delete. Matches the
        // lock order in `put` (blocks → session_blocks) and `delete`,
        // so concurrent put + purge + delete can't deadlock. Also
        // forces any concurrent `put` that dedups to one of these
        // blocks to block at its `DO UPDATE` until purge commits —
        // closing the orphan-check race where put's session_blocks
        // insert could commit between purge's check and purge's
        // block delete. See `PgStore::purge_session` for the long
        // rationale.
        tx.execute(
            "SELECT 1 FROM llm386_blocks WHERE id = ANY($1) FOR UPDATE",
            &[&id_slices],
        )
        .await
        .map_err(|e| StoreError::Backend(format!("lock blocks: {e}")))?;

        // Step 3: lock every session_blocks row that references any
        // of these block ids. Without this, two concurrent
        // purge_session calls on shared blocks would both see each
        // other's uncommitted deletes as "still alive" in their
        // orphan checks, skip the block delete, and leak the block.
        tx.execute(
            "SELECT 1 FROM llm386_session_blocks
             WHERE block_id = ANY($1)
             FOR UPDATE",
            &[&id_slices],
        )
        .await
        .map_err(|e| StoreError::Backend(format!("lock session refs: {e}")))?;

        // Step 4: drop this session's references.
        tx.execute(
            "DELETE FROM llm386_session_blocks WHERE session_id = $1",
            &[&session_slice],
        )
        .await
        .map_err(|e| StoreError::Backend(format!("delete session refs: {e}")))?;

        // Step 5: find which of those blocks are now orphans in a
        // single query (vs. the per-id N+1 round-trips this used to
        // do).
        let orphan_rows = tx
            .query(
                "SELECT id FROM llm386_blocks
                 WHERE id = ANY($1)
                   AND NOT EXISTS (
                     SELECT 1 FROM llm386_session_blocks sb
                     WHERE sb.block_id = llm386_blocks.id
                   )",
                &[&id_slices],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("collect orphans: {e}")))?;
        let orphan_byte_bufs: Vec<Vec<u8>> = orphan_rows
            .iter()
            .map(|r| {
                let b: &[u8] = r.get(0);
                b.to_vec()
            })
            .collect();

        // Step 6: scrub edges + blocks for the orphan set in two
        // statements, regardless of how many orphans there are.
        if !orphan_byte_bufs.is_empty() {
            let orphan_slices: Vec<&[u8]> =
                orphan_byte_bufs.iter().map(Vec::as_slice).collect();
            tx.execute(
                "DELETE FROM llm386_edges
                 WHERE from_id = ANY($1) OR to_id = ANY($1)",
                &[&orphan_slices],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("delete orphan edges: {e}")))?;
            tx.execute(
                "DELETE FROM llm386_blocks WHERE id = ANY($1)",
                &[&orphan_slices],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("delete orphan blocks: {e}")))?;
        }

        tx.commit()
            .await
            .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;
        Ok(count)
    }

    pub async fn put_edge(&self, edge: Edge) -> Result<(), StoreError> {
        let conn = self.conn().await?;
        let kind_str = edge_kind_to_str(edge.kind);
        let from_b = id_bytes(edge.from);
        let to_b = id_bytes(edge.to);
        conn.execute(
            "INSERT INTO llm386_edges (from_id, kind, to_id)
             VALUES ($1, $2, $3)
             ON CONFLICT (from_id, kind, to_id) DO NOTHING",
            &[&from_b.as_slice(), &kind_str, &to_b.as_slice()],
        )
        .await
        .map_err(|e| StoreError::Backend(format!("put_edge: {e}")))?;
        Ok(())
    }

    pub async fn edges_from(&self, from: BlockId) -> Result<Vec<Edge>, StoreError> {
        let conn = self.conn().await?;
        let rows = conn
            .query(
                "SELECT to_id, kind FROM llm386_edges
                 WHERE from_id = $1 ORDER BY kind, to_id",
                &[&id_bytes(from).as_slice()],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("edges_from: {e}")))?;
        let mut out = Vec::with_capacity(rows.len());
        for r in &rows {
            let to_b: &[u8] = r.get(0);
            let kind_str: &str = r.get(1);
            out.push(Edge {
                from,
                to: decode_block_id(to_b)?,
                kind: edge_kind_from_str(kind_str)?,
            });
        }
        Ok(out)
    }

    pub async fn edges_to(&self, to: BlockId) -> Result<Vec<Edge>, StoreError> {
        let conn = self.conn().await?;
        let rows = conn
            .query(
                "SELECT from_id, kind FROM llm386_edges
                 WHERE to_id = $1 ORDER BY kind, from_id",
                &[&id_bytes(to).as_slice()],
            )
            .await
            .map_err(|e| StoreError::Backend(format!("edges_to: {e}")))?;
        let mut out = Vec::with_capacity(rows.len());
        for r in &rows {
            let from_b: &[u8] = r.get(0);
            let kind_str: &str = r.get(1);
            out.push(Edge {
                from: decode_block_id(from_b)?,
                to,
                kind: edge_kind_from_str(kind_str)?,
            });
        }
        Ok(out)
    }
}

/// Render a tokio_postgres error including the SQLSTATE and detail
/// from the underlying DbError, which the bare Display impl strips
/// down to a useless `"db error"`.
fn format_pg_err(e: &tokio_postgres::Error) -> String {
    if let Some(db) = e.as_db_error() {
        format!(
            "{} (SQLSTATE {}): {}",
            db.severity(),
            db.code().code(),
            db.message(),
        )
    } else {
        e.to_string()
    }
}

