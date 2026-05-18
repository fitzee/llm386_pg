//! `PgStore` — synchronous `BlockStore` implementation backed by PostgreSQL.
//!
//! Uses the `postgres` crate (a blocking wrapper around `tokio-postgres`)
//! and `r2d2_postgres` for connection pooling. The trait surface is
//! synchronous; for pipelined / concurrent access against the same
//! Postgres database, enable the `async` feature and use
//! [`crate::AsyncPgStore`] instead.

use llm386_core::{
    BlockId, BlockStore, ContentHash, ContextBlock, Edge, SessionId, StoreError,
};
use postgres::NoTls;
use postgres::types::ToSql;
use r2d2::{CustomizeConnection, Pool, PooledConnection};
use r2d2_postgres::PostgresConnectionManager;
use tracing::{debug, instrument};

use crate::common::{
    CURRENT_SCHEMA, MIGRATION_SQL, PgStoreConfig, StoreOpenError, block_kind_to_str,
    decode_block_id, edge_kind_from_str, edge_kind_to_str, id_bytes, is_valid_ident,
    provenance_to_json, row_to_block, session_bytes, token_counts_to_json, ts_to_i64,
};

type PgPool = Pool<PostgresConnectionManager<NoTls>>;
type PgConn = PooledConnection<PostgresConnectionManager<NoTls>>;

/// Synchronous PostgreSQL-backed implementation of the `BlockStore` trait.
///
/// Cheap to clone (clones share the underlying r2d2 pool).
#[derive(Clone)]
pub struct PgStore {
    pool: PgPool,
}

impl std::fmt::Debug for PgStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PgStore")
            .field("schema", &CURRENT_SCHEMA)
            .finish_non_exhaustive()
    }
}

impl PgStore {
    /// Connect to `url` and prepare the schema. Idempotent: subsequent
    /// opens against the same database reuse the existing tables.
    #[instrument(skip(config), fields(schema = ?config.schema))]
    pub fn open(url: &str, config: &PgStoreConfig) -> Result<Self, StoreOpenError> {
        if let Some(schema) = &config.schema
            && !is_valid_ident(schema)
        {
            return Err(StoreOpenError::InvalidSchemaName(schema.clone()));
        }

        // Bootstrap the optional schema on a single dedicated
        // connection *before* the pool exists. `CREATE SCHEMA IF NOT
        // EXISTS` is documented as not transactional — running it
        // concurrently from N pool connections races on
        // `pg_namespace_nspname_index` and dumps "db error" lines
        // even though the schema does end up created.
        if let Some(schema) = &config.schema {
            bootstrap_schema(url, schema)?;
        }

        let manager = PostgresConnectionManager::new(
            url.parse().map_err(|e: postgres::Error| {
                StoreOpenError::Connect(format!("invalid Postgres URL: {e}"))
            })?,
            NoTls,
        );

        let mut builder = Pool::builder().max_size(config.max_pool_size);
        if let Some(schema) = config.schema.clone() {
            builder = builder.connection_customizer(Box::new(SearchPathCustomizer { schema }));
        }
        let pool = builder
            .build(manager)
            .map_err(|e| StoreOpenError::Connect(format!("r2d2 pool: {e}")))?;

        let mut conn = pool.get().map_err(|e| StoreOpenError::Connect(e.to_string()))?;
        run_migrations(&mut conn)?;
        check_schema_version(&mut conn)?;
        debug!(schema = CURRENT_SCHEMA, "PgStore ready");

        Ok(Self { pool })
    }

    fn conn(&self) -> Result<PgConn, StoreError> {
        self.pool
            .get()
            .map_err(|e| StoreError::Backend(format!("pool checkout: {e}")))
    }
}

impl BlockStore for PgStore {
    #[instrument(skip(self, block), fields(id = %block.id, kind = ?block.kind))]
    fn put(&self, session: SessionId, block: ContextBlock) -> Result<BlockId, StoreError> {
        let mut conn = self.conn()?;
        let mut tx = conn
            .transaction()
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

        let rows = tx
            .query(
                "INSERT INTO llm386_blocks
                    (id, kind, bytes, token_counts, priority,
                     created_at, updated_at, provenance, hash)
                 VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                 ON CONFLICT (hash) DO NOTHING
                 RETURNING id",
                params,
            )
            .map_err(|e| StoreError::Backend(format!("insert block: {e}")))?;

        let stored_id = if let Some(row) = rows.first() {
            let bytes: &[u8] = row.get(0);
            decode_block_id(bytes)?
        } else {
            let lookup = tx
                .query_opt(
                    "SELECT id FROM llm386_blocks WHERE hash = $1",
                    &[&hash_bytes.as_slice()],
                )
                .map_err(|e| StoreError::Backend(format!("lookup hash: {e}")))?
                .ok_or_else(|| {
                    StoreError::Backend("dedup conflict but hash row missing".into())
                })?;
            let bytes: &[u8] = lookup.get(0);
            let id = decode_block_id(bytes)?;
            debug!(?id, "deduped on content hash");
            id
        };

        tx.execute(
            "INSERT INTO llm386_session_blocks (session_id, block_id)
             VALUES ($1, $2)
             ON CONFLICT DO NOTHING",
            &[&session_bytes(session).as_slice(), &id_bytes(stored_id).as_slice()],
        )
        .map_err(|e| StoreError::Backend(format!("insert session_block: {e}")))?;

        tx.commit()
            .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;
        Ok(stored_id)
    }

    fn get(&self, id: BlockId) -> Result<Option<ContextBlock>, StoreError> {
        let mut conn = self.conn()?;
        let row = conn
            .query_opt(
                "SELECT id, kind, bytes, token_counts, priority,
                        created_at, updated_at, provenance, hash
                 FROM llm386_blocks WHERE id = $1",
                &[&id_bytes(id).as_slice()],
            )
            .map_err(|e| StoreError::Backend(format!("get block: {e}")))?;
        match row {
            Some(r) => Ok(Some(row_to_block(&r)?)),
            None => Ok(None),
        }
    }

    fn list_session(&self, session: SessionId) -> Result<Vec<BlockId>, StoreError> {
        let mut conn = self.conn()?;
        let rows = conn
            .query(
                "SELECT block_id FROM llm386_session_blocks
                 WHERE session_id = $1 ORDER BY block_id ASC",
                &[&session_bytes(session).as_slice()],
            )
            .map_err(|e| StoreError::Backend(format!("list_session: {e}")))?;
        let mut ids = Vec::with_capacity(rows.len());
        for row in &rows {
            let bytes: &[u8] = row.get(0);
            ids.push(decode_block_id(bytes)?);
        }
        Ok(ids)
    }

    fn list_sessions(&self) -> Result<Vec<SessionId>, StoreError> {
        let mut conn = self.conn()?;
        let rows = conn
            .query(
                "SELECT DISTINCT session_id FROM llm386_session_blocks
                 ORDER BY session_id ASC",
                &[],
            )
            .map_err(|e| StoreError::Backend(format!("list_sessions: {e}")))?;
        let mut sessions = Vec::with_capacity(rows.len());
        for row in &rows {
            let bytes: &[u8] = row.get(0);
            let id = decode_block_id(bytes)?;
            sessions.push(SessionId(id.0));
        }
        Ok(sessions)
    }

    fn lookup_hash(&self, hash: ContentHash) -> Result<Option<BlockId>, StoreError> {
        let mut conn = self.conn()?;
        let row = conn
            .query_opt(
                "SELECT id FROM llm386_blocks WHERE hash = $1",
                &[&hash.0.as_slice()],
            )
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
    fn delete(&self, id: BlockId) -> Result<bool, StoreError> {
        let mut conn = self.conn()?;
        let mut tx = conn
            .transaction()
            .map_err(|e| StoreError::Backend(format!("begin tx: {e}")))?;

        let id_b = id_bytes(id);
        let id_slice: &[u8] = id_b.as_slice();

        let sessions_deleted = tx
            .execute(
                "DELETE FROM llm386_session_blocks WHERE block_id = $1",
                &[&id_slice],
            )
            .map_err(|e| StoreError::Backend(format!("delete session refs: {e}")))?;
        tx.execute(
            "DELETE FROM llm386_edges WHERE from_id = $1 OR to_id = $1",
            &[&id_slice],
        )
        .map_err(|e| StoreError::Backend(format!("delete edges: {e}")))?;
        let block_deleted = tx
            .execute(
                "DELETE FROM llm386_blocks WHERE id = $1",
                &[&id_slice],
            )
            .map_err(|e| StoreError::Backend(format!("delete block: {e}")))?;

        tx.commit()
            .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;

        let existed = block_deleted > 0 || sessions_deleted > 0;
        if existed {
            debug!(?id, sessions = sessions_deleted, "block deleted");
        }
        Ok(existed)
    }

    #[instrument(skip(self), fields(session = %session))]
    fn purge_session(&self, session: SessionId) -> Result<usize, StoreError> {
        let mut conn = self.conn()?;
        let mut tx = conn
            .transaction()
            .map_err(|e| StoreError::Backend(format!("begin tx: {e}")))?;

        let session_b = session_bytes(session);
        let session_slice: &[u8] = session_b.as_slice();

        let rows = tx
            .query(
                "SELECT block_id FROM llm386_session_blocks WHERE session_id = $1",
                &[&session_slice],
            )
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
                .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;
            return Ok(0);
        }

        tx.execute(
            "DELETE FROM llm386_session_blocks WHERE session_id = $1",
            &[&session_slice],
        )
        .map_err(|e| StoreError::Backend(format!("delete session refs: {e}")))?;

        for id in &block_ids {
            let id_b = id_bytes(*id);
            let id_slice: &[u8] = id_b.as_slice();
            let still_referenced: bool = tx
                .query_one(
                    "SELECT EXISTS(SELECT 1 FROM llm386_session_blocks WHERE block_id = $1)",
                    &[&id_slice],
                )
                .map_err(|e| StoreError::Backend(format!("orphan check: {e}")))?
                .get(0);
            if !still_referenced {
                tx.execute(
                    "DELETE FROM llm386_edges WHERE from_id = $1 OR to_id = $1",
                    &[&id_slice],
                )
                .map_err(|e| StoreError::Backend(format!("delete orphan edges: {e}")))?;
                tx.execute(
                    "DELETE FROM llm386_blocks WHERE id = $1",
                    &[&id_slice],
                )
                .map_err(|e| StoreError::Backend(format!("delete orphan block: {e}")))?;
            }
        }

        tx.commit()
            .map_err(|e| StoreError::Backend(format!("commit: {e}")))?;
        Ok(count)
    }

    fn put_edge(&self, edge: Edge) -> Result<(), StoreError> {
        let mut conn = self.conn()?;
        let kind_str = edge_kind_to_str(edge.kind);
        let from_b = id_bytes(edge.from);
        let to_b = id_bytes(edge.to);
        conn.execute(
            "INSERT INTO llm386_edges (from_id, kind, to_id)
             VALUES ($1, $2, $3)
             ON CONFLICT (from_id, kind, to_id) DO NOTHING",
            &[&from_b.as_slice(), &kind_str, &to_b.as_slice()],
        )
        .map_err(|e| StoreError::Backend(format!("put_edge: {e}")))?;
        Ok(())
    }

    fn edges_from(&self, from: BlockId) -> Result<Vec<Edge>, StoreError> {
        let mut conn = self.conn()?;
        let rows = conn
            .query(
                "SELECT to_id, kind FROM llm386_edges
                 WHERE from_id = $1 ORDER BY kind, to_id",
                &[&id_bytes(from).as_slice()],
            )
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

    fn edges_to(&self, to: BlockId) -> Result<Vec<Edge>, StoreError> {
        let mut conn = self.conn()?;
        let rows = conn
            .query(
                "SELECT from_id, kind FROM llm386_edges
                 WHERE to_id = $1 ORDER BY kind, from_id",
                &[&id_bytes(to).as_slice()],
            )
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

/// Per-connection setup: scope every checked-out connection to the
/// configured schema. Idempotent and race-free — the schema itself
/// is created up-front by [`bootstrap_schema`].
#[derive(Debug)]
struct SearchPathCustomizer {
    schema: String,
}

impl CustomizeConnection<postgres::Client, postgres::Error> for SearchPathCustomizer {
    fn on_acquire(&self, conn: &mut postgres::Client) -> Result<(), postgres::Error> {
        // schema is validated by is_valid_ident() before we get here,
        // so this interpolation is safe.
        conn.batch_execute(&format!("SET search_path TO \"{}\";", self.schema))
    }
}

/// Create `schema` if it doesn't already exist, using a single
/// dedicated connection. Tolerant of the `CREATE SCHEMA IF NOT
/// EXISTS` race (Postgres documents this statement as not
/// transactional — concurrent callers can still collide on
/// `pg_namespace_nspname_index`): if the unique-violation fires the
/// schema clearly exists, so it's safe to swallow.
fn bootstrap_schema(url: &str, schema: &str) -> Result<(), StoreOpenError> {
    let mut conn = postgres::Client::connect(url, NoTls)
        .map_err(|e| StoreOpenError::Connect(format!("bootstrap connection: {e}")))?;
    let sql = format!("CREATE SCHEMA IF NOT EXISTS \"{schema}\";");
    match conn.batch_execute(&sql) {
        Ok(()) => Ok(()),
        Err(e) => {
            // SQLSTATE 23505 = unique_violation; the schema already
            // exists because another caller won the race.
            if e.code().map(postgres::error::SqlState::code) == Some("23505") {
                Ok(())
            } else {
                Err(StoreOpenError::Connect(format!("CREATE SCHEMA: {e}")))
            }
        }
    }
}

fn run_migrations(conn: &mut PgConn) -> Result<(), StoreOpenError> {
    conn.batch_execute(MIGRATION_SQL)?;
    conn.execute(
        "INSERT INTO llm386_meta (key, value)
         VALUES ('schema_version', $1)
         ON CONFLICT (key) DO NOTHING",
        &[&CURRENT_SCHEMA.to_be_bytes().to_vec()],
    )?;
    Ok(())
}

fn check_schema_version(conn: &mut PgConn) -> Result<(), StoreOpenError> {
    let row = conn
        .query_opt(
            "SELECT value FROM llm386_meta WHERE key = 'schema_version'",
            &[],
        )?
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
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm386_core::{BlockKind, EdgeKind, Provenance, Timestamp, TokenCounts};
    use std::sync::atomic::{AtomicU32, Ordering};

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

    fn open_test() -> Option<(PgStore, String, String)> {
        let url = std::env::var("TEST_DATABASE_URL").ok()?;
        let n = SCHEMA_COUNTER.fetch_add(1, Ordering::SeqCst);
        let pid = std::process::id();
        let schema = format!("llm386_test_{pid}_{n}");
        let store = PgStore::open(
            &url,
            &PgStoreConfig {
                max_pool_size: 4,
                schema: Some(schema.clone()),
            },
        )
        .expect("open PgStore");
        Some((store, url, schema))
    }

    fn cleanup(url: &str, schema: &str) {
        let Ok(mut client) = postgres::Client::connect(url, NoTls) else {
            return;
        };
        let _ = client.batch_execute(&format!(
            "DROP SCHEMA IF EXISTS \"{schema}\" CASCADE;",
        ));
    }

    macro_rules! pg_test {
        ($body:expr) => {{
            let Some((store, url, schema)) = open_test() else {
                eprintln!("skipped: TEST_DATABASE_URL not set");
                return;
            };
            let _guard = scopeguard_local(&url, &schema);
            $body(store);
        }};
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
    fn scopeguard_local<'a>(url: &'a str, schema: &'a str) -> DropGuard<'a> {
        DropGuard { url, schema }
    }

    #[test]
    fn put_then_get_roundtrips() {
        pg_test!(|store: PgStore| {
            let session = SessionId(1);
            let block = make_block(b"hello", BlockKind::UserMessage, 1_000, 42);
            let id = store.put(session, block.clone()).unwrap();
            let fetched = store.get(id).unwrap().unwrap();
            assert_eq!(fetched.bytes, block.bytes);
            assert_eq!(fetched.kind, block.kind);
            assert_eq!(fetched.hash, block.hash);
            assert_eq!(fetched.created_at, block.created_at);
        });
    }

    #[test]
    fn duplicate_content_returns_existing_id() {
        pg_test!(|store: PgStore| {
            let session = SessionId(1);
            let first = make_block(b"hello", BlockKind::UserMessage, 1_000, 42);
            let id1 = store.put(session, first).unwrap();
            let dup = make_block(b"hello", BlockKind::UserMessage, 2_000, 99);
            let id2 = store.put(session, dup).unwrap();
            assert_eq!(id1, id2);
        });
    }

    #[test]
    fn list_session_returns_chronologically() {
        pg_test!(|store: PgStore| {
            let session = SessionId(7);
            let a = make_block(b"a", BlockKind::UserMessage, 1, 1);
            let b = make_block(b"b", BlockKind::UserMessage, 2, 2);
            let c = make_block(b"c", BlockKind::UserMessage, 3, 3);
            let id_a = store.put(session, a).unwrap();
            let id_b = store.put(session, b).unwrap();
            let id_c = store.put(session, c).unwrap();
            assert_eq!(store.list_session(session).unwrap(), vec![id_a, id_b, id_c]);
        });
    }

    #[test]
    fn list_sessions_returns_unique_sorted() {
        pg_test!(|store: PgStore| {
            let s_a = SessionId(7);
            let s_b = SessionId(3);
            let s_c = SessionId(11);
            store.put(s_a, make_block(b"x", BlockKind::Fact, 1, 1)).unwrap();
            store.put(s_a, make_block(b"y", BlockKind::Fact, 2, 2)).unwrap();
            store.put(s_b, make_block(b"z", BlockKind::Fact, 3, 3)).unwrap();
            store.put(s_c, make_block(b"w", BlockKind::Fact, 4, 4)).unwrap();
            assert_eq!(
                store.list_sessions().unwrap(),
                vec![SessionId(3), SessionId(7), SessionId(11)],
            );
        });
    }

    #[test]
    fn delete_removes_block_from_all_indexes() {
        pg_test!(|store: PgStore| {
            let session = SessionId(1);
            let block = make_block(b"to-be-deleted", BlockKind::Fact, 1, 1);
            let hash = block.hash;
            let id = store.put(session, block).unwrap();
            assert!(store.get(id).unwrap().is_some());
            assert_eq!(store.lookup_hash(hash).unwrap(), Some(id));
            assert!(store.delete(id).unwrap());
            assert!(store.get(id).unwrap().is_none());
            assert_eq!(store.lookup_hash(hash).unwrap(), None);
            assert!(store.list_session(session).unwrap().is_empty());
        });
    }

    #[test]
    fn delete_returns_false_for_unknown() {
        pg_test!(|store: PgStore| {
            let bogus = BlockId::from_parts(99, 99);
            assert!(!store.delete(bogus).unwrap());
        });
    }

    #[test]
    fn delete_scrubs_block_from_every_session_referencing_it() {
        pg_test!(|store: PgStore| {
            let s1 = SessionId(1);
            let s2 = SessionId(2);
            let block = make_block(b"shared", BlockKind::Fact, 1, 1);
            let id_a = store.put(s1, block.clone()).unwrap();
            let id_b = store.put(s2, block).unwrap();
            assert_eq!(id_a, id_b);
            store.delete(id_a).unwrap();
            assert!(store.list_session(s1).unwrap().is_empty());
            assert!(store.list_session(s2).unwrap().is_empty());
        });
    }

    #[test]
    fn purge_session_removes_blocks_unique_to_that_session() {
        pg_test!(|store: PgStore| {
            let session = SessionId(7);
            store.put(session, make_block(b"a", BlockKind::Fact, 1, 1)).unwrap();
            store.put(session, make_block(b"b", BlockKind::Fact, 2, 2)).unwrap();
            store.put(session, make_block(b"c", BlockKind::Fact, 3, 3)).unwrap();
            assert_eq!(store.purge_session(session).unwrap(), 3);
            assert!(store.list_session(session).unwrap().is_empty());
            assert!(store.list_sessions().unwrap().is_empty());
        });
    }

    #[test]
    fn purge_session_keeps_blocks_referenced_by_other_sessions() {
        pg_test!(|store: PgStore| {
            let s1 = SessionId(1);
            let s2 = SessionId(2);
            let id = store
                .put(s1, make_block(b"shared", BlockKind::Fact, 1, 1))
                .unwrap();
            let id_b = store
                .put(s2, make_block(b"shared", BlockKind::Fact, 2, 2))
                .unwrap();
            assert_eq!(id, id_b);
            let _solo = store
                .put(s1, make_block(b"solo", BlockKind::Fact, 3, 3))
                .unwrap();
            store.purge_session(s1).unwrap();
            assert!(store.list_session(s1).unwrap().is_empty());
            assert_eq!(store.list_session(s2).unwrap(), vec![id]);
            assert!(store.get(id).unwrap().is_some());
        });
    }

    #[test]
    fn lookup_hash_finds_inserted_block() {
        pg_test!(|store: PgStore| {
            let session = SessionId(1);
            let block = make_block(b"findme", BlockKind::Fact, 1_000, 42);
            let id = store.put(session, block.clone()).unwrap();
            assert_eq!(store.lookup_hash(block.hash).unwrap(), Some(id));
        });
    }

    #[test]
    fn lookup_hash_unknown_is_none() {
        pg_test!(|store: PgStore| {
            let unknown = ContentHash::of(b"never inserted");
            assert!(store.lookup_hash(unknown).unwrap().is_none());
        });
    }

    #[test]
    fn put_edge_then_edges_from_and_to_roundtrip() {
        pg_test!(|store: PgStore| {
            let s = SessionId(1);
            let a = store.put(s, make_block(b"A", BlockKind::Fact, 1, 1)).unwrap();
            let b = store.put(s, make_block(b"B", BlockKind::Fact, 2, 2)).unwrap();
            let c = store.put(s, make_block(b"C", BlockKind::Fact, 3, 3)).unwrap();
            store
                .put_edge(Edge { from: a, to: b, kind: EdgeKind::Supports })
                .unwrap();
            store
                .put_edge(Edge { from: a, to: c, kind: EdgeKind::DerivedFrom })
                .unwrap();
            let outgoing = store.edges_from(a).unwrap();
            assert_eq!(outgoing.len(), 2);
            assert!(outgoing.iter().any(|e| e.to == b && e.kind == EdgeKind::Supports));
            assert!(outgoing.iter().any(|e| e.to == c && e.kind == EdgeKind::DerivedFrom));
            assert_eq!(
                store.edges_to(b).unwrap(),
                vec![Edge { from: a, to: b, kind: EdgeKind::Supports }],
            );
            assert!(store.edges_to(a).unwrap().is_empty());
        });
    }

    #[test]
    fn put_edge_is_idempotent() {
        pg_test!(|store: PgStore| {
            let s = SessionId(1);
            let a = store.put(s, make_block(b"A", BlockKind::Fact, 1, 1)).unwrap();
            let b = store.put(s, make_block(b"B", BlockKind::Fact, 2, 2)).unwrap();
            let edge = Edge { from: a, to: b, kind: EdgeKind::Parent };
            store.put_edge(edge).unwrap();
            store.put_edge(edge).unwrap();
            assert_eq!(store.edges_from(a).unwrap().len(), 1);
            assert_eq!(store.edges_to(b).unwrap().len(), 1);
        });
    }

    #[test]
    fn delete_block_purges_edges_in_both_directions() {
        pg_test!(|store: PgStore| {
            let s = SessionId(1);
            let a = store.put(s, make_block(b"A", BlockKind::Fact, 1, 1)).unwrap();
            let b = store.put(s, make_block(b"B", BlockKind::Fact, 2, 2)).unwrap();
            store
                .put_edge(Edge { from: a, to: b, kind: EdgeKind::Supports })
                .unwrap();
            store
                .put_edge(Edge { from: b, to: a, kind: EdgeKind::Contradicts })
                .unwrap();
            assert!(store.delete(a).unwrap());
            assert!(store.edges_from(a).unwrap().is_empty());
            assert!(store.edges_to(a).unwrap().is_empty());
            assert!(store.edges_from(b).unwrap().is_empty());
            assert!(store.edges_to(b).unwrap().is_empty());
        });
    }

    #[test]
    fn provenance_with_parents_and_labels_roundtrips() {
        pg_test!(|store: PgStore| {
            let s = SessionId(42);
            let parent_id = BlockId::from_parts(123, 456);
            let block = ContextBlock {
                id: BlockId::from_parts(999, 0),
                kind: BlockKind::Summary,
                bytes: b"with-provenance".to_vec(),
                token_counts: TokenCounts::new(),
                priority: 0.5,
                created_at: Timestamp(999),
                updated_at: Timestamp(999),
                provenance: Provenance {
                    source: Some("doc:42".into()),
                    parents: vec![parent_id],
                    labels: vec!["workspace:abc".into(), "kind:summary".into()],
                },
                hash: ContentHash::of(b"with-provenance"),
            };
            let id = store.put(s, block.clone()).unwrap();
            let fetched = store.get(id).unwrap().unwrap();
            assert_eq!(fetched.provenance.source.as_deref(), Some("doc:42"));
            assert_eq!(fetched.provenance.parents, vec![parent_id]);
            assert_eq!(
                fetched.provenance.labels,
                vec!["workspace:abc".to_string(), "kind:summary".to_string()],
            );
            assert!((fetched.priority - 0.5).abs() < f32::EPSILON);
        });
    }
}
