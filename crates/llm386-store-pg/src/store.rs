//! `PgStore` — `BlockStore` implementation backed by PostgreSQL.

use llm386_core::{
    BlockId, BlockKind, BlockStore, ContentHash, ContextBlock, Edge, EdgeKind, Provenance,
    SessionId, StoreError, Timestamp, TokenCount, TokenCounts,
};
use postgres::NoTls;
use postgres::types::ToSql;
use r2d2::{CustomizeConnection, Pool, PooledConnection};
use r2d2_postgres::PostgresConnectionManager;
use thiserror::Error;
use tracing::{debug, instrument};

/// Schema version written to the `llm386_meta` table on first connect.
///
/// Bump this whenever the on-disk layout changes incompatibly. Older
/// stores will refuse to open with the new code.
const CURRENT_SCHEMA: i32 = 1;

const DEFAULT_POOL_SIZE: u32 = 8;

type PgPool = Pool<PostgresConnectionManager<NoTls>>;
type PgConn = PooledConnection<PostgresConnectionManager<NoTls>>;

/// Configuration for opening a [`PgStore`].
#[derive(Clone, Debug)]
pub struct PgStoreConfig {
    /// Maximum connections in the r2d2 pool.
    pub max_pool_size: u32,
    /// Optional schema to scope every connection to. Useful for test
    /// isolation. The schema is created if missing. The name must be a
    /// valid Postgres identifier: ASCII letters/digits/underscores only,
    /// and must start with a letter or underscore.
    pub schema: Option<String>,
}

impl Default for PgStoreConfig {
    fn default() -> Self {
        Self {
            max_pool_size: DEFAULT_POOL_SIZE,
            schema: None,
        }
    }
}

/// PostgreSQL-backed implementation of the `BlockStore` trait.
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

        let manager = PostgresConnectionManager::new(
            url.parse().map_err(|e: postgres::Error| {
                StoreOpenError::Connect(format!("invalid Postgres URL: {e}"))
            })?,
            NoTls,
        );

        let mut builder = Pool::builder().max_size(config.max_pool_size);
        if let Some(schema) = config.schema.clone() {
            builder = builder.connection_customizer(Box::new(SchemaCustomizer { schema }));
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

        // ON CONFLICT (hash) DO NOTHING -- dedup. RETURNING id is empty
        // when a row with this hash already exists; in that case fall
        // back to a hash lookup.
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
            // Dedup hit: look up the existing id by hash.
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

        // Record session membership unconditionally — same block may
        // appear in many sessions.
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

        // Step 1: capture every block id this session references.
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

        // Step 2: drop this session's references.
        tx.execute(
            "DELETE FROM llm386_session_blocks WHERE session_id = $1",
            &[&session_slice],
        )
        .map_err(|e| StoreError::Backend(format!("delete session refs: {e}")))?;

        // Step 3: for each block, if no other session refs, drop the
        // block + any edges referencing it.
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

/// Errors that can occur while opening a [`PgStore`].
#[derive(Debug, Error)]
pub enum StoreOpenError {
    #[error("could not connect: {0}")]
    Connect(String),
    #[error("Postgres error: {0}")]
    Postgres(#[from] postgres::Error),
    #[error("on-disk schema version {found} does not match expected {expected}")]
    SchemaMismatch { expected: i32, found: i32 },
    #[error("meta row is corrupt: {0}")]
    CorruptMeta(String),
    #[error("invalid schema name `{0}`: must be ASCII [a-zA-Z_][a-zA-Z0-9_]*")]
    InvalidSchemaName(String),
}

#[derive(Debug)]
struct SchemaCustomizer {
    schema: String,
}

impl CustomizeConnection<postgres::Client, postgres::Error> for SchemaCustomizer {
    fn on_acquire(&self, conn: &mut postgres::Client) -> Result<(), postgres::Error> {
        // schema is validated by is_valid_ident() before we get here,
        // so this interpolation is safe.
        conn.batch_execute(&format!(
            "CREATE SCHEMA IF NOT EXISTS \"{}\"; SET search_path TO \"{}\";",
            self.schema, self.schema,
        ))
    }
}

fn is_valid_ident(s: &str) -> bool {
    if s.is_empty() {
        return false;
    }
    let mut chars = s.chars();
    let first = chars.next().unwrap();
    if !(first.is_ascii_alphabetic() || first == '_') {
        return false;
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn run_migrations(conn: &mut PgConn) -> Result<(), StoreOpenError> {
    conn.batch_execute(
        "
        CREATE TABLE IF NOT EXISTS llm386_blocks (
            id              BYTEA PRIMARY KEY,
            kind            TEXT NOT NULL,
            bytes           BYTEA NOT NULL,
            token_counts    JSONB NOT NULL DEFAULT '{}'::jsonb,
            priority        REAL NOT NULL DEFAULT 0.0,
            created_at      BIGINT NOT NULL,
            updated_at      BIGINT NOT NULL,
            provenance      JSONB NOT NULL DEFAULT '{}'::jsonb,
            hash            BYTEA NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS llm386_blocks_hash_idx
            ON llm386_blocks (hash);

        CREATE TABLE IF NOT EXISTS llm386_session_blocks (
            session_id  BYTEA NOT NULL,
            block_id    BYTEA NOT NULL,
            PRIMARY KEY (session_id, block_id)
        );
        CREATE INDEX IF NOT EXISTS llm386_session_blocks_sid
            ON llm386_session_blocks (session_id, block_id);

        CREATE TABLE IF NOT EXISTS llm386_edges (
            from_id     BYTEA NOT NULL,
            to_id       BYTEA NOT NULL,
            kind        TEXT NOT NULL,
            PRIMARY KEY (from_id, kind, to_id)
        );
        CREATE INDEX IF NOT EXISTS llm386_edges_to
            ON llm386_edges (to_id, kind);

        CREATE TABLE IF NOT EXISTS llm386_meta (
            key   TEXT PRIMARY KEY,
            value BYTEA NOT NULL
        );
        ",
    )?;

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

fn id_bytes(id: BlockId) -> [u8; 16] {
    id.0.to_be_bytes()
}

fn session_bytes(s: SessionId) -> [u8; 16] {
    s.0.to_be_bytes()
}

fn decode_block_id(bytes: &[u8]) -> Result<BlockId, StoreError> {
    let arr: [u8; 16] = bytes
        .try_into()
        .map_err(|_| StoreError::Backend(format!("BlockId width {}", bytes.len())))?;
    Ok(BlockId(u128::from_be_bytes(arr)))
}

fn ts_to_i64(ts: Timestamp) -> i64 {
    // Timestamp is u64 ms-since-epoch; postgres BIGINT is signed.
    // Saturating to i64::MAX preserves ordering for the ~292 million
    // year horizon when this matters.
    i64::try_from(ts.0).unwrap_or(i64::MAX)
}

fn ts_from_i64(v: i64) -> Timestamp {
    Timestamp(u64::try_from(v).unwrap_or(0))
}

fn block_kind_to_str(k: BlockKind) -> &'static str {
    match k {
        BlockKind::System => "System",
        BlockKind::UserMessage => "UserMessage",
        BlockKind::AssistantMessage => "AssistantMessage",
        BlockKind::ToolResult => "ToolResult",
        BlockKind::Summary => "Summary",
        BlockKind::Fact => "Fact",
        BlockKind::DocumentChunk => "DocumentChunk",
        BlockKind::Plan => "Plan",
        BlockKind::State => "State",
        BlockKind::Trace => "Trace",
    }
}

fn block_kind_from_str(s: &str) -> Result<BlockKind, StoreError> {
    Ok(match s {
        "System" => BlockKind::System,
        "UserMessage" => BlockKind::UserMessage,
        "AssistantMessage" => BlockKind::AssistantMessage,
        "ToolResult" => BlockKind::ToolResult,
        "Summary" => BlockKind::Summary,
        "Fact" => BlockKind::Fact,
        "DocumentChunk" => BlockKind::DocumentChunk,
        "Plan" => BlockKind::Plan,
        "State" => BlockKind::State,
        "Trace" => BlockKind::Trace,
        other => return Err(StoreError::Backend(format!("unknown BlockKind `{other}`"))),
    })
}

fn edge_kind_to_str(k: EdgeKind) -> &'static str {
    match k {
        EdgeKind::Parent => "Parent",
        EdgeKind::DerivedFrom => "DerivedFrom",
        EdgeKind::Supports => "Supports",
        EdgeKind::Contradicts => "Contradicts",
        EdgeKind::ToolInvocation => "ToolInvocation",
    }
}

fn edge_kind_from_str(s: &str) -> Result<EdgeKind, StoreError> {
    Ok(match s {
        "Parent" => EdgeKind::Parent,
        "DerivedFrom" => EdgeKind::DerivedFrom,
        "Supports" => EdgeKind::Supports,
        "Contradicts" => EdgeKind::Contradicts,
        "ToolInvocation" => EdgeKind::ToolInvocation,
        other => return Err(StoreError::Backend(format!("unknown EdgeKind `{other}`"))),
    })
}

fn provenance_to_json(p: &Provenance) -> serde_json::Value {
    let parents: Vec<String> = p.parents.iter().map(BlockId::to_string).collect();
    serde_json::json!({
        "source": p.source,
        "parents": parents,
        "labels": p.labels,
    })
}

fn provenance_from_json(v: &serde_json::Value) -> Result<Provenance, StoreError> {
    let source = v
        .get("source")
        .and_then(serde_json::Value::as_str)
        .map(String::from);
    let parents = match v.get("parents").and_then(serde_json::Value::as_array) {
        None => Vec::new(),
        Some(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let s = item.as_str().ok_or_else(|| {
                    StoreError::Backend("provenance.parents: non-string entry".into())
                })?;
                out.push(s.parse::<BlockId>().map_err(|e| {
                    StoreError::Backend(format!("provenance.parents: parse: {e}"))
                })?);
            }
            out
        }
    };
    let labels = match v.get("labels").and_then(serde_json::Value::as_array) {
        None => Vec::new(),
        Some(arr) => arr
            .iter()
            .filter_map(serde_json::Value::as_str)
            .map(String::from)
            .collect(),
    };
    Ok(Provenance {
        source,
        parents,
        labels,
    })
}

fn token_counts_to_json(tc: &TokenCounts) -> serde_json::Value {
    // Serialize as a flat {tokenizer_id: count} object. TokenizerId is
    // a transparent string newtype so this round-trips cleanly without
    // the wrapping struct that serde derive would produce.
    let map: serde_json::Map<String, serde_json::Value> = tc
        .iter()
        .map(|(id, count)| (id.as_str().to_string(), serde_json::json!(count.0)))
        .collect();
    serde_json::Value::Object(map)
}

fn token_counts_from_json(v: &serde_json::Value) -> Result<TokenCounts, StoreError> {
    let mut out = TokenCounts::new();
    let Some(obj) = v.as_object() else {
        return Ok(out);
    };
    for (k, val) in obj {
        let count = val
            .as_u64()
            .ok_or_else(|| StoreError::Backend(format!("token_counts.{k}: non-u64")))?;
        let n = u32::try_from(count).map_err(|_| {
            StoreError::Backend(format!("token_counts.{k}: exceeds u32"))
        })?;
        out.insert(k.as_str().into(), TokenCount(n));
    }
    Ok(out)
}

fn row_to_block(row: &postgres::Row) -> Result<ContextBlock, StoreError> {
    let id_b: &[u8] = row.get(0);
    let kind_s: &str = row.get(1);
    let bytes: Vec<u8> = row.get(2);
    let tc_json: serde_json::Value = row.get(3);
    let priority: f32 = row.get(4);
    let created_at: i64 = row.get(5);
    let updated_at: i64 = row.get(6);
    let prov_json: serde_json::Value = row.get(7);
    let hash_b: &[u8] = row.get(8);

    let id = decode_block_id(id_b)?;
    let kind = block_kind_from_str(kind_s)?;
    let token_counts = token_counts_from_json(&tc_json)?;
    let provenance = provenance_from_json(&prov_json)?;
    let hash_arr: [u8; 32] = hash_b
        .try_into()
        .map_err(|_| StoreError::Backend(format!("hash width {}", hash_b.len())))?;
    let hash = ContentHash(hash_arr);

    Ok(ContextBlock {
        id,
        kind,
        bytes,
        token_counts,
        priority,
        created_at: ts_from_i64(created_at),
        updated_at: ts_from_i64(updated_at),
        provenance,
        hash,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm386_core::{BlockKind, Provenance, Timestamp, TokenCounts};
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

    /// Open a PgStore against a fresh per-test schema, or `None` if
    /// `TEST_DATABASE_URL` isn't set. Tests skip cleanly in that case.
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

    /// Drop the per-test schema. Called from each test's tail; skipped
    /// in the no-PG-available case.
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

    // Minimal local scopeguard equivalent so we don't pull in the crate.
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
