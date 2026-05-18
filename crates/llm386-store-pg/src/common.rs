//! Shared helpers between the sync (`PgStore`) and async (`AsyncPgStore`)
//! backends. The sync `postgres` crate re-exports its core types from
//! `tokio_postgres`, so `Row` / `Error` / `ToSql` are identical types
//! across the two driver styles and these helpers compile against both.

use llm386_core::{
    BlockId, BlockKind, ContentHash, ContextBlock, EdgeKind, Provenance, SessionId, StoreError,
    Timestamp, TokenCount, TokenCounts,
};
use thiserror::Error;
use tokio_postgres::Row;

/// Schema version written to the `llm386_meta` table on first connect.
///
/// Bump this whenever the on-disk layout changes incompatibly. Older
/// stores will refuse to open with the new code.
pub(crate) const CURRENT_SCHEMA: i32 = 1;

pub(crate) const DEFAULT_POOL_SIZE: u32 = 8;

/// How to secure the Postgres connection. Default is [`TlsMode::Disable`]
/// for back-compat; production deployments **should** opt in to
/// [`TlsMode::Require`] (or stronger) unless the Postgres server runs
/// on localhost or over a trusted private socket.
///
/// `Require` / `RequireCustomCa` need the `tls-native-tls` feature.
/// Without it, opening with either mode returns
/// [`StoreOpenError::TlsUnsupported`] — never silently downgrades.
#[derive(Clone, Debug, Default)]
pub enum TlsMode {
    /// Plaintext connection. Only safe on localhost or a trusted
    /// private network. This is the default for back-compat; explicit
    /// opt-in for clarity.
    #[default]
    Disable,
    /// Require TLS; verify the server certificate against the system
    /// root CA store. Works out of the box with most managed Postgres
    /// providers (RDS, Cloud SQL, Supabase, Neon).
    Require,
    /// Require TLS; verify the server certificate against a custom
    /// PEM-encoded CA bundle. Use this when the server presents a
    /// cert chain rooted at a private CA not in the system store.
    RequireCustomCa {
        /// Path to a PEM file containing one or more CA certificates.
        ca_path: std::path::PathBuf,
    },
}

/// Configuration shared by [`crate::PgStore`] and
/// (with the `async` feature) [`crate::AsyncPgStore`].
#[derive(Clone, Debug)]
pub struct PgStoreConfig {
    /// Maximum connections in the pool.
    pub max_pool_size: u32,
    /// Optional schema to scope every connection to. Useful for test
    /// isolation. The schema is created if missing. The name must be a
    /// valid Postgres identifier: ASCII letters/digits/underscores only,
    /// and must start with a letter or underscore.
    pub schema: Option<String>,
    /// TLS posture. Defaults to [`TlsMode::Disable`] for back-compat;
    /// **set this explicitly for any non-localhost deployment**.
    pub tls: TlsMode,
}

impl Default for PgStoreConfig {
    fn default() -> Self {
        Self {
            max_pool_size: DEFAULT_POOL_SIZE,
            schema: None,
            tls: TlsMode::default(),
        }
    }
}

/// Errors that can occur while opening a Postgres-backed store.
#[derive(Debug, Error)]
pub enum StoreOpenError {
    #[error("could not connect: {0}")]
    Connect(String),
    #[error("Postgres error: {0}")]
    Postgres(#[from] tokio_postgres::Error),
    #[error("on-disk schema version {found} does not match expected {expected}")]
    SchemaMismatch { expected: i32, found: i32 },
    #[error("meta row is corrupt: {0}")]
    CorruptMeta(String),
    #[error("invalid schema name `{0}`: must be ASCII [a-zA-Z_][a-zA-Z0-9_]*")]
    InvalidSchemaName(String),
    #[error(
        "TLS mode {0:?} requires the `tls-native-tls` feature; \
         rebuild llm386-store-pg (or the consuming crate) with \
         --features tls-native-tls, or set TlsMode::Disable"
    )]
    TlsUnsupported(TlsMode),
    #[error("TLS configuration error: {0}")]
    TlsConfig(String),
}

/// Full schema bootstrap. Idempotent — safe to issue on every open.
pub(crate) const MIGRATION_SQL: &str = "
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
";

pub(crate) fn is_valid_ident(s: &str) -> bool {
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

pub(crate) fn id_bytes(id: BlockId) -> [u8; 16] {
    id.0.to_be_bytes()
}

pub(crate) fn session_bytes(s: SessionId) -> [u8; 16] {
    s.0.to_be_bytes()
}

pub(crate) fn decode_block_id(bytes: &[u8]) -> Result<BlockId, StoreError> {
    let arr: [u8; 16] = bytes
        .try_into()
        .map_err(|_| StoreError::Backend(format!("BlockId width {}", bytes.len())))?;
    Ok(BlockId(u128::from_be_bytes(arr)))
}

pub(crate) fn ts_to_i64(ts: Timestamp) -> i64 {
    i64::try_from(ts.0).unwrap_or(i64::MAX)
}

pub(crate) fn ts_from_i64(v: i64) -> Timestamp {
    Timestamp(u64::try_from(v).unwrap_or(0))
}

pub(crate) fn block_kind_to_str(k: BlockKind) -> &'static str {
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

pub(crate) fn block_kind_from_str(s: &str) -> Result<BlockKind, StoreError> {
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

pub(crate) fn edge_kind_to_str(k: EdgeKind) -> &'static str {
    match k {
        EdgeKind::Parent => "Parent",
        EdgeKind::DerivedFrom => "DerivedFrom",
        EdgeKind::Supports => "Supports",
        EdgeKind::Contradicts => "Contradicts",
        EdgeKind::ToolInvocation => "ToolInvocation",
    }
}

pub(crate) fn edge_kind_from_str(s: &str) -> Result<EdgeKind, StoreError> {
    Ok(match s {
        "Parent" => EdgeKind::Parent,
        "DerivedFrom" => EdgeKind::DerivedFrom,
        "Supports" => EdgeKind::Supports,
        "Contradicts" => EdgeKind::Contradicts,
        "ToolInvocation" => EdgeKind::ToolInvocation,
        other => return Err(StoreError::Backend(format!("unknown EdgeKind `{other}`"))),
    })
}

pub(crate) fn provenance_to_json(p: &Provenance) -> serde_json::Value {
    let parents: Vec<String> = p.parents.iter().map(BlockId::to_string).collect();
    serde_json::json!({
        "source": p.source,
        "parents": parents,
        "labels": p.labels,
    })
}

pub(crate) fn provenance_from_json(v: &serde_json::Value) -> Result<Provenance, StoreError> {
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
                out.push(
                    s.parse::<BlockId>()
                        .map_err(|e| StoreError::Backend(format!("provenance.parents: parse: {e}")))?,
                );
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

pub(crate) fn token_counts_to_json(tc: &TokenCounts) -> serde_json::Value {
    let map: serde_json::Map<String, serde_json::Value> = tc
        .iter()
        .map(|(id, count)| (id.as_str().to_string(), serde_json::json!(count.0)))
        .collect();
    serde_json::Value::Object(map)
}

pub(crate) fn token_counts_from_json(v: &serde_json::Value) -> Result<TokenCounts, StoreError> {
    let mut out = TokenCounts::new();
    let Some(obj) = v.as_object() else {
        return Ok(out);
    };
    for (k, val) in obj {
        let count = val
            .as_u64()
            .ok_or_else(|| StoreError::Backend(format!("token_counts.{k}: non-u64")))?;
        let n = u32::try_from(count)
            .map_err(|_| StoreError::Backend(format!("token_counts.{k}: exceeds u32")))?;
        out.insert(k.as_str().into(), TokenCount(n));
    }
    Ok(out)
}

pub(crate) fn row_to_block(row: &Row) -> Result<ContextBlock, StoreError> {
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
