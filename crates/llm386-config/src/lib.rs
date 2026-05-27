//! Shared TOML config schema + backend-selection helpers used by both
//! the `llm386` CLI and the Python (`llm386-py`) wrapper.
//!
//! All errors are returned as `String` so each caller can wrap them in
//! whatever error type they prefer (`anyhow::Error` for the CLI,
//! `PyErr` for the Python bindings) without this crate pulling either
//! one in.
//!
//! Top-level entry points:
//!
//! - [`parse`] — read a TOML file into a [`ConfigFile`].
//! - [`apply`] — fold a parsed [`ConfigFile`] into a model + tokenizer
//!   registry, returning the leftovers ([`Applied`]) that need to be
//!   re-bound per-call (retrievers) or applied at packer-construction
//!   time (section budgets, packer options).
//! - [`open_backend`] — resolve a [`StoreEntry`] + CLI/kwarg overrides
//!   into an opened [`StoreBackend`].
//! - [`build_retrievers`] — materialize a `Vec<Arc<dyn Retriever>>`
//!   from parsed `[[retriever]]` entries, bound to any concrete
//!   `BlockStore`.
//! - [`dispatch!`] — pattern-match a `StoreBackend` and bind the inner
//!   `Arc<S>` to a name, so call sites can stay backend-agnostic.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use llm386_core::{BlockStore, ModelProfile, ModelRegistry, Retriever, SectionKind, TokenizerId};
use llm386_packer::{CacheOptions, PackerOptions};
use llm386_pager::{
    Bm25Retriever, LexicalRetriever, RecencyRetriever, SectionBudgetTable, SessionRetriever,
};
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_store_pg::{PgStore, PgStoreConfig, TlsMode};
use llm386_tokenizer::{HfTokenizer, TokenizerRegistry};
use serde::Deserialize;

// ---------- TOML schema ----------

/// Top-level shape of the LLM386 config TOML file. Every field is
/// optional; an empty file is valid and produces an empty `Applied`.
#[derive(Debug, Default, Deserialize)]
pub struct ConfigFile {
    #[serde(default)]
    pub profile: Vec<ModelProfile>,
    #[serde(default)]
    pub hf_tokenizer: Vec<HfTokenizerEntry>,
    #[serde(default)]
    pub retriever: Vec<RetrieverEntry>,
    #[serde(default)]
    pub section_budgets: Option<SectionBudgetEntry>,
    #[serde(default)]
    pub packer: Option<PackerEntry>,
    #[serde(default)]
    pub cache: Option<CacheEntry>,
    #[serde(default)]
    pub store: Option<StoreEntry>,
}

/// `[store]` selects the persistent block-store backend. Optional —
/// when absent, callers fall back to their own positional/kwarg
/// arguments.
#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "backend", rename_all = "kebab-case")]
pub enum StoreEntry {
    Lmdb {
        #[serde(default)]
        path: Option<PathBuf>,
    },
    Pg {
        #[serde(default)]
        url: Option<String>,
        #[serde(default)]
        pool_size: Option<u32>,
        #[serde(default)]
        schema: Option<String>,
        /// `"disable"` (default), `"require"`, or `"require-custom-ca"`.
        /// The last requires `tls_ca_path` to be set. Modes other
        /// than `disable` need the `tls-native-tls` cargo feature on
        /// `llm386-store-pg` — opening with TLS in a build without
        /// the feature returns `StoreOpenError::TlsUnsupported`,
        /// never silently downgrades.
        #[serde(default)]
        tls: Option<String>,
        /// PEM-encoded CA bundle, required when `tls =
        /// "require-custom-ca"`.
        #[serde(default)]
        tls_ca_path: Option<PathBuf>,
    },
}

/// Translate the TOML `tls = "..."` string + optional `tls_ca_path`
/// into a [`TlsMode`]. Unknown strings raise a clear error rather than
/// silently defaulting to `Disable`.
fn parse_tls_mode(
    tls: Option<&str>,
    tls_ca_path: Option<&std::path::Path>,
) -> Result<TlsMode, String> {
    match tls.map(str::to_ascii_lowercase).as_deref() {
        None | Some("disable") => Ok(TlsMode::Disable),
        Some("require") => Ok(TlsMode::Require),
        Some("require-noverify") => Ok(TlsMode::RequireNoverify),
        Some("require-custom-ca") => {
            let ca_path = tls_ca_path.ok_or_else(|| {
                "[store] tls = \"require-custom-ca\" requires `tls_ca_path` \
                 to be set (PEM bundle)"
                    .to_string()
            })?;
            Ok(TlsMode::RequireCustomCa {
                ca_path: ca_path.to_path_buf(),
            })
        }
        Some(other) => Err(format!(
            "[store] tls = \"{other}\" — expected one of: disable, require, require-noverify, require-custom-ca",
        )),
    }
}

#[derive(Debug, Deserialize)]
pub struct HfTokenizerEntry {
    pub name: String,
    pub path: PathBuf,
}

/// `[[retriever]]` — one entry per retriever to plug into the pager.
/// `kind` selects the implementation; the remaining fields are
/// per-kind knobs.
#[derive(Clone, Debug, Deserialize)]
pub struct RetrieverEntry {
    pub kind: String,
    #[serde(default)]
    pub half_life_secs: Option<f32>,
    #[serde(default)]
    pub min_word_len: Option<usize>,
    #[serde(default)]
    pub k1: Option<f32>,
    #[serde(default)]
    pub b: Option<f32>,
    #[serde(default)]
    pub score: Option<f32>,
}

#[derive(Debug, Default, Deserialize)]
pub struct PackerEntry {
    #[serde(default)]
    pub include_timestamps: bool,
}

impl PackerEntry {
    pub fn build(self) -> PackerOptions {
        PackerOptions {
            include_timestamps: self.include_timestamps,
            now_ms: None,
            cache: CacheOptions::default(),
        }
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct CacheEntry {
    #[serde(default)]
    pub stable_sections: Option<Vec<String>>,
}

impl CacheEntry {
    pub fn build(self) -> Result<CacheOptions, String> {
        let mut out = CacheOptions::default();
        if let Some(names) = self.stable_sections {
            let mut sections = Vec::with_capacity(names.len());
            for name in names {
                let s = match name.to_ascii_lowercase().as_str() {
                    "system" => SectionKind::System,
                    "state" => SectionKind::State,
                    "plan" => SectionKind::Plan,
                    "retrieved" => SectionKind::Retrieved,
                    "background" => SectionKind::Background,
                    other => {
                        return Err(format!(
                            "[cache].stable_sections: unsupported section `{other}` — valid: system, state, plan, retrieved, background",
                        ));
                    }
                };
                sections.push(s);
            }
            out.stable_sections = sections;
        }
        Ok(out)
    }
}

#[derive(Debug, Default, Deserialize)]
pub struct SectionBudgetEntry {
    #[serde(default)]
    pub state: Option<f32>,
    #[serde(default)]
    pub plan: Option<f32>,
    #[serde(default)]
    pub recent: Option<f32>,
    #[serde(default)]
    pub retrieved: Option<f32>,
    #[serde(default)]
    pub tools: Option<f32>,
    #[serde(default)]
    pub background: Option<f32>,
    #[serde(default)]
    pub slack: Option<f32>,
}

impl SectionBudgetEntry {
    pub fn build(self) -> SectionBudgetTable {
        let mut table = SectionBudgetTable::empty();
        if let Some(v) = self.state {
            table.set(SectionKind::State, v);
        }
        if let Some(v) = self.plan {
            table.set(SectionKind::Plan, v);
        }
        if let Some(v) = self.recent {
            table.set(SectionKind::Recent, v);
        }
        if let Some(v) = self.retrieved {
            table.set(SectionKind::Retrieved, v);
        }
        if let Some(v) = self.tools {
            table.set(SectionKind::Tools, v);
        }
        if let Some(v) = self.background {
            table.set(SectionKind::Background, v);
        }
        if let Some(v) = self.slack {
            table.set(SectionKind::Slack, v);
        }
        table
    }
}

// ---------- Parsing + applying ----------

/// Read a TOML config file from disk.
pub fn parse(path: &Path) -> Result<ConfigFile, String> {
    let s = std::fs::read_to_string(path)
        .map_err(|e| format!("reading config file at {}: {e}", path.display()))?;
    toml::from_str(&s).map_err(|e| format!("parsing config file at {}: {e}", path.display()))
}

/// What's left after [`apply`] has folded the [[profile]] and
/// [[hf_tokenizer]] entries into the registries. Retrievers are
/// returned as TOML entries (not built) because they bind to a
/// specific store, which the caller materializes at command time via
/// [`build_retrievers`].
#[derive(Debug)]
pub struct Applied {
    pub retrievers: Vec<RetrieverEntry>,
    pub section_budgets: Option<SectionBudgetTable>,
    pub packer_options: Option<PackerOptions>,
    pub store: Option<StoreEntry>,
}

/// Fold `[[profile]]` and `[[hf_tokenizer]]` entries into the given
/// registries (mutating in place). Returns the rest of the parsed
/// config as [`Applied`].
pub fn apply(
    parsed: ConfigFile,
    models: &mut ModelRegistry,
    tokenizers: &mut TokenizerRegistry,
) -> Result<Applied, String> {
    for profile in parsed.profile {
        models.register(profile);
    }
    for entry in parsed.hf_tokenizer {
        let tok = HfTokenizer::from_file(&entry.path, TokenizerId::new(&entry.name)).map_err(
            |e| {
                format!(
                    "loading huggingface tokenizer `{}` from {}: {e}",
                    entry.name,
                    entry.path.display(),
                )
            },
        )?;
        tokenizers.register(Arc::new(tok));
    }
    let mut packer_options = parsed.packer.map(PackerEntry::build);
    if let Some(cache) = parsed.cache {
        let cache_opts = cache.build()?;
        let opts = packer_options.get_or_insert_with(PackerOptions::default);
        opts.cache = cache_opts;
    }
    Ok(Applied {
        retrievers: parsed.retriever,
        section_budgets: parsed.section_budgets.map(SectionBudgetEntry::build),
        packer_options,
        store: parsed.store,
    })
}

// ---------- Backend selection ----------

/// Persistent block-store backend. Cloning an `Arc<S>` is cheap (a
/// pointer bump) so the enum is cheap to pattern-match per call.
pub enum StoreBackend {
    Lmdb(Arc<LmdbStore>),
    Pg(Arc<PgStore>),
}

impl StoreBackend {
    /// Short label for diagnostics (`"lmdb"` / `"pg"`).
    pub fn label(&self) -> &'static str {
        match self {
            Self::Lmdb(_) => "lmdb",
            Self::Pg(_) => "pg",
        }
    }

    /// Borrow the inner LMDB store, or `Err(self.label())` if the
    /// active backend isn't LMDB. Used by commands that depend on
    /// LMDB-only surface area (`verify`, `repair`, etc).
    pub fn as_lmdb(&self) -> Result<&Arc<LmdbStore>, &'static str> {
        match self {
            Self::Lmdb(s) => Ok(s),
            Self::Pg(_) => Err(self.label()),
        }
    }
}

impl std::fmt::Debug for StoreBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StoreBackend::{}", self.label())
    }
}

/// Resolve the chosen backend from a TOML `[store]` section plus
/// CLI/kwarg overrides, then open it.
///
/// Precedence: an explicit `path` / `url` override **wins** over the
/// matching field in the TOML, on the principle that command-line
/// flags and kwargs are more specific than config. The TOML still
/// selects the *backend kind* — passing `--pg-url` with no TOML uses
/// PG; passing it when the TOML says `backend="lmdb"` is an error.
pub fn open_backend(
    store_entry: Option<StoreEntry>,
    path: Option<PathBuf>,
    url: Option<String>,
) -> Result<StoreBackend, String> {
    match store_entry {
        Some(StoreEntry::Pg {
            url: cfg_url,
            pool_size,
            schema,
            tls,
            tls_ca_path,
        }) => {
            if path.is_some() {
                return Err(
                    "[store] backend=\"pg\" is incompatible with an LMDB path override; \
                     pass `url`/`--pg-url` instead".into(),
                );
            }
            let connect_url = url.or(cfg_url).ok_or_else(|| {
                "[store] backend=\"pg\" requires a Postgres URL \
                 (set `url` in the TOML or pass `--pg-url`/`url=`)"
                    .to_string()
            })?;
            let mut cfg = PgStoreConfig::default();
            if let Some(n) = pool_size {
                cfg.max_pool_size = n;
            }
            if let Some(s) = schema {
                cfg.schema = Some(s);
            }
            cfg.tls = parse_tls_mode(tls.as_deref(), tls_ca_path.as_deref())?;
            let store = PgStore::open(&connect_url, &cfg)
                .map_err(|e| format!("open pg store: {e}"))?;
            Ok(StoreBackend::Pg(Arc::new(store)))
        }
        Some(StoreEntry::Lmdb { path: cfg_path }) => {
            if url.is_some() {
                return Err(
                    "[store] backend=\"lmdb\" is incompatible with a Postgres URL override; \
                     pass `path`/`--store` instead".into(),
                );
            }
            let p = path.or(cfg_path).ok_or_else(|| {
                "[store] backend=\"lmdb\" requires a path \
                 (set `path` in the TOML or pass `--store`/positional)"
                    .to_string()
            })?;
            let store = LmdbStore::open(&p, StoreConfig::default())
                .map_err(|e| format!("open lmdb store: {e}"))?;
            Ok(StoreBackend::Lmdb(Arc::new(store)))
        }
        None => match (path, url) {
            (Some(_), Some(_)) => Err(
                "pass either an LMDB path or a Postgres URL, not both — \
                 or pin one in a [store] section"
                    .into(),
            ),
            (None, Some(connect_url)) => {
                let store = PgStore::open(&connect_url, &PgStoreConfig::default())
                    .map_err(|e| format!("open pg store: {e}"))?;
                Ok(StoreBackend::Pg(Arc::new(store)))
            }
            (Some(p), None) => {
                let store = LmdbStore::open(&p, StoreConfig::default())
                    .map_err(|e| format!("open lmdb store: {e}"))?;
                Ok(StoreBackend::Lmdb(Arc::new(store)))
            }
            (None, None) => Err(
                "no block-store backend selected: pass an LMDB path, a `--pg-url`/`url=`, \
                 or a [store] section in the config TOML"
                    .into(),
            ),
        },
    }
}

/// Dispatch a method call to whichever concrete backend is active.
/// Binds the inner `Arc<S>` to `$s` inside `$body`.
///
/// ```ignore
/// let id = dispatch!(backend, |s| s.put(session, block))?;
/// ```
#[macro_export]
macro_rules! dispatch {
    ($backend:expr, |$s:ident| $body:expr) => {
        match $backend {
            $crate::StoreBackend::Lmdb($s) => $body,
            $crate::StoreBackend::Pg($s) => $body,
        }
    };
}

// ---------- Retrievers ----------

/// Materialize a `Vec<Arc<dyn Retriever>>` from the parsed
/// `[[retriever]]` entries, bound to any concrete `BlockStore`.
/// Returns `None` when no entries were configured — callers fall back
/// to the `GreedyPager` default (`RecencyRetriever`).
pub fn build_retrievers<S>(
    entries: &[RetrieverEntry],
    store: &Arc<S>,
) -> Result<Option<Vec<Arc<dyn Retriever>>>, String>
where
    S: BlockStore + 'static,
{
    if entries.is_empty() {
        return Ok(None);
    }
    let mut out: Vec<Arc<dyn Retriever>> = Vec::with_capacity(entries.len());
    for entry in entries {
        let r: Arc<dyn Retriever> = match entry.kind.as_str() {
            "recency" => {
                let mut r = RecencyRetriever::new(store.clone());
                if let Some(h) = entry.half_life_secs {
                    r = r.with_half_life(h);
                }
                Arc::new(r)
            }
            "lexical" => {
                let mut r = LexicalRetriever::new(store.clone());
                if let Some(n) = entry.min_word_len {
                    r = r.with_min_word_len(n);
                }
                Arc::new(r)
            }
            "bm25" => {
                let mut r = Bm25Retriever::new(store.clone());
                if let Some(k) = entry.k1 {
                    r = r.with_k1(k);
                }
                if let Some(b) = entry.b {
                    r = r.with_b(b);
                }
                if let Some(n) = entry.min_word_len {
                    r = r.with_min_word_len(n);
                }
                Arc::new(r)
            }
            "session" => {
                let mut r = SessionRetriever::new(store.clone());
                if let Some(s) = entry.score {
                    r = r.with_score(s);
                }
                Arc::new(r)
            }
            other => {
                return Err(format!(
                    "unknown retriever kind `{other}`; expected one of: recency, lexical, bm25, session",
                ));
            }
        };
        out.push(r);
    }
    Ok(Some(out))
}
