//! `Store` PyO3 class — the main entry point for the Python SDK.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use llm386_compress::{NoopSummarizer, TruncatingSummarizer};
use llm386_compress_anthropic::AnthropicSummarizer;
use llm386_core::{
    BlockId, BlockKind, BlockStore, CallId, ContentHash, ContextBlock as RustBlock, Edge,
    EdgeKind, ModelRegistry, PagePlan as RustPagePlan, Packer, PageRequest, Pager, Provenance,
    SessionId, Summarizer, Timestamp, TokenCount, TokenCounts, Tokenizer,
    TraceRecord as RustTraceRecord, TraceSink, default_registry,
};
use llm386_config::{StoreBackend, dispatch, open_backend};
use llm386_packer::SimplePacker;
use llm386_pager::GreedyPager;
use llm386_tokenizer::{TokenizerRegistry, default_registry as default_tokenizers};
use llm386_trace::LmdbTraceSink;

use crate::config;
use crate::python_traits::wrap_retriever;
use crate::types::{ChatMessage, ContextBlock, PackResult, PagePlan, TraceRecord};

use llm386_core::Retriever;
use parking_lot::RwLock;

create_exception!(llm386, LLM386Error, PyException);

#[pyclass]
pub struct Store {
    inner: StoreBackend,
    tokenizers: TokenizerRegistry,
    models: ModelRegistry,
    retriever_entries: Vec<config::RetrieverEntry>,
    section_budgets: Option<llm386_pager::SectionBudgetTable>,
    packer_options: llm386_packer::PackerOptions,
    python_retrievers: RwLock<Vec<Arc<dyn Retriever>>>,
}

#[pymethods]
impl Store {
    /// Open a persistent block store. Idempotent.
    ///
    /// Backend selection:
    ///   - A `[store]` section in the `profiles` TOML pins the
    ///     backend kind. Within the chosen kind, explicit `url` /
    ///     `path` kwargs take precedence over the TOML's value;
    ///     missing fields fall back to the TOML.
    ///   - With no `[store]` section, `url=` opens PostgreSQL and the
    ///     positional `path` opens LMDB.
    ///   - Passing both `path` and `url` (without a TOML pin), or
    ///     mixing them across kinds (e.g. `url=` with
    ///     `backend="lmdb"`), raises `LLM386Error`.
    ///
    /// `profiles` optionally points at a TOML config file with the
    /// same schema the CLI accepts (`[[profile]]`, `[[hf_tokenizer]]`,
    /// `[[retriever]]`, `[store]`). User profiles override built-ins
    /// by name; user retriever entries replace the default
    /// RecencyRetriever stack.
    #[new]
    #[pyo3(signature = (path = None, *, url = None, profiles = None))]
    fn new(
        path: Option<PathBuf>,
        url: Option<String>,
        profiles: Option<PathBuf>,
    ) -> PyResult<Self> {
        let mut tokenizers = default_tokenizers()
            .map_err(|e| LLM386Error::new_err(format!("init tokenizers: {e}")))?;
        let mut models = default_registry();
        let (retriever_entries, section_budgets, packer_options, store_entry) =
            if let Some(cfg_path) = profiles {
                let parsed = config::parse(&cfg_path).map_err(LLM386Error::new_err)?;
                let applied = config::apply(parsed, &mut models, &mut tokenizers)
                    .map_err(LLM386Error::new_err)?;
                (
                    applied.retrievers,
                    applied.section_budgets,
                    applied.packer_options.unwrap_or_default(),
                    applied.store,
                )
            } else {
                (Vec::new(), None, llm386_packer::PackerOptions::default(), None)
            };

        let inner = open_backend(store_entry, path, url).map_err(LLM386Error::new_err)?;
        Ok(Self {
            inner,
            tokenizers,
            models,
            retriever_entries,
            section_budgets,
            packer_options,
            python_retrievers: RwLock::new(Vec::new()),
        })
    }

    /// Register a Python object as a retriever. The object must
    /// expose a `name: str` attribute and a
    /// `retrieve(session: int, task: str, limit: int)` method that
    /// returns `list[tuple[str, float]]` (block-id hex string +
    /// score). The retriever runs alongside any TOML-configured
    /// retrievers and the default RecencyRetriever fallback,
    /// merged by max score per BlockId.
    fn add_python_retriever(&self, py: Python<'_>, obj: Py<PyAny>) -> PyResult<()> {
        let wrapped = wrap_retriever(py, obj)?;
        self.python_retrievers.write().push(wrapped);
        Ok(())
    }

    /// Drop every previously-registered Python retriever.
    fn clear_python_retrievers(&self) {
        self.python_retrievers.write().clear();
    }

    /// Insert a block. Returns the assigned BlockId (32-char hex).
    #[pyo3(signature = (session, kind, body, *, priority = 0.0))]
    fn put(
        &self,
        session: u128,
        kind: &str,
        body: &Bound<'_, PyAny>,
        priority: f32,
    ) -> PyResult<String> {
        let bytes = if let Ok(b) = body.cast::<PyBytes>() {
            b.as_bytes().to_vec()
        } else if let Ok(s) = body.cast::<PyString>() {
            s.extract::<String>()?.into_bytes()
        } else {
            return Err(PyTypeError::new_err("body must be bytes or str"));
        };
        let block_kind = parse_kind(kind)?;
        let session = SessionId(session);
        let id = new_block_id();
        let now = Timestamp(now_ms());
        let block = RustBlock {
            id,
            kind: block_kind,
            bytes: bytes.clone(),
            token_counts: TokenCounts::new(),
            priority,
            created_at: now,
            updated_at: now,
            provenance: Provenance::default(),
            hash: ContentHash::of(&bytes),
        };
        let stored = dispatch!(&self.inner, |s| s.put(session, block))
            .map_err(|e| LLM386Error::new_err(format!("put: {e}")))?;
        Ok(format!("{stored}"))
    }

    /// Fetch a block by id (32-char hex string).
    fn show(&self, block_id: &str) -> PyResult<ContextBlock> {
        let id = parse_block_id(block_id)?;
        let block = dispatch!(&self.inner, |s| s.get(id))
            .map_err(|e| LLM386Error::new_err(format!("get: {e}")))?
            .ok_or_else(|| LLM386Error::new_err(format!("block not found: {block_id}")))?;
        Ok(ContextBlock::from_rust(block))
    }

    /// Every distinct session id with at least one block.
    fn list_sessions(&self) -> PyResult<Vec<String>> {
        let sessions = dispatch!(&self.inner, |s| s.list_sessions())
            .map_err(|e| LLM386Error::new_err(format!("list_sessions: {e}")))?;
        Ok(sessions.into_iter().map(|s| format!("{s}")).collect())
    }

    /// Delete a block entirely: from the primary table, the
    /// content-hash index, and every session that referenced it.
    /// Returns True if the block existed.
    fn delete(&self, block_id: &str) -> PyResult<bool> {
        let id = parse_block_id(block_id)?;
        dispatch!(&self.inner, |s| s.delete(id))
            .map_err(|e| LLM386Error::new_err(format!("delete: {e}")))
    }

    /// Remove every block belonging to `session`. Returns the count
    /// of blocks affected. Blocks still referenced by other
    /// sessions are kept; ones with no remaining references are
    /// removed entirely (including from the content-hash index).
    fn purge_session(&self, session: u128) -> PyResult<usize> {
        dispatch!(&self.inner, |s| s.purge_session(SessionId(session)))
            .map_err(|e| LLM386Error::new_err(format!("purge_session: {e}")))
    }

    /// Persist a typed directed edge from `from_id` to `to_id`. The
    /// `kind` string is one of `"parent"`, `"derived-from"`,
    /// `"supports"`, `"contradicts"`, `"tool-invocation"`. Idempotent:
    /// re-adding the same triple is a no-op.
    fn add_edge(&self, from_id: &str, to_id: &str, kind: &str) -> PyResult<()> {
        let edge = Edge {
            from: parse_block_id(from_id)?,
            to: parse_block_id(to_id)?,
            kind: parse_edge_kind(kind)?,
        };
        dispatch!(&self.inner, |s| s.put_edge(edge))
            .map_err(|e| LLM386Error::new_err(format!("put_edge: {e}")))
    }

    /// Outgoing edges from `block_id`. Returns a list of `(to_id_hex,
    /// kind)` tuples.
    fn edges_from(&self, block_id: &str) -> PyResult<Vec<(String, String)>> {
        let id = parse_block_id(block_id)?;
        let edges = dispatch!(&self.inner, |s| s.edges_from(id))
            .map_err(|e| LLM386Error::new_err(format!("edges_from: {e}")))?;
        Ok(edges
            .into_iter()
            .map(|e| (format!("{}", e.to), edge_kind_to_str(e.kind).to_string()))
            .collect())
    }

    /// Incoming edges into `block_id`. Returns a list of `(from_id_hex,
    /// kind)` tuples.
    fn edges_to(&self, block_id: &str) -> PyResult<Vec<(String, String)>> {
        let id = parse_block_id(block_id)?;
        let edges = dispatch!(&self.inner, |s| s.edges_to(id))
            .map_err(|e| LLM386Error::new_err(format!("edges_to: {e}")))?;
        Ok(edges
            .into_iter()
            .map(|e| (format!("{}", e.from), edge_kind_to_str(e.kind).to_string()))
            .collect())
    }

    /// Run the pager and return the resulting plan.
    fn page(&self, session: u128, model: &str, task: &str) -> PyResult<PagePlan> {
        let (profile, tokenizer) = self.profile_and_tokenizer(model)?;
        dispatch!(&self.inner, |s| self.page_with(s.clone(), profile, tokenizer, session, task))
    }

    /// Run page+pack and return either a rendered prompt or a list
    /// of role-tagged chat messages, optionally recording a trace.
    ///
    /// `timestamps=True` opts in to ISO 8601 UTC timestamp prefixes
    /// on each rendered block plus a "Current time" anchor in the
    /// task message. Overrides any `[packer]` setting in the config
    /// file.
    #[pyo3(signature = (session, model, task, *, chat = false, timestamps = false, trace = None))]
    fn pack(
        &self,
        session: u128,
        model: &str,
        task: &str,
        chat: bool,
        timestamps: bool,
        trace: Option<PathBuf>,
    ) -> PyResult<PackResult> {
        let (profile, tokenizer) = self.profile_and_tokenizer(model)?;
        dispatch!(&self.inner, |s| self.pack_with(
            s.clone(),
            profile,
            tokenizer,
            session,
            task,
            chat,
            timestamps,
            trace,
        ))
    }

    /// Re-render an existing [`PagePlan`] for a (possibly different)
    /// model — same selected blocks, new tokenizer, new model
    /// budget, new `cache_boundary` from the new model's section
    /// table.
    ///
    /// The cascade-routing pattern: `page()` once for the cheapest
    /// model in your tier; call `pack_with_plan(plan, model=cheap)`;
    /// if confidence is low, escalate via
    /// `pack_with_plan(plan, model=expensive)` without re-running
    /// retrieval.
    ///
    /// Note: blocks are selected against the original page request's
    /// budget. If the new model has a *smaller* budget the re-pack
    /// may return [`PackerError::BudgetExceeded`]; in that case
    /// page again for the smaller model.
    #[pyo3(signature = (plan, session, model, task, *, chat = false, timestamps = false, trace = None))]
    fn pack_with_plan(
        &self,
        plan: Py<PagePlan>,
        session: u128,
        model: &str,
        task: &str,
        chat: bool,
        timestamps: bool,
        trace: Option<PathBuf>,
        py: Python<'_>,
    ) -> PyResult<PackResult> {
        let (profile, tokenizer) = self.profile_and_tokenizer(model)?;
        let request = PageRequest {
            session_id: SessionId(session),
            task: task.to_string(),
            model: profile,
            required_blocks: vec![],
        };

        // Reconstruct the Rust PagePlan from the Python wrapper.
        // Only `selected` is needed for rendering — `selections`,
        // `omitted`, and `estimated_tokens` are informational and the
        // packer doesn't read them during pack/pack_chat.
        let plan_ref = plan.borrow(py);
        let selected: Result<Vec<BlockId>, _> = plan_ref
            .selected
            .iter()
            .map(|s| s.parse::<BlockId>())
            .collect();
        let selected = selected.map_err(|e| LLM386Error::new_err(format!("invalid block id: {e}")))?;
        let estimated_tokens = plan_ref.estimated_tokens;
        drop(plan_ref);
        let rust_plan = RustPagePlan {
            selected,
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount(estimated_tokens),
        };

        let started_at = Timestamp(now_ms());
        let started = Instant::now();
        dispatch!(&self.inner, |s| self.render(
            s.clone(),
            request.clone(),
            rust_plan.clone(),
            tokenizer.clone(),
            chat,
            timestamps,
            trace.clone(),
            started_at,
            started,
        ))
    }

    /// Summarize a session via the named summarizer.
    #[pyo3(signature = (
        session,
        *,
        summarizer = "truncating",
        max_chars = 80,
        last = None,
        store_summary = false,
        anthropic_model = None,
        anthropic_max_tokens = None,
    ))]
    #[allow(clippy::too_many_arguments)] // PyO3 kwargs surface; bundling would obscure the API
    fn summarize(
        &self,
        session: u128,
        summarizer: &str,
        max_chars: usize,
        last: Option<usize>,
        store_summary: bool,
        anthropic_model: Option<&str>,
        anthropic_max_tokens: Option<u32>,
    ) -> PyResult<String> {
        let session = SessionId(session);
        let mut ids = dispatch!(&self.inner, |s| s.list_session(session))
            .map_err(|e| LLM386Error::new_err(format!("list_session: {e}")))?;
        ids.sort();
        if let Some(n) = last {
            let from = ids.len().saturating_sub(n);
            ids.drain(0..from);
        }
        let mut blocks: Vec<RustBlock> = Vec::with_capacity(ids.len());
        for &id in &ids {
            if let Some(b) = dispatch!(&self.inner, |s| s.get(id))
                .map_err(|e| LLM386Error::new_err(format!("get: {e}")))?
            {
                blocks.push(b);
            }
        }

        let (text, name) = match summarizer {
            "noop" => {
                let s = NoopSummarizer;
                (
                    s.summarize(&blocks)
                        .map_err(|e| LLM386Error::new_err(format!("summarize: {e}")))?,
                    s.name(),
                )
            }
            "truncating" => {
                let s = TruncatingSummarizer::new(max_chars);
                (
                    s.summarize(&blocks)
                        .map_err(|e| LLM386Error::new_err(format!("summarize: {e}")))?,
                    s.name(),
                )
            }
            "anthropic" => {
                let mut s = AnthropicSummarizer::from_env()
                    .map_err(|e| LLM386Error::new_err(format!("anthropic init: {e}")))?;
                if let Some(model) = anthropic_model {
                    s = s.with_model(model);
                }
                if let Some(n) = anthropic_max_tokens {
                    s = s.with_max_tokens(n);
                }
                (
                    s.summarize(&blocks)
                        .map_err(|e| LLM386Error::new_err(format!("summarize: {e}")))?,
                    s.name(),
                )
            }
            other => {
                return Err(LLM386Error::new_err(format!(
                    "unknown summarizer `{other}` (expected: noop | truncating | anthropic)",
                )));
            }
        };

        if store_summary {
            let bytes = text.clone().into_bytes();
            let now = Timestamp(now_ms());
            let id = new_block_id();
            let block = RustBlock {
                id,
                kind: BlockKind::Summary,
                bytes: bytes.clone(),
                token_counts: TokenCounts::new(),
                priority: 0.0,
                created_at: now,
                updated_at: now,
                provenance: Provenance {
                    source: Some(format!("summarize:{name}")),
                    parents: ids,
                    labels: vec![],
                },
                hash: ContentHash::of(&bytes),
            };
            dispatch!(&self.inner, |s| s.put(session, block))
                .map_err(|e| LLM386Error::new_err(format!("store summary: {e}")))?;
        }
        Ok(text)
    }
}

impl Store {
    fn profile_and_tokenizer(
        &self,
        model_name: &str,
    ) -> PyResult<(llm386_core::ModelProfile, Arc<dyn Tokenizer>)> {
        let profile = self
            .models
            .get(model_name)
            .ok_or_else(|| LLM386Error::new_err(format!("unknown model: {model_name}")))?
            .clone();
        let tokenizer = self.tokenizers.get(&profile.tokenizer).ok_or_else(|| {
            LLM386Error::new_err(format!(
                "no tokenizer adapter for {} (used by model {})",
                profile.tokenizer, profile.name,
            ))
        })?;
        Ok((profile, tokenizer))
    }

    /// Build a configured `GreedyPager<S>` for a concrete backend.
    fn make_pager<S>(
        &self,
        store: Arc<S>,
        tokenizer: Arc<dyn Tokenizer>,
    ) -> PyResult<GreedyPager<S>>
    where
        S: BlockStore + 'static,
    {
        let mut pager = GreedyPager::new(store.clone(), tokenizer);
        let mut retrievers = config::build_retrievers(&self.retriever_entries, &store)
            .map_err(LLM386Error::new_err)?
            .unwrap_or_default();
        retrievers.extend(self.python_retrievers.read().iter().cloned());
        if !retrievers.is_empty() {
            pager = pager.with_retrievers(retrievers);
        }
        if let Some(budgets) = &self.section_budgets {
            pager = pager.with_budgets(budgets.clone());
        }
        Ok(pager)
    }

    /// Backend-specific `page()` core. Built and dispatched once per
    /// `StoreBackend` arm by [`Store::page`].
    fn page_with<S>(
        &self,
        store: Arc<S>,
        profile: llm386_core::ModelProfile,
        tokenizer: Arc<dyn Tokenizer>,
        session: u128,
        task: &str,
    ) -> PyResult<PagePlan>
    where
        S: BlockStore + 'static,
    {
        let pager = self.make_pager(store, tokenizer)?;
        let request = PageRequest {
            session_id: SessionId(session),
            task: task.to_string(),
            model: profile,
            required_blocks: vec![],
        };
        let plan = pager.page(request).map_err(|e| LLM386Error::new_err(format!("page: {e}")))?;
        Ok(PagePlan::from_rust(plan))
    }

    /// Backend-specific `pack()` core: pages, then hands off to
    /// [`Store::render`].
    #[allow(clippy::too_many_arguments)] // matches the pyfn surface
    fn pack_with<S>(
        &self,
        store: Arc<S>,
        profile: llm386_core::ModelProfile,
        tokenizer: Arc<dyn Tokenizer>,
        session: u128,
        task: &str,
        chat: bool,
        timestamps: bool,
        trace: Option<PathBuf>,
    ) -> PyResult<PackResult>
    where
        S: BlockStore + 'static,
    {
        let pager = self.make_pager(store.clone(), tokenizer.clone())?;
        let request = PageRequest {
            session_id: SessionId(session),
            task: task.to_string(),
            model: profile,
            required_blocks: vec![],
        };
        let started_at = Timestamp(now_ms());
        let started = Instant::now();
        let plan = pager
            .page(request.clone())
            .map_err(|e| LLM386Error::new_err(format!("page: {e}")))?;
        self.render(store, request, plan, tokenizer, chat, timestamps, trace, started_at, started)
    }

    /// Shared rendering core for `pack` and `pack_with_plan`. Builds
    /// a `SimplePacker<S>` for the concrete backend, dispatches
    /// `pack` or `pack_chat`, optionally records a trace, and
    /// packages the result for Python.
    #[allow(clippy::too_many_arguments)] // factored from the callers; bundling obscures the API
    fn render<S>(
        &self,
        store: Arc<S>,
        request: PageRequest,
        plan: RustPagePlan,
        tokenizer: Arc<dyn Tokenizer>,
        chat: bool,
        timestamps: bool,
        trace: Option<PathBuf>,
        started_at: Timestamp,
        started: Instant,
    ) -> PyResult<PackResult>
    where
        S: BlockStore + 'static,
    {
        let mut packer_options = self.packer_options.clone();
        if timestamps {
            packer_options.include_timestamps = true;
        }
        let packer = SimplePacker::new(store, tokenizer).with_options(packer_options);

        if chat {
            let chat_prompt = packer
                .pack_chat(&request, &plan)
                .map_err(|e| LLM386Error::new_err(format!("pack_chat: {e}")))?;
            let cache_boundary = chat_prompt.cache_boundary;
            let messages: Vec<ChatMessage> =
                chat_prompt.messages.into_iter().map(ChatMessage::from_rust).collect();
            let trace_id = if let Some(trace_path) = trace {
                Some(record_trace(
                    &trace_path,
                    request.session_id,
                    &request.model.name,
                    request.model.tokenizer.as_str(),
                    &plan,
                    chat_prompt.input_tokens,
                    started_at,
                    u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
                )?)
            } else {
                None
            };
            Ok(PackResult {
                rendered: None,
                messages: Some(messages),
                cache_boundary,
                trace_id,
            })
        } else {
            let prompt = packer
                .pack(&request, &plan)
                .map_err(|e| LLM386Error::new_err(format!("pack: {e}")))?;
            let trace_id = if let Some(trace_path) = trace {
                Some(record_trace(
                    &trace_path,
                    request.session_id,
                    &request.model.name,
                    request.model.tokenizer.as_str(),
                    &plan,
                    prompt.input_tokens,
                    started_at,
                    u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX),
                )?)
            } else {
                None
            };
            Ok(PackResult {
                rendered: Some(prompt.rendered),
                messages: None,
                cache_boundary: None,
                trace_id,
            })
        }
    }
}

#[allow(clippy::too_many_arguments)] // mirrors the trace record fields by design
fn record_trace(
    path: &std::path::Path,
    session: SessionId,
    model: &str,
    tokenizer_version: &str,
    plan: &llm386_core::PagePlan,
    prompt_tokens: llm386_core::TokenCount,
    started_at: Timestamp,
    duration_ms: u32,
) -> PyResult<String> {
    let sink = LmdbTraceSink::open(path)
        .map_err(|e| LLM386Error::new_err(format!("open trace: {e}")))?;
    let call_id = new_call_id();
    sink.record(RustTraceRecord {
        call_id,
        session,
        model: model.to_string(),
        plan: plan.clone(),
        prompt_tokens,
        prompt_hash: ContentHash::of(&[]),
        started_at,
        duration_ms,
        model_version: model.to_string(),
        tokenizer_version: tokenizer_version.to_string(),
        output: None,
        output_tokens: None,
    })
    .map_err(|e| LLM386Error::new_err(format!("record trace: {e}")))?;
    Ok(format!("{call_id}"))
}

/// Read-only wrapper around an LMDB trace store. Pair with
/// `Store.pack(trace="./traces")` to inspect a recorded trace.
#[pyclass]
pub struct Trace {
    sink: LmdbTraceSink,
}

#[pymethods]
impl Trace {
    /// Open (or create) a trace store at `path`.
    #[new]
    fn new(path: PathBuf) -> PyResult<Self> {
        let sink = LmdbTraceSink::open(&path)
            .map_err(|e| LLM386Error::new_err(format!("open trace store: {e}")))?;
        Ok(Self { sink })
    }

    /// Fetch a trace record by call id.
    fn show(&self, call_id: &str) -> PyResult<TraceRecord> {
        let id = parse_call_id(call_id)?;
        let record = self
            .sink
            .fetch(id)
            .map_err(|e| LLM386Error::new_err(format!("fetch trace: {e}")))?
            .ok_or_else(|| LLM386Error::new_err(format!("no trace for {call_id}")))?;
        Ok(TraceRecord::from_rust(record))
    }

    /// Patch the model output and output token count back into a
    /// previously-recorded trace. Use after the model call returns
    /// so the trace is replay-complete.
    fn update_output(&self, call_id: &str, output: String, output_tokens: u32) -> PyResult<()> {
        let id = parse_call_id(call_id)?;
        self.sink
            .update_output(id, output, llm386_core::TokenCount(output_tokens))
            .map_err(|e| LLM386Error::new_err(format!("update_output: {e}")))
    }
}

fn parse_call_id(s: &str) -> PyResult<CallId> {
    let n = u128::from_str_radix(s, 16)
        .map_err(|e| LLM386Error::new_err(format!("invalid call id `{s}`: {e}")))?;
    Ok(CallId(n))
}

fn parse_kind(s: &str) -> PyResult<BlockKind> {
    let kind = match s {
        "system" | "System" => BlockKind::System,
        "user-message" | "UserMessage" => BlockKind::UserMessage,
        "assistant-message" | "AssistantMessage" => BlockKind::AssistantMessage,
        "tool-result" | "ToolResult" => BlockKind::ToolResult,
        "summary" | "Summary" => BlockKind::Summary,
        "fact" | "Fact" => BlockKind::Fact,
        "document-chunk" | "DocumentChunk" => BlockKind::DocumentChunk,
        "plan" | "Plan" => BlockKind::Plan,
        "state" | "State" => BlockKind::State,
        "trace" | "Trace" => BlockKind::Trace,
        other => {
            return Err(LLM386Error::new_err(format!("unknown block kind: {other}")));
        }
    };
    Ok(kind)
}

fn parse_block_id(s: &str) -> PyResult<BlockId> {
    let n = u128::from_str_radix(s, 16)
        .map_err(|e| LLM386Error::new_err(format!("invalid block id `{s}`: {e}")))?;
    Ok(BlockId(n))
}

fn parse_edge_kind(s: &str) -> PyResult<EdgeKind> {
    let kind = match s {
        "parent" | "Parent" => EdgeKind::Parent,
        "derived-from" | "DerivedFrom" => EdgeKind::DerivedFrom,
        "supports" | "Supports" => EdgeKind::Supports,
        "contradicts" | "Contradicts" => EdgeKind::Contradicts,
        "tool-invocation" | "ToolInvocation" => EdgeKind::ToolInvocation,
        other => {
            return Err(LLM386Error::new_err(format!("unknown edge kind: {other}")));
        }
    };
    Ok(kind)
}

const fn edge_kind_to_str(kind: EdgeKind) -> &'static str {
    match kind {
        EdgeKind::Parent => "parent",
        EdgeKind::DerivedFrom => "derived-from",
        EdgeKind::Supports => "supports",
        EdgeKind::Contradicts => "contradicts",
        EdgeKind::ToolInvocation => "tool-invocation",
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
}

fn new_block_id() -> BlockId {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("getrandom failed");
    BlockId::from_parts(now_ms(), u128::from_be_bytes(buf))
}

fn new_call_id() -> CallId {
    let mut buf = [0u8; 16];
    getrandom::fill(&mut buf).expect("getrandom failed");
    CallId(u128::from_be_bytes(buf))
}
