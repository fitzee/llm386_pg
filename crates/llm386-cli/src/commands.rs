//! Subcommand handlers for `llm386`.

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, anyhow};
use llm386_compress::{NoopSummarizer, TruncatingSummarizer};
use llm386_compress_anthropic::AnthropicSummarizer;
use llm386_config::{
    Applied, RetrieverEntry, StoreBackend, StoreEntry, build_retrievers, dispatch, open_backend,
};
use llm386_core::{
    BlockId, BlockKind, BlockStore, CallId, ContentHash, ContextBlock, ModelProfile, ModelRegistry,
    Packer, PageRequest, Pager, Provenance, SessionId, Summarizer, Timestamp, TokenCounts,
    Tokenizer, TraceRecord, TraceSink, default_registry,
};
use llm386_packer::SimplePacker;
use llm386_pager::GreedyPager;
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::{TokenizerRegistry, default_registry as tokenizer_registry};
use llm386_trace::LmdbTraceSink;

use crate::cli::{Command, SummarizerArg, TraceSub};

const PROFILES_ENV: &str = "LLM386_PROFILES";

/// Bundle of registries the CLI hands off to every subcommand
/// handler. Built once at startup from defaults + (optional) user
/// config file. Retrievers can't be pre-built because they hold a
/// store reference — the CLI rebuilds them per-command from
/// `retriever_entries`.
pub(crate) struct LoadedConfig {
    pub models: ModelRegistry,
    pub tokenizers: TokenizerRegistry,
    pub retriever_entries: Vec<RetrieverEntry>,
    pub section_budgets: Option<llm386_pager::SectionBudgetTable>,
    pub packer_options: llm386_packer::PackerOptions,
    /// Optional `[store]` from the TOML — combined with the global
    /// `--store` / `--pg-url` flags by [`open_block_store`].
    pub store: Option<StoreEntry>,
}

/// Load the built-in registries, then fold in user-supplied
/// `[[profile]]` and `[[hf_tokenizer]]` entries from `--profiles
/// <path>` (or, if absent, the `LLM386_PROFILES` env var).
pub(crate) fn load_config(flag_path: Option<&Path>) -> Result<LoadedConfig> {
    let mut models = default_registry();
    let mut tokenizers = tokenizer_registry().context("initializing default tokenizer registry")?;

    let path = flag_path
        .map(Path::to_path_buf)
        .or_else(|| std::env::var_os(PROFILES_ENV).map(PathBuf::from));

    let applied = if let Some(path) = path {
        let parsed = llm386_config::parse(&path).map_err(|e| anyhow!(e))?;
        llm386_config::apply(parsed, &mut models, &mut tokenizers).map_err(|e| anyhow!(e))?
    } else {
        Applied {
            retrievers: Vec::new(),
            section_budgets: None,
            packer_options: None,
            store: None,
        }
    };

    Ok(LoadedConfig {
        models,
        tokenizers,
        retriever_entries: applied.retrievers,
        section_budgets: applied.section_budgets,
        packer_options: applied.packer_options.unwrap_or_default(),
        store: applied.store,
    })
}

/// Resolve the chosen block store from TOML `[store]` + global CLI
/// flags. Called by every subcommand that touches block storage.
fn open_block_store(
    config: &LoadedConfig,
    cli_store: Option<PathBuf>,
    cli_pg_url: Option<String>,
) -> Result<StoreBackend> {
    open_backend(config.store.clone(), cli_store, cli_pg_url).map_err(|e| anyhow!(e))
}

pub(crate) fn dispatch(
    command: Command,
    cli_store: Option<PathBuf>,
    cli_pg_url: Option<String>,
    config: &LoadedConfig,
) -> Result<()> {
    // Commands that don't need a block-store backend.
    match command {
        Command::Init { path } => return init(&path),
        Command::ListModels => return list_models(&config.models),
        Command::Trace(TraceSub::Show { store, call_id }) => {
            return trace_show(&store, CallId(call_id));
        }
        Command::Trace(TraceSub::Diff { store, prev, next }) => {
            return trace_diff(&store, CallId(prev), CallId(next));
        }
        _ => {}
    }

    // Everything below operates on the block store. Open it once.
    let backend = open_block_store(config, cli_store, cli_pg_url)?;

    match command {
        Command::Init { .. } | Command::ListModels | Command::Trace(_) => {
            unreachable!("handled above")
        }
        Command::Put {
            session,
            kind,
            priority,
            file,
        } => put(&backend, SessionId(session), kind.into(), priority, &file),
        Command::Page {
            session,
            model,
            task,
            json,
        } => page(&backend, SessionId(session), &model, &task, json, config),
        Command::Pack {
            session,
            model,
            task,
            prompt_only,
            chat,
            timestamps,
            trace,
        } => pack(
            &backend,
            SessionId(session),
            &model,
            &task,
            prompt_only,
            chat,
            timestamps,
            trace.as_deref(),
            config,
        ),
        Command::ListSessions => list_sessions(&backend),
        Command::Verify => verify(&backend),
        Command::Repair { yes } => repair(&backend, yes),
        Command::Purge {
            block,
            session,
            yes,
        } => purge(&backend, block, session, yes),
        Command::Show { id, json } => show(&backend, BlockId(id), json),
        Command::AddEdge { from, to, kind } => {
            add_edge(&backend, BlockId(from), BlockId(to), kind.into())
        }
        Command::Edges { id, incoming } => edges(&backend, BlockId(id), incoming),
        Command::Summarize {
            session,
            summarizer,
            max_chars,
            last,
            store_summary,
            anthropic_model,
            anthropic_max_tokens,
        } => summarize(&SummarizeArgs {
            backend: &backend,
            session: SessionId(session),
            summarizer,
            max_chars,
            last,
            store_summary,
            anthropic_model: anthropic_model.as_deref(),
            anthropic_max_tokens,
        }),
    }
}

fn init(path: &Path) -> Result<()> {
    let _store = LmdbStore::open(path, StoreConfig::default())
        .with_context(|| format!("opening store at {}", path.display()))?;
    println!("initialized LMDB store at {}", path.display());
    Ok(())
}

fn put(
    backend: &StoreBackend,
    session: SessionId,
    kind: BlockKind,
    priority: f32,
    file: &Path,
) -> Result<()> {
    let bytes = if file == Path::new("-") {
        let mut buf = Vec::new();
        std::io::stdin()
            .read_to_end(&mut buf)
            .context("reading stdin")?;
        buf
    } else {
        std::fs::read(file).with_context(|| format!("reading {}", file.display()))?
    };

    let id = new_block_id();
    let now = Timestamp(now_ms());
    let block = ContextBlock {
        id,
        kind,
        bytes: bytes.clone(),
        token_counts: TokenCounts::new(),
        priority,
        created_at: now,
        updated_at: now,
        provenance: Provenance::default(),
        hash: ContentHash::of(&bytes),
    };
    let stored = dispatch!(backend, |s| s.put(session, block))?;
    println!("{stored}");
    Ok(())
}

#[allow(clippy::unnecessary_wraps)] // matches sibling-handler signatures.
fn list_models(reg: &ModelRegistry) -> Result<()> {
    let mut profiles: Vec<&ModelProfile> = reg.profiles().collect();
    profiles.sort_by(|a, b| a.name.cmp(&b.name));
    println!(
        "{:<24}  {:>8}  {:>6}  {:>6}  {:<14}",
        "name", "ctx", "out", "margin", "tokenizer"
    );
    for p in profiles {
        println!(
            "{:<24}  {:>8}  {:>6}  {:>6}  {:<14}",
            p.name,
            p.max_context_tokens,
            p.reserved_output_tokens,
            p.safety_margin_tokens,
            p.tokenizer,
        );
    }
    Ok(())
}

fn profile_and_tokenizer(
    config: &LoadedConfig,
    model_name: &str,
) -> Result<(ModelProfile, Arc<dyn Tokenizer>)> {
    // `resolve` is infallible — strips provider prefixes, exact-
    // matches, then family-fallbacks, then default-fallbacks with a
    // deduped warning per unknown input.
    let profile = config.models.resolve(model_name).clone();
    let tokenizer = config.tokenizers.get(&profile.tokenizer).ok_or_else(|| {
        anyhow!(
            "no tokenizer adapter for {} (used by model {})",
            profile.tokenizer,
            profile.name,
        )
    })?;
    Ok((profile, tokenizer))
}

/// Build a configured `GreedyPager<S>` for a concrete backend.
fn make_pager<S: BlockStore + 'static>(
    store: &Arc<S>,
    tokenizer: Arc<dyn Tokenizer>,
    config: &LoadedConfig,
) -> Result<GreedyPager<S>> {
    let mut pager = GreedyPager::new(store.clone(), tokenizer);
    if let Some(retrievers) =
        build_retrievers(&config.retriever_entries, store).map_err(|e| anyhow!(e))?
    {
        pager = pager.with_retrievers(retrievers);
    }
    if let Some(budgets) = &config.section_budgets {
        pager = pager.with_budgets(budgets.clone());
    }
    Ok(pager)
}

fn page(
    backend: &StoreBackend,
    session: SessionId,
    model_name: &str,
    task: &str,
    json: bool,
    config: &LoadedConfig,
) -> Result<()> {
    let (profile, tokenizer) = profile_and_tokenizer(config, model_name)?;
    let plan = dispatch!(backend, |s| {
        let pager = make_pager(s, tokenizer, config)?;
        pager.page(PageRequest {
            session_id: session,
            task: task.to_string(),
            model: profile,
            required_blocks: vec![],
        })?
    });

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&plan).context("serializing plan")?
        );
        return Ok(());
    }

    println!("selected ({}):", plan.selected.len());
    for id in &plan.selected {
        println!("  {id}");
    }
    println!("omitted ({}):", plan.omitted.len());
    for o in &plan.omitted {
        println!("  {} ({:?}, score={:.4})", o.block_id, o.reason, o.score);
    }
    println!("estimated_tokens: {}", plan.estimated_tokens);
    Ok(())
}

#[allow(clippy::fn_params_excessive_bools, clippy::too_many_arguments)] // CLI flags map 1:1 to handler args; refactoring to a struct buys nothing here.
fn pack(
    backend: &StoreBackend,
    session: SessionId,
    model_name: &str,
    task: &str,
    prompt_only: bool,
    chat: bool,
    timestamps_flag: bool,
    trace_path: Option<&Path>,
    config: &LoadedConfig,
) -> Result<()> {
    let (profile, tokenizer) = profile_and_tokenizer(config, model_name)?;

    let request = PageRequest {
        session_id: session,
        task: task.to_string(),
        model: profile,
        required_blocks: vec![],
    };
    let mut packer_options = config.packer_options.clone();
    if timestamps_flag {
        packer_options.include_timestamps = true;
    }
    let started_at = Timestamp(now_ms());
    let started = Instant::now();

    let (plan, prompt_rendered, prompt_input_tokens, chat_prompt_json) =
        dispatch!(backend, |s| {
            let pager = make_pager(s, tokenizer.clone(), config)?;
            let packer =
                SimplePacker::new(s.clone(), tokenizer.clone()).with_options(packer_options.clone());
            let plan = pager.page(request.clone())?;
            let prompt = packer.pack(&request, &plan)?;
            let chat_json = if chat {
                let chat_prompt = packer.pack_chat(&request, &plan)?;
                Some((
                    chat_prompt.model.clone(),
                    chat_prompt.input_tokens,
                    chat_prompt.messages.len(),
                    chat_prompt.cache_boundary,
                    serde_json::to_string_pretty(&chat_prompt)
                        .context("serializing chat prompt")?,
                ))
            } else {
                None
            };
            (plan, prompt.rendered, prompt.input_tokens, chat_json)
        });

    let duration_ms = u32::try_from(started.elapsed().as_millis()).unwrap_or(u32::MAX);

    let trace_id = if let Some(path) = trace_path {
        let sink = LmdbTraceSink::open(path)
            .with_context(|| format!("opening trace store at {}", path.display()))?;
        let call_id = new_call_id();
        sink.record(TraceRecord {
            call_id,
            session,
            model: request.model.name.clone(),
            plan: plan.clone(),
            prompt_tokens: prompt_input_tokens,
            prompt_hash: ContentHash::of(prompt_rendered.as_bytes()),
            started_at,
            duration_ms,
            model_version: request.model.name.clone(),
            tokenizer_version: request.model.tokenizer.as_str().to_string(),
            output: None,
            output_tokens: None,
        })?;
        Some(call_id)
    } else {
        None
    };

    if let Some((model_name, input_tokens, message_count, cache_boundary, json)) =
        chat_prompt_json
    {
        eprintln!("# model:          {model_name}");
        eprintln!("# input_tokens:   {input_tokens}");
        eprintln!("# messages:       {message_count}");
        match cache_boundary {
            Some(n) => eprintln!("# cache_boundary: {n} (messages[0..={n}] cacheable)"),
            None => eprintln!("# cache_boundary: none"),
        }
        eprintln!("# duration_ms:    {duration_ms}");
        if let Some(id) = trace_id {
            eprintln!("# trace_id:       {id}");
        }
        eprintln!("---");
        println!("{json}");
    } else if prompt_only {
        print!("{prompt_rendered}");
    } else {
        eprintln!("# model:         {}", request.model.name);
        eprintln!("# input_tokens:  {prompt_input_tokens}");
        eprintln!("# duration_ms:   {duration_ms}");
        if let Some(id) = trace_id {
            eprintln!("# trace_id:      {id}");
        }
        eprintln!("---");
        print!("{prompt_rendered}");
    }
    Ok(())
}

struct SummarizeArgs<'a> {
    backend: &'a StoreBackend,
    session: SessionId,
    summarizer: SummarizerArg,
    max_chars: usize,
    last: Option<usize>,
    store_summary: bool,
    anthropic_model: Option<&'a str>,
    anthropic_max_tokens: Option<u32>,
}

fn summarize(args: &SummarizeArgs<'_>) -> Result<()> {
    let mut ids = dispatch!(args.backend, |s| s.list_session(args.session))?;
    ids.sort(); // BlockId order is chronological.
    if let Some(n) = args.last {
        let from = ids.len().saturating_sub(n);
        ids.drain(0..from);
    }
    let mut blocks: Vec<ContextBlock> = Vec::with_capacity(ids.len());
    for &id in &ids {
        if let Some(b) = dispatch!(args.backend, |s| s.get(id))? {
            blocks.push(b);
        }
    }

    let (summary_text, summarizer_name) = match args.summarizer {
        SummarizerArg::Noop => {
            let s = NoopSummarizer;
            (s.summarize(&blocks)?, s.name())
        }
        SummarizerArg::Truncating => {
            let s = TruncatingSummarizer::new(args.max_chars);
            (s.summarize(&blocks)?, s.name())
        }
        SummarizerArg::Anthropic => {
            let mut s =
                AnthropicSummarizer::from_env().context("constructing AnthropicSummarizer")?;
            if let Some(model) = args.anthropic_model {
                s = s.with_model(model);
            }
            if let Some(n) = args.anthropic_max_tokens {
                s = s.with_max_tokens(n);
            }
            (s.summarize(&blocks)?, s.name())
        }
    };

    print!("{summary_text}");

    if args.store_summary {
        let bytes = summary_text.into_bytes();
        let now = Timestamp(now_ms());
        let id = new_block_id();
        let block = ContextBlock {
            id,
            kind: BlockKind::Summary,
            bytes: bytes.clone(),
            token_counts: TokenCounts::new(),
            priority: 0.0,
            created_at: now,
            updated_at: now,
            provenance: Provenance {
                source: Some(format!("summarize:{summarizer_name}")),
                parents: ids,
                labels: vec![],
            },
            hash: ContentHash::of(&bytes),
        };
        let stored = dispatch!(args.backend, |s| s.put(args.session, block))?;
        eprintln!("# summary stored: {stored}");
    }

    Ok(())
}

fn list_sessions(backend: &StoreBackend) -> Result<()> {
    let sessions = dispatch!(backend, |s| s.list_sessions())?;
    for s in sessions {
        println!("{s}");
    }
    Ok(())
}

fn verify(backend: &StoreBackend) -> Result<()> {
    let store = backend
        .as_lmdb()
        .map_err(|label| anyhow!("verify is LMDB-only (active backend: {label})"))?;
    let report = store.verify()?;
    println!("blocks checked:           {}", report.blocks_checked);
    println!("hash mismatches:          {}", report.hash_mismatches.len());
    println!(
        "missing from hash index:  {}",
        report.missing_from_hash_index.len()
    );
    println!(
        "hash index misroutes:     {}",
        report.hash_index_misroutes.len()
    );
    println!(
        "orphan session entries:   {}",
        report.orphan_session_entries
    );
    println!(
        "blocks with no session:   {}",
        report.blocks_with_no_session.len()
    );
    if !report.hash_mismatches.is_empty() {
        eprintln!("\nhash mismatches:");
        for id in &report.hash_mismatches {
            eprintln!("  {id}");
        }
    }
    if !report.missing_from_hash_index.is_empty() {
        eprintln!("\nmissing from hash index:");
        for id in &report.missing_from_hash_index {
            eprintln!("  {id}");
        }
    }
    if report.is_clean() {
        println!("\nOK");
        Ok(())
    } else {
        Err(anyhow!("integrity check failed"))
    }
}

fn repair(backend: &StoreBackend, yes: bool) -> Result<()> {
    if !yes {
        return Err(anyhow!("destructive operation: pass --yes to confirm"));
    }
    let store = backend
        .as_lmdb()
        .map_err(|label| anyhow!("repair is LMDB-only (active backend: {label})"))?;
    let report = store.repair()?;
    println!(
        "hash index rebuilt:                  {}",
        report.hash_index_rebuilt
    );
    println!(
        "hash entries written:                {}",
        report.hash_entries_written
    );
    println!(
        "orphan session entries removed:      {}",
        report.orphan_session_entries_removed
    );
    println!(
        "blocks with no session (untouched):  {}",
        report.blocks_with_no_session.len()
    );
    println!(
        "hash mismatches quarantined:         {}",
        report.hash_mismatches_quarantined.len()
    );
    if !report.hash_mismatches_quarantined.is_empty() {
        eprintln!("\nhash mismatches (left as-is, need human review):");
        for id in &report.hash_mismatches_quarantined {
            eprintln!("  {id}");
        }
    }
    Ok(())
}

fn purge(
    backend: &StoreBackend,
    block: Option<u128>,
    session: Option<u128>,
    yes: bool,
) -> Result<()> {
    if !yes {
        return Err(anyhow!("destructive operation: pass --yes to confirm"));
    }
    match (block, session) {
        (Some(_), Some(_)) | (None, None) => {
            Err(anyhow!("specify exactly one of --block or --session"))
        }
        (Some(id), None) => {
            let deleted = dispatch!(backend, |s| s.delete(BlockId(id)))?;
            if deleted {
                println!("deleted block {}", BlockId(id));
            } else {
                eprintln!("block not found: {}", BlockId(id));
            }
            Ok(())
        }
        (None, Some(sid)) => {
            let count = dispatch!(backend, |s| s.purge_session(SessionId(sid)))?;
            println!("purged {count} blocks from session {}", SessionId(sid));
            Ok(())
        }
    }
}

fn show(backend: &StoreBackend, id: BlockId, json: bool) -> Result<()> {
    let block = dispatch!(backend, |s| s.get(id))?
        .ok_or_else(|| anyhow!("block not found: {id}"))?;

    if json {
        println!(
            "{}",
            serde_json::to_string_pretty(&block).context("serializing block")?
        );
        return Ok(());
    }

    println!("id:            {}", block.id);
    println!("kind:          {:?}", block.kind);
    println!("priority:      {:.4}", block.priority);
    println!("created_at:    {}", block.created_at.0);
    println!("updated_at:    {}", block.updated_at.0);
    println!("hash:          {:?}", block.hash);
    println!("body_bytes:    {}", block.bytes.len());

    if block.provenance.source.is_some()
        || !block.provenance.parents.is_empty()
        || !block.provenance.labels.is_empty()
    {
        println!("provenance:");
        if let Some(src) = &block.provenance.source {
            println!("  source:      {src}");
        }
        if !block.provenance.parents.is_empty() {
            println!("  parents ({}):", block.provenance.parents.len());
            for p in &block.provenance.parents {
                println!("    {p}");
            }
        }
        if !block.provenance.labels.is_empty() {
            println!("  labels:      {}", block.provenance.labels.join(", "));
        }
    }

    if !block.token_counts.is_empty() {
        println!("token_counts:");
        for (tid, count) in block.token_counts.iter() {
            println!("  {tid}: {count}");
        }
    }

    println!("---");
    if let Ok(text) = std::str::from_utf8(&block.bytes) {
        print!("{text}");
        if !text.ends_with('\n') {
            println!();
        }
    } else {
        for chunk in block.bytes.chunks(16).take(16) {
            for b in chunk {
                print!("{b:02x} ");
            }
            println!();
        }
        if block.bytes.len() > 256 {
            println!("... ({} more bytes)", block.bytes.len() - 256);
        }
    }
    Ok(())
}

fn trace_diff(store_path: &Path, prev: CallId, next: CallId) -> Result<()> {
    let sink = LmdbTraceSink::open(store_path)
        .with_context(|| format!("opening trace store at {}", store_path.display()))?;
    let prev_rec = sink.fetch(prev)?.ok_or_else(|| anyhow!("no trace for {prev}"))?;
    let next_rec = sink.fetch(next)?.ok_or_else(|| anyhow!("no trace for {next}"))?;

    let diff = llm386_diff::diff_traces(&prev_rec, &next_rec);
    println!("prev:    {prev}");
    println!("next:    {next}");
    println!("summary: {}", diff.summary());

    if !diff.added.is_empty() {
        println!("added ({}):", diff.added.len());
        for entry in &diff.added {
            println!(
                "  + {} ({:?})",
                entry.block_id,
                entry.reason_next.expect("added entries have a next reason"),
            );
        }
    }
    if !diff.removed.is_empty() {
        println!("removed ({}):", diff.removed.len());
        for entry in &diff.removed {
            println!(
                "  - {} ({:?})",
                entry.block_id,
                entry.reason_prev.expect("removed entries have a prev reason"),
            );
        }
    }
    let changed: Vec<_> = diff.kept.iter().filter(|e| e.reason_changed()).collect();
    if !changed.is_empty() {
        println!("reason changes ({}):", changed.len());
        for entry in changed {
            println!(
                "  ~ {} ({:?} -> {:?})",
                entry.block_id,
                entry.reason_prev.expect("kept entries have a prev reason"),
                entry.reason_next.expect("kept entries have a next reason"),
            );
        }
    }
    Ok(())
}

fn add_edge(
    backend: &StoreBackend,
    from: BlockId,
    to: BlockId,
    kind: llm386_core::EdgeKind,
) -> Result<()> {
    dispatch!(backend, |s| s.put_edge(llm386_core::Edge { from, to, kind }))?;
    println!("edge added: {from} --{kind:?}--> {to}");
    Ok(())
}

fn edges(backend: &StoreBackend, id: BlockId, incoming: bool) -> Result<()> {
    let edges = if incoming {
        dispatch!(backend, |s| s.edges_to(id))?
    } else {
        dispatch!(backend, |s| s.edges_from(id))?
    };
    if edges.is_empty() {
        println!("no edges");
        return Ok(());
    }
    for edge in edges {
        println!("{} --{:?}--> {}", edge.from, edge.kind, edge.to);
    }
    Ok(())
}

fn trace_show(store_path: &Path, call_id: CallId) -> Result<()> {
    let sink = LmdbTraceSink::open(store_path)
        .with_context(|| format!("opening trace store at {}", store_path.display()))?;
    let trace = sink
        .fetch(call_id)?
        .ok_or_else(|| anyhow!("no trace for {call_id}"))?;

    println!("call_id:         {}", trace.call_id);
    println!("session:         {}", trace.session);
    println!("model:           {}", trace.model);
    println!("started_at_ms:   {}", trace.started_at.0);
    println!("duration_ms:     {}", trace.duration_ms);
    println!("prompt_tokens:   {}", trace.prompt_tokens);
    println!("prompt_hash:     {:?}", trace.prompt_hash);
    println!("estimated:       {}", trace.plan.estimated_tokens);
    println!("plan.selected ({}):", trace.plan.selected.len());
    for id in &trace.plan.selected {
        println!("  {id}");
    }
    println!("plan.omitted ({}):", trace.plan.omitted.len());
    for o in &trace.plan.omitted {
        println!("  {} ({:?}, score={:.4})", o.block_id, o.reason, o.score);
    }
    Ok(())
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
