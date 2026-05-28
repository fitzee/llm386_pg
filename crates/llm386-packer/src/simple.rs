//! `SimplePacker` — string-rendering [`Packer`].
//!
//! Walks `PagePlan::selected`, classifies each block by its
//! [`BlockKind`] into one of nine canonical sections, renders the
//! sections in fixed order with `## <name>` headers, and verifies
//! that the rendered token count does not exceed
//! [`ModelProfile::input_budget`].
//!
//! Determinism: within each section, blocks render in `BlockId`
//! order (chronological). Identical inputs always produce a
//! byte-identical `rendered` string. With timestamps enabled
//! (`PackerOptions::include_timestamps`), the "current time" anchor
//! also affects the bytes — set `now_ms` explicitly for byte-stable
//! replay.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use llm386_core::BlockKind;
use llm386_core::{
    BlockId, BlockStore, ChatMessage, ChatPrompt, ChatRole, ContextBlock, PackedBlock,
    PackedPrompt, Packer, PackerError, PagePlan, PageRequest, SectionKind, StoreError, TokenCount,
    Tokenizer,
};
use tracing::instrument;

/// Canonical render order. `Slack` is intentional headroom and is
/// never emitted; `Task` is synthesized from `PageRequest::task`
/// rather than drawn from blocks.
const SECTION_ORDER: [SectionKind; 9] = [
    SectionKind::System,
    SectionKind::Task,
    SectionKind::State,
    SectionKind::Plan,
    SectionKind::Retrieved,
    SectionKind::Tools,
    SectionKind::Recent,
    SectionKind::Background,
    SectionKind::Slack,
];

/// Optional knobs for [`SimplePacker`].
///
/// Defaults preserve the original behavior — bodies are rendered as
/// stored, with no temporal annotations.
#[derive(Clone, Default, Debug)]
pub struct PackerOptions {
    /// When `true`, prepend each rendered block with its
    /// `created_at` timestamp in ISO 8601 UTC form
    /// (`[2026-05-05T11:32:00Z] ...`), and emit a "Current time"
    /// line at the start of the Task section so the model can
    /// compute relative times. Defaults to `false`.
    pub include_timestamps: bool,
    /// Override the "current time" anchor used when
    /// `include_timestamps` is `true`. `None` reads `SystemTime::now()`
    /// at pack time. Set explicitly (e.g. `Some(request_started_at)`)
    /// for byte-stable replay across invocations.
    pub now_ms: Option<u64>,
    /// Cache-boundary knobs for `pack_chat`. Defaults to
    /// [`CacheOptions::default()`].
    pub cache: CacheOptions,
}

/// Prompt-cache knobs for [`SimplePacker::pack_chat`].
///
/// Only sections listed in `stable_sections` are emitted as part of
/// the stable prefix that [`ChatPrompt::cache_boundary`] points at.
/// Within that prefix, messages are ordered by the canonical
/// section order (System → State → Plan → Retrieved → Background);
/// non-stable sections are emitted after the stable prefix so they
/// don't break the cache key.
///
/// Default: `System` and `Background` are considered stable.
/// `Tools` is not the default since "tools" here covers
/// `BlockKind::ToolResult` (tool outputs from prior turns), which
/// are call-dependent and not generally cache-stable. Tool *schemas*
/// would be cache-stable but aren't a distinct section yet.
#[derive(Clone, Debug)]
pub struct CacheOptions {
    pub stable_sections: Vec<SectionKind>,
}

impl Default for CacheOptions {
    fn default() -> Self {
        Self {
            stable_sections: vec![SectionKind::System, SectionKind::Background],
        }
    }
}

/// String-rendering [`Packer`].
pub struct SimplePacker<S: BlockStore> {
    store: Arc<S>,
    tokenizer: Arc<dyn Tokenizer>,
    options: PackerOptions,
}

impl<S: BlockStore> SimplePacker<S> {
    pub fn new(store: Arc<S>, tokenizer: Arc<dyn Tokenizer>) -> Self {
        Self { store, tokenizer, options: PackerOptions::default() }
    }

    /// Replace the packer's options. Builder-style.
    #[must_use]
    pub fn with_options(mut self, options: PackerOptions) -> Self {
        self.options = options;
        self
    }

    /// Convenience: enable [`PackerOptions::include_timestamps`].
    #[must_use]
    pub fn with_timestamps(mut self) -> Self {
        self.options.include_timestamps = true;
        self
    }

    fn now_ms(&self) -> u64 {
        if let Some(ms) = self.options.now_ms {
            return ms;
        }
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX))
    }
}

impl<S: BlockStore> fmt::Debug for SimplePacker<S> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SimplePacker")
            .field("tokenizer", &self.tokenizer.id())
            .finish_non_exhaustive()
    }
}

impl<S: BlockStore + 'static> Packer for SimplePacker<S> {
    #[instrument(skip(self, request, plan), fields(model = %request.model.name))]
    fn pack(&self, request: &PageRequest, plan: &PagePlan) -> Result<PackedPrompt, PackerError> {
        // Load every selected block, grouped by section.
        let mut by_section: HashMap<SectionKind, Vec<ContextBlock>> = HashMap::new();
        for &id in &plan.selected {
            let block = self.store.get(id)?.ok_or_else(|| {
                PackerError::Storage(StoreError::Backend(format!(
                    "selected block {id} missing from store",
                )))
            })?;
            by_section
                .entry(block.kind.default_section())
                .or_default()
                .push(block);
        }
        for blocks in by_section.values_mut() {
            blocks.sort_by_key(|b| b.id);
        }

        let notes = collect_notes(plan);

        let mut rendered = String::new();
        let mut packed_blocks: Vec<PackedBlock> = Vec::new();

        for &section in &SECTION_ORDER {
            match section {
                SectionKind::Slack => (), // Intentional headroom — never emitted.
                SectionKind::Task => {
                    if !request.task.is_empty() {
                        write_header(&mut rendered, section);
                        if self.options.include_timestamps {
                            rendered.push_str("Current time: ");
                            rendered.push_str(&format_iso8601_utc(self.now_ms()));
                            rendered.push_str("\n\n");
                        }
                        rendered.push_str(&request.task);
                        rendered.push_str("\n\n");
                    }
                }
                _ => {
                    let Some(blocks) = by_section.get(&section) else {
                        continue;
                    };
                    if blocks.is_empty() {
                        continue;
                    }
                    write_header(&mut rendered, section);
                    for block in blocks {
                        let body = std::str::from_utf8(&block.bytes).map_err(|e| {
                            PackerError::Storage(StoreError::Backend(format!(
                                "non-utf8 block {}: {e}",
                                block.id,
                            )))
                        })?;
                        if let Some(note) = notes.get(&block.id) {
                            push_note(&mut rendered, note);
                        }
                        if self.options.include_timestamps {
                            rendered.push('[');
                            rendered.push_str(&format_iso8601_utc(block.created_at.0));
                            rendered.push_str("] ");
                        }
                        rendered.push_str(body);
                        rendered.push_str("\n\n");

                        let tokens = self.tokenizer.count(&block.bytes)?;
                        packed_blocks.push(PackedBlock {
                            block_id: block.id,
                            section,
                            tokens,
                            score: 0.0,
                        });
                    }
                }
            }
        }

        let total = self.tokenizer.count(rendered.as_bytes())?;
        let budget = request.model.input_budget();
        if total.0 > budget.0 {
            return Err(PackerError::BudgetExceeded {
                tokens: total.0,
                budget: budget.0,
            });
        }

        Ok(PackedPrompt {
            model: request.model.name.clone(),
            input_tokens: total,
            blocks: packed_blocks,
            rendered,
        })
    }
}

impl<S: BlockStore + 'static> SimplePacker<S> {
    /// Render the same plan as a sequence of role-tagged
    /// [`ChatMessage`]s, suitable for chat-completion APIs.
    ///
    /// Mapping:
    ///
    /// - Each system-side section (System / State / Plan /
    ///   Retrieved / Background) → one `system` message (or `user`
    ///   when the model's `supports_system_role` is false), tagged
    ///   with its [`SectionKind`]. Sections in
    ///   [`CacheOptions::stable_sections`] are emitted first (in
    ///   canonical order); the remaining system-side sections
    ///   follow. This puts the cache-stable prefix at the front of
    ///   the message list.
    /// - Recent (`UserMessage` / `AssistantMessage`) → individual
    ///   role-tagged messages in `BlockId` order.
    /// - Tool results → individual `tool` messages, emitted just
    ///   before the final task message.
    /// - Task → final `user` message containing `request.task`.
    ///
    /// Every message carries its originating
    /// [`ChatMessage::section`] for downstream provider adapters.
    /// `ChatPrompt::cache_boundary` is the index of the last
    /// message belonging to the stable prefix.
    ///
    /// As with [`pack`], the result is verified against
    /// `ModelProfile::input_budget`; over-budget plans return
    /// [`PackerError::BudgetExceeded`].
    #[instrument(skip(self, request, plan), fields(model = %request.model.name))]
    #[allow(clippy::too_many_lines)] // a single linear render pipeline; splitting hurts readability
    pub fn pack_chat(
        &self,
        request: &PageRequest,
        plan: &PagePlan,
    ) -> Result<ChatPrompt, PackerError> {
        let mut by_section: HashMap<SectionKind, Vec<ContextBlock>> = HashMap::new();
        for &id in &plan.selected {
            let block = self.store.get(id)?.ok_or_else(|| {
                PackerError::Storage(StoreError::Backend(format!(
                    "selected block {id} missing from store",
                )))
            })?;
            by_section
                .entry(block.kind.default_section())
                .or_default()
                .push(block);
        }
        for blocks in by_section.values_mut() {
            blocks.sort_by_key(|b| b.id);
        }

        let notes = collect_notes(plan);
        let mut messages: Vec<ChatMessage> = Vec::new();
        let stamps = self.options.include_timestamps;

        // (a) System-side sections, one message per section.
        //     Stable sections first (in canonical order), then the
        //     rest. The stable prefix sits at the front of the
        //     message list so `cache_boundary` is a contiguous
        //     `0..=N` range that a provider adapter can mark as a
        //     cache key.
        const SYSTEM_SIDE: [SectionKind; 5] = [
            SectionKind::System,
            SectionKind::State,
            SectionKind::Plan,
            SectionKind::Retrieved,
            SectionKind::Background,
        ];
        let stable = &self.options.cache.stable_sections;
        let role_for_sys = if request.model.supports_system_role {
            ChatRole::System
        } else {
            ChatRole::User
        };
        let ordered = SYSTEM_SIDE
            .iter()
            .filter(|s| stable.contains(s))
            .chain(SYSTEM_SIDE.iter().filter(|s| !stable.contains(s)));

        let mut stable_message_count: usize = 0;
        let mut still_in_stable_prefix = true;

        for &section in ordered {
            let Some(blocks) = by_section.get(&section) else {
                continue;
            };
            if blocks.is_empty() {
                continue;
            }
            // Preserve the `## SectionName` header inside the
            // message content so a model that ingests multiple
            // system-role messages can still see explicit
            // structure. The header is stable bytes — it doesn't
            // hurt cache hits.
            let mut content = String::new();
            content.push_str("## ");
            content.push_str(section_label(section));
            content.push_str("\n\n");
            for block in blocks {
                if let Some(note) = notes.get(&block.id) {
                    push_note(&mut content, note);
                }
                if stamps {
                    content.push('[');
                    content.push_str(&format_iso8601_utc(block.created_at.0));
                    content.push_str("] ");
                }
                content.push_str(block_text(block)?);
                content.push_str("\n\n");
            }
            messages.push(ChatMessage {
                role: role_for_sys,
                content: content.trim_end().to_string(),
                section: Some(section),
            });
            if still_in_stable_prefix && stable.contains(&section) {
                stable_message_count += 1;
            } else {
                still_in_stable_prefix = false;
            }
        }

        // (b) Recent — preserve user/assistant alternation.
        //     Note: timestamps are NOT prefixed onto assistant
        //     messages even when `include_timestamps` is on. The
        //     model would otherwise see its own past replies
        //     formatted with a `[ISO]` prefix and mimic that prefix
        //     in subsequent outputs. User and tool messages are
        //     fine to prefix because the model treats them as input
        //     context, not as a template for its own voice.
        if let Some(recent) = by_section.get(&SectionKind::Recent) {
            for block in recent {
                let body = block_text(block)?;
                let role = match block.kind {
                    BlockKind::AssistantMessage => ChatRole::Assistant,
                    _ => ChatRole::User,
                };
                let mut content = if stamps && role != ChatRole::Assistant {
                    format!("[{}] {body}", format_iso8601_utc(block.created_at.0))
                } else {
                    body.into()
                };
                if let Some(note) = notes.get(&block.id) {
                    content = format!("[!] {note}\n{content}");
                }
                messages.push(ChatMessage {
                    role,
                    content,
                    section: Some(SectionKind::Recent),
                });
            }
        }

        // (c) Tools — emitted as tool-role messages.
        if let Some(tools) = by_section.get(&SectionKind::Tools) {
            for block in tools {
                let body = block_text(block)?;
                let mut content = if stamps {
                    format!("[{}] {body}", format_iso8601_utc(block.created_at.0))
                } else {
                    body.into()
                };
                if let Some(note) = notes.get(&block.id) {
                    content = format!("[!] {note}\n{content}");
                }
                messages.push(ChatMessage {
                    role: ChatRole::Tool,
                    content,
                    section: Some(SectionKind::Tools),
                });
            }
        }

        // (d) Task — final user message.
        if !request.task.is_empty() {
            let content = if stamps {
                format!(
                    "Current time: {}\n\n{}",
                    format_iso8601_utc(self.now_ms()),
                    request.task,
                )
            } else {
                request.task.clone()
            };
            messages.push(ChatMessage {
                role: ChatRole::User,
                content,
                section: Some(SectionKind::Task),
            });
        }

        // Total tokens across all message contents.
        let mut total = TokenCount::ZERO;
        for m in &messages {
            total = total.saturating_add(self.tokenizer.count(m.content.as_bytes())?);
        }
        let budget = request.model.input_budget();
        if total.0 > budget.0 {
            return Err(PackerError::BudgetExceeded {
                tokens: total.0,
                budget: budget.0,
            });
        }

        let cache_boundary = stable_message_count.checked_sub(1);

        Ok(ChatPrompt {
            model: request.model.name.clone(),
            input_tokens: total,
            messages,
            cache_boundary,
        })
    }
}

/// Format a unix-ms timestamp as ISO 8601 UTC,
/// e.g. `2026-05-05T11:32:00Z`.
///
/// Hand-rolled (Howard Hinnant's days-from-epoch ↔ y/m/d algorithm)
/// so the packer doesn't need a date crate.
fn format_iso8601_utc(ms_since_epoch: u64) -> String {
    let secs = ms_since_epoch / 1000;
    let days = secs / 86_400;
    let s_of_day = secs % 86_400;
    let hour = s_of_day / 3600;
    let minute = (s_of_day / 60) % 60;
    let second = s_of_day % 60;

    // Days since 1970-01-01 → (year, month, day) via Hinnant's algorithm.
    let z = days + 719_468;
    let era = z / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let (month, year) = if mp < 10 { (mp + 3, y) } else { (mp - 9, y + 1) };

    format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z")
}

/// Map of block id → inline note set by the pager (e.g. an
/// edge-reconciliation `Contradicts` flag). Empty when the plan
/// carries no `selections` (the `pack_with_plan` re-render path) or no
/// block was annotated.
fn collect_notes(plan: &PagePlan) -> HashMap<BlockId, &str> {
    plan.selections
        .iter()
        .filter_map(|s| s.note.as_deref().map(|n| (s.block_id, n)))
        .collect()
}

/// Render an inline note marker on its own line.
fn push_note(buf: &mut String, note: &str) {
    buf.push_str("[!] ");
    buf.push_str(note);
    buf.push('\n');
}

fn block_text(block: &ContextBlock) -> Result<&str, PackerError> {
    std::str::from_utf8(&block.bytes).map_err(|e| {
        PackerError::Storage(StoreError::Backend(format!(
            "non-utf8 block {}: {e}",
            block.id,
        )))
    })
}

fn write_header(buf: &mut String, section: SectionKind) {
    buf.push_str("## ");
    buf.push_str(section_label(section));
    buf.push_str("\n\n");
}

const fn section_label(s: SectionKind) -> &'static str {
    match s {
        SectionKind::System => "System",
        SectionKind::Task => "Task",
        SectionKind::State => "State",
        SectionKind::Plan => "Plan",
        SectionKind::Retrieved => "Retrieved",
        SectionKind::Tools => "Tools",
        SectionKind::Recent => "Recent",
        SectionKind::Background => "Background",
        SectionKind::Slack => "Slack",
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use llm386_core::{
        BlockId, ContentHash, ContextBlock, ModelProfile, PagePlan, PageRequest, Pager, Provenance,
        SessionId, Timestamp, TokenCount, TokenCounts, Tokenizer, TokenizerId,
    };
    use llm386_store_lmdb::{LmdbStore, StoreConfig};
    use llm386_tokenizer::cl100k_base;
    use tempfile::TempDir;

    use super::*;

    // macOS returns `LMDB: Invalid argument (os error 22)` once too
    // many LMDB envs are mmap-active at once across the process — a
    // platform concurrency ceiling, not a logic bug. Cap how many test
    // stores are live simultaneously. The permit rides on the guard
    // bound in each test's `_dir` slot, so no test body changes.
    struct Semaphore {
        permits: std::sync::Mutex<usize>,
        avail: std::sync::Condvar,
    }
    impl Semaphore {
        const fn new(n: usize) -> Self {
            Self {
                permits: std::sync::Mutex::new(n),
                avail: std::sync::Condvar::new(),
            }
        }
        fn acquire(&self) {
            let mut n = self.permits.lock().unwrap();
            while *n == 0 {
                n = self.avail.wait(n).unwrap();
            }
            *n -= 1;
        }
        fn release(&self) {
            *self.permits.lock().unwrap() += 1;
            self.avail.notify_one();
        }
    }
    static STORE_SEM: Semaphore = Semaphore::new(6);

    struct TestEnv {
        _dir: TempDir,
    }
    impl Drop for TestEnv {
        fn drop(&mut self) {
            STORE_SEM.release();
        }
    }

    fn setup() -> (Arc<LmdbStore>, TestEnv, Arc<dyn Tokenizer>) {
        STORE_SEM.acquire();
        let dir = TempDir::new().unwrap();
        let store = Arc::new(LmdbStore::open(dir.path(), StoreConfig::default()).unwrap());
        let tok: Arc<dyn Tokenizer> = Arc::new(cl100k_base().unwrap());
        (store, TestEnv { _dir: dir }, tok)
    }

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

    fn profile(max: u32) -> ModelProfile {
        ModelProfile {
            name: "test".into(),
            max_context_tokens: max,
            reserved_output_tokens: 0,
            safety_margin_tokens: 0,
            tokenizer: TokenizerId::new("cl100k_base"),
            family: None,
            supports_system_role: true,
            supports_tools: true,
        }
    }

    #[test]
    fn empty_plan_renders_only_task() {
        let (store, _dir, tok) = setup();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: SessionId(1),
            task: "answer the user".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let prompt = packer.pack(&request, &plan).unwrap();
        assert!(prompt.rendered.contains("## Task"));
        assert!(prompt.rendered.contains("answer the user"));
        assert!(prompt.blocks.is_empty());
        assert!(prompt.input_tokens.0 > 0);
    }

    #[test]
    fn plan_renders_sections_in_canonical_order() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"SYS", BlockKind::System, 1, 1))
            .unwrap();
        let user_id = store
            .put(session, make_block(b"USR", BlockKind::UserMessage, 2, 2))
            .unwrap();
        let plan_id = store
            .put(session, make_block(b"PLN", BlockKind::Plan, 3, 3))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "TASKBODY".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            // Deliberately in wrong order — packer must canonicalize.
            selected: vec![plan_id, user_id, sys_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let prompt = packer.pack(&request, &plan).unwrap();
        let sys_pos = prompt.rendered.find("SYS").unwrap();
        let task_pos = prompt.rendered.find("TASKBODY").unwrap();
        let plan_pos = prompt.rendered.find("PLN").unwrap();
        let user_pos = prompt.rendered.find("USR").unwrap();
        // System < Task < Plan < Recent
        assert!(sys_pos < task_pos);
        assert!(task_pos < plan_pos);
        assert!(plan_pos < user_pos);
    }

    #[test]
    fn pack_is_deterministic() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let a = store
            .put(session, make_block(b"AAA", BlockKind::Fact, 1, 1))
            .unwrap();
        let b = store
            .put(session, make_block(b"BBB", BlockKind::Fact, 2, 2))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "t".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![b, a],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let p1 = packer.pack(&request, &plan).unwrap();
        let p2 = packer.pack(&request, &plan).unwrap();
        assert_eq!(p1.rendered, p2.rendered);
        assert_eq!(p1.input_tokens, p2.input_tokens);
    }

    #[test]
    fn budget_exceeded_returns_error() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        // Build a plan that's way larger than the budget.
        let big = "lorem ipsum dolor sit amet ".repeat(200);
        let id = store
            .put(session, make_block(big.as_bytes(), BlockKind::Fact, 1, 1))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "t".into(),
            model: profile(10), // tiny budget
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let err = packer.pack(&request, &plan).unwrap_err();
        assert!(matches!(err, PackerError::BudgetExceeded { .. }));
    }

    #[test]
    fn end_to_end_with_pager() {
        use llm386_pager::GreedyPager;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        store
            .put(session, make_block(b"system rule", BlockKind::System, 1, 1))
            .unwrap();
        store
            .put(
                session,
                make_block(b"hello there", BlockKind::UserMessage, 2, 2),
            )
            .unwrap();
        store
            .put(
                session,
                make_block(b"sure thing", BlockKind::AssistantMessage, 3, 3),
            )
            .unwrap();
        let pager = GreedyPager::new(store.clone(), tok.clone());
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "summarize the conversation".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = pager.page(request.clone()).unwrap();
        let prompt = packer.pack(&request, &plan).unwrap();
        assert!(prompt.rendered.contains("system rule"));
        assert!(prompt.rendered.contains("hello there"));
        assert!(prompt.rendered.contains("summarize the conversation"));
        assert!(prompt.input_tokens.0 <= request.model.input_budget().0);
    }

    #[test]
    fn pack_chat_emits_role_tagged_messages() {
        use llm386_core::ChatRole;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"be brief", BlockKind::System, 1, 1))
            .unwrap();
        let user_id = store
            .put(session, make_block(b"hello?", BlockKind::UserMessage, 2, 2))
            .unwrap();
        let asst_id = store
            .put(
                session,
                make_block(b"hi there", BlockKind::AssistantMessage, 3, 3),
            )
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "what's next?".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![sys_id, user_id, asst_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // System (with `be brief`), then user, assistant, then task user.
        let roles: Vec<ChatRole> = chat.messages.iter().map(|m| m.role).collect();
        assert_eq!(
            roles,
            vec![
                ChatRole::System,
                ChatRole::User,
                ChatRole::Assistant,
                ChatRole::User
            ],
        );
        assert!(chat.messages[0].content.contains("be brief"));
        assert_eq!(chat.messages[1].content, "hello?");
        assert_eq!(chat.messages[2].content, "hi there");
        assert_eq!(chat.messages[3].content, "what's next?");
        assert!(chat.input_tokens.0 > 0);
    }

    #[test]
    fn pack_chat_folds_system_into_user_when_unsupported() {
        use llm386_core::ChatRole;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"be brief", BlockKind::System, 1, 1))
            .unwrap();
        let mut p = profile(1_000);
        p.supports_system_role = false;
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "go".into(),
            model: p,
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![sys_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // No system role; first message is user with the system content.
        assert!(chat.messages.iter().all(|m| m.role != ChatRole::System));
        assert_eq!(chat.messages[0].role, ChatRole::User);
        assert!(chat.messages[0].content.contains("be brief"));
    }

    #[test]
    fn iso8601_formatter_known_values() {
        assert_eq!(format_iso8601_utc(0), "1970-01-01T00:00:00Z");
        assert_eq!(format_iso8601_utc(1_000), "1970-01-01T00:00:01Z");
        assert_eq!(format_iso8601_utc(86_400_000), "1970-01-02T00:00:00Z");
        // 2000-01-01T00:00:00Z = 946684800 unix seconds.
        assert_eq!(format_iso8601_utc(946_684_800_000), "2000-01-01T00:00:00Z");
        // 2024-01-01T00:00:00Z, immediately after a leap day on
        // 2020-02-29 — exercises the era/leap arithmetic.
        assert_eq!(format_iso8601_utc(1_704_067_200_000), "2024-01-01T00:00:00Z");
        // 2024-02-29T00:00:00Z — the leap day itself.
        assert_eq!(format_iso8601_utc(1_709_164_800_000), "2024-02-29T00:00:00Z");
    }

    #[test]
    fn pack_with_timestamps_prefixes_blocks_and_anchors_now() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let fact_id = store
            .put(session, make_block(b"important fact", BlockKind::Fact, 1_500_000_000_000, 1))
            .unwrap();
        let packer = SimplePacker::new(store, tok).with_options(PackerOptions {
            include_timestamps: true,
            now_ms: Some(1_700_000_000_000),
            cache: CacheOptions::default(),
        });
        let request = PageRequest {
            session_id: session,
            task: "what did we discuss?".into(),
            model: profile(10_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![fact_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let prompt = packer.pack(&request, &plan).unwrap();
        // Per-block timestamp prefix.
        assert!(
            prompt.rendered.contains("[2017-07-14T02:40:00Z] important fact"),
            "missing block timestamp: {}",
            prompt.rendered,
        );
        // Current-time anchor in the Task section.
        assert!(
            prompt.rendered.contains("Current time: 2023-11-14T22:13:20Z"),
            "missing current-time anchor: {}",
            prompt.rendered,
        );
    }

    #[test]
    fn pack_without_timestamps_unchanged() {
        // Default behavior must not emit timestamp prefixes.
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let fact_id = store
            .put(session, make_block(b"plain content", BlockKind::Fact, 1_500_000_000_000, 1))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "go".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![fact_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let prompt = packer.pack(&request, &plan).unwrap();
        assert!(!prompt.rendered.contains("Current time:"));
        assert!(!prompt.rendered.contains("Z]"));
        assert!(prompt.rendered.contains("plain content"));
    }

    #[test]
    fn pack_chat_with_timestamps_prefixes_user_and_tool_only() {
        // Past assistant messages must NOT be prefixed even with
        // timestamps on, to avoid the model mimicking the prefix
        // in its own voice. User and tool messages are fine to
        // prefix because the model treats them as input context.
        use llm386_core::ChatRole;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let user_id = store
            .put(
                session,
                make_block(b"hello?", BlockKind::UserMessage, 1_500_000_000_000, 2),
            )
            .unwrap();
        let asst_id = store
            .put(
                session,
                make_block(b"hi", BlockKind::AssistantMessage, 1_500_000_001_000, 3),
            )
            .unwrap();
        let tool_id = store
            .put(
                session,
                make_block(
                    b"{\"result\": 42}",
                    BlockKind::ToolResult,
                    1_500_000_002_000,
                    4,
                ),
            )
            .unwrap();
        let packer = SimplePacker::new(store, tok).with_options(PackerOptions {
            include_timestamps: true,
            now_ms: Some(1_700_000_000_000),
            cache: CacheOptions::default(),
        });
        let request = PageRequest {
            session_id: session,
            task: "next?".into(),
            model: profile(10_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![user_id, asst_id, tool_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        let by_role: Vec<(ChatRole, String)> = chat
            .messages
            .iter()
            .map(|m| (m.role, m.content.clone()))
            .collect();

        // User Recent message: prefixed.
        assert!(
            by_role.iter().any(|(r, c)| {
                *r == ChatRole::User && c.starts_with("[2017-07-14T02:40:00Z] hello?")
            }),
            "user message missing timestamp prefix: {by_role:?}",
        );
        // Assistant Recent message: NOT prefixed — the assistant
        // message has the bare body so the model never sees its own
        // voice with a `[ISO]` prefix to copy.
        let asst_msg = chat
            .messages
            .iter()
            .find(|m| m.role == ChatRole::Assistant)
            .expect("assistant message present");
        assert_eq!(asst_msg.content, "hi");
        assert!(!asst_msg.content.starts_with('['));
        // Tool message: prefixed.
        let tool_msg = chat
            .messages
            .iter()
            .find(|m| m.role == ChatRole::Tool)
            .expect("tool message present");
        assert!(tool_msg.content.starts_with("[2017-07-14T02:40:02Z]"));

        // The final task user message has the current-time anchor.
        let task_msg = chat.messages.last().unwrap();
        assert_eq!(task_msg.role, ChatRole::User);
        assert!(task_msg.content.starts_with("Current time: 2023-11-14T22:13:20Z"));
        assert!(task_msg.content.contains("next?"));
    }

    #[test]
    fn pack_chat_emits_tool_role_for_tool_results() {
        use llm386_core::ChatRole;
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let tool_id = store
            .put(
                session,
                make_block(b"{\"result\": 42}", BlockKind::ToolResult, 1, 1),
            )
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "x".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![tool_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        let tool_msg = chat
            .messages
            .iter()
            .find(|m| m.role == ChatRole::Tool)
            .unwrap();
        assert_eq!(tool_msg.content, "{\"result\": 42}");
    }

    #[test]
    fn pack_chat_splits_system_side_sections_into_separate_messages() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"SYS-CONTENT", BlockKind::System, 1, 1))
            .unwrap();
        let state_id = store
            .put(session, make_block(b"STATE-CONTENT", BlockKind::State, 2, 2))
            .unwrap();
        let bg_id = store
            .put(session, make_block(b"BG-CONTENT", BlockKind::DocumentChunk, 3, 3))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "t".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![sys_id, state_id, bg_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // No section's content is consolidated with another's:
        // each system-side section produces its own message
        // containing only that section's body.
        let by_section: std::collections::HashMap<_, _> = chat
            .messages
            .iter()
            .filter_map(|m| m.section.map(|s| (s, m.content.clone())))
            .collect();
        let sys = by_section.get(&SectionKind::System).expect("system message");
        assert!(sys.contains("SYS-CONTENT"));
        assert!(!sys.contains("STATE-CONTENT") && !sys.contains("BG-CONTENT"));
        let state = by_section.get(&SectionKind::State).expect("state message");
        assert!(state.contains("STATE-CONTENT"));
        assert!(!state.contains("SYS-CONTENT") && !state.contains("BG-CONTENT"));
        let bg = by_section
            .get(&SectionKind::Background)
            .expect("background message");
        assert!(bg.contains("BG-CONTENT"));
        assert!(!bg.contains("SYS-CONTENT") && !bg.contains("STATE-CONTENT"));
        // Every message carries its section tag.
        assert!(chat.messages.iter().all(|m| m.section.is_some()));
    }

    #[test]
    fn pack_chat_cache_boundary_covers_default_stable_sections() {
        // Default stable_sections = [System, Background]. With both
        // sections populated plus a State block in between, the
        // boundary should point at the 2nd message (System then
        // Background, then State after the boundary).
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"sys", BlockKind::System, 1, 1))
            .unwrap();
        let state_id = store
            .put(session, make_block(b"state", BlockKind::State, 2, 2))
            .unwrap();
        let bg_id = store
            .put(session, make_block(b"bg", BlockKind::DocumentChunk, 3, 3))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "go".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![sys_id, state_id, bg_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // Stable prefix is the first 2 messages (System then Background).
        assert_eq!(chat.cache_boundary, Some(1));
        assert_eq!(
            chat.messages[0].section,
            Some(llm386_core::SectionKind::System),
        );
        assert_eq!(
            chat.messages[1].section,
            Some(llm386_core::SectionKind::Background),
        );
        // Unstable sections (State) follow the stable prefix.
        assert_eq!(
            chat.messages[2].section,
            Some(llm386_core::SectionKind::State),
        );
    }

    #[test]
    fn pack_chat_cache_boundary_is_none_when_no_stable_sections_present() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        // Only State + Recent — neither is in the default stable set.
        let state_id = store
            .put(session, make_block(b"state", BlockKind::State, 1, 1))
            .unwrap();
        let user_id = store
            .put(session, make_block(b"hi", BlockKind::UserMessage, 2, 2))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "t".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![state_id, user_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        assert_eq!(chat.cache_boundary, None);
    }

    #[test]
    fn pack_chat_cache_boundary_respects_custom_stable_sections() {
        // Mark only State as stable; System and Background should
        // land *after* the stable prefix.
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let sys_id = store
            .put(session, make_block(b"sys", BlockKind::System, 1, 1))
            .unwrap();
        let state_id = store
            .put(session, make_block(b"state", BlockKind::State, 2, 2))
            .unwrap();
        let packer = SimplePacker::new(store, tok).with_options(PackerOptions {
            include_timestamps: false,
            now_ms: None,
            cache: CacheOptions {
                stable_sections: vec![SectionKind::State],
            },
        });
        let request = PageRequest {
            session_id: session,
            task: "t".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![sys_id, state_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // State first (stable), then System (unstable in this config).
        assert_eq!(chat.cache_boundary, Some(0));
        assert_eq!(
            chat.messages[0].section,
            Some(llm386_core::SectionKind::State),
        );
        assert_eq!(
            chat.messages[1].section,
            Some(llm386_core::SectionKind::System),
        );
    }

    #[test]
    fn pack_chat_recent_tools_task_carry_section_tags() {
        let (store, _dir, tok) = setup();
        let session = SessionId(1);
        let user_id = store
            .put(session, make_block(b"hi", BlockKind::UserMessage, 1, 1))
            .unwrap();
        let asst_id = store
            .put(session, make_block(b"hey", BlockKind::AssistantMessage, 2, 2))
            .unwrap();
        let tool_id = store
            .put(session, make_block(b"{}", BlockKind::ToolResult, 3, 3))
            .unwrap();
        let packer = SimplePacker::new(store, tok);
        let request = PageRequest {
            session_id: session,
            task: "next?".into(),
            model: profile(1_000),
            required_blocks: vec![],
        };
        let plan = PagePlan {
            selected: vec![user_id, asst_id, tool_id],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount::ZERO,
        };
        let chat = packer.pack_chat(&request, &plan).unwrap();
        // Recent messages tagged as Recent (regardless of role).
        let recent_count = chat
            .messages
            .iter()
            .filter(|m| m.section == Some(SectionKind::Recent))
            .count();
        assert_eq!(recent_count, 2);
        // Tool message tagged as Tools.
        assert!(
            chat.messages
                .iter()
                .any(|m| m.section == Some(SectionKind::Tools)),
        );
        // Final message tagged as Task.
        assert_eq!(
            chat.messages.last().unwrap().section,
            Some(SectionKind::Task),
        );
    }

    proptest::proptest! {
        #![proptest_config(proptest::test_runner::Config { cases: 12, ..proptest::test_runner::Config::default() })]

        /// Pack must be byte-deterministic across two calls with the
        /// same store state, request, and plan — this is the property
        /// the trace layer relies on for replay.
        #[test]
        fn pack_is_byte_deterministic(
            n_blocks in 0u64..10,
            task_seed in 0u32..100,
        ) {
            use llm386_pager::GreedyPager;
            let (store, _dir, tok) = setup();
            let session = SessionId(1);
            for i in 0..n_blocks {
                let bytes = format!("p{task_seed}/{i} content");
                store
                    .put(session, make_block(bytes.as_bytes(), BlockKind::Fact, i, u128::from(i)))
                    .unwrap();
            }
            let pager = GreedyPager::new(store.clone(), tok.clone());
            let packer = SimplePacker::new(store, tok);
            let request = PageRequest {
                session_id: session,
                task: format!("task {task_seed}"),
                model: profile(10_000),
                required_blocks: vec![],
            };
            let plan = pager.page(request.clone()).unwrap();
            let a = packer.pack(&request, &plan).unwrap();
            let b = packer.pack(&request, &plan).unwrap();
            proptest::prop_assert_eq!(a.rendered, b.rendered);
            proptest::prop_assert_eq!(a.input_tokens, b.input_tokens);
        }

        /// If pack succeeds, its rendered prompt must fit within the
        /// model's input budget. (Covers the BudgetExceeded short-
        /// circuit and the surrounding accounting.)
        #[test]
        fn successful_pack_fits_budget(
            n_blocks in 0u64..10,
            budget in 80u32..2_000,
        ) {
            use llm386_pager::GreedyPager;
            let (store, _dir, tok) = setup();
            let session = SessionId(1);
            for i in 0..n_blocks {
                let bytes = format!("block {i} content");
                store
                    .put(session, make_block(bytes.as_bytes(), BlockKind::Fact, i, u128::from(i)))
                    .unwrap();
            }
            let pager = GreedyPager::new(store.clone(), tok.clone());
            let packer = SimplePacker::new(store, tok);
            let request = PageRequest {
                session_id: session,
                task: "task".into(),
                model: profile(budget),
                required_blocks: vec![],
            };
            let plan = pager.page(request.clone()).unwrap();
            if let Ok(prompt) = packer.pack(&request, &plan) {
                proptest::prop_assert!(
                    prompt.input_tokens.0 <= budget,
                    "input_tokens={} > budget={}",
                    prompt.input_tokens.0,
                    budget,
                );
            }
        }
    }
}
