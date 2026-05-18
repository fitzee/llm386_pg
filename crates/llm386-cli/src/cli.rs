//! Clap argument definitions for `llm386`.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};

#[derive(Parser, Debug)]
#[command(
    name = "llm386",
    version,
    about = "LLM386 — context virtualization runtime"
)]
pub(crate) struct Cli {
    /// Optional path to a TOML file with `[[profile]]`,
    /// `[[hf_tokenizer]]`, `[[retriever]]`, `[store]`, `[packer]`,
    /// `[cache]`, and `[section_budgets]` entries. May also be set
    /// via the `LLM386_PROFILES` environment variable; the flag wins
    /// when both are present.
    #[arg(long, global = true)]
    pub(crate) profiles: Option<PathBuf>,

    /// LMDB block-store path. Overrides `path` in the TOML
    /// `[store]` section. Mutually exclusive with `--pg-url`.
    #[arg(long, global = true, conflicts_with = "pg_url")]
    pub(crate) store: Option<PathBuf>,

    /// PostgreSQL connection URL for the block store. Overrides
    /// `url` in the TOML `[store]` section. Mutually exclusive with
    /// `--store`.
    #[arg(long, global = true)]
    pub(crate) pg_url: Option<String>,

    #[command(subcommand)]
    pub(crate) command: Command,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Command {
    /// Create (or open) an LMDB store at the given path.
    /// LMDB-only — Postgres bootstraps its schema on first connect.
    Init {
        /// Path to the store directory.
        path: PathBuf,
    },

    /// Insert a block from a file (or `-` for stdin).
    Put {
        /// Session id (decimal, or `0x`-prefixed hex).
        #[arg(long, value_parser = parse_u128)]
        session: u128,
        /// Kind of block.
        #[arg(long, value_enum)]
        kind: KindArg,
        /// Priority in [0.0, 1.0]; higher is preferred during paging.
        #[arg(long, default_value_t = 0.0)]
        priority: f32,
        /// File to read; `-` for stdin.
        file: PathBuf,
    },

    /// List built-in model profiles.
    ListModels,

    /// Run the pager and print the resulting plan.
    Page {
        #[arg(long, value_parser = parse_u128)]
        session: u128,
        /// Built-in model profile name (see `list-models`).
        #[arg(long)]
        model: String,
        #[arg(long)]
        task: String,
        /// Print the PagePlan as JSON instead of the human table.
        #[arg(long)]
        json: bool,
    },

    /// Run page + pack and print the resulting prompt.
    Pack {
        #[arg(long, value_parser = parse_u128)]
        session: u128,
        #[arg(long)]
        model: String,
        #[arg(long)]
        task: String,
        /// Print only the rendered prompt (no header / manifest).
        #[arg(long)]
        prompt_only: bool,
        /// Render as a JSON list of role-tagged chat messages instead
        /// of a single string (suitable for chat-completion APIs).
        #[arg(long, conflicts_with = "prompt_only")]
        chat: bool,
        /// Prepend each rendered block with its ISO 8601 UTC creation
        /// timestamp and emit a "Current time" anchor in the Task
        /// section, so the model can reason about when things
        /// happened. Adds ~12 tokens per block. Overrides any
        /// `[packer]` setting in the config file.
        #[arg(long)]
        timestamps: bool,
        /// Optional trace store path. When set, the call is recorded
        /// and its CallId is printed on stderr.
        #[arg(long)]
        trace: Option<PathBuf>,
    },

    /// Inspect persisted traces.
    #[command(subcommand)]
    Trace(TraceSub),

    /// List every session id with at least one block in the store.
    ListSessions,

    /// Read-only integrity check: walks every block, recomputes
    /// content hashes, and verifies the indexes are consistent.
    /// LMDB-only.
    Verify,

    /// Rebuilds derivable indexes (hash index) from the primary
    /// table and removes orphan session entries. Destructive —
    /// requires `--yes`. LMDB-only.
    Repair {
        /// Required confirmation flag.
        #[arg(long)]
        yes: bool,
    },

    /// Delete a single block or every block in a session.
    /// Destructive — requires `--yes`.
    Purge {
        /// Block id (decimal, hex with `0x`, or bare 32-char hex).
        /// Mutually exclusive with --session.
        #[arg(long, value_parser = parse_u128, conflicts_with = "session")]
        block: Option<u128>,
        /// Session id. Removes every block belonging to this session.
        /// Mutually exclusive with --block.
        #[arg(long, value_parser = parse_u128, conflicts_with = "block")]
        session: Option<u128>,
        /// Required confirmation flag for destructive operations.
        #[arg(long)]
        yes: bool,
    },

    /// Print the full contents of a single block by id.
    Show {
        /// Block id (decimal, hex with `0x`, or bare 32-char hex).
        #[arg(value_parser = parse_u128)]
        id: u128,
        /// Print the ContextBlock as JSON instead of the human view.
        #[arg(long)]
        json: bool,
    },

    /// Add a typed directed edge between two blocks.
    AddEdge {
        /// Source block id (decimal, hex with `0x`, or bare 32-char hex).
        #[arg(long, value_parser = parse_u128)]
        from: u128,
        /// Destination block id.
        #[arg(long, value_parser = parse_u128)]
        to: u128,
        /// Edge kind.
        #[arg(long, value_enum)]
        kind: EdgeKindArg,
    },

    /// List edges incident to a block (outgoing by default).
    Edges {
        /// Block id.
        #[arg(value_parser = parse_u128)]
        id: u128,
        /// Show incoming edges instead of outgoing.
        #[arg(long)]
        incoming: bool,
    },

    /// Summarize a session's blocks via the configured summarizer.
    Summarize {
        /// Session id (decimal, hex with `0x`, or bare 32-char hex).
        #[arg(long, value_parser = parse_u128)]
        session: u128,
        /// Which summarizer to use.
        #[arg(long, value_enum, default_value_t = SummarizerArg::Truncating)]
        summarizer: SummarizerArg,
        /// For TruncatingSummarizer: max characters per block bullet.
        #[arg(long, default_value_t = 80)]
        max_chars: usize,
        /// Only summarize the most recent N blocks (default: all).
        #[arg(long)]
        last: Option<usize>,
        /// Also persist the summary as a new Summary block whose
        /// Provenance.parents reference the originals.
        #[arg(long)]
        store_summary: bool,
        /// For AnthropicSummarizer: model id (default `claude-haiku-4-5`).
        #[arg(long)]
        anthropic_model: Option<String>,
        /// For AnthropicSummarizer: response token cap (default 1024).
        #[arg(long)]
        anthropic_max_tokens: Option<u32>,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum TraceSub {
    /// Show a single trace record by CallId.
    Show {
        /// Path to the trace store (distinct from the block store).
        #[arg(long = "trace-store")]
        store: PathBuf,
        /// Call id (decimal, or `0x`-prefixed hex).
        #[arg(value_parser = parse_u128)]
        call_id: u128,
    },

    /// Diff two trace records' page plans. Prints which blocks were
    /// added, removed, or kept (and which kept-block selection
    /// reasons changed) plus the input-token delta.
    Diff {
        /// Path to the trace store (distinct from the block store).
        #[arg(long = "trace-store")]
        store: PathBuf,
        /// Older / baseline call id.
        #[arg(value_parser = parse_u128)]
        prev: u128,
        /// Newer call id.
        #[arg(value_parser = parse_u128)]
        next: u128,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum SummarizerArg {
    Noop,
    Truncating,
    /// Anthropic Claude (requires `ANTHROPIC_API_KEY`).
    Anthropic,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum EdgeKindArg {
    Parent,
    DerivedFrom,
    Supports,
    Contradicts,
    ToolInvocation,
}

impl From<EdgeKindArg> for llm386_core::EdgeKind {
    fn from(k: EdgeKindArg) -> Self {
        match k {
            EdgeKindArg::Parent => Self::Parent,
            EdgeKindArg::DerivedFrom => Self::DerivedFrom,
            EdgeKindArg::Supports => Self::Supports,
            EdgeKindArg::Contradicts => Self::Contradicts,
            EdgeKindArg::ToolInvocation => Self::ToolInvocation,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub(crate) enum KindArg {
    System,
    UserMessage,
    AssistantMessage,
    ToolResult,
    Summary,
    Fact,
    DocumentChunk,
    Plan,
    State,
    Trace,
}

impl From<KindArg> for llm386_core::BlockKind {
    fn from(k: KindArg) -> Self {
        match k {
            KindArg::System => Self::System,
            KindArg::UserMessage => Self::UserMessage,
            KindArg::AssistantMessage => Self::AssistantMessage,
            KindArg::ToolResult => Self::ToolResult,
            KindArg::Summary => Self::Summary,
            KindArg::Fact => Self::Fact,
            KindArg::DocumentChunk => Self::DocumentChunk,
            KindArg::Plan => Self::Plan,
            KindArg::State => Self::State,
            KindArg::Trace => Self::Trace,
        }
    }
}

fn parse_u128(s: &str) -> Result<u128, String> {
    if let Some(hex) = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        u128::from_str_radix(hex, 16).map_err(|e| e.to_string())
    } else if s.len() == 32 && s.chars().all(|c| c.is_ascii_hexdigit()) {
        // Bare 32-char hex matches the BlockId/SessionId/CallId Display
        // form, so accept it without requiring the `0x` prefix.
        u128::from_str_radix(s, 16).map_err(|e| e.to_string())
    } else {
        s.parse::<u128>().map_err(|e| e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_u128_decimal_and_hex() {
        assert_eq!(parse_u128("42").unwrap(), 42);
        assert_eq!(parse_u128("0xff").unwrap(), 255);
        assert_eq!(parse_u128("0XFF").unwrap(), 255);
        assert!(parse_u128("not-a-number").is_err());
    }

    #[test]
    fn parse_u128_accepts_bare_32_char_hex() {
        let hex = "7b732fd4d8b1f1b734909ba162113e76";
        assert_eq!(
            parse_u128(hex).unwrap(),
            u128::from_str_radix(hex, 16).unwrap()
        );
    }
}
