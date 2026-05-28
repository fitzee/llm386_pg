//! `llm386-pager` — working-set selection for LLM386.
//!
//! Picks the subset of stored blocks that fits within a model's input
//! budget for a given session and task. The first cut is a recency-
//! weighted greedy pager; section budgets and richer scoring will
//! land in follow-on phases.

#![doc(html_root_url = "https://docs.rs/llm386-pager/1.0.0-alpha")]

mod budget;
mod edges;
mod greedy;
mod retrievers;

pub use budget::{SectionAllocation, SectionBudgetTable};
pub use edges::{
    ContradictMode, DerivedMode, EdgePolicy, MAX_DEPTH_CEILING, ParentMode, SupportsMode, ToolMode,
};
pub use greedy::{GreedyPager, ScoringPolicy};
pub use retrievers::{
    Bm25Retriever, LexicalRetriever, PinnedRetriever, RecencyRetriever, SessionRetriever,
};
