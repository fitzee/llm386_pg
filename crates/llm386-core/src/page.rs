//! `PageRequest` / `PagePlan` — the pager's input and output.

use serde::{Deserialize, Serialize};

use crate::ids::{BlockId, SessionId, TokenCount};
use crate::model::ModelProfile;

/// Request to the pager — what to assemble context for.
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct PageRequest {
    pub session_id: SessionId,
    pub task: String,
    pub model: ModelProfile,
    /// Block ids that must appear in the plan.
    pub required_blocks: Vec<BlockId>,
}

/// The pager's decision: which blocks were selected, which were
/// omitted, and the aggregate token estimate.
///
/// `selected` and `selections` are kept in lockstep — `selected[i]`
/// is the same block as `selections[i].block_id`. The plain id list
/// is preserved for backward compatibility; new consumers should
/// prefer `selections`, which carries the per-block score and the
/// reason the pager included it.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PagePlan {
    pub selected: Vec<BlockId>,
    #[serde(default)]
    pub selections: Vec<Selection>,
    pub omitted: Vec<OmittedBlock>,
    pub estimated_tokens: TokenCount,
}

/// One selected block plus the score and reason the pager used.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct Selection {
    pub block_id: BlockId,
    pub score: f32,
    pub reason: SelectionReason,
    /// Optional human-readable annotation the packer renders inline
    /// with the block. Set by edge-aware reconciliation — e.g. a
    /// `Contradicts` edge in `Flag` mode tags the demoted block with
    /// `"contradicted by newer block <id>"`. `None` for the common
    /// case. Defaulted on deserialize so older serialized plans load.
    #[serde(default)]
    pub note: Option<String>,
}

/// Why the pager included a particular block in the working set.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum SelectionReason {
    /// Listed in `PageRequest::required_blocks`.
    Pinned,
    /// Surfaced by a retriever and ranked into a section budget.
    HighRelevance,
    /// Selected by recency-based scoring.
    Recency,
    /// Pulled in as an ancestor of an already-selected block via
    /// edge-aware inclusion.
    Dependency,
    /// Surfaced by a retriever explicitly tagged as global facts.
    GlobalFact,
    /// Tool result paired with an assistant message that called it.
    ToolResult,
}

/// A candidate block that the pager considered but did not include.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct OmittedBlock {
    pub block_id: BlockId,
    pub reason: OmissionReason,
    pub score: f32,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Serialize, Deserialize)]
pub enum OmissionReason {
    /// Section budget was full when the block was considered.
    Budget,
    /// Block was older than the configured staleness threshold.
    Stale,
    /// Block was redundant with another already in the plan.
    Redundant,
    /// Block scored below the configured threshold.
    LowScore,
    /// Block was filtered out by kind / labels.
    FilteredByKind,
    /// Block depended on another block that was not included.
    DependencyMissing,
    /// Block did not fit but a Summary block referencing it was
    /// included in its place.
    Compressed,
    /// Block was dropped because a `Contradicts` edge linked it to a
    /// newer / higher-priority block that the pager kept instead
    /// (edge policy `Contradicts = PreferNewer`).
    Contradicted,
    /// Block was dropped because a `DerivedFrom` edge marked it as the
    /// source of a derivative (summary / distillation) that was
    /// already selected (edge policy `DerivedFrom = SuppressSource`).
    SupersededByDerived,
}
