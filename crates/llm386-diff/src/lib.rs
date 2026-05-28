//! `llm386-diff` — structured diffs between two paged contexts.
//!
//! The runtime makes the working set computed and inspectable, but
//! you usually want to know not just *what* the working set is on
//! turn N, but *what changed* between turn N-1 and turn N. That's
//! the question this crate answers.
//!
//! Inputs are two [`PagePlan`]s (or two [`TraceRecord`]s, via the
//! convenience wrapper). Output is a [`PromptDiff`]: which blocks
//! were added, which were dropped, which stayed put, and how the
//! input-token estimate moved.
//!
//! The diff operates on `BlockId`s and the per-block
//! [`SelectionReason`]s recorded in `PagePlan.selections`, so a
//! "kept" block whose inclusion reason changed (e.g. promoted from
//! `HighRelevance` to `Pinned`) shows up with both reasons.

#![doc(html_root_url = "https://docs.rs/llm386-diff/1.0.0-alpha")]

use std::collections::HashMap;

use llm386_core::{BlockId, PagePlan, SelectionReason, TokenCount, TraceRecord};
use serde::{Deserialize, Serialize};

/// Per-block entry in a [`PromptDiff`].
#[derive(Clone, Copy, PartialEq, Debug, Serialize, Deserialize)]
pub struct DiffEntry {
    pub block_id: BlockId,
    /// Selection reason on the previous turn, if the block was
    /// present then.
    pub reason_prev: Option<SelectionReason>,
    /// Selection reason on the next turn, if the block is present
    /// now.
    pub reason_next: Option<SelectionReason>,
}

/// Structured diff between two [`PagePlan`]s.
#[derive(Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct PromptDiff {
    /// Blocks that appear in `next` but not in `prev`.
    pub added: Vec<DiffEntry>,
    /// Blocks that appeared in `prev` but are gone from `next`.
    pub removed: Vec<DiffEntry>,
    /// Blocks present in both. When the inclusion reason changed,
    /// `reason_prev != reason_next`; check
    /// [`DiffEntry::reason_changed`] for that case.
    pub kept: Vec<DiffEntry>,
    /// `next.estimated_tokens - prev.estimated_tokens`.
    pub token_delta: i64,
}

impl DiffEntry {
    /// True when the block stayed in the working set but the pager's
    /// reason for including it changed (e.g. previously surfaced by
    /// a retriever, now pinned).
    #[must_use]
    pub fn reason_changed(&self) -> bool {
        self.reason_prev != self.reason_next
    }
}

impl PromptDiff {
    /// True when `added`, `removed`, and any `reason_changed` entry
    /// in `kept` are all empty.
    #[must_use]
    pub fn is_noop(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.kept.iter().all(|e| !e.reason_changed())
    }

    /// One-line human summary suitable for log output.
    #[must_use]
    pub fn summary(&self) -> String {
        let kept_changed = self.kept.iter().filter(|e| e.reason_changed()).count();
        format!(
            "+{} -{} ~{} ({:+} tokens)",
            self.added.len(),
            self.removed.len(),
            kept_changed,
            self.token_delta,
        )
    }
}

/// Diff two [`PagePlan`]s. The input ordering of `selected` /
/// `selections` is ignored — diffs are set-based on `BlockId`. The
/// `kept` list is sorted by `BlockId` for determinism.
#[must_use]
pub fn diff_plans(prev: &PagePlan, next: &PagePlan) -> PromptDiff {
    let prev_reasons = reasons_by_id(prev);
    let next_reasons = reasons_by_id(next);

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut kept = Vec::new();

    for (id, &reason_next) in &next_reasons {
        if let Some(&reason_prev) = prev_reasons.get(id) {
            kept.push(DiffEntry {
                block_id: *id,
                reason_prev: Some(reason_prev),
                reason_next: Some(reason_next),
            });
        } else {
            added.push(DiffEntry {
                block_id: *id,
                reason_prev: None,
                reason_next: Some(reason_next),
            });
        }
    }
    for (id, &reason_prev) in &prev_reasons {
        if !next_reasons.contains_key(id) {
            removed.push(DiffEntry {
                block_id: *id,
                reason_prev: Some(reason_prev),
                reason_next: None,
            });
        }
    }

    added.sort_by_key(|e| e.block_id);
    removed.sort_by_key(|e| e.block_id);
    kept.sort_by_key(|e| e.block_id);

    PromptDiff {
        added,
        removed,
        kept,
        token_delta: token_delta(prev.estimated_tokens, next.estimated_tokens),
    }
}

/// Convenience wrapper: diff two [`TraceRecord`]s by their plans.
#[must_use]
pub fn diff_traces(prev: &TraceRecord, next: &TraceRecord) -> PromptDiff {
    diff_plans(&prev.plan, &next.plan)
}

/// Build an `id → SelectionReason` map. Falls back to a synthetic
/// reason (`SelectionReason::HighRelevance`) for plans that pre-date
/// the `selections` field — those leave `selections` empty but still
/// have populated `selected` ids.
fn reasons_by_id(plan: &PagePlan) -> HashMap<BlockId, SelectionReason> {
    let mut map = HashMap::with_capacity(plan.selected.len());
    if plan.selections.is_empty() {
        for &id in &plan.selected {
            map.insert(id, SelectionReason::HighRelevance);
        }
    } else {
        for sel in &plan.selections {
            map.insert(sel.block_id, sel.reason);
        }
        // Selected ids without a matching Selection (shouldn't happen
        // with the current pager, but keep the diff sound).
        for &id in &plan.selected {
            map.entry(id).or_insert(SelectionReason::HighRelevance);
        }
    }
    map
}

fn token_delta(prev: TokenCount, next: TokenCount) -> i64 {
    // `TokenCount.0` is `u32`, so the conversion to `i64` is exact
    // and can never fail; `from` is the right tool here.
    let prev_i = i64::from(prev.0);
    let next_i = i64::from(next.0);
    next_i - prev_i
}

#[cfg(test)]
mod tests {
    use super::*;
    use llm386_core::{OmittedBlock, OmissionReason, Selection};

    fn id(n: u128) -> BlockId {
        BlockId::from_parts(0, n)
    }

    fn plan_with(selections: Vec<Selection>, tokens: u32) -> PagePlan {
        PagePlan {
            selected: selections.iter().map(|s| s.block_id).collect(),
            selections,
            omitted: vec![],
            estimated_tokens: TokenCount(tokens),
        }
    }

    fn sel(block_id: BlockId, reason: SelectionReason) -> Selection {
        Selection { block_id, score: 0.5, reason, note: None }
    }

    #[test]
    fn empty_vs_empty_is_noop() {
        let prev = plan_with(vec![], 0);
        let next = plan_with(vec![], 0);
        let diff = diff_plans(&prev, &next);
        assert!(diff.is_noop());
        assert_eq!(diff.token_delta, 0);
    }

    #[test]
    fn added_blocks_appear_in_added() {
        let prev = plan_with(vec![sel(id(1), SelectionReason::Pinned)], 10);
        let next = plan_with(
            vec![
                sel(id(1), SelectionReason::Pinned),
                sel(id(2), SelectionReason::HighRelevance),
            ],
            18,
        );
        let diff = diff_plans(&prev, &next);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].block_id, id(2));
        assert_eq!(diff.added[0].reason_prev, None);
        assert_eq!(diff.added[0].reason_next, Some(SelectionReason::HighRelevance));
        assert!(diff.removed.is_empty());
        assert_eq!(diff.token_delta, 8);
    }

    #[test]
    fn removed_blocks_appear_in_removed() {
        let prev = plan_with(
            vec![
                sel(id(1), SelectionReason::Pinned),
                sel(id(2), SelectionReason::HighRelevance),
            ],
            18,
        );
        let next = plan_with(vec![sel(id(1), SelectionReason::Pinned)], 10);
        let diff = diff_plans(&prev, &next);
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].block_id, id(2));
        assert_eq!(diff.removed[0].reason_prev, Some(SelectionReason::HighRelevance));
        assert_eq!(diff.token_delta, -8);
    }

    #[test]
    fn kept_with_reason_change_is_flagged() {
        let prev = plan_with(vec![sel(id(1), SelectionReason::HighRelevance)], 10);
        let next = plan_with(vec![sel(id(1), SelectionReason::Pinned)], 10);
        let diff = diff_plans(&prev, &next);
        assert!(diff.added.is_empty());
        assert!(diff.removed.is_empty());
        assert_eq!(diff.kept.len(), 1);
        assert!(diff.kept[0].reason_changed());
        assert!(!diff.is_noop());
    }

    #[test]
    fn kept_with_same_reason_is_noop() {
        let prev = plan_with(vec![sel(id(1), SelectionReason::Pinned)], 10);
        let next = plan_with(vec![sel(id(1), SelectionReason::Pinned)], 10);
        let diff = diff_plans(&prev, &next);
        assert!(diff.is_noop());
    }

    #[test]
    fn legacy_plan_without_selections_still_diffs() {
        // Pre-v2 plans had no `selections` field. The diff should
        // fall back to treating selected ids as HighRelevance.
        let prev = PagePlan {
            selected: vec![id(1), id(2)],
            selections: vec![],
            omitted: vec![],
            estimated_tokens: TokenCount(10),
        };
        let next = PagePlan {
            selected: vec![id(2), id(3)],
            selections: vec![],
            omitted: vec![OmittedBlock {
                block_id: id(1),
                reason: OmissionReason::Budget,
                score: 0.0,
            }],
            estimated_tokens: TokenCount(15),
        };
        let diff = diff_plans(&prev, &next);
        assert_eq!(diff.added.len(), 1);
        assert_eq!(diff.added[0].block_id, id(3));
        assert_eq!(diff.removed.len(), 1);
        assert_eq!(diff.removed[0].block_id, id(1));
        assert_eq!(diff.kept.len(), 1);
        assert_eq!(diff.kept[0].block_id, id(2));
        assert_eq!(diff.token_delta, 5);
    }

    #[test]
    fn summary_string_format() {
        let prev = plan_with(vec![sel(id(1), SelectionReason::Pinned)], 10);
        let next = plan_with(
            vec![
                sel(id(1), SelectionReason::HighRelevance),
                sel(id(2), SelectionReason::Recency),
            ],
            22,
        );
        let diff = diff_plans(&prev, &next);
        assert_eq!(diff.summary(), "+1 -0 ~1 (+12 tokens)");
    }
}
