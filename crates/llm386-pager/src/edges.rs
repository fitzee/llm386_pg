//! `EdgePolicy` — how the pager treats each [`EdgeKind`] when
//! assembling a working set.
//!
//! Typed edges ([`EdgeKind`]) carry meaning beyond "these two blocks
//! are related": a `Supports` edge says one block is *evidence* for
//! another; a `Contradicts` edge says two blocks can't both be true.
//! The pager turns that meaning into concrete pack-time behavior via
//! two passes (see `greedy.rs`):
//!
//! - **Expansion** pulls dependent blocks *in* (claim → its evidence,
//!   assistant turn → its tool result, child → its container, derived
//!   block → its source). Bounded by [`EdgePolicy::max_depth`] and
//!   [`EdgePolicy::max_fanout`].
//! - **Reconciliation** acts on blocks *already* selected: it can drop
//!   a source that a selected summary supersedes, and resolve
//!   `Contradicts` pairs by demoting the older / lower-priority side.
//!
//! Every kind has an independent mode, so a consumer can enable some
//! and disable others. The defaults turn every kind on with the most
//! useful mode; flip [`EdgePolicy::enabled`] off (or any single mode
//! to `Off`) to opt out.

use llm386_core::EdgeKind;

/// Hard ceiling on traversal depth, enforced regardless of the
/// configured [`EdgePolicy::max_depth`]. Stops a misconfigured profile
/// from requesting an unbounded transitive walk that could blow the
/// token budget or the latency floor.
pub const MAX_DEPTH_CEILING: u8 = 5;

/// What `Parent` edges (child `from` → container `to`) do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ParentMode {
    /// Ignore `Parent` edges.
    Off,
    /// When a child is selected, pull in its container (the `to`
    /// endpoint). The cheap, bounded direction.
    ContainerOnly,
    /// `ContainerOnly` plus the reverse: when a container is selected,
    /// pull in its children (the `from` endpoints), capped by
    /// [`EdgePolicy::max_fanout`]. The expensive direction — a single
    /// document can have hundreds of chunk children.
    Bidirectional,
}

/// What `DerivedFrom` edges (derivative `from` → source `to`) do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DerivedMode {
    /// Ignore `DerivedFrom` edges.
    Off,
    /// When a derivative (summary / distillation) is selected, also
    /// pull in its source so the model can drill into the original.
    /// Higher fidelity, costs tokens.
    CoRetrieveSource,
    /// When both a derivative and its source are selected, drop the
    /// source — the derivative already covers it. Token-efficient,
    /// loses the source's detail. Omitted as
    /// [`OmissionReason::SupersededByDerived`](llm386_core::OmissionReason::SupersededByDerived).
    SuppressSource,
}

/// What `Supports` edges (claim `from` → evidence `to`) do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SupportsMode {
    /// Ignore `Supports` edges.
    Off,
    /// When a claim / conclusion is selected, co-retrieve its
    /// supporting evidence so the model isn't asked to trust an
    /// unsupported claim.
    CoRetrieve,
}

/// What `Contradicts` edges (`from` contradicts `to`) do when both
/// endpoints are in the selected set. Never pulls anything in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ContradictMode {
    /// Ignore `Contradicts` edges.
    Off,
    /// Keep the newer / higher-priority block, drop the other
    /// (omitted as
    /// [`OmissionReason::Contradicted`](llm386_core::OmissionReason::Contradicted)).
    PreferNewer,
    /// Keep both, but tag the older / lower-priority block with a note
    /// the packer renders inline ("contradicted by newer block …") so
    /// the model can weigh them rather than seeing both as equally
    /// true.
    Flag,
}

/// What `ToolInvocation` edges (assistant `from` → tool result `to`)
/// do.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ToolMode {
    /// Ignore `ToolInvocation` edges.
    Off,
    /// When an assistant message that called a tool is selected, pull
    /// in the tool result it consumed so the call/result pair travels
    /// together.
    CoRetrieve,
}

/// Per-[`EdgeKind`] pack-time behavior plus the traversal bounds that
/// keep edge-following from running away.
///
/// `Default` enables every kind with its most useful mode. Construct
/// [`EdgePolicy::disabled`] for the legacy "ignore all edges" posture.
#[derive(Clone, Copy, Debug)]
pub struct EdgePolicy {
    /// Master switch. When `false`, neither pass runs and edges have
    /// no effect on selection (they remain queryable in the store).
    pub enabled: bool,
    /// Maximum hops from any originally-selected block during the
    /// expansion pass. `0` disables expansion. Clamped to
    /// [`MAX_DEPTH_CEILING`].
    pub max_depth: u8,
    /// Maximum neighbors expanded per block *per kind* in the
    /// expansion pass. Caps the fan-out of a high-degree hub (e.g. a
    /// claim with hundreds of `Supports` edges). When exceeded,
    /// neighbors are ranked newest-first (by block id) and the top
    /// `max_fanout` are taken.
    pub max_fanout: usize,
    /// Also follow `Provenance.parents` lineage during expansion (the
    /// pre-typed-edge dependency signal). Off by default: the reducer
    /// records both a `Parent` edge and `provenance.parents`, so
    /// following typed edges alone avoids double-pulling. Turn on for
    /// legacy data that only has provenance links.
    pub follow_provenance_parents: bool,
    pub parent: ParentMode,
    pub derived_from: DerivedMode,
    pub supports: SupportsMode,
    pub contradicts: ContradictMode,
    pub tool_invocation: ToolMode,
}

impl Default for EdgePolicy {
    fn default() -> Self {
        Self {
            enabled: true,
            max_depth: 1,
            max_fanout: 8,
            follow_provenance_parents: false,
            parent: ParentMode::ContainerOnly,
            derived_from: DerivedMode::CoRetrieveSource,
            supports: SupportsMode::CoRetrieve,
            contradicts: ContradictMode::Flag,
            tool_invocation: ToolMode::CoRetrieve,
        }
    }
}

impl EdgePolicy {
    /// Ignore every edge kind — reproduces the pre-edge-policy
    /// behavior where selection never followed edges.
    #[must_use]
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }

    /// Depth clamped to the hard ceiling.
    #[must_use]
    pub(crate) fn effective_depth(&self) -> u8 {
        self.max_depth.min(MAX_DEPTH_CEILING)
    }

    /// Whether an outgoing edge of `kind` should pull its `to` endpoint
    /// into the working set during the expansion pass.
    pub(crate) fn pulls_outgoing(&self, kind: EdgeKind) -> bool {
        match kind {
            EdgeKind::Parent => matches!(
                self.parent,
                ParentMode::ContainerOnly | ParentMode::Bidirectional
            ),
            // SuppressSource demotes in reconciliation; it never pulls.
            EdgeKind::DerivedFrom => self.derived_from == DerivedMode::CoRetrieveSource,
            EdgeKind::Supports => self.supports == SupportsMode::CoRetrieve,
            EdgeKind::Contradicts => false,
            EdgeKind::ToolInvocation => self.tool_invocation == ToolMode::CoRetrieve,
        }
    }

    /// Whether the expansion pass needs to look at incoming edges at
    /// all (only `Parent` in `Bidirectional` mode, to find children).
    pub(crate) fn pulls_children(&self) -> bool {
        self.parent == ParentMode::Bidirectional
    }

    /// Whether the reconciliation pass has any work to do.
    pub(crate) fn reconciles(&self) -> bool {
        self.derived_from == DerivedMode::SuppressSource
            || self.contradicts != ContradictMode::Off
    }
}
