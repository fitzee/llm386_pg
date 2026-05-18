//! `ModelProfile`, `ModelRegistry`, and resolution of caller-supplied
//! model names into registered profiles.
//!
//! The built-in profile set is data-driven: see
//! [`crates/llm386-core/data/models.toml`](../data/models.toml).
//! Add a model by editing the TOML and rebuilding; no Rust changes
//! needed for new entries.
//!
//! Resolution (`ModelRegistry::resolve`) accepts any caller-supplied
//! string and always returns a profile:
//!
//! 1. **Strip provider segments.** `anthropic/claude-sonnet-4-6`,
//!    `openrouter/anthropic/claude-3.5-sonnet` → take the last
//!    `/`-separated segment.
//! 2. **Exact match.** If the normalized name exists in the registry,
//!    return it.
//! 3. **Family fallback.** Find the profile whose `family` field is
//!    the longest prefix of the normalized name (e.g.
//!    `claude-sonnet-4.5` → family `claude-sonnet`). Among profiles
//!    sharing a family, the last one registered wins. Emits a
//!    deduped `tracing::warn!`.
//! 4. **Default fallback.** Return the profile named by `default` in
//!    the data file (currently `gpt-4o`). Emits a deduped warning.
//!
//! The dedup set lives on the `ModelRegistry`; cloning a registry
//! resets the seen-warnings set, so tests get fresh isolation by
//! constructing a new `default_registry()`.

use std::collections::{HashMap, HashSet};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::ids::TokenCount;
use crate::tokenizer::TokenizerId;

/// Constraints and capabilities of a target model.
///
/// `safety_margin_tokens`, `family`, `supports_system_role`, and
/// `supports_tools` carry serde defaults so user-supplied TOML / JSON
/// profile files only need to set the load-bearing fields (`name`,
/// `max_context_tokens`, `reserved_output_tokens`, `tokenizer`).
#[derive(Clone, PartialEq, Eq, Debug, Serialize, Deserialize)]
pub struct ModelProfile {
    pub name: String,
    pub max_context_tokens: u32,
    pub reserved_output_tokens: u32,
    #[serde(default)]
    pub safety_margin_tokens: u32,
    pub tokenizer: TokenizerId,
    #[serde(default = "default_true")]
    pub supports_system_role: bool,
    #[serde(default = "default_true")]
    pub supports_tools: bool,
    /// Optional family-prefix string for fallback resolution.
    /// `claude-sonnet-4-6` would set `family = "claude-sonnet"` so
    /// future names like `claude-sonnet-4-9` family-resolve to this
    /// profile. `None` means this profile only matches its exact
    /// `name`.
    #[serde(default)]
    pub family: Option<String>,
}

const fn default_true() -> bool {
    true
}

impl ModelProfile {
    /// Effective input budget after subtracting reserved output and
    /// the safety margin. Saturates at zero if the profile is
    /// misconfigured (sum of reservations ≥ context window).
    #[must_use]
    pub const fn input_budget(&self) -> TokenCount {
        let avail = self
            .max_context_tokens
            .saturating_sub(self.reserved_output_tokens)
            .saturating_sub(self.safety_margin_tokens);
        TokenCount(avail)
    }
}

/// How a `resolve` call landed on a profile. Returned by
/// [`ModelRegistry::resolve_with_outcome`] so callers (mostly tests)
/// can distinguish the four code paths without parsing warning
/// output.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Resolution {
    /// The input (after stripping provider segments) was already a
    /// registered name. No warning emitted.
    Exact,
    /// No exact match; resolved by `family` prefix to a registered
    /// profile. Warning emitted (deduped per input).
    FamilyFallback,
    /// No exact or family match; returned the registry's default
    /// profile. Warning emitted (deduped per input).
    DefaultFallback,
}

/// Name-keyed lookup of [`ModelProfile`]s plus the resolver state
/// (default fallback, dedup set for fallback warnings).
#[derive(Default, Debug)]
pub struct ModelRegistry {
    /// Insertion-ordered list. Index into here is what `by_name`
    /// holds. Order matters: family fallback picks the *last*
    /// matching family entry, so newer entries override older ones.
    profiles: Vec<ModelProfile>,
    by_name: HashMap<String, usize>,
    /// Name of the profile used when nothing matches. Set by the
    /// `default = "…"` field in `models.toml`; `None` means the
    /// registry has no fallback, in which case [`Self::resolve`]
    /// returns whichever profile was registered first.
    default_name: Option<String>,
    /// Inputs that have already produced a fallback warning in this
    /// process — deduped so a noisy caller doesn't flood the log.
    warned: Mutex<HashSet<String>>,
}

impl ModelRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a profile under its own [`ModelProfile::name`]. Later
    /// registrations of the same name replace earlier ones (the
    /// `by_name` index updates) but do not remove the older entry
    /// from `profiles` — family-fallback iteration still finds the
    /// newest matching entry by scanning in reverse order.
    pub fn register(&mut self, profile: ModelProfile) {
        let idx = self.profiles.len();
        self.by_name.insert(profile.name.clone(), idx);
        self.profiles.push(profile);
    }

    /// Set the default fallback profile name. Must reference a
    /// profile that has been (or will be) registered. The lookup
    /// happens in [`Self::resolve`], not here.
    pub fn set_default(&mut self, name: impl Into<String>) {
        self.default_name = Some(name.into());
    }

    /// Strict lookup by canonical name. Returns `None` if there is
    /// no profile with this exact name. Prefer [`Self::resolve`] in
    /// most call sites — it's tolerant of provider prefixes and
    /// unknown names.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&ModelProfile> {
        self.by_name.get(name).map(|&i| &self.profiles[i])
    }

    /// Iterate over registered profile names.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.profiles.iter().map(|p| p.name.as_str())
    }

    /// Iterate over all registered profiles in insertion order.
    pub fn profiles(&self) -> impl Iterator<Item = &ModelProfile> {
        self.profiles.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }

    /// Resolve a caller-supplied model name to a registered profile.
    /// Always succeeds — falls back to family-prefix match, then to
    /// the registry's default profile if both fail. Fallbacks emit a
    /// deduped `tracing::warn!` so operators can see when the
    /// runtime is guessing.
    ///
    /// See [`Self::resolve_with_outcome`] for the version that also
    /// returns *how* the resolution landed.
    ///
    /// # Panics
    ///
    /// Panics only if the registry is empty AND has no default set —
    /// in which case there is no profile to return. This is treated
    /// as a programmer error (the built-in `default_registry()`
    /// ships with both).
    #[must_use]
    pub fn resolve(&self, input: &str) -> &ModelProfile {
        let (_outcome, profile) = self.resolve_with_outcome(input);
        profile
    }

    /// Like [`Self::resolve`] but also returns a [`Resolution`]
    /// describing which branch fired. Used by tests; production
    /// callers want [`Self::resolve`].
    ///
    /// Side effect: emits at most one `tracing::warn!` per input
    /// string per registry lifetime when the resolution is a
    /// fallback (family or default).
    #[must_use]
    pub fn resolve_with_outcome(&self, input: &str) -> (Resolution, &ModelProfile) {
        let normalized = strip_provider_prefix(input);

        // 2. Exact match against normalized name.
        if let Some(p) = self.get(normalized) {
            return (Resolution::Exact, p);
        }

        // 3. Family fallback: longest matching `family` prefix.
        if let Some(p) = self.family_match(normalized) {
            self.warn_once(input, &p.name, "family fallback");
            return (Resolution::FamilyFallback, p);
        }

        // 4. Default fallback.
        let default = self
            .default_profile()
            .expect("ModelRegistry is empty and has no default — cannot resolve");
        self.warn_once(input, &default.name, "default fallback");
        (Resolution::DefaultFallback, default)
    }

    /// Find the profile whose `family` field is the longest prefix of
    /// `name`. Among ties, the last-registered entry wins (so newer
    /// TOML entries override older ones for the same family).
    fn family_match(&self, name: &str) -> Option<&ModelProfile> {
        let mut best: Option<(usize, usize)> = None; // (family_len, profile_idx)
        for (idx, p) in self.profiles.iter().enumerate() {
            let Some(family) = p.family.as_deref() else {
                continue;
            };
            if family.is_empty() || !name.starts_with(family) {
                continue;
            }
            let len = family.len();
            // Use `>=` so a later registration with the same family
            // length overrides the earlier one — "latest registered
            // wins" within a family.
            match best {
                Some((best_len, _)) if len < best_len => {}
                _ => best = Some((len, idx)),
            }
        }
        best.map(|(_, idx)| &self.profiles[idx])
    }

    fn default_profile(&self) -> Option<&ModelProfile> {
        self.default_name
            .as_deref()
            .and_then(|n| self.get(n))
            .or_else(|| self.profiles.first())
    }

    fn warn_once(&self, input: &str, resolved: &str, reason: &str) {
        let mut warned = self.warned.lock();
        if warned.insert(input.to_string()) {
            warn!(
                model = %input,
                resolved = %resolved,
                reason,
                "model name resolved by fallback — not in the built-in registry",
            );
        }
    }
}

/// Strip provider segments — `openrouter/anthropic/claude-3.5-sonnet`
/// returns `claude-3.5-sonnet`. An input with no `/` is returned
/// unchanged.
fn strip_provider_prefix(input: &str) -> &str {
    input.rsplit('/').next().unwrap_or(input)
}

// ---------- Data-driven loading from models.toml ----------

const MODELS_TOML: &str = include_str!("../data/models.toml");

#[derive(Deserialize)]
struct ModelsFile {
    #[serde(default)]
    default: Option<String>,
    #[serde(default, rename = "model")]
    models: Vec<ModelProfile>,
}

fn parse_models_file() -> ModelsFile {
    toml::from_str(MODELS_TOML).expect(
        "BUG: built-in crates/llm386-core/data/models.toml failed to parse — \
         check the diff that touched the data file or the ModelProfile struct",
    )
}

/// Built-in model profiles loaded from
/// `crates/llm386-core/data/models.toml`.
///
/// Anthropic profiles reference `cl100k_base` as a tokenizer
/// approximation (Anthropic does not publish an exact public
/// tokenizer) and bump `safety_margin_tokens` accordingly. Llama and
/// Qwen profiles reference tokenizer ids that are not yet shipped by
/// `llm386-tokenizer`; using these profiles before those adapters
/// land will fail at the lookup site, not silently miscount.
///
/// These numbers are starting points; tune per workload.
#[must_use]
pub fn default_profiles() -> Vec<ModelProfile> {
    parse_models_file().models
}

/// Build a [`ModelRegistry`] preloaded with [`default_profiles`] and
/// the default-fallback name from `models.toml`.
#[must_use]
pub fn default_registry() -> ModelRegistry {
    let file = parse_models_file();
    let mut reg = ModelRegistry::new();
    for p in file.models {
        reg.register(p);
    }
    if let Some(name) = file.default {
        reg.set_default(name);
    }
    reg
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile(max: u32, reserved: u32, margin: u32) -> ModelProfile {
        ModelProfile {
            name: "test".to_string(),
            max_context_tokens: max,
            reserved_output_tokens: reserved,
            safety_margin_tokens: margin,
            tokenizer: TokenizerId::new("test"),
            supports_system_role: true,
            supports_tools: true,
            family: None,
        }
    }

    #[test]
    fn input_budget_subtracts_output_and_margin() {
        let p = profile(128_000, 4_000, 1_000);
        assert_eq!(p.input_budget(), TokenCount(123_000));
    }

    #[test]
    fn input_budget_saturates_at_zero_when_misconfigured() {
        let p = profile(1_000, 4_000, 0);
        assert_eq!(p.input_budget(), TokenCount(0));
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut reg = ModelRegistry::new();
        reg.register(profile(1_000, 100, 10));
        assert!(reg.get("test").is_some());
        assert!(reg.get("nope").is_none());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn default_registry_includes_known_models() {
        let reg = default_registry();
        for name in [
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o3",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "llama-3.1-70b",
            "qwen-2.5-72b",
        ] {
            assert!(reg.get(name).is_some(), "missing built-in profile: {name}");
        }
    }

    #[test]
    fn all_default_profiles_have_positive_input_budget() {
        for p in default_profiles() {
            assert!(
                p.input_budget().0 > 0,
                "profile {} has non-positive input budget",
                p.name,
            );
        }
    }

    #[test]
    fn anthropic_profiles_use_bumped_safety_margin() {
        let reg = default_registry();
        for name in ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5"] {
            let p = reg.get(name).unwrap();
            assert!(
                p.safety_margin_tokens >= 1024,
                "expected bumped margin on {name} (cl100k approximation)",
            );
        }
    }

    // ---------- Resolver tests ----------

    #[test]
    fn all_currently_registered_models_resolve_exactly() {
        let reg = default_registry();
        for name in [
            "gpt-4.1",
            "gpt-4o",
            "gpt-4o-mini",
            "o1",
            "o3",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "claude-haiku-4-5",
            "llama-3.1-70b",
            "qwen-2.5-72b",
        ] {
            let (outcome, profile) = reg.resolve_with_outcome(name);
            assert_eq!(outcome, Resolution::Exact, "{name} should match exactly");
            assert_eq!(profile.name, name);
        }
    }

    #[test]
    fn provider_prefix_is_stripped_before_lookup() {
        let reg = default_registry();
        let (outcome, profile) = reg.resolve_with_outcome("openai/gpt-4o-mini");
        assert_eq!(outcome, Resolution::Exact);
        assert_eq!(profile.name, "gpt-4o-mini");

        let (outcome, profile) = reg.resolve_with_outcome("anthropic/claude-sonnet-4-6");
        assert_eq!(outcome, Resolution::Exact);
        assert_eq!(profile.name, "claude-sonnet-4-6");

        // Nested provider segments — only the last segment matters.
        let (outcome, profile) =
            reg.resolve_with_outcome("openrouter/anthropic/claude-sonnet-4-6");
        assert_eq!(outcome, Resolution::Exact);
        assert_eq!(profile.name, "claude-sonnet-4-6");
    }

    #[test]
    fn unknown_version_family_resolves_to_family_default() {
        let reg = default_registry();
        let (outcome, profile) = reg.resolve_with_outcome("claude-sonnet-4-9");
        assert_eq!(outcome, Resolution::FamilyFallback);
        assert_eq!(profile.family.as_deref(), Some("claude-sonnet"));
        // Currently the only `claude-sonnet` entry is 4-6.
        assert_eq!(profile.name, "claude-sonnet-4-6");

        // With a provider prefix too:
        let (outcome, profile) =
            reg.resolve_with_outcome("anthropic/claude-sonnet-4-9");
        assert_eq!(outcome, Resolution::FamilyFallback);
        assert_eq!(profile.name, "claude-sonnet-4-6");
    }

    #[test]
    fn truly_unknown_name_falls_back_to_default() {
        let reg = default_registry();
        let (outcome, profile) =
            reg.resolve_with_outcome("something-totally-new-2099");
        assert_eq!(outcome, Resolution::DefaultFallback);
        // models.toml sets default = "gpt-4o".
        assert_eq!(profile.name, "gpt-4o");

        // Provider-prefixed unknowns hit the same path.
        let (outcome, profile) =
            reg.resolve_with_outcome("openrouter/google/gemini-2.5");
        assert_eq!(outcome, Resolution::DefaultFallback);
        assert_eq!(profile.name, "gpt-4o");
    }

    #[test]
    fn family_match_prefers_longest_prefix() {
        // Construct a registry where two families overlap by prefix.
        let mut reg = ModelRegistry::new();
        reg.register(ModelProfile {
            name: "broad".to_string(),
            family: Some("foo".to_string()),
            ..profile(1000, 100, 10)
        });
        reg.register(ModelProfile {
            name: "specific".to_string(),
            family: Some("foo-bar".to_string()),
            ..profile(1000, 100, 10)
        });
        reg.set_default("broad");

        // `foo-bar-baz` matches both families; longest wins.
        let (outcome, profile) = reg.resolve_with_outcome("foo-bar-baz");
        assert_eq!(outcome, Resolution::FamilyFallback);
        assert_eq!(profile.name, "specific");

        // `foo-other` only matches `foo`.
        let (outcome, profile) = reg.resolve_with_outcome("foo-other");
        assert_eq!(outcome, Resolution::FamilyFallback);
        assert_eq!(profile.name, "broad");
    }

    #[test]
    fn family_match_picks_last_registered_within_family() {
        // Two profiles in the same family — the newer one wins for
        // family-resolution. Lets `models.toml` express "latest in
        // family" by ordering entries newest-first... or in our case,
        // by ordering them in any consistent way and trusting "last
        // registered" semantics.
        let mut reg = ModelRegistry::new();
        reg.register(ModelProfile {
            name: "claude-sonnet-old".to_string(),
            family: Some("claude-sonnet".to_string()),
            ..profile(1000, 100, 10)
        });
        reg.register(ModelProfile {
            name: "claude-sonnet-new".to_string(),
            family: Some("claude-sonnet".to_string()),
            ..profile(1000, 100, 10)
        });
        reg.set_default("claude-sonnet-old");

        let (outcome, profile) = reg.resolve_with_outcome("claude-sonnet-99");
        assert_eq!(outcome, Resolution::FamilyFallback);
        assert_eq!(profile.name, "claude-sonnet-new");
    }

    #[test]
    fn fallback_emits_one_warning_per_unique_name() {
        let reg = default_registry();

        // Two repeats of the same unknown name → one warning.
        let _ = reg.resolve_with_outcome("claude-sonnet-4-9");
        let _ = reg.resolve_with_outcome("claude-sonnet-4-9");
        // A different unknown name → another warning.
        let _ = reg.resolve_with_outcome("claude-sonnet-4-10");
        // An exact-match resolution → no warning.
        let _ = reg.resolve_with_outcome("gpt-4o");
        // A default-fallback name → another warning.
        let _ = reg.resolve_with_outcome("entirely-novel-2099");

        let warned = reg.warned.lock();
        // claude-sonnet-4-9, claude-sonnet-4-10, entirely-novel-2099.
        assert_eq!(warned.len(), 3, "warned set: {warned:?}");
        assert!(warned.contains("claude-sonnet-4-9"));
        assert!(warned.contains("claude-sonnet-4-10"));
        assert!(warned.contains("entirely-novel-2099"));
        assert!(!warned.contains("gpt-4o"));
    }

    #[test]
    fn stripping_treats_input_without_slash_unchanged() {
        assert_eq!(strip_provider_prefix("plain-name"), "plain-name");
        assert_eq!(strip_provider_prefix("a/b"), "b");
        assert_eq!(strip_provider_prefix("a/b/c"), "c");
        assert_eq!(strip_provider_prefix(""), "");
    }

    #[test]
    fn token_counts_for_known_models_are_unchanged() {
        // Pin the per-model budget numbers so a future TOML edit that
        // accidentally changes one trips a test rather than silently
        // shifting input-token math.
        let reg = default_registry();
        let cases = &[
            ("gpt-4.1", 1_048_576, 32_768, 256, "o200k_base"),
            ("gpt-4o", 128_000, 16_384, 256, "o200k_base"),
            ("gpt-4o-mini", 128_000, 16_384, 256, "o200k_base"),
            ("claude-opus-4-7", 200_000, 8_192, 4_096, "cl100k_base"),
            ("claude-sonnet-4-6", 200_000, 8_192, 4_096, "cl100k_base"),
            ("claude-haiku-4-5", 200_000, 8_192, 4_096, "cl100k_base"),
        ];
        for &(name, ctx, out, margin, tok) in cases {
            let p = reg.get(name).unwrap_or_else(|| panic!("missing {name}"));
            assert_eq!(p.max_context_tokens, ctx, "{name}: max_context_tokens");
            assert_eq!(
                p.reserved_output_tokens, out,
                "{name}: reserved_output_tokens",
            );
            assert_eq!(
                p.safety_margin_tokens, margin,
                "{name}: safety_margin_tokens",
            );
            assert_eq!(p.tokenizer.as_str(), tok, "{name}: tokenizer");
        }
    }
}
