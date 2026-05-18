//! Thin pass-through over `llm386_config` for the Python `Store`.
//!
//! All TOML schema + retriever building lives in the shared
//! `llm386-config` crate so the CLI and Python wrapper share one
//! source of truth. This module just narrows the error type to
//! `String` and re-exports what the rest of the `python/` crate uses.

use std::path::Path;

use llm386_core::{ModelRegistry};
use llm386_tokenizer::TokenizerRegistry;

pub(crate) use llm386_config::{Applied, RetrieverEntry, build_retrievers};

pub(crate) fn parse(path: &Path) -> Result<llm386_config::ConfigFile, String> {
    llm386_config::parse(path)
}

pub(crate) fn apply(
    parsed: llm386_config::ConfigFile,
    models: &mut ModelRegistry,
    tokenizers: &mut TokenizerRegistry,
) -> Result<Applied, String> {
    llm386_config::apply(parsed, models, tokenizers)
}
