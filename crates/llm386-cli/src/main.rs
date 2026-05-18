//! `llm386` — command-line interface for the LLM386 runtime.

mod cli;
mod commands;

use clap::Parser;
use tracing_subscriber::EnvFilter;

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = cli::Cli::parse();
    let config = commands::load_config(args.profiles.as_deref())?;
    commands::dispatch(args.command, args.store, args.pg_url, &config)
}
