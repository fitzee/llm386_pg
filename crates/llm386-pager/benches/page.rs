//! End-to-end paging throughput against a populated LMDB store.

use std::sync::Arc;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use llm386_core::{
    BlockId, BlockKind, BlockStore, ContentHash, ContextBlock, ModelProfile, PageRequest, Pager,
    Provenance, SessionId, Timestamp, TokenCounts, Tokenizer, TokenizerId,
};
use llm386_pager::GreedyPager;
use llm386_store_lmdb::{LmdbStore, StoreConfig};
use llm386_tokenizer::cl100k_base;
use tempfile::TempDir;

fn populate(n: u64) -> (TempDir, Arc<LmdbStore>, Arc<dyn Tokenizer>) {
    let dir = TempDir::new().expect("tempdir");
    let store = Arc::new(LmdbStore::open(dir.path(), StoreConfig::default()).expect("open store"));
    let tok: Arc<dyn Tokenizer> = Arc::new(cl100k_base().expect("cl100k_base"));
    let session = SessionId(1);
    for i in 0..n {
        let bytes = format!("block content number {i} with padding text").into_bytes();
        let hash = ContentHash::of(&bytes);
        store
            .put(
                session,
                ContextBlock {
                    id: BlockId::from_parts(i, u128::from(i)),
                    kind: BlockKind::UserMessage,
                    bytes,
                    token_counts: TokenCounts::new(),
                    priority: 0.0,
                    created_at: Timestamp(i),
                    updated_at: Timestamp(i),
                    provenance: Provenance::default(),
                    hash,
                },
            )
            .expect("put");
    }
    (dir, store, tok)
}

fn bench_page_profile() -> ModelProfile {
    ModelProfile {
        name: "bench".into(),
        max_context_tokens: 1_000_000,
        reserved_output_tokens: 0,
        safety_margin_tokens: 0,
        tokenizer: TokenizerId::new("cl100k_base"),
        family: None,
        supports_system_role: true,
        supports_tools: true,
    }
}

fn bench_page(c: &mut Criterion) {
    for &n in &[100u64, 1_000] {
        let (_dir, store, tok) = populate(n);
        let pager = GreedyPager::new(store, tok);
        let request = PageRequest {
            session_id: SessionId(1),
            task: "bench task".into(),
            model: bench_page_profile(),
            required_blocks: vec![],
        };
        c.bench_function(&format!("page_{n}_blocks"), |b| {
            b.iter(|| pager.page(black_box(request.clone())).expect("page"));
        });
    }
}

criterion_group!(benches, bench_page);
criterion_main!(benches);
