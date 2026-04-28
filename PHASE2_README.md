# Phase 2 — SPARC-LTM Implementation

**Salience and Provenance Aware Reconciliation and Compression for Long-Term Memory**

This directory contains the complete Phase 2 implementation as specified in `PHASE2_METHODOLOGY.md`.

## Architecture Overview

```
Phase 2 Pipeline
├── Ingestion
│   ├── Semantic Chunker (topic-boundary detection)
│   ├── Candidate Detector (cheap filter before LLM)
│   ├── Agentic Chunker (LLM fact extraction)
│   ├── Salience Scorer (multi-factor, no LLM)
│   ├── Contradiction Resolver (FAISS + LLM classifier)
│   └── Graph Linking (NetworkX relationships)
├── Storage
│   ├── SQLite (source of truth)
│   ├── FAISS (active + compressed indexes)
│   └── NetworkX (relationship graph)
├── Lifecycle
│   └── Transition Engine (90% capacity trigger)
└── Retrieval
    ├── Parallel 4-Worker Pipeline
    ├── RRF Fusion
    ├── Cross-Encoder Reranking
    └── Context Builder (structured prompt)
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

### 2. Set Up API Keys

```bash
cp .env.example .env
# Edit .env and add:
# OPENROUTER_API_KEY=your_key_here
# ANTHROPIC_API_KEY=your_key_here (for answer generation)
```

### 3. Run Phase 2 Experiment

```bash
# Full SPARC-LTM with all features
python -m locomo_memory.phase2.experiments.run_phase2_qa \
  --config configs/phase2_full.yaml

# Ablation: no LLM extraction (rule-based only)
python -m locomo_memory.phase2.experiments.run_phase2_qa \
  --config configs/phase2_no_llm_extraction.yaml
```

## Implementation Status

### ✅ Completed Components

- **Schemas** (`phase2/schemas.py`)
  - MemoryUnit, CompressedLabel, ArchivedEntry, EdgeRecord
  - Pydantic v2 validation with strict mode
  - Timezone-aware datetimes
  - 5-state lifecycle (Active, Compressed, Archived, Forgotten, Deleted)

- **Storage Layer** (`phase2/store/`)
  - SQLite source-of-truth with atomic transactions
  - NetworkX graph index (rebuildable from SQLite)
  - Atomic compound operations (compress, restore, forget, delete)
  - Full referential integrity checks

- **Ingestion Pipeline** (`phase2/ingestion/`)
  - Semantic chunker (cosine similarity topic boundaries)
  - Memory candidate detector (rule-based scoring)
  - Agentic chunker (LLM fact extraction via OpenRouter)
  - Salience scorer (multi-factor, no LLM)
  - Contradiction resolver (FAISS + LLM classifier)

- **Lifecycle Management** (`phase2/lifecycle/`)
  - Transition engine with 90% capacity trigger
  - Demotion scoring (salience + frequency + recency + centrality)
  - Automatic compress/forget decisions
  - User pin protection

- **Retrieval Pipeline** (`phase2/retrieval/`)
  - Parallel 4-worker retrieval (Dense, BM25, Compressed, Graph)
  - RRF fusion
  - Forgotten tier fallback
  - Context builder (structured prompt sections)

- **Orchestration** (`phase2/pipeline.py`)
  - Complete end-to-end pipeline
  - Ingestion + Query interfaces
  - Automatic state transitions

- **Experiment Runner** (`phase2/experiments/`)
  - LoCoMo QA evaluation
  - Integrates with Phase 1 metrics
  - Ablation support via config flags

### 🚧 Partial / Placeholder Components

- **Cross-Encoder Reranker**
  - Interface exists but returns candidates as-is
  - Full implementation: use `BAAI/bge-reranker-base`

- **LLM Label Generation**
  - Currently uses first 10 words as label
  - Full implementation: call cheap LLM for smart summary

- **Uniqueness Scoring**
  - Currently returns 1.0 (assume unique)
  - Full implementation: compute cosine similarity to existing MUs

### ❌ Not Implemented (Out of Scope for Benchmark)

- **FastAPI Backend** (`/chat`, `/memory`, `/override` endpoints)
- **Streamlit UI** (chat interface + memory inspector)
- **Storage Gauge Visualization**
- **Manual Override Controls** (pin, compress, forget, delete buttons)
- **Provenance Trail Viewer**

These UI components are specified in the methodology but not required for the LoCoMo benchmark evaluation.

## Configuration

Phase 2 experiments use YAML configs with these sections:

```yaml
phase2:
  # Memory extraction
  enable_llm_extraction: true        # false → rule-based
  candidate_detector_threshold: 0.35 # 0.0 = always call LLM

  # Contradiction detection
  enable_contradiction_llm: true     # false → similarity only
  contradiction_similarity_threshold: 0.85

  # Retrieval pipeline
  enable_reranker: true
  enable_compressed_label_search: true
  enable_graph_traversal_worker: true
  enable_forgotten_tier_fallback: true

  # Lifecycle
  storage_cap: 500
  transition_trigger_pct: 0.90
  enable_compression_llm: false      # true → call LLM for labels

  # Caching
  cache_dir: data/processed/phase2_cache
```

## Key Design Decisions

### 1. SQLite as Source of Truth

All writes go to SQLite first (transactional, atomic). FAISS and NetworkX are derived indexes that can be rebuilt from SQLite at any time.

### 2. 90% Capacity Trigger

Automatic transitions only fire when active memory hits ~90% of capacity. Below that threshold, only user overrides are honored.

### 3. Forgotten ≠ Deleted

Forgotten data is dormant but recoverable. Only the user can trigger permanent deletion.

### 4. Compression = Label + Archive

Compressed memory is a searchable label that points to the full archived data. When matched, full data is restored from archive.

### 5. NetworkX is Prototype-Grade

For production scale (millions of MUs), migrate to Neo4j or Memgraph. NetworkX is sufficient for LoCoMo benchmark + demo.

## Testing

Phase 2 has comprehensive test coverage:

```bash
# Run all Phase 2 tests
pytest tests/phase2/ -v

# Specific test modules
pytest tests/phase2/test_schemas.py -v
pytest tests/phase2/test_store.py -v
pytest tests/phase2/test_graph_index.py -v
```

## Expected Performance

Based on Phase 1 results and methodology targets:

| Metric | Phase 1 Best | Phase 2 Target |
|--------|--------------|----------------|
| Overall Recall@5 | 0.591 | 0.72-0.78 |
| Single-hop | 0.246 | ~0.55 |
| Multi-hop | 0.565 | ~0.72 |
| Temporal | 0.233 | ~0.55 |
| Adversarial | 0.714 | ~0.80 |
| Retrieval p95 | ~5ms | <100ms |

**Note:** These are design targets, not guarantees. Actual results depend on LLM extraction quality, contradiction detector precision, and embedding model behavior on LoCoMo's vocabulary.

## Cost Optimization

Phase 2 minimizes LLM costs through:

1. **Trivial filter** (~15% of turns skipped)
2. **Candidate detector** (~20-30% additional skipped)
3. **diskcache** for all LLM calls (extraction, contradiction, answers)
4. **Cheap models** for ingestion (llama-3.1-8b @ $0.07/M tokens)
5. **Similarity gate** for contradiction detection (only call LLM if cosine > 0.85)

Expected cost per QA item: ~$0.005 with caching.

## Troubleshooting

### "OPENROUTER_API_KEY not set"

Add your OpenRouter API key to `.env`:
```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

### "No semantic chunks produced"

Check that the conversation has non-summary turns. The semantic chunker filters out turns with `speaker="summary"`.

### "Storage pressure below threshold"

The transition engine only runs at 90% capacity. If you have fewer than 450 active MUs (with cap=500), no automatic transitions will occur.

### "Failed to parse LLM response as JSON"

The LLM occasionally returns markdown fences. The code strips these, but if parsing still fails, the system logs a warning and continues with empty facts.

## Next Steps

To complete the full SPARC-LTM vision:

1. **Implement cross-encoder reranking** using `BAAI/bge-reranker-base`
2. **Add LLM label generation** for smarter compressed summaries
3. **Implement uniqueness scoring** via embedding similarity
4. **Build FastAPI backend** for `/chat`, `/memory`, `/override` endpoints
5. **Create Streamlit UI** with memory inspector and manual controls
6. **Add storage gauge visualization** with 90% warning
7. **Implement provenance trail viewer** showing source dialog IDs

See `PHASE2_METHODOLOGY.md` §12 for the complete build order.

## References

- **Methodology**: `PHASE2_METHODOLOGY.md` (904 lines, complete design)
- **Progress**: `PROJECT_PROGRESS.md` (Phase 1 results and failure analysis)
- **Phase 1 Code**: `src/locomo_memory/` (baseline implementation)
- **Tests**: `tests/phase2/` (schemas, store, graph)
