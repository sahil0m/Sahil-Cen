# Phase 2 Quick Start Guide

Get SPARC-LTM running in 5 minutes.

## 1. Install Dependencies

```bash
pip install -e ".[dev]"
```

## 2. Set Up API Keys

Create a `.env` file in the project root:

```bash
# OpenRouter (for fact extraction and contradiction detection)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Anthropic (for answer generation)
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

Get your keys:
- OpenRouter: https://openrouter.ai/keys
- Anthropic: https://console.anthropic.com/

## 3. Download LoCoMo Dataset

```bash
# Follow instructions in scripts/download_locomo.sh
# Place locomo10.json in data/raw/
```

## 4. Run Phase 2 Experiment

```bash
# Full SPARC-LTM with all features
python -m locomo_memory.phase2.experiments.run_phase2_qa \
  --config configs/phase2_full.yaml
```

This will:
1. Load LoCoMo conversations
2. Ingest into Phase 2 memory (semantic chunking → fact extraction → salience scoring)
3. Query memory for each QA item
4. Generate answers using Claude
5. Evaluate and save results to `results/`

## 5. Check Results

```bash
# View metrics
cat results/metrics/phase2_sparc_ltm_full_metrics.json

# View predictions
cat results/raw_predictions/phase2_sparc_ltm_full.json

# View by-category scores
cat results/tables/phase2_sparc_ltm_full_by_category.csv
```

## Expected Output

```
[Stage 1] Loading dataset from data/raw/locomo10.json
Loaded 10 conversations, 5882 total turns, 1986 total QA items

[Stage 2] Ingesting 10 conversations
Ingestion complete: 1234 MUs created across 10 conversations

[Stage 3] Processing QA items
  Progress: 50 / 1986 QA items processed
  Progress: 100 / 1986 QA items processed
  ...

[Stage 4] Processed 1986 QA items

[Stage 5] Saving outputs
Saved 1986 predictions to results/raw_predictions/phase2_sparc_ltm_full.json
Saved metrics to results/metrics/phase2_sparc_ltm_full_metrics.json

Phase 2 experiment complete: phase2_sparc_ltm_full
  Avg F1            : 0.XXXX
  Exact Match       : 0.XXXX
  Evidence Recall@k : 0.XXXX
```

## Run Ablation (No LLM Extraction)

To test without expensive LLM calls:

```bash
python -m locomo_memory.phase2.experiments.run_phase2_qa \
  --config configs/phase2_no_llm_extraction.yaml
```

This uses rule-based fact extraction instead of LLM, and runs in retrieval-only mode (no answer generation).

## Run Tests

```bash
# All Phase 2 tests
pytest tests/phase2/ -v

# Specific test file
pytest tests/phase2/test_store.py -v
```

## Troubleshooting

### "OPENROUTER_API_KEY not set"

Make sure you created `.env` in the project root (not in a subdirectory) and added your API key.

### "Dataset loaded 0 conversations"

Check that `data/raw/locomo10.json` exists. Run `bash scripts/download_locomo.sh` for download instructions.

### "No semantic chunks produced"

This is normal if a conversation has only summary turns. The semantic chunker filters out turns with `speaker="summary"`.

### Slow ingestion

The first run will be slow because:
1. Embeddings are being computed and cached
2. LLM calls are being made and cached

Subsequent runs will be much faster due to caching.

## What's Happening Under the Hood

### Ingestion Pipeline

```
Conversation turns
  ↓
Semantic chunking (topic boundaries)
  ↓
Candidate detection (cheap filter)
  ↓
Agentic chunking (LLM fact extraction)
  ↓
Salience scoring (multi-factor)
  ↓
Contradiction detection (FAISS + LLM)
  ↓
Graph linking (relationships)
  ↓
SQLite store + FAISS index + NetworkX graph
  ↓
State transitions (if at 90% capacity)
```

### Query Pipeline

```
Question
  ↓
Parallel 4-worker retrieval:
  - Dense FAISS (active MUs)
  - BM25 (active MUs)
  - Compressed labels (with restoration)
  - Graph traversal (neighbors)
  ↓
RRF fusion
  ↓
Reranking (placeholder)
  ↓
Context building (structured sections)
  ↓
Answer generation (Claude)
  ↓
Evaluation (F1, EM, Evidence Recall)
```

## Configuration Options

Edit `configs/phase2_full.yaml` to customize:

```yaml
phase2:
  # Disable LLM extraction (use rule-based)
  enable_llm_extraction: false
  
  # Lower threshold = more LLM calls
  candidate_detector_threshold: 0.35
  
  # Disable contradiction LLM (similarity only)
  enable_contradiction_llm: false
  
  # Disable specific retrieval workers
  enable_reranker: false
  enable_compressed_label_search: false
  enable_graph_traversal_worker: false
  enable_forgotten_tier_fallback: false
  
  # Adjust storage capacity
  storage_cap: 500
  transition_trigger_pct: 0.90
```

## Next Steps

1. **Compare with Phase 1**: Run Phase 1 baseline and compare metrics
2. **Analyze failures**: Check `results/tables/failure_cases.csv`
3. **Inspect memory**: Query the SQLite database to see stored MUs
4. **Try ablations**: Disable components to measure their impact
5. **Tune parameters**: Adjust thresholds, capacity, scoring weights

## Documentation

- **Complete methodology**: `PHASE2_METHODOLOGY.md`
- **Implementation details**: `PHASE2_README.md`
- **What was built**: `IMPLEMENTATION_SUMMARY.md`
- **Phase 1 baseline**: `README.md`

## Support

If you encounter issues:

1. Check the logs in `results/logs/`
2. Review the troubleshooting section in `PHASE2_README.md`
3. Verify your API keys are correct
4. Make sure all dependencies are installed

Happy experimenting! 🚀
