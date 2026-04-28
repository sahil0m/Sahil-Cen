#!/bin/bash
# Run Phase 2 SPARC-LTM experiments

set -e

echo "=========================================="
echo "Phase 2 SPARC-LTM Experiments"
echo "=========================================="

# Full SPARC-LTM
echo ""
echo "Running: phase2_sparc_ltm_full"
python -m locomo_memory.phase2.experiments.run_phase2_qa \
  --config configs/phase2_full.yaml

# Ablation: no LLM extraction
echo ""
echo "Running: phase2_no_llm_extraction"
python -m locomo_memory.phase2.experiments.run_phase2_qa \
  --config configs/phase2_no_llm_extraction.yaml

echo ""
echo "=========================================="
echo "Phase 2 experiments complete!"
echo "Results saved to results/"
echo "=========================================="
