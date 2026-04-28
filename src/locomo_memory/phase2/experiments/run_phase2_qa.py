"""Phase 2 experiment runner for LoCoMo QA evaluation.

Usage:
    python -m locomo_memory.phase2.experiments.run_phase2_qa --config configs/phase2_full.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_phase2_experiment(config_path: str) -> dict:
    """Run Phase 2 SPARC-LTM experiment on LoCoMo."""
    from locomo_memory.data.load_locomo import load_locomo
    from locomo_memory.data.schemas import PredictionRow
    from locomo_memory.evaluation.evidence_recall import evidence_recall_at_k
    from locomo_memory.evaluation.qa_metrics import exact_match, token_f1
    from locomo_memory.evaluation.report import (
        compute_and_save_metrics,
        save_predictions,
        save_retrieval_debug,
    )
    from locomo_memory.generation.llm_client import LLMClient
    from locomo_memory.generation.prompts import build_answer_prompt, count_prompt_tokens
    from locomo_memory.indexing.embeddings import EmbeddingGenerator
    from locomo_memory.phase2.pipeline import Phase2Pipeline
    
    # Load config
    cfg = _load_config(config_path)
    experiment_name = cfg["experiment"]["name"]
    output_dir = Path(cfg.get("output", {}).get("dir", "results"))
    
    logger.info("=" * 60)
    logger.info("Phase 2 Experiment: %s", experiment_name)
    logger.info("=" * 60)
    
    # Load dataset
    dataset_path = cfg["dataset"]["path"]
    logger.info("[Stage 1] Loading dataset from %s", dataset_path)
    conversations = load_locomo(dataset_path)
    
    if not conversations:
        raise RuntimeError("Dataset loaded 0 conversations")
    
    # Initialize embedder
    emb_cfg = cfg.get("embedding", {})
    embedder = EmbeddingGenerator(
        model_name=emb_cfg.get("model_name", "BAAI/bge-small-en-v1.5"),
        batch_size=emb_cfg.get("batch_size", 64),
        normalize=emb_cfg.get("normalize_embeddings", True),
        cache_dir=emb_cfg.get("cache_dir"),
    )
    
    # Initialize Phase 2 pipeline
    phase2_cfg = cfg.get("phase2", {})
    db_path = output_dir / "phase2_memory" / f"{experiment_name}.sqlite"
    
    pipeline = Phase2Pipeline(
        db_path=db_path,
        embedder=embedder,
        storage_cap=phase2_cfg.get("storage_cap", 500),
        enable_llm_extraction=phase2_cfg.get("enable_llm_extraction", True),
        enable_contradiction_llm=phase2_cfg.get("enable_contradiction_llm", True),
        candidate_detector_threshold=phase2_cfg.get("candidate_detector_threshold", 0.35),
        cache_dir=phase2_cfg.get("cache_dir"),
    )
    
    # Ingest all conversations
    logger.info("[Stage 2] Ingesting %d conversations", len(conversations))
    ingestion_stats = []
    
    for conv in conversations:
        stats = pipeline.ingest_conversation(conv)
        ingestion_stats.append(stats)
    
    logger.info(
        "Ingestion complete: %d MUs created across %d conversations",
        sum(s["memory_units_created"] for s in ingestion_stats),
        len(conversations),
    )
    
    # Query and evaluate
    gen_cfg = cfg.get("generation", {})
    generation_enabled = gen_cfg.get("enabled", True)
    
    llm_client = None
    if generation_enabled:
        llm_client = LLMClient(
            provider=gen_cfg.get("provider", "anthropic"),
            model_name=gen_cfg.get("model_name", "claude-3-5-sonnet-latest"),
            temperature=gen_cfg.get("temperature", 0.0),
            max_output_tokens=gen_cfg.get("max_output_tokens", 120),
            cache_dir=gen_cfg.get("cache_dir"),
        )
    
    logger.info("[Stage 3] Processing QA items")
    predictions: list[PredictionRow] = []
    
    for conv in conversations:
        for qa in conv.qa_items:
            t_start = time.perf_counter()
            
            # Query Phase 2 pipeline
            query_result = pipeline.query(conv.conversation_id, qa.question)
            
            # Generate answer
            predicted_answer = ""
            input_tokens = 0
            output_tokens = 0
            gen_latency_ms = 0.0
            
            if generation_enabled and llm_client:
                # Build prompt from Phase 2 context
                prompt = f"{query_result['context']}\n\nQuestion: {qa.question}\n\nAnswer:"
                input_tokens = count_prompt_tokens(prompt)
                
                try:
                    gen_result = llm_client.generate(prompt)
                    predicted_answer = gen_result.answer
                    input_tokens = gen_result.input_tokens or input_tokens
                    output_tokens = gen_result.output_tokens
                    gen_latency_ms = gen_result.generation_latency_ms
                except Exception as exc:
                    logger.error("LLM generation failed for qa_id=%s: %s", qa.qa_id, exc)
                    predicted_answer = "GENERATION_ERROR"
            
            e2e_ms = (time.perf_counter() - t_start) * 1000.0
            
            # Evaluate
            f1 = token_f1(predicted_answer, qa.answer) if generation_enabled else 0.0
            em = exact_match(predicted_answer, qa.answer) if generation_enabled else False
            
            # Evidence recall (check if gold dia_ids are in retrieved MUs)
            retrieved_dia_ids = set()
            for mu in query_result["retrieved_mus"]:
                retrieved_dia_ids.update(mu.source_dia_ids)
            
            recall = None
            if qa.gold_evidence_ids:
                hits = sum(1 for gid in qa.gold_evidence_ids if gid in retrieved_dia_ids)
                recall = hits / len(qa.gold_evidence_ids)
            
            predictions.append(
                PredictionRow(
                    experiment_name=experiment_name,
                    conversation_id=conv.conversation_id,
                    qa_id=qa.qa_id,
                    question=qa.question,
                    gold_answer=qa.answer,
                    predicted_answer=predicted_answer,
                    category=qa.category,
                    gold_evidence_ids=qa.gold_evidence_ids,
                    retrieved_chunks=[],  # Phase 2 uses MUs, not chunks
                    f1=f1,
                    exact_match=em,
                    evidence_recall=recall,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    retrieval_latency_ms=query_result["retrieval_latency_ms"],
                    generation_latency_ms=gen_latency_ms,
                    end_to_end_latency_ms=e2e_ms,
                )
            )
    
    logger.info("[Stage 4] Processed %d QA items", len(predictions))
    
    # Save outputs
    logger.info("[Stage 5] Saving outputs")
    save_predictions(predictions, output_dir, experiment_name)
    save_retrieval_debug(predictions, output_dir, experiment_name)
    
    run_metadata = {
        "experiment_name": experiment_name,
        "phase": 2,
        "config": cfg,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ingestion_stats": ingestion_stats,
    }
    
    metrics = compute_and_save_metrics(
        predictions,
        output_dir,
        experiment_name,
        run_metadata,
        generation_enabled=generation_enabled,
    )
    
    logger.info("=" * 60)
    logger.info("Phase 2 experiment complete: %s", experiment_name)
    if generation_enabled and metrics:
        logger.info("  Avg F1            : %.4f", metrics.get("avg_f1", 0))
        logger.info("  Exact Match       : %.4f", metrics.get("exact_match", 0))
        if metrics.get("mean_evidence_recall") is not None:
            logger.info("  Evidence Recall@k : %.4f", metrics["mean_evidence_recall"])
    logger.info("=" * 60)
    
    return metrics


def _load_config(config_path: str) -> dict:
    """Load YAML config file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run Phase 2 SPARC-LTM experiment on LoCoMo QA."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML experiment config (e.g. configs/phase2_full.yaml)",
    )
    args = parser.parse_args()
    run_phase2_experiment(args.config)


if __name__ == "__main__":
    main()
