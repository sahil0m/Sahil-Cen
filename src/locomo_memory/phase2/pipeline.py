"""Phase 2 complete ingestion and query pipeline orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from locomo_memory.data.schemas import Conversation
from locomo_memory.indexing.embeddings import EmbeddingGenerator
from locomo_memory.phase2.ingestion import (
    AgenticChunker,
    ContradictionResolver,
    MemoryCandidateDetector,
    SalienceScorer,
    SemanticChunker,
)
from locomo_memory.phase2.lifecycle import TransitionEngine
from locomo_memory.phase2.retrieval import ContextBuilder, ParallelRetriever
from locomo_memory.phase2.schemas import EdgeRecord, EdgeType, MemoryUnit, new_mu_id
from locomo_memory.phase2.store import MemoryGraphIndex, MemoryStore

logger = logging.getLogger(__name__)


class Phase2Pipeline:
    """Complete Phase 2 SPARC-LTM pipeline.
    
    Ingestion:
    1. Semantic chunking (topic boundaries)
    2. Candidate detection (cheap filter)
    3. Agentic chunking (LLM fact extraction)
    4. Salience scoring
    5. Contradiction detection
    6. Graph linking
    7. Store write
    8. State transitions (at 90% capacity)
    
    Query:
    1. Parallel 4-worker retrieval
    2. RRF fusion
    3. Reranking
    4. Context building
    5. Answer generation (external)
    """
    
    def __init__(
        self,
        db_path: str | Path,
        embedder: EmbeddingGenerator,
        storage_cap: int = 500,
        enable_llm_extraction: bool = True,
        enable_contradiction_llm: bool = True,
        candidate_detector_threshold: float = 0.35,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.store = MemoryStore(db_path)
        self.graph = MemoryGraphIndex()
        self.embedder = embedder
        self.storage_cap = storage_cap
        
        # Ingestion components
        self.semantic_chunker = SemanticChunker(embedder)
        self.candidate_detector = MemoryCandidateDetector(candidate_detector_threshold)
        self.salience_scorer = SalienceScorer()
        
        if enable_llm_extraction:
            self.agentic_chunker = AgenticChunker(cache_dir=cache_dir)
        else:
            self.agentic_chunker = None
        
        if enable_contradiction_llm:
            self.contradiction_resolver = ContradictionResolver(cache_dir=cache_dir)
        else:
            self.contradiction_resolver = None
        
        # Lifecycle
        self.transition_engine = TransitionEngine(
            self.store,
            self.graph,
            storage_cap,
        )
        
        # Retrieval
        self.retriever = ParallelRetriever(
            self.store,
            self.graph,
            embedder,
        )
        self.context_builder = ContextBuilder(self.store)
        
        # Rebuild graph from store
        self.graph.rebuild_from_store(self.store)
    
    def ingest_conversation(self, conversation: Conversation) -> dict:
        """Ingest a conversation into memory.
        
        Returns:
            Statistics dict
        """
        logger.info("Ingesting conversation %s", conversation.conversation_id)
        
        stats = {
            "conversation_id": conversation.conversation_id,
            "total_turns": len(conversation.turns),
            "semantic_chunks": 0,
            "candidates": 0,
            "facts_extracted": 0,
            "memory_units_created": 0,
            "contradictions_detected": 0,
            "transitions_executed": 0,
        }
        
        # Step 1: Semantic chunking
        chunks = self.semantic_chunker.chunk_turns(conversation.turns)
        stats["semantic_chunks"] = len(chunks)
        
        if not chunks:
            logger.warning("No semantic chunks produced for %s", conversation.conversation_id)
            return stats
        
        # Step 2-3: Candidate detection + fact extraction
        all_facts: list[tuple[str, list[str]]] = []  # (session_id, facts)
        
        for chunk in chunks:
            # Candidate detection
            candidate_score = self.candidate_detector.is_candidate(chunk.text)
            
            if not candidate_score.is_candidate:
                logger.debug("Skipping chunk %s: %s", chunk.dia_ids, candidate_score.reason)
                continue
            
            stats["candidates"] += 1
            
            # Fact extraction
            if self.agentic_chunker:
                extraction = self.agentic_chunker.extract_facts(chunk)
                facts = extraction.facts
            else:
                # Fallback: use chunk text as single fact
                facts = [chunk.text]
            
            if facts:
                all_facts.append((chunk.session_id, facts))
                stats["facts_extracted"] += len(facts)
        
        # Step 4-7: Create Memory Units
        for session_id, facts in all_facts:
            for fact in facts:
                mu = self._create_memory_unit(
                    conversation.conversation_id,
                    session_id,
                    fact,
                )
                
                # Salience scoring
                mu = self.salience_scorer.update_salience(mu)
                
                # Embed claim
                claim_emb = self.embedder.embed_query(mu.claim)
                
                # Contradiction detection
                if self.contradiction_resolver:
                    existing = [
                        (existing_mu, self.embedder.embed_query(existing_mu.claim))
                        for existing_mu in self.store.list_active(conversation.conversation_id)
                    ]
                    
                    contradictions = self.contradiction_resolver.check_contradiction(
                        mu, claim_emb, existing
                    )
                    
                    for result in contradictions:
                        stats["contradictions_detected"] += 1
                        
                        # Create edge based on relationship
                        if result.relationship.value in ("updated", "superseded_by"):
                            edge = EdgeRecord(
                                source_mu_id=result.mu_a.mu_id,
                                target_mu_id=result.mu_b.mu_id,
                                edge_type=EdgeType.SUPERSEDED_BY,
                            )
                            self.store.insert_edge(edge)
                            self.graph.add_edge(edge)
                        
                        elif result.relationship.value == "contradiction":
                            edge = EdgeRecord(
                                source_mu_id=result.mu_a.mu_id,
                                target_mu_id=result.mu_b.mu_id,
                                edge_type=EdgeType.CONFLICTS_WITH,
                            )
                            self.store.insert_edge(edge)
                            self.graph.add_edge(edge)
                        
                        elif result.relationship.value == "related":
                            edge = EdgeRecord(
                                source_mu_id=result.mu_a.mu_id,
                                target_mu_id=result.mu_b.mu_id,
                                edge_type=EdgeType.RELATED_TO,
                            )
                            self.store.insert_edge(edge)
                            self.graph.add_edge(edge)
                
                # Store MU
                self.store.insert_memory_unit(mu)
                self.graph.upsert_memory_unit(mu)
                stats["memory_units_created"] += 1
        
        # Step 8: Check for state transitions
        decisions = self.transition_engine.check_and_transition(conversation.conversation_id)
        stats["transitions_executed"] = len(decisions)
        
        logger.info(
            "Ingestion complete: %d turns → %d chunks → %d facts → %d MUs",
            stats["total_turns"],
            stats["semantic_chunks"],
            stats["facts_extracted"],
            stats["memory_units_created"],
        )
        
        return stats
    
    def query(self, conversation_id: str, question: str) -> dict:
        """Query memory for a conversation.
        
        Returns:
            Dict with retrieved_mus, context, latency
        """
        result = self.retriever.retrieve(conversation_id, question)
        
        context = self.context_builder.build_context(
            result.retrieved_mus,
            result.restored_from_compressed,
        )
        
        return {
            "conversation_id": conversation_id,
            "question": question,
            "retrieved_mus": result.retrieved_mus,
            "context": context,
            "retrieval_latency_ms": result.retrieval_latency_ms,
            "restored_from_compressed": result.restored_from_compressed,
        }
    
    def _create_memory_unit(
        self,
        conversation_id: str,
        session_id: str,
        claim: str,
    ) -> MemoryUnit:
        """Create a new Memory Unit from an extracted fact."""
        return MemoryUnit(
            mu_id=new_mu_id(),
            conversation_id=conversation_id,
            session_id=session_id,
            claim=claim,
            original_text=claim,
            source_dia_ids=[],
            source_speaker="",
        )
