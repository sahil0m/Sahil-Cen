"""Parallel retrieval pipeline with 4 workers + RRF fusion + reranking.

Workers:
1. Dense FAISS over active MUs
2. BM25 over active MUs
3. Compressed label FAISS
4. Graph traversal (1-hop neighbors of top candidates)
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from locomo_memory.indexing.embeddings import EmbeddingGenerator
from locomo_memory.phase2.schemas import MemoryUnit
from locomo_memory.phase2.store import MemoryGraphIndex, MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Result of parallel retrieval."""
    
    conversation_id: str
    question: str
    retrieved_mus: list[MemoryUnit]
    retrieval_latency_ms: float
    restored_from_compressed: list[str]  # MU IDs restored


class ParallelRetriever:
    """4-worker parallel retrieval with RRF fusion.
    
    Args:
        store: SQLite memory store
        graph: NetworkX graph index
        embedder: Embedding generator
        top_k: Final number of MUs to return
        candidate_k: Candidates per worker before fusion
        enable_reranker: Use cross-encoder reranking
        enable_compressed_search: Search compressed labels
        enable_graph_traversal: Use graph worker
        enable_forgotten_fallback: Search forgotten tier if confidence low
    """
    
    def __init__(
        self,
        store: MemoryStore,
        graph: MemoryGraphIndex,
        embedder: EmbeddingGenerator,
        top_k: int = 5,
        candidate_k: int = 30,
        enable_reranker: bool = True,
        enable_compressed_search: bool = True,
        enable_graph_traversal: bool = True,
        enable_forgotten_fallback: bool = True,
    ) -> None:
        self.store = store
        self.graph = graph
        self.embedder = embedder
        self.top_k = top_k
        self.candidate_k = candidate_k
        self.enable_reranker = enable_reranker
        self.enable_compressed_search = enable_compressed_search
        self.enable_graph_traversal = enable_graph_traversal
        self.enable_forgotten_fallback = enable_forgotten_fallback
        
        # Build active FAISS index (will be rebuilt as MUs change)
        self._active_index: dict[str, tuple[list[MemoryUnit], np.ndarray]] = {}
        self._compressed_index: dict[str, tuple[list, np.ndarray]] = {}
    
    def retrieve(
        self,
        conversation_id: str,
        question: str,
    ) -> RetrievalResult:
        """Retrieve top-k Memory Units for a question."""
        t0 = time.perf_counter()
        
        # Embed question
        query_emb = self.embedder.embed_query(question)
        
        # Run workers in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            # Worker 1: Dense FAISS
            futures.append(executor.submit(
                self._worker_dense_faiss,
                conversation_id,
                query_emb,
            ))
            
            # Worker 2: BM25
            futures.append(executor.submit(
                self._worker_bm25,
                conversation_id,
                question,
            ))
            
            # Worker 3: Compressed labels (if enabled)
            if self.enable_compressed_search:
                futures.append(executor.submit(
                    self._worker_compressed_labels,
                    conversation_id,
                    query_emb,
                ))
            
            # Collect results
            all_candidates: list[list[tuple[MemoryUnit, float]]] = []
            for future in as_completed(futures):
                candidates = future.result()
                if candidates:
                    all_candidates.append(candidates)
        
        # RRF fusion
        fused = self._rrf_fuse(all_candidates, self.candidate_k)
        
        # Worker 4: Graph traversal (after we have initial candidates)
        if self.enable_graph_traversal and fused:
            graph_candidates = self._worker_graph_traversal(
                conversation_id,
                [mu for mu, _ in fused[:5]],
            )
            if graph_candidates:
                all_candidates.append(graph_candidates)
                fused = self._rrf_fuse(all_candidates, self.candidate_k)
        
        # Reranking (if enabled)
        if self.enable_reranker and fused:
            reranked = self._rerank(question, fused)
        else:
            reranked = fused
        
        # Take top-k
        top_mus = [mu for mu, _ in reranked[:self.top_k]]
        
        # Check if we need forgotten fallback
        restored_ids: list[str] = []
        if self.enable_forgotten_fallback and top_mus:
            mean_score = sum(score for _, score in reranked[:self.top_k]) / len(reranked[:self.top_k])
            if mean_score < 0.5:
                logger.info("Low confidence (%.3f) — searching forgotten tier", mean_score)
                forgotten_mus = self._search_forgotten(conversation_id, query_emb)
                top_mus.extend(forgotten_mus[:2])
        
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        logger.info(
            "Retrieved %d MUs for conversation %s in %.1f ms",
            len(top_mus),
            conversation_id,
            latency_ms,
        )
        
        return RetrievalResult(
            conversation_id=conversation_id,
            question=question,
            retrieved_mus=top_mus,
            retrieval_latency_ms=latency_ms,
            restored_from_compressed=restored_ids,
        )
    
    def _worker_dense_faiss(
        self,
        conversation_id: str,
        query_emb: np.ndarray,
    ) -> list[tuple[MemoryUnit, float]]:
        """Worker 1: Dense FAISS search over active MUs."""
        # Rebuild index if needed
        if conversation_id not in self._active_index:
            self._rebuild_active_index(conversation_id)
        
        mus, embeddings = self._active_index.get(conversation_id, ([], np.array([])))
        if len(mus) == 0:
            return []
        
        # Cosine similarity (embeddings are normalized)
        scores = embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:self.candidate_k]
        
        return [(mus[i], float(scores[i])) for i in top_indices]
    
    def _worker_bm25(
        self,
        conversation_id: str,
        question: str,
    ) -> list[tuple[MemoryUnit, float]]:
        """Worker 2: BM25 search over active MU claims."""
        from rank_bm25 import BM25Okapi  # type: ignore
        
        mus = self.store.list_active(conversation_id)
        if not mus:
            return []
        
        # Tokenize
        corpus = [mu.claim.lower().split() for mu in mus]
        query_tokens = question.lower().split()
        
        # BM25 scoring
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query_tokens)
        
        # Top candidates
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(mus[i], float(s)) for i, s in ranked[:self.candidate_k] if s > 0]
    
    def _worker_compressed_labels(
        self,
        conversation_id: str,
        query_emb: np.ndarray,
    ) -> list[tuple[MemoryUnit, float]]:
        """Worker 3: Search compressed labels, restore on match."""
        # Rebuild compressed index if needed
        if conversation_id not in self._compressed_index:
            self._rebuild_compressed_index(conversation_id)
        
        labels, embeddings = self._compressed_index.get(conversation_id, ([], np.array([])))
        if len(labels) == 0:
            return []
        
        # Search labels
        scores = embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:10]  # Top 10 labels
        
        # Restore full MUs from archive
        restored: list[tuple[MemoryUnit, float]] = []
        for i in top_indices:
            label = labels[i]
            score = float(scores[i])
            
            # Fetch archive
            archive = self.store.get_archived_entry(label.archived_pointer)
            if archive:
                # Restore to active
                mu = self.store.restore_atomic(label.mu_id)
                restored.append((mu, score))
                logger.info("Restored %s from compressed (label match)", mu.mu_id)
        
        return restored
    
    def _worker_graph_traversal(
        self,
        conversation_id: str,
        seed_mus: list[MemoryUnit],
    ) -> list[tuple[MemoryUnit, float]]:
        """Worker 4: Graph traversal from seed MUs."""
        neighbor_ids: set[str] = set()
        
        for mu in seed_mus:
            neighbors = self.graph.neighbors(mu.mu_id)
            neighbor_ids.update(neighbors)
        
        # Remove seeds
        neighbor_ids -= {mu.mu_id for mu in seed_mus}
        
        # Fetch neighbor MUs
        neighbors: list[tuple[MemoryUnit, float]] = []
        for mu_id in neighbor_ids:
            mu = self.store.get_memory_unit(mu_id)
            if mu and mu.status.value == "active":
                # Score based on graph centrality
                centrality = self.graph.degree_centrality().get(mu_id, 0.0)
                neighbors.append((mu, centrality))
        
        return neighbors
    
    def _rrf_fuse(
        self,
        rankings: list[list[tuple[MemoryUnit, float]]],
        top_k: int,
    ) -> list[tuple[MemoryUnit, float]]:
        """RRF fusion across multiple ranked lists."""
        RRF_K = 60
        
        doc_scores: dict[str, tuple[MemoryUnit, float]] = {}
        
        for ranking in rankings:
            for rank, (mu, _) in enumerate(ranking):
                rrf_contrib = 1.0 / (RRF_K + rank + 1)
                if mu.mu_id not in doc_scores:
                    doc_scores[mu.mu_id] = (mu, 0.0)
                prev_mu, prev_score = doc_scores[mu.mu_id]
                doc_scores[mu.mu_id] = (prev_mu, prev_score + rrf_contrib)
        
        fused = sorted(doc_scores.values(), key=lambda x: x[1], reverse=True)
        return fused[:top_k]
    
    def _rerank(
        self,
        question: str,
        candidates: list[tuple[MemoryUnit, float]],
    ) -> list[tuple[MemoryUnit, float]]:
        """Cross-encoder reranking (placeholder for now)."""
        # In full implementation, use BAAI/bge-reranker-base
        # For now, return as-is
        return candidates
    
    def _search_forgotten(
        self,
        conversation_id: str,
        query_emb: np.ndarray,
    ) -> list[MemoryUnit]:
        """Search forgotten tier as fallback."""
        forgotten = self.store.list_by_status(conversation_id, MemoryStatus.FORGOTTEN)
        if not forgotten:
            return []
        
        # Embed and search
        texts = [mu.claim for mu in forgotten]
        embeddings = self.embedder.embed_texts(texts)
        
        scores = embeddings @ query_emb
        top_indices = np.argsort(scores)[::-1][:5]
        
        return [forgotten[i] for i in top_indices]
    
    def _rebuild_active_index(self, conversation_id: str) -> None:
        """Rebuild FAISS index for active MUs."""
        mus = self.store.list_active(conversation_id)
        if not mus:
            self._active_index[conversation_id] = ([], np.array([]))
            return
        
        texts = [mu.claim for mu in mus]
        embeddings = self.embedder.embed_texts(texts)
        
        self._active_index[conversation_id] = (mus, embeddings)
        logger.debug("Rebuilt active index for %s: %d MUs", conversation_id, len(mus))
    
    def _rebuild_compressed_index(self, conversation_id: str) -> None:
        """Rebuild FAISS index for compressed labels."""
        labels = self.store.list_compressed_labels(conversation_id)
        if not labels:
            self._compressed_index[conversation_id] = ([], np.array([]))
            return
        
        texts = [label.short_summary for label in labels]
        embeddings = self.embedder.embed_texts(texts)
        
        self._compressed_index[conversation_id] = (labels, embeddings)
        logger.debug("Rebuilt compressed index for %s: %d labels", conversation_id, len(labels))
