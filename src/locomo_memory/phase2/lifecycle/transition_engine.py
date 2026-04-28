"""State transition engine: manages memory lifecycle at 90% capacity.

Automatic transitions only fire when active memory hits ~90% of capacity.
Below that threshold, only user overrides are honored.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    MemoryStatus,
    MemoryUnit,
)
from locomo_memory.phase2.store import MemoryGraphIndex, MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class TransitionDecision:
    """Decision about what to do with a Memory Unit."""
    
    mu_id: str
    current_status: MemoryStatus
    target_status: MemoryStatus
    demotion_score: float
    reason: str


class TransitionEngine:
    """Manages automatic memory state transitions at capacity threshold.
    
    Args:
        store: SQLite memory store
        graph: NetworkX graph index
        storage_cap: Maximum active MUs per conversation
        trigger_pct: Capacity percentage that triggers transitions (default 0.90)
        target_pct: Target capacity after transitions (default 0.70)
    """
    
    def __init__(
        self,
        store: MemoryStore,
        graph: MemoryGraphIndex,
        storage_cap: int = 500,
        trigger_pct: float = 0.90,
        target_pct: float = 0.70,
    ) -> None:
        self.store = store
        self.graph = graph
        self.storage_cap = storage_cap
        self.trigger_pct = trigger_pct
        self.target_pct = target_pct
    
    def check_and_transition(self, conversation_id: str) -> list[TransitionDecision]:
        """Check if transitions are needed and execute them.
        
        Returns:
            List of transition decisions made
        """
        pressure = self.store.storage_pressure(conversation_id, self.storage_cap)
        
        if pressure < self.trigger_pct:
            logger.debug(
                "Storage pressure %.1f%% < %.1f%% threshold — no auto-transitions",
                pressure * 100,
                self.trigger_pct * 100,
            )
            return []
        
        logger.info(
            "Storage pressure %.1f%% >= %.1f%% — running transition engine",
            pressure * 100,
            self.trigger_pct * 100,
        )
        
        # Get all active MUs
        active_mus = self.store.list_active(conversation_id)
        
        # Compute demotion scores
        scored = [
            (mu, self._compute_demotion_score(mu))
            for mu in active_mus
        ]
        
        # Sort by demotion score (highest first = most demotable)
        scored.sort(key=lambda x: x[1], reverse=True)
        
        # Determine how many to demote
        current_count = len(active_mus)
        target_count = int(self.storage_cap * self.target_pct)
        n_to_demote = max(0, current_count - target_count)
        
        logger.info(
            "Active: %d, Target: %d, Will demote: %d",
            current_count,
            target_count,
            n_to_demote,
        )
        
        # Process top candidates
        decisions: list[TransitionDecision] = []
        for mu, score in scored[:n_to_demote]:
            decision = self._decide_target_status(mu, score)
            decisions.append(decision)
            
            # Execute transition
            self._execute_transition(mu, decision)
        
        return decisions
    
    def _compute_demotion_score(self, mu: MemoryUnit) -> float:
        """Compute demotion score for a Memory Unit.
        
        Higher score = more likely to be demoted.
        """
        # Never demote pinned MUs
        if mu.user_pinned:
            return -1000.0
        
        # Get graph centrality
        centrality = self._get_centrality(mu.mu_id)
        
        # Multi-factor demotion score
        w1, w2, w3, w4, w5 = 0.25, 0.25, 0.20, 0.15, 0.15
        
        score = (
            w1 * (1 - mu.salience_score)
            + w2 * (1 - mu.prompt_frequency)
            + w3 * (1 - mu.recency_weight)
            + w4 * (1 - centrality)
            + w5 * self._redundancy_score(mu)
        )
        
        return score
    
    def _get_centrality(self, mu_id: str) -> float:
        """Get graph centrality for a MU (0-1 normalized)."""
        if not self.graph.has_node(mu_id):
            return 0.0
        
        # Use degree centrality (fast)
        centralities = self.graph.degree_centrality()
        return centralities.get(mu_id, 0.0)
    
    def _redundancy_score(self, mu: MemoryUnit) -> float:
        """Estimate redundancy (placeholder for now)."""
        # In full implementation, check similarity to other active MUs
        return 0.0
    
    def _decide_target_status(
        self,
        mu: MemoryUnit,
        demotion_score: float,
    ) -> TransitionDecision:
        """Decide what status to transition to."""
        # Very low value + never retrieved → forget
        if mu.salience_score < 0.15 and mu.retrieval_count == 0:
            return TransitionDecision(
                mu_id=mu.mu_id,
                current_status=mu.status,
                target_status=MemoryStatus.FORGOTTEN,
                demotion_score=demotion_score,
                reason="low_salience_never_used",
            )
        
        # Low value or old + rarely used → compress
        import math
        age_days = (mu.updated_at - mu.extracted_at).total_seconds() / 86400.0
        
        if mu.salience_score < 0.40 or (age_days > 30 and mu.retrieval_count < 2):
            return TransitionDecision(
                mu_id=mu.mu_id,
                current_status=mu.status,
                target_status=MemoryStatus.COMPRESSED,
                demotion_score=demotion_score,
                reason="low_value_or_old_unused",
            )
        
        # Default: keep active (shouldn't reach here if scoring is correct)
        return TransitionDecision(
            mu_id=mu.mu_id,
            current_status=mu.status,
            target_status=MemoryStatus.ACTIVE,
            demotion_score=demotion_score,
            reason="keep_active",
        )
    
    def _execute_transition(
        self,
        mu: MemoryUnit,
        decision: TransitionDecision,
    ) -> None:
        """Execute a state transition."""
        if decision.target_status == MemoryStatus.FORGOTTEN:
            self.store.forget_atomic(mu.mu_id)
            logger.info(
                "Forgot %s (score=%.3f, reason=%s)",
                mu.mu_id,
                decision.demotion_score,
                decision.reason,
            )
        
        elif decision.target_status == MemoryStatus.COMPRESSED:
            # Generate label and archive
            label = self._generate_label(mu)
            archive = self._generate_archive(mu, label.label_id)
            
            self.store.compress_atomic(mu.mu_id, label, archive)
            logger.info(
                "Compressed %s (score=%.3f, reason=%s)",
                mu.mu_id,
                decision.demotion_score,
                decision.reason,
            )
    
    def _generate_label(self, mu: MemoryUnit) -> CompressedLabel:
        """Generate a compressed label for a Memory Unit.
        
        In full implementation, this would call an LLM to generate a
        smart summary. For now, use first 10 words as label.
        """
        words = mu.claim.split()
        short_summary = " ".join(words[:10])
        if len(words) > 10:
            short_summary += "..."
        
        # Extract entities (simple heuristic)
        entities = [
            w for w in words
            if w and w[0].isupper() and len(w) > 1
        ][:5]
        
        from locomo_memory.phase2.schemas import new_archive_id, new_label_id
        
        archive_id = new_archive_id()
        label_id = new_label_id()
        
        return CompressedLabel(
            label_id=label_id,
            archived_pointer=archive_id,
            mu_id=mu.mu_id,
            conversation_id=mu.conversation_id,
            topic="general",  # Placeholder
            short_summary=short_summary,
            key_entities=entities,
            time_range=mu.timestamp,
            original_dia_ids=mu.source_dia_ids,
        )
    
    def _generate_archive(
        self,
        mu: MemoryUnit,
        label_id: str,
    ) -> ArchivedEntry:
        """Generate an archived entry for a Memory Unit."""
        from locomo_memory.phase2.schemas import new_archive_id
        
        archive_id = new_archive_id()
        
        return ArchivedEntry(
            archived_entry_id=archive_id,
            label_pointer=label_id,
            mu_id=mu.mu_id,
            conversation_id=mu.conversation_id,
            full_memory_unit_json=mu.model_dump_json(),
            full_original_text=mu.original_text,
        )
