"""Salience scorer: multi-factor scoring for Memory Units.

No LLM calls — pure math based on:
- Entity density
- Recency weight
- Topic importance
- Uniqueness
- Prompt frequency
- User pin bonus
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from locomo_memory.phase2.schemas import MemoryUnit

logger = logging.getLogger(__name__)


# Topic importance weights (rule-based)
TOPIC_WEIGHTS = {
    "work": 0.9,
    "job": 0.9,
    "career": 0.9,
    "family": 0.95,
    "health": 1.0,
    "surgery": 1.0,
    "medical": 1.0,
    "home": 0.8,
    "move": 0.85,
    "school": 0.8,
    "project": 0.7,
    "meeting": 0.6,
}


class SalienceScorer:
    """Multi-factor salience scoring for Memory Units.
    
    Args:
        recency_decay_days: Half-life for recency weight (default 30 days)
    """
    
    def __init__(self, recency_decay_days: float = 30.0) -> None:
        self.recency_decay_days = recency_decay_days
    
    def score(
        self,
        mu: MemoryUnit,
        existing_embeddings: list[tuple[str, list[float]]] | None = None,
    ) -> float:
        """Compute salience score for a Memory Unit.
        
        Args:
            mu: Memory Unit to score
            existing_embeddings: List of (mu_id, embedding) for uniqueness calculation
        
        Returns:
            Salience score in [0, 1]
        """
        entity_density = self._entity_density(mu.claim)
        recency_weight = self._recency_weight(mu.extracted_at)
        topic_importance = self._topic_importance(mu.claim)
        uniqueness = self._uniqueness(mu, existing_embeddings)
        prompt_frequency = mu.prompt_frequency
        user_pin_bonus = 1.0 if mu.user_pinned else 0.0
        
        salience = (
            0.25 * entity_density
            + 0.20 * recency_weight
            + 0.20 * topic_importance
            + 0.15 * uniqueness
            + 0.10 * prompt_frequency
            + 0.10 * user_pin_bonus
        )
        
        return min(1.0, max(0.0, salience))
    
    def _entity_density(self, text: str) -> float:
        """Score based on named entity density."""
        words = text.split()
        if not words:
            return 0.0
        
        # Count capitalized words (simple entity proxy)
        capitalized = sum(
            1 for w in words
            if w and w[0].isupper() and len(w) > 1
        )
        
        # Normalize by word count
        density = capitalized / len(words)
        return min(1.0, density * 3)
    
    def _recency_weight(self, extracted_at: datetime) -> float:
        """Exponential decay based on age."""
        now = datetime.now(timezone.utc)
        age_days = (now - extracted_at).total_seconds() / 86400.0
        
        # Exponential decay: weight = exp(-age / half_life)
        import math
        decay_rate = math.log(2) / self.recency_decay_days
        weight = math.exp(-decay_rate * age_days)
        
        return min(1.0, max(0.0, weight))
    
    def _topic_importance(self, text: str) -> float:
        """Rule-based topic importance."""
        text_lower = text.lower()
        
        # Find highest-weight topic mentioned
        max_weight = 0.5  # default for unrecognized topics
        for topic, weight in TOPIC_WEIGHTS.items():
            if topic in text_lower:
                max_weight = max(max_weight, weight)
        
        return max_weight
    
    def _uniqueness(
        self,
        mu: MemoryUnit,
        existing_embeddings: list[tuple[str, list[float]]] | None,
    ) -> float:
        """Uniqueness based on similarity to existing MUs.
        
        If no embeddings provided, returns 1.0 (assume unique).
        Otherwise, returns 1 - max_similarity.
        """
        if not existing_embeddings:
            return 1.0
        
        # This would require embedding the new MU's claim
        # For now, return a placeholder
        # In full implementation, compute cosine similarity to all existing
        return 1.0
    
    def update_salience(self, mu: MemoryUnit) -> MemoryUnit:
        """Update all salience-related fields on a Memory Unit."""
        mu.salience_score = self.score(mu)
        mu.importance = self._topic_importance(mu.claim)
        mu.recency_weight = self._recency_weight(mu.extracted_at)
        mu.uniqueness = 1.0  # Placeholder
        
        return mu
