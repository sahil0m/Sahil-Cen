"""Contradiction resolver: detects and classifies conflicting facts.

Two-pass pipeline:
1. FAISS similarity search (cheap filter, threshold 0.85)
2. LLM classifier (llama-3.3-70b-instruct) for relationship type
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from tenacity import (  # type: ignore
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from locomo_memory.phase2.schemas import MemoryUnit

load_dotenv()
logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """Relationship between two memory claims."""
    
    SAME = "same"
    UPDATED = "updated"
    CONTRADICTION = "contradiction"
    TEMPORAL_CHANGE = "temporal_change"
    RELATED = "related"
    UNRELATED = "unrelated"


CONTRADICTION_PROMPT_TEMPLATE = """\
You are analyzing two memory claims to determine their relationship.

Claim A ({timestamp_a}): {claim_a}
Claim B ({timestamp_b}): {claim_b}

Classify the relationship as ONE of:
- same: identical or nearly identical facts
- updated: B updates/corrects A (A is now outdated)
- contradiction: genuinely conflicting facts that cannot both be true
- temporal_change: both true but at different times (e.g., job change)
- related: related topics but not contradictory
- unrelated: completely different topics

Return ONLY valid JSON, no markdown fences:
{{"relationship": "...", "reason": "brief explanation"}}
"""


@dataclass
class ContradictionResult:
    """Result of contradiction detection between two MUs."""
    
    mu_a: MemoryUnit
    mu_b: MemoryUnit
    similarity: float
    relationship: RelationshipType
    reason: str
    llm_latency_ms: float
    cache_hit: bool


class ContradictionResolver:
    """Two-pass contradiction detector with LLM classification.
    
    Args:
        similarity_threshold: Minimum cosine similarity to trigger LLM (default 0.85)
        model_name: OpenRouter model for classification (default: llama-3.3-70b-instruct)
        api_key: OpenRouter API key
        cache_dir: Directory for response caching
    """
    
    def __init__(
        self,
        similarity_threshold: float = 0.85,
        model_name: str = "meta-llama/llama-3.3-70b-instruct",
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. Add it to .env or pass as argument."
            )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def check_contradiction(
        self,
        new_mu: MemoryUnit,
        new_embedding: np.ndarray,
        existing_mus: list[tuple[MemoryUnit, np.ndarray]],
    ) -> list[ContradictionResult]:
        """Check if new MU contradicts any existing MUs.
        
        Args:
            new_mu: New Memory Unit to check
            new_embedding: Embedding of new MU's claim
            existing_mus: List of (MemoryUnit, embedding) tuples
        
        Returns:
            List of contradiction results (only for high-similarity pairs)
        """
        results: list[ContradictionResult] = []
        
        for existing_mu, existing_emb in existing_mus:
            # Pass 1: Similarity filter
            similarity = float(np.dot(new_embedding, existing_emb))
            
            if similarity < self.similarity_threshold:
                continue
            
            # Pass 2: LLM classification
            relationship, reason, latency_ms, cache_hit = self._classify_relationship(
                new_mu, existing_mu
            )
            
            results.append(ContradictionResult(
                mu_a=existing_mu,
                mu_b=new_mu,
                similarity=similarity,
                relationship=relationship,
                reason=reason,
                llm_latency_ms=latency_ms,
                cache_hit=cache_hit,
            ))
            
            logger.info(
                "Contradiction check: %s vs %s → %s (sim=%.3f)",
                existing_mu.mu_id,
                new_mu.mu_id,
                relationship.value,
                similarity,
            )
        
        return results
    
    def _classify_relationship(
        self,
        mu_a: MemoryUnit,
        mu_b: MemoryUnit,
    ) -> tuple[RelationshipType, str, float, bool]:
        """Classify relationship between two MUs using LLM.
        
        Returns:
            (relationship_type, reason, latency_ms, cache_hit)
        """
        prompt = self._build_prompt(mu_a, mu_b)
        
        # Check cache
        cached = self._load_cache(prompt)
        if cached is not None:
            return (
                RelationshipType(cached["relationship"]),
                cached["reason"],
                0.0,
                True,
            )
        
        # Call LLM
        t0 = time.perf_counter()
        result = self._call_llm_with_retry(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        # Save to cache
        self._save_cache(prompt, result)
        
        return (
            RelationshipType(result["relationship"]),
            result["reason"],
            latency_ms,
            False,
        )
    
    def _build_prompt(self, mu_a: MemoryUnit, mu_b: MemoryUnit) -> str:
        """Build classification prompt."""
        return CONTRADICTION_PROMPT_TEMPLATE.format(
            timestamp_a=mu_a.timestamp or mu_a.extracted_at.isoformat(),
            claim_a=mu_a.claim,
            timestamp_b=mu_b.timestamp or mu_b.extracted_at.isoformat(),
            claim_b=mu_b.claim,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_llm_with_retry(self, prompt: str) -> dict:
        """Call OpenRouter API with retries."""
        import httpx
        
        response = httpx.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.0,
                "max_tokens": 200,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        try:
            parsed = json.loads(content)
            return {
                "relationship": parsed.get("relationship", "unrelated"),
                "reason": parsed.get("reason", ""),
            }
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response: %s", content[:200])
            return {"relationship": "unrelated", "reason": "parse_error"}
    
    def _cache_key(self, prompt: str) -> str:
        """Generate cache key."""
        key_str = f"{self.model_name}||{prompt}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _cache_path(self, prompt: str) -> Path | None:
        """Get cache file path."""
        if self.cache_dir is None:
            return None
        key = self._cache_key(prompt)
        return self.cache_dir / f"contradict_{key}.json"
    
    def _load_cache(self, prompt: str) -> dict | None:
        """Load cached result."""
        path = self._cache_path(prompt)
        if path is None or not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    
    def _save_cache(self, prompt: str, result: dict) -> None:
        """Save result to cache."""
        path = self._cache_path(prompt)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps(result, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save contradiction cache: %s", exc)
