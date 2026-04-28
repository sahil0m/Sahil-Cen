"""Agentic chunker: LLM-based fact extraction from semantic chunks.

Uses a cheap LLM (llama-3.1-8b-instruct) to extract atomic facts from
conversation chunks. The LLM makes decisions about what counts as a
complete fact, what to merge, and what to skip.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from tenacity import (  # type: ignore
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from locomo_memory.phase2.ingestion.semantic_chunker import SemanticChunk

load_dotenv()
logger = logging.getLogger(__name__)


EXTRACTION_PROMPT_TEMPLATE = """\
You are a memory extraction agent. Read this conversation chunk and extract atomic facts.

Rules:
1. One fact per line, complete and standalone
2. Normalize entity names to full form (resolve pronouns)
3. Merge facts that say the same thing differently
4. Skip opinions, questions, uncertain statements
5. Maximum 7 facts per chunk
6. Return ONLY valid JSON, no markdown fences

Chunk [Session {session_id} | {timestamp}]:
{chunk_text}

Return exactly this JSON structure:
{{"facts": ["fact 1", "fact 2", ...]}}
"""


@dataclass
class ExtractionResult:
    """Result of fact extraction from a chunk."""
    
    chunk: SemanticChunk
    facts: list[str]
    extraction_latency_ms: float
    cache_hit: bool


class AgenticChunker:
    """LLM-based fact extractor with caching.
    
    Args:
        model_name: OpenRouter model name (default: llama-3.1-8b-instruct)
        api_key: OpenRouter API key (reads from OPENROUTER_API_KEY env var)
        cache_dir: Directory for response caching
        max_retries: Maximum retry attempts on failure
    """
    
    def __init__(
        self,
        model_name: str = "meta-llama/llama-3.1-8b-instruct",
        api_key: str | None = None,
        cache_dir: str | Path | None = None,
        max_retries: int = 3,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise EnvironmentError(
                "OPENROUTER_API_KEY not set. Add it to .env or pass as argument."
            )
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_retries = max_retries
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def extract_facts(self, chunk: SemanticChunk) -> ExtractionResult:
        """Extract atomic facts from a semantic chunk."""
        prompt = self._build_prompt(chunk)
        
        # Check cache
        cached = self._load_cache(prompt)
        if cached is not None:
            logger.debug("Cache hit for chunk %s", chunk.dia_ids)
            return ExtractionResult(
                chunk=chunk,
                facts=cached,
                extraction_latency_ms=0.0,
                cache_hit=True,
            )
        
        # Call LLM
        t0 = time.perf_counter()
        facts = self._call_llm_with_retry(prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        # Save to cache
        self._save_cache(prompt, facts)
        
        logger.info(
            "Extracted %d facts from chunk %s in %.1f ms",
            len(facts),
            chunk.dia_ids,
            latency_ms,
        )
        
        return ExtractionResult(
            chunk=chunk,
            facts=facts,
            extraction_latency_ms=latency_ms,
            cache_hit=False,
        )
    
    def _build_prompt(self, chunk: SemanticChunk) -> str:
        """Build extraction prompt for a chunk."""
        timestamp = chunk.turns[0].timestamp if chunk.turns else ""
        return EXTRACTION_PROMPT_TEMPLATE.format(
            session_id=chunk.session_id,
            timestamp=timestamp,
            chunk_text=chunk.text,
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def _call_llm_with_retry(self, prompt: str) -> list[str]:
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
                "max_tokens": 500,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        
        data = response.json()
        content = data["choices"][0]["message"]["content"].strip()
        
        # Parse JSON response
        # Remove markdown fences if present
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1]) if len(lines) > 2 else content
        
        try:
            parsed = json.loads(content)
            facts = parsed.get("facts", [])
            return [f.strip() for f in facts if f and f.strip()]
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON: %s", content[:200])
            return []
    
    def _cache_key(self, prompt: str) -> str:
        """Generate cache key from model name and prompt."""
        key_str = f"{self.model_name}||{prompt}"
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _cache_path(self, prompt: str) -> Path | None:
        """Get cache file path for a prompt."""
        if self.cache_dir is None:
            return None
        key = self._cache_key(prompt)
        return self.cache_dir / f"extract_{key}.json"
    
    def _load_cache(self, prompt: str) -> list[str] | None:
        """Load cached extraction result."""
        path = self._cache_path(prompt)
        if path is None or not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return data.get("facts", [])
        except Exception:
            return None
    
    def _save_cache(self, prompt: str, facts: list[str]) -> None:
        """Save extraction result to cache."""
        path = self._cache_path(prompt)
        if path is None:
            return
        try:
            path.write_text(
                json.dumps({"facts": facts}, ensure_ascii=False),
                encoding="utf-8",
            )
        except Exception as exc:
            logger.warning("Failed to save extraction cache: %s", exc)
