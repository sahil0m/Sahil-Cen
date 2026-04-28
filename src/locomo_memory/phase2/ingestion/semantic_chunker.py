"""Semantic chunker: groups consecutive turns by topic similarity.

Uses BGE-small embeddings and cosine similarity to detect topic boundaries.
When similarity drops below threshold, starts a new chunk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from locomo_memory.data.schemas import Turn
from locomo_memory.indexing.embeddings import EmbeddingGenerator

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """A group of consecutive turns about the same topic."""
    
    conversation_id: str
    session_id: str
    turns: list[Turn]
    start_index: int
    end_index: int
    
    @property
    def text(self) -> str:
        """Combined text of all turns in the chunk."""
        return " ".join(t.text for t in self.turns)
    
    @property
    def dia_ids(self) -> list[str]:
        return [t.dia_id for t in self.turns]


class SemanticChunker:
    """Groups consecutive conversation turns by topic similarity.
    
    Args:
        embedder: EmbeddingGenerator for turn embeddings
        similarity_threshold: Cosine similarity threshold for same topic (default 0.65)
        min_chunk_size: Minimum turns per chunk (default 1)
        max_chunk_size: Maximum turns per chunk (default 10)
    """
    
    def __init__(
        self,
        embedder: EmbeddingGenerator,
        similarity_threshold: float = 0.65,
        min_chunk_size: int = 1,
        max_chunk_size: int = 10,
    ) -> None:
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_turns(self, turns: list[Turn]) -> list[SemanticChunk]:
        """Chunk a list of turns by topic similarity.
        
        Algorithm:
        1. Embed each turn individually
        2. Walk through turns sequentially
        3. If cosine(turn_i, turn_i-1) > threshold → extend current chunk
        4. Otherwise → close current chunk, start new one
        5. Force chunk boundary at max_chunk_size
        """
        if not turns:
            return []
        
        # Filter out summary turns
        dialogue_turns = [t for t in turns if t.speaker.lower() != "summary"]
        if not dialogue_turns:
            return []
        
        # Embed all turns
        texts = [t.text for t in dialogue_turns]
        embeddings = self.embedder.embed_texts(texts)
        
        chunks: list[SemanticChunk] = []
        current_chunk_turns: list[Turn] = [dialogue_turns[0]]
        current_chunk_start = 0
        
        for i in range(1, len(dialogue_turns)):
            prev_emb = embeddings[i - 1]
            curr_emb = embeddings[i]
            
            # Cosine similarity (embeddings are already normalized)
            similarity = float(np.dot(prev_emb, curr_emb))
            
            # Check if we should extend current chunk or start new one
            should_extend = (
                similarity >= self.similarity_threshold
                and len(current_chunk_turns) < self.max_chunk_size
            )
            
            if should_extend:
                current_chunk_turns.append(dialogue_turns[i])
            else:
                # Close current chunk if it meets min size
                if len(current_chunk_turns) >= self.min_chunk_size:
                    chunks.append(self._make_chunk(
                        current_chunk_turns,
                        current_chunk_start,
                        i - 1,
                    ))
                
                # Start new chunk
                current_chunk_turns = [dialogue_turns[i]]
                current_chunk_start = i
        
        # Close final chunk
        if len(current_chunk_turns) >= self.min_chunk_size:
            chunks.append(self._make_chunk(
                current_chunk_turns,
                current_chunk_start,
                len(dialogue_turns) - 1,
            ))
        
        logger.info(
            "Semantic chunking: %d turns → %d chunks (threshold=%.2f)",
            len(dialogue_turns),
            len(chunks),
            self.similarity_threshold,
        )
        
        return chunks
    
    def _make_chunk(
        self,
        turns: list[Turn],
        start_idx: int,
        end_idx: int,
    ) -> SemanticChunk:
        """Create a SemanticChunk from a list of turns."""
        return SemanticChunk(
            conversation_id=turns[0].conversation_id,
            session_id=turns[0].session_id,
            turns=turns,
            start_index=start_idx,
            end_index=end_idx,
        )
