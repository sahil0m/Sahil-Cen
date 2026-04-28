"""Memory candidate detector: cheap filter before LLM extraction.

Rule-based scoring to avoid calling the extraction LLM on chunks that
obviously contain no extractable facts (e.g., pure greetings, emotional
venting with no concrete facts, speculation).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Common greetings and affirmations to skip
TRIVIAL_PATTERNS = frozenset({
    "hi", "hello", "hey", "ok", "okay", "yeah", "yep", "sure",
    "thanks", "thank you", "bye", "goodbye", "haha", "lol",
})


@dataclass
class CandidateScore:
    """Scoring result for a memory candidate chunk."""
    
    text: str
    score: float
    is_candidate: bool
    reason: str


class MemoryCandidateDetector:
    """Rule-based detector for memory-worthy chunks.
    
    Scores chunks on multiple factors without calling an LLM:
    - Named entity presence
    - Verb density (action vs filler)
    - Factual statement markers
    - Concrete topic markers
    - Length normalization
    - Specific numbers/dates
    
    Args:
        threshold: Minimum score to be considered a candidate (default 0.35)
    """
    
    def __init__(self, threshold: float = 0.35) -> None:
        self.threshold = threshold
    
    def is_candidate(self, text: str) -> CandidateScore:
        """Check if text is worth sending to LLM for extraction."""
        # Trivial filter first
        if self._is_trivial(text):
            return CandidateScore(
                text=text,
                score=0.0,
                is_candidate=False,
                reason="trivial",
            )
        
        # Multi-factor scoring
        score = (
            0.30 * self._has_named_entity(text)
            + 0.20 * self._verb_density(text)
            + 0.15 * self._is_factual_statement(text)
            + 0.15 * self._has_concrete_topic(text)
            + 0.10 * self._length_normalized(text)
            + 0.10 * self._has_specific_number_or_date(text)
        )
        
        is_candidate = score >= self.threshold
        reason = "candidate" if is_candidate else "low_score"
        
        return CandidateScore(
            text=text,
            score=score,
            is_candidate=is_candidate,
            reason=reason,
        )
    
    def _is_trivial(self, text: str) -> bool:
        """Check if text is a trivial greeting/affirmation."""
        words = text.lower().split()
        if len(words) < 5:
            # Check if all words are trivial
            non_trivial = [w for w in words if w not in TRIVIAL_PATTERNS]
            if len(non_trivial) <= 1:
                return True
        return False
    
    def _has_named_entity(self, text: str) -> float:
        """Score based on named entity presence (simple heuristics)."""
        # Capitalized words (excluding sentence starts)
        words = text.split()
        if not words:
            return 0.0
        
        capitalized = sum(
            1 for i, w in enumerate(words)
            if i > 0 and w and w[0].isupper() and len(w) > 1
        )
        
        # Date patterns
        has_date = bool(re.search(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\b',
            text,
            re.IGNORECASE,
        ))
        
        # Organization markers
        has_org = bool(re.search(
            r'\b(?:Inc|LLC|Corp|Company|University|Hospital|School)\b',
            text,
            re.IGNORECASE,
        ))
        
        score = min(1.0, (capitalized / max(len(words), 1)) * 3)
        if has_date:
            score = min(1.0, score + 0.3)
        if has_org:
            score = min(1.0, score + 0.3)
        
        return score
    
    def _verb_density(self, text: str) -> float:
        """Score based on action verb density."""
        # Common action verbs
        action_verbs = {
            "work", "works", "worked", "working",
            "start", "starts", "started", "starting",
            "join", "joins", "joined", "joining",
            "move", "moves", "moved", "moving",
            "buy", "buys", "bought", "buying",
            "sell", "sells", "sold", "selling",
            "meet", "meets", "met", "meeting",
            "call", "calls", "called", "calling",
            "visit", "visits", "visited", "visiting",
            "go", "goes", "went", "going",
            "come", "comes", "came", "coming",
            "leave", "leaves", "left", "leaving",
            "quit", "quits", "quitting",
            "hire", "hires", "hired", "hiring",
        }
        
        words = text.lower().split()
        if not words:
            return 0.0
        
        verb_count = sum(1 for w in words if w in action_verbs)
        return min(1.0, verb_count / max(len(words), 1) * 10)
    
    def _is_factual_statement(self, text: str) -> float:
        """Score based on factual statement markers."""
        # Questions reduce score
        if text.strip().endswith("?"):
            return 0.3
        
        # Opinion markers reduce score
        opinion_markers = ["think", "feel", "believe", "maybe", "perhaps", "might"]
        has_opinion = any(marker in text.lower() for marker in opinion_markers)
        if has_opinion:
            return 0.5
        
        # Declarative statements score higher
        return 1.0
    
    def _has_concrete_topic(self, text: str) -> float:
        """Score based on concrete topic markers."""
        topic_markers = {
            "work", "job", "career", "company", "office",
            "family", "child", "parent", "spouse", "sibling",
            "health", "doctor", "hospital", "surgery", "medicine",
            "home", "house", "apartment", "move", "city",
            "school", "university", "degree", "study",
            "project", "meeting", "deadline", "client",
        }
        
        words = text.lower().split()
        matches = sum(1 for w in words if w in topic_markers)
        return min(1.0, matches * 0.5)
    
    def _length_normalized(self, text: str) -> float:
        """Score based on ideal length (30-150 words)."""
        words = text.split()
        word_count = len(words)
        
        if word_count < 10:
            return 0.3
        if 30 <= word_count <= 150:
            return 1.0
        if word_count > 200:
            return 0.5
        
        # Linear interpolation for 10-30 and 150-200
        if word_count < 30:
            return 0.3 + (word_count - 10) / 20 * 0.7
        else:  # 150 < word_count <= 200
            return 1.0 - (word_count - 150) / 50 * 0.5
    
    def _has_specific_number_or_date(self, text: str) -> float:
        """Score based on specific numbers or dates."""
        # Numbers
        has_number = bool(re.search(r'\b\d+\b', text))
        
        # Dates
        has_date = bool(re.search(
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|'
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b|'
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            text,
            re.IGNORECASE,
        ))
        
        if has_date:
            return 1.0
        if has_number:
            return 0.7
        return 0.0
