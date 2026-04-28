"""Context builder: formats retrieved MUs into structured prompt sections."""

from __future__ import annotations

import logging

from locomo_memory.phase2.schemas import MemoryStatus, MemoryUnit
from locomo_memory.phase2.store import MemoryStore

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds structured context from retrieved Memory Units.
    
    Sections:
    - ACTIVE MEMORIES (use these first)
    - HISTORICAL CONTEXT (superseded, kept for reference)
    - CONFLICTING (treat with caution)
    - RESTORED (from compressed, label match)
    """
    
    def __init__(self, store: MemoryStore) -> None:
        self.store = store
    
    def build_context(
        self,
        retrieved_mus: list[MemoryUnit],
        restored_ids: list[str],
    ) -> str:
        """Build structured context prompt from retrieved MUs."""
        sections: list[str] = []
        
        # Separate by status and flags
        active: list[MemoryUnit] = []
        superseded: list[MemoryUnit] = []
        conflicted: list[MemoryUnit] = []
        restored: list[MemoryUnit] = []
        
        for mu in retrieved_mus:
            if mu.mu_id in restored_ids:
                restored.append(mu)
            elif mu.status == MemoryStatus.ACTIVE:
                # Check if superseded or conflicted
                edges_from = self.store.edges_from(mu.mu_id)
                has_superseded = any(e.edge_type.value == "superseded_by" for e in edges_from)
                has_conflict = any(e.edge_type.value == "conflicts_with" for e in edges_from)
                
                if has_superseded:
                    superseded.append(mu)
                elif has_conflict:
                    conflicted.append(mu)
                else:
                    active.append(mu)
        
        # Build sections
        if active:
            sections.append(self._format_active_section(active))
        
        if superseded:
            sections.append(self._format_superseded_section(superseded))
        
        if conflicted:
            sections.append(self._format_conflicted_section(conflicted))
        
        if restored:
            sections.append(self._format_restored_section(restored))
        
        return "\n\n".join(sections)
    
    def _format_active_section(self, mus: list[MemoryUnit]) -> str:
        """Format active memories section."""
        lines = ["ACTIVE MEMORIES (use these first):"]
        
        for i, mu in enumerate(mus, 1):
            lines.append(
                f"[{i}] {mu.claim}\n"
                f"    Source: Session {mu.session_id}, {mu.timestamp or 'unknown date'} | "
                f"Confidence: {mu.confidence:.2f}"
            )
        
        return "\n".join(lines)
    
    def _format_superseded_section(self, mus: list[MemoryUnit]) -> str:
        """Format superseded memories section."""
        lines = ["HISTORICAL CONTEXT (superseded, kept for reference):"]
        
        for i, mu in enumerate(mus, 1):
            # Find what superseded it
            edges = self.store.edges_from(mu.mu_id)
            superseded_by_edges = [e for e in edges if e.edge_type.value == "superseded_by"]
            
            superseded_by_text = ""
            if superseded_by_edges:
                target_mu = self.store.get_memory_unit(superseded_by_edges[0].target_mu_id)
                if target_mu:
                    superseded_by_text = f"\n    SUPERSEDED by: {target_mu.claim}"
            
            lines.append(
                f"[{i}] {mu.claim}{superseded_by_text}"
            )
        
        return "\n".join(lines)
    
    def _format_conflicted_section(self, mus: list[MemoryUnit]) -> str:
        """Format conflicted memories section."""
        lines = ["CONFLICTING (treat with caution):"]
        
        for i, mu in enumerate(mus, 1):
            # Find conflicts
            edges = self.store.edges_from(mu.mu_id)
            conflict_edges = [e for e in edges if e.edge_type.value == "conflicts_with"]
            
            conflict_text = ""
            if conflict_edges:
                target_mu = self.store.get_memory_unit(conflict_edges[0].target_mu_id)
                if target_mu:
                    conflict_text = f" — CONFLICTS WITH: {target_mu.claim}"
            
            lines.append(f"[{i}] {mu.claim}{conflict_text}")
        
        return "\n".join(lines)
    
    def _format_restored_section(self, mus: list[MemoryUnit]) -> str:
        """Format restored memories section."""
        lines = ["RESTORED FROM COMPRESSED (label match → full data fetched):"]
        
        for i, mu in enumerate(mus, 1):
            lines.append(
                f"[{i}] {mu.claim}\n"
                f"    Restored because query matched compressed label\n"
                f"    Full original: {mu.original_text[:200]}"
            )
        
        return "\n".join(lines)
