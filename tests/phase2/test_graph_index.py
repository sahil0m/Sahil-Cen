"""Phase 2 NetworkX graph derived-index tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from locomo_memory.phase2.schemas import EdgeRecord, EdgeType, MemoryUnit
from locomo_memory.phase2.store import MemoryGraphIndex, MemoryIntegrityError, MemoryStore


def _mu(mu_id: str, claim: str = "claim") -> MemoryUnit:
    return MemoryUnit(
        mu_id=mu_id,
        conversation_id="conv-1",
        session_id="session_1",
        claim=claim,
        original_text=claim,
        source_dia_ids=["D1:1"],
        source_speaker="Caroline",
    )


def _store_with_graph_data(tmp_path: Path) -> tuple[MemoryStore, list[MemoryUnit]]:
    store = MemoryStore(tmp_path / "graph.sqlite")
    mus = [
        _mu("mu_1", "Caroline worked at Google"),
        _mu("mu_2", "Caroline works at Microsoft"),
        _mu("mu_3", "Caroline moved to Seattle"),
    ]
    for mu in mus:
        store.insert_memory_unit(mu)
    store.insert_edge(
        EdgeRecord(
            edge_id="edg_1",
            source_mu_id="mu_1",
            target_mu_id="mu_2",
            edge_type=EdgeType.SUPERSEDED_BY,
        )
    )
    store.insert_edge(
        EdgeRecord(
            edge_id="edg_2",
            source_mu_id="mu_2",
            target_mu_id="mu_3",
            edge_type=EdgeType.RELATED_TO,
        )
    )
    return store, mus


def test_rebuild_from_store_loads_nodes_and_edges(tmp_path: Path) -> None:
    store, _ = _store_with_graph_data(tmp_path)
    graph = MemoryGraphIndex()

    graph.rebuild_from_store(store)

    assert graph.node_count() == 3
    assert graph.edge_count() == 2
    assert graph.has_node("mu_1")
    assert graph.node_attrs("mu_1")["conversation_id"] == "conv-1"
    assert graph.node_attrs("mu_1")["status"] == "active"


def test_neighbors_edge_type_filters_and_k_hop(tmp_path: Path) -> None:
    store, _ = _store_with_graph_data(tmp_path)
    graph = MemoryGraphIndex()
    graph.rebuild_from_store(store)

    assert graph.neighbors("mu_2") == {"mu_1", "mu_3"}
    assert graph.neighbors("mu_2", EdgeType.SUPERSEDED_BY) == {"mu_1"}
    assert graph.neighbors("mu_2", EdgeType.RELATED_TO) == {"mu_3"}
    assert graph.k_hop_neighbors("mu_1", k=2) == {"mu_2", "mu_3"}
    assert graph.neighbors("missing") == set()


def test_manual_edge_add_requires_existing_nodes() -> None:
    graph = MemoryGraphIndex()
    edge = EdgeRecord(
        edge_id="edg_1",
        source_mu_id="mu_1",
        target_mu_id="mu_2",
        edge_type=EdgeType.RELATED_TO,
    )

    with pytest.raises(MemoryIntegrityError):
        graph.add_edge(edge)

    graph.upsert_memory_unit(_mu("mu_1"))
    graph.upsert_memory_unit(_mu("mu_2"))
    graph.add_edge(edge)

    assert graph.edge_count() == 1
    assert graph.remove_edge("edg_1") is True
    assert graph.remove_edge("edg_1") is False


def test_rebuild_rejects_dangling_edges_without_swapping_live_graph(
    tmp_path: Path,
) -> None:
    store = MemoryStore(tmp_path / "dangling.sqlite")
    graph = MemoryGraphIndex()
    graph.upsert_memory_unit(_mu("mu_existing"))

    with store.transaction() as conn:
        conn.execute(
            """
            INSERT INTO edges (
                edge_id, source_mu_id, target_mu_id, edge_type, weight,
                created_at, metadata_json
            ) VALUES (
                'edg_bad', 'mu_missing_a', 'mu_missing_b', 'related_to', 1.0,
                '2024-01-01T00:00:00+00:00', NULL
            )
            """
        )

    with pytest.raises(MemoryIntegrityError):
        graph.rebuild_from_store(store)

    assert graph.node_count() == 1
    assert graph.has_node("mu_existing")


def test_centrality_empty_graph_is_safe() -> None:
    graph = MemoryGraphIndex()

    assert graph.degree("missing") == 0
    assert graph.degree_centrality() == {}
    assert graph.betweenness_centrality() == {}
