"""Phase 2 SQLite MemoryStore tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    EdgeRecord,
    EdgeType,
    MemoryStatus,
    MemoryUnit,
)
from locomo_memory.phase2.store import (
    IllegalStateTransitionError,
    MemoryIntegrityError,
    MemoryStore,
    MemoryStoreError,
    MemoryUnitNotFoundError,
)
from locomo_memory.phase2.store.sqlite_store import DELETED_PLACEHOLDER


def _store(tmp_path: Path) -> MemoryStore:
    return MemoryStore(tmp_path / "phase2_memory.sqlite")


def _mu(
    *,
    mu_id: str = "mu_1",
    conversation_id: str = "conv-1",
    session_id: str = "session_1",
    claim: str = "Caroline works at Microsoft",
) -> MemoryUnit:
    return MemoryUnit(
        mu_id=mu_id,
        conversation_id=conversation_id,
        session_id=session_id,
        claim=claim,
        original_text=f"Original: {claim}",
        source_dia_ids=["D1:1"],
        source_speaker="Caroline",
        salience_score=0.8,
        importance=0.7,
    )


def _archive_for(mu: MemoryUnit, archive_id: str, label_id: str) -> ArchivedEntry:
    return ArchivedEntry(
        archived_entry_id=archive_id,
        label_pointer=label_id,
        mu_id=mu.mu_id,
        conversation_id=mu.conversation_id,
        full_memory_unit_json=mu.model_dump_json(),
        full_original_text=mu.original_text,
    )


def _label_for(mu: MemoryUnit, label_id: str, archive_id: str) -> CompressedLabel:
    return CompressedLabel(
        label_id=label_id,
        archived_pointer=archive_id,
        mu_id=mu.mu_id,
        conversation_id=mu.conversation_id,
        topic="Career",
        short_summary="Caroline works at Microsoft",
        key_entities=["Caroline", "Microsoft"],
        original_dia_ids=mu.source_dia_ids,
    )


def test_schema_initialization_is_idempotent(tmp_path: Path) -> None:
    store = _store(tmp_path)
    MemoryStore(store.db_path)

    with store.reader() as conn:
        rows = conn.execute("SELECT version FROM schema_version").fetchall()

    assert [row["version"] for row in rows] == [1]


def test_insert_get_update_and_counts(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()

    store.insert_memory_unit(mu)
    loaded = store.get_memory_unit(mu.mu_id)
    assert loaded is not None
    assert loaded.claim == mu.claim
    assert loaded.source_dia_ids == ["D1:1"]

    loaded.salience_score = 0.55
    store.update_memory_unit(loaded)
    assert store.get_memory_unit_or_raise(mu.mu_id).salience_score == 0.55

    store.increment_retrieval_count(mu.mu_id)
    loaded = store.get_memory_unit_or_raise(mu.mu_id)
    assert loaded.retrieval_count == 1
    assert loaded.last_accessed is not None

    store.set_pinned(mu.mu_id, True)
    assert store.get_memory_unit_or_raise(mu.mu_id).user_pinned is True
    assert store.count_by_status("conv-1")[MemoryStatus.ACTIVE] == 1
    assert store.storage_pressure("conv-1", cap=2) == 0.5


def test_missing_memory_unit_raises_meaningful_exception(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(MemoryUnitNotFoundError):
        store.get_memory_unit_or_raise("mu_missing")

    with pytest.raises(MemoryUnitNotFoundError):
        store.increment_retrieval_count("mu_missing")


def test_label_insert_requires_matching_archive(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    store.insert_memory_unit(mu)
    label = _label_for(mu, label_id="lbl_1", archive_id="arc_missing")

    with pytest.raises(MemoryIntegrityError):
        store.insert_compressed_label(label)


def test_archive_insert_requires_existing_memory_unit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    archive = _archive_for(mu, archive_id="arc_1", label_id="lbl_1")

    with pytest.raises(MemoryUnitNotFoundError):
        store.insert_archived_entry(archive)


def test_compress_atomic_writes_label_archive_and_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    store.insert_memory_unit(mu)
    archive = _archive_for(mu, archive_id="arc_1", label_id="lbl_1")
    label = _label_for(mu, label_id="lbl_1", archive_id="arc_1")

    store.compress_atomic(mu.mu_id, label, archive)

    compressed = store.get_memory_unit_or_raise(mu.mu_id)
    assert compressed.status == MemoryStatus.COMPRESSED
    assert compressed.compressed_label_id == "lbl_1"
    assert compressed.archived_entry_id == "arc_1"
    assert store.get_compressed_label("lbl_1") is not None
    assert store.get_archived_entry("arc_1") is not None


def test_compress_atomic_rolls_back_on_bad_pointers(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    store.insert_memory_unit(mu)
    archive = _archive_for(mu, archive_id="arc_1", label_id="lbl_1")
    bad_label = _label_for(mu, label_id="lbl_1", archive_id="arc_wrong")

    with pytest.raises(MemoryStoreError):
        store.compress_atomic(mu.mu_id, bad_label, archive)

    assert store.get_memory_unit_or_raise(mu.mu_id).status == MemoryStatus.ACTIVE
    assert store.get_compressed_label("lbl_1") is None
    assert store.get_archived_entry("arc_1") is None


def test_restore_compressed_memory_is_atomic(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    store.insert_memory_unit(mu)
    archive = _archive_for(mu, archive_id="arc_1", label_id="lbl_1")
    label = _label_for(mu, label_id="lbl_1", archive_id="arc_1")
    store.compress_atomic(mu.mu_id, label, archive)

    restored = store.restore_atomic(mu.mu_id)

    assert restored.status == MemoryStatus.ACTIVE
    assert restored.needs_reindex is True
    assert restored.compressed_label_id is None
    assert restored.archived_entry_id is None
    assert store.get_compressed_label("lbl_1") is None
    assert store.get_archived_entry("arc_1") is None


def test_illegal_state_transitions_raise(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    store.insert_memory_unit(mu)

    with pytest.raises(IllegalStateTransitionError):
        store.restore_atomic(mu.mu_id)

    with pytest.raises(IllegalStateTransitionError):
        store.update_status(mu.mu_id, MemoryStatus.COMPRESSED)


def test_archive_status_requires_archive_pointer(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    store.insert_memory_unit(mu)

    with pytest.raises(IllegalStateTransitionError):
        store.update_status(mu.mu_id, MemoryStatus.ARCHIVED)

    archive = _archive_for(mu, archive_id="arc_1", label_id="lbl_future")
    store.insert_archived_entry(archive)
    store.update_status(mu.mu_id, MemoryStatus.ARCHIVED, archived_entry_id="arc_1")

    archived = store.get_memory_unit_or_raise(mu.mu_id)
    assert archived.status == MemoryStatus.ARCHIVED
    assert archived.archived_entry_id == "arc_1"
    assert archived.compressed_label_id is None


def test_forget_restore_and_delete_lifecycle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu = _mu()
    other = _mu(mu_id="mu_2", claim="Caroline used to work at Google")
    store.insert_memory_unit(mu)
    store.insert_memory_unit(other)
    edge = EdgeRecord(
        edge_id="edg_1",
        source_mu_id=mu.mu_id,
        target_mu_id=other.mu_id,
        edge_type=EdgeType.SUPERSEDED_BY,
    )
    store.insert_edge(edge)

    store.forget_atomic(mu.mu_id)
    assert store.get_memory_unit_or_raise(mu.mu_id).status == MemoryStatus.FORGOTTEN

    restored = store.restore_from_forgotten(mu.mu_id)
    assert restored.status == MemoryStatus.ACTIVE
    assert restored.needs_reindex is True

    store.delete_atomic(mu.mu_id, deleted_by="test")
    deleted = store.get_memory_unit_or_raise(mu.mu_id)
    assert deleted.status == MemoryStatus.DELETED
    assert deleted.claim == DELETED_PLACEHOLDER
    assert deleted.original_text == DELETED_PLACEHOLDER
    assert store.edges_from(mu.mu_id) == []
    assert store.edges_to(mu.mu_id) == []

    audit = store.list_deletion_audit("conv-1")
    assert len(audit) == 1
    assert audit[0].mu_id == mu.mu_id
    assert audit[0].deleted_by == "test"

    with pytest.raises(IllegalStateTransitionError):
        store.forget_atomic(mu.mu_id)


def test_edge_insert_validates_endpoints_and_uniqueness(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mu1 = _mu(mu_id="mu_1")
    mu2 = _mu(mu_id="mu_2")
    cross_conv = _mu(mu_id="mu_3", conversation_id="conv-2")
    store.insert_memory_unit(mu1)
    store.insert_memory_unit(mu2)
    store.insert_memory_unit(cross_conv)

    edge = EdgeRecord(
        edge_id="edg_1",
        source_mu_id="mu_1",
        target_mu_id="mu_2",
        edge_type=EdgeType.RELATED_TO,
        metadata_json='{"reason": "same topic"}',
    )
    store.insert_edge(edge)
    assert store.get_edge("edg_1") is not None

    duplicate = EdgeRecord(
        edge_id="edg_2",
        source_mu_id="mu_1",
        target_mu_id="mu_2",
        edge_type=EdgeType.RELATED_TO,
    )
    with pytest.raises(MemoryStoreError):
        store.insert_edge(duplicate)

    missing = EdgeRecord(
        edge_id="edg_3",
        source_mu_id="mu_1",
        target_mu_id="mu_missing",
        edge_type=EdgeType.RELATED_TO,
    )
    with pytest.raises(MemoryUnitNotFoundError):
        store.insert_edge(missing)

    cross = EdgeRecord(
        edge_id="edg_4",
        source_mu_id="mu_1",
        target_mu_id="mu_3",
        edge_type=EdgeType.RELATED_TO,
    )
    with pytest.raises(MemoryIntegrityError):
        store.insert_edge(cross)


def test_transaction_rolls_back_on_exception(tmp_path: Path) -> None:
    store = _store(tmp_path)

    with pytest.raises(sqlite3.OperationalError):
        with store.transaction() as conn:
            conn.execute(
                """
                INSERT INTO memory_units (
                    mu_id, conversation_id, session_id, claim, original_text,
                    source_dia_ids, source_speaker, extracted_at,
                    salience_score, importance, recency_weight, uniqueness,
                    status, confidence, created_at, updated_at
                ) VALUES (
                    'mu_rollback', 'conv-1', 'session_1', 'claim', 'text',
                    '[]', '', '2024-01-01T00:00:00+00:00',
                    0.5, 0.5, 1.0, 1.0,
                    'active', 0.9, '2024-01-01T00:00:00+00:00',
                    '2024-01-01T00:00:00+00:00'
                )
                """
            )
            conn.execute("SELECT * FROM table_that_does_not_exist")

    assert store.get_memory_unit("mu_rollback") is None
