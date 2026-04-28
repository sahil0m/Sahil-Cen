"""Phase 2 schema validation tests."""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from locomo_memory.phase2.schemas import (
    ArchivedEntry,
    CompressedLabel,
    DeletionAudit,
    EdgeRecord,
    EdgeType,
    MemoryStatus,
    MemoryUnit,
)


def _memory_unit(**overrides: object) -> MemoryUnit:
    data: dict[str, object] = {
        "conversation_id": "conv-1",
        "session_id": "session_1",
        "claim": "Caroline works at Microsoft",
        "original_text": "Caroline: I work at Microsoft now.",
        "source_dia_ids": [" D1:1 ", "", "D1:2"],
        "source_speaker": "Caroline",
    }
    data.update(overrides)
    return MemoryUnit(**data)


def test_memory_unit_defaults_and_id_prefix() -> None:
    mu = _memory_unit()

    assert mu.mu_id.startswith("mu_")
    assert mu.status == MemoryStatus.ACTIVE
    assert mu.source_dia_ids == ["D1:1", "D1:2"]
    assert mu.extracted_at.tzinfo is not None


def test_memory_unit_rejects_out_of_range_scores() -> None:
    with pytest.raises(ValidationError):
        _memory_unit(salience_score=1.2)


def test_memory_unit_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        _memory_unit(unexpected_field=True)


def test_memory_unit_rejects_naive_datetimes() -> None:
    with pytest.raises(ValidationError):
        _memory_unit(extracted_at=datetime(2024, 1, 1))


def test_compressed_memory_unit_requires_both_pointers() -> None:
    with pytest.raises(ValidationError):
        _memory_unit(status=MemoryStatus.COMPRESSED)

    mu = _memory_unit(
        status=MemoryStatus.COMPRESSED,
        compressed_label_id="lbl_abc123",
        archived_entry_id="arc_abc123",
    )
    assert mu.compressed_label_id == "lbl_abc123"
    assert mu.archived_entry_id == "arc_abc123"


def test_active_memory_unit_rejects_compression_pointers() -> None:
    with pytest.raises(ValidationError):
        _memory_unit(compressed_label_id="lbl_abc123")


def test_archived_memory_unit_requires_archive_pointer_only() -> None:
    with pytest.raises(ValidationError):
        _memory_unit(status=MemoryStatus.ARCHIVED)

    archived = _memory_unit(
        status=MemoryStatus.ARCHIVED,
        archived_entry_id="arc_abc123",
    )
    assert archived.archived_entry_id == "arc_abc123"

    with pytest.raises(ValidationError):
        _memory_unit(
            status=MemoryStatus.ARCHIVED,
            compressed_label_id="lbl_abc123",
            archived_entry_id="arc_abc123",
        )


def test_compressed_label_strips_lists() -> None:
    label = CompressedLabel(
        archived_pointer="arc_abc123",
        mu_id="mu_abc123",
        conversation_id="conv-1",
        topic="Career",
        short_summary="Caroline works at Microsoft",
        key_entities=[" Caroline ", "", "Microsoft"],
        original_dia_ids=[" D1:1 ", ""],
    )

    assert label.label_id.startswith("lbl_")
    assert label.key_entities == ["Caroline", "Microsoft"]
    assert label.original_dia_ids == ["D1:1"]


def test_archived_entry_requires_valid_json_payload() -> None:
    with pytest.raises(ValidationError):
        ArchivedEntry(
            label_pointer="lbl_abc123",
            mu_id="mu_abc123",
            conversation_id="conv-1",
            full_memory_unit_json="{not-json",
        )


def test_edge_rejects_self_loop_and_invalid_metadata_json() -> None:
    with pytest.raises(ValidationError):
        EdgeRecord(
            source_mu_id="mu_1",
            target_mu_id="mu_1",
            edge_type=EdgeType.RELATED_TO,
        )

    with pytest.raises(ValidationError):
        EdgeRecord(
            source_mu_id="mu_1",
            target_mu_id="mu_2",
            edge_type=EdgeType.RELATED_TO,
            metadata_json="{bad-json",
        )


def test_deletion_audit_strips_source_ids() -> None:
    audit = DeletionAudit(
        mu_id="mu_abc123",
        conversation_id="conv-1",
        source_dia_ids=[" D1:1 ", ""],
    )

    assert audit.source_dia_ids == ["D1:1"]
