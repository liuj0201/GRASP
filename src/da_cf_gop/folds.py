"""Deterministic speaker-disjoint nested LOSO fold definitions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence


FOLD_SCHEMA_VERSION = "da-cf-gop.folds.v1"


@dataclass(frozen=True)
class InnerFold:
    held_out_patient: str
    training_patients: tuple[str, ...]
    healthy_references: tuple[str, ...]


@dataclass(frozen=True)
class OuterFold:
    cohort: str
    held_out_patient: str
    training_patients: tuple[str, ...]
    healthy_references: tuple[str, ...]
    inner_folds: tuple[InnerFold, ...]

    @property
    def fold_id(self) -> str:
        return f"{self.cohort}__outer_{self.held_out_patient}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": FOLD_SCHEMA_VERSION,
            "fold_id": self.fold_id,
            **asdict(self),
        }


def nested_loso_folds(
    cohort: str,
    patients: Sequence[str],
    healthy_references: Sequence[str],
) -> tuple[OuterFold, ...]:
    """Create outer and inner patient LOSO folds in a stable order.

    Healthy speakers are training references in every fold and are never part
    of the population on which an inner hyperparameter is selected.
    """

    patient_ids = tuple(sorted(str(value).upper() for value in patients))
    healthy_ids = tuple(sorted(str(value).upper() for value in healthy_references))
    if len(patient_ids) < 3:
        raise ValueError("nested patient LOSO requires at least three patients")
    if not healthy_ids:
        raise ValueError("at least one healthy reference speaker is required")
    if len(patient_ids) != len(set(patient_ids)) or len(healthy_ids) != len(set(healthy_ids)):
        raise ValueError("speaker lists must not contain duplicates")
    if set(patient_ids) & set(healthy_ids):
        raise ValueError("patient and healthy speaker lists must be disjoint")

    outer: list[OuterFold] = []
    for test in patient_ids:
        training = tuple(value for value in patient_ids if value != test)
        inner = tuple(
            InnerFold(
                held_out_patient=validation,
                training_patients=tuple(value for value in training if value != validation),
                healthy_references=healthy_ids,
            )
            for validation in training
        )
        outer.append(OuterFold(str(cohort), test, training, healthy_ids, inner))
    return tuple(outer)


def validate_fold_rows(
    fold: OuterFold,
    rows: Iterable[Any],
    *,
    speaker_field: str = "speaker_id",
) -> None:
    """Reject missing or unexpected speakers before a fold is executed."""

    seen: set[str] = set()
    for row in rows:
        if isinstance(row, Mapping):
            speaker = row.get(speaker_field)
        else:
            speaker = getattr(row, speaker_field, None)
        if speaker is None:
            raise ValueError(f"row has no {speaker_field}")
        seen.add(str(speaker).upper())
    expected = {
        fold.held_out_patient,
        *fold.training_patients,
        *fold.healthy_references,
    }
    missing = expected - seen
    unexpected = seen - expected
    if missing or unexpected:
        raise ValueError(
            f"fold row speakers disagree with declaration; missing={sorted(missing)}, "
            f"unexpected={sorted(unexpected)}"
        )


def training_artifact_hash(
    *,
    stage: str,
    config_sha256: str,
    checkpoint_sha256: str,
    training_rows: Iterable[Mapping[str, Any]],
) -> str:
    """Hash only training inputs, so held-out data cannot select a cache.

    Callers should pass a deliberately reduced row representation containing
    the fields consumed by ``stage``.  The helper sorts those records before
    hashing and never accepts a test-row argument.
    """

    def canonical(record: Mapping[str, Any]) -> str:
        return json.dumps(record, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":"))

    payload = {
        "schema_version": FOLD_SCHEMA_VERSION,
        "stage": str(stage),
        "config_sha256": str(config_sha256),
        "checkpoint_sha256": str(checkpoint_sha256),
        "training_rows": sorted((dict(row) for row in training_rows), key=canonical),
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def folds_document(folds: Sequence[OuterFold]) -> dict[str, Any]:
    return {
        "schema_version": FOLD_SCHEMA_VERSION,
        "folds": [fold.to_dict() for fold in folds],
    }
