"""Fail-closed, local-only contract for downstream research LLM evidence.

Evaluation artifacts may contain PHN labels and identities.  Runtime records
are instead constructed from an explicit whitelist and recursively checked for
gold/identity leakage.  Their deterministic SHA-256 identifiers are only
stable local join keys: without a secret they provide no protection suitable
for public release and must not be treated as anonymization or pseudonymization.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metrics import DELETE, PHONE_VOCAB
from .provenance import canonical_json_bytes


SCHEMA_VERSION = "da-cf-gop.llm.v1"
LOCAL_IDENTIFIER_SCHEME = "local_deterministic_sha256.v1"
DECISIONS = frozenset(
    {"correct", "substitution", "deletion", "atypical_unspecified", "abstain"}
)
ARPABET_39 = PHONE_VOCAB
RUNTIME_KEYS = frozenset(
    {
        "schema_version",
        "record_id",
        "utterance_record_id",
        "phone_index",
        "canonical_phone",
        "decision",
        "error_probability",
        "alternative_phone",
        "alternative_probability_given_error",
        "abstain",
        "abstain_reason",
        "permissions",
    }
)
PERMISSION_KEYS = frozenset(
    {
        "offline_research_llm",
        "patient_facing",
        "mention_error",
        "mention_alternative",
        "articulatory_claim",
        "diagnosis_or_treatment",
    }
)
FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "speaker",
        "speaker_id",
        "patient",
        "patient_id",
        "participant",
        "participant_id",
        "severity",
        "severity_rank",
        "fold",
        "cohort",
        "event",
        "utterance_id",
        "phn",
        "phn_path",
        "textgrid",
        "textgrid_path",
        "audio_path",
        "wav_path",
        "raw_gop",
        "gop",
        "candidate_probabilities",
        "candidate_scores",
        "gold_event",
        "gold_realized",
        "ground_truth",
        "reference_label",
        "stable",
    }
)


class LLMContractError(ValueError):
    """Raised when a runtime record is unsafe or schema-incompatible."""


@dataclass(frozen=True)
class SpecificOutputGate:
    """Frozen inner-OOF authorization for substitution/deletion statements."""

    enabled: bool
    probability_threshold: float | None
    min_margin: float
    precision: float | None
    coverage: float
    n_eligible: int
    n_authorized: int
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _probability(value: object, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise LLMContractError(f"{name} must be numeric") from error
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise LLMContractError(f"{name} must be a finite probability")
    return result


def local_identifier(
    prefix: str,
    *parts: object,
    namespace: str = SCHEMA_VERSION,
    length: int = 20,
) -> str:
    """Return a domain-separated deterministic identifier for local joins.

    This is deliberately unkeyed.  ``namespace`` lets a caller isolate local
    artifact families, but neither it nor the digest makes the identifier safe
    for public release.
    """

    if prefix not in {"phone", "utt"} or not 16 <= length <= 64:
        raise LLMContractError("invalid local identifier configuration")
    if not isinstance(namespace, str) or not namespace:
        raise LLMContractError("local identifier namespace must be non-empty")
    payload = canonical_json_bytes(
        {
            "scheme": LOCAL_IDENTIFIER_SCHEME,
            "namespace": namespace,
            "kind": prefix,
            "parts": list(parts),
        }
    )
    digest = hashlib.sha256(payload).hexdigest()
    return f"{prefix}_{digest[:length]}"


def _specific_prediction_correct(row: Mapping[str, object]) -> bool:
    if row.get("top_alt") == DELETE:
        return row.get("gold_event") == "deletion"
    return (
        row.get("gold_event") == "substitution"
        and row.get("gold_realized") == row.get("top_alt")
    )


def fit_specific_output_gate(
    rows: Sequence[Mapping[str, object]],
    *,
    min_precision: float = 0.80,
    min_coverage: float = 0.10,
    min_margin: float = 0.10,
    error_threshold: float = 0.50,
    threshold_grid: Sequence[float] = (
        0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95,
    ),
) -> SpecificOutputGate:
    """Fit the predeclared specific-output gate on inner-OOF rows only.

    Coverage is measured over stable tokens classified as errors.  Gold is
    used only here, never in :func:`build_llm_record`.
    """

    for value, name in (
        (min_precision, "min_precision"), (min_coverage, "min_coverage"),
        (min_margin, "min_margin"), (error_threshold, "error_threshold"),
    ):
        _probability(value, name)
    eligible = [
        row for row in rows
        if row.get("stable") is True
        and _probability(row.get("p_error"), "p_error") >= error_threshold
    ]
    if not eligible:
        return SpecificOutputGate(
            False, None, min_margin, None, 0.0, 0, 0,
            "no_stable_inner_oof_error_predictions",
        )
    candidates = [
        row for row in eligible
        if row.get("top_alt") is not None
        and _probability(row.get("top2_margin"), "top2_margin") >= min_margin
    ]
    if not candidates:
        return SpecificOutputGate(
            False, None, min_margin, None, 0.0, len(eligible), 0,
            "no_candidate_passed_margin",
        )

    best: tuple[float, float, float, int] | None = None
    if not threshold_grid:
        raise LLMContractError("threshold_grid must not be empty")
    thresholds = sorted(
        {_probability(value, "threshold_grid") for value in threshold_grid}, reverse=True
    )
    for threshold in thresholds:
        selected = [
            row for row in candidates
            if float(row["p_alt_given_error"]) >= threshold
        ]
        if not selected:
            continue
        precision = sum(_specific_prediction_correct(row) for row in selected) / len(selected)
        coverage = len(selected) / len(eligible)
        if precision >= min_precision and coverage >= min_coverage:
            proposal = (coverage, precision, threshold, len(selected))
            if best is None or proposal > best:
                best = proposal
    if best is None:
        return SpecificOutputGate(
            False, None, min_margin, None, 0.0, len(eligible), 0,
            "inner_oof_precision_or_coverage_requirement_not_met",
        )
    coverage, precision, threshold, selected = best
    return SpecificOutputGate(
        True, threshold, min_margin, precision, coverage, len(eligible), selected, None
    )


def _permissions(decision: str) -> dict[str, bool]:
    mention_error = decision in {"substitution", "deletion", "atypical_unspecified"}
    mention_alternative = decision in {"substitution", "deletion"}
    return {
        "offline_research_llm": True,
        "patient_facing": False,
        "mention_error": mention_error,
        "mention_alternative": mention_alternative,
        "articulatory_claim": False,
        "diagnosis_or_treatment": False,
    }


def _base_identifiers(
    row: Mapping[str, object], identifier_namespace: str
) -> tuple[str, str]:
    required = ("speaker", "event", "phone_index", "target")
    missing = [name for name in required if name not in row]
    if missing:
        raise LLMContractError(f"prediction lacks identifier inputs: {missing}")
    index = row["phone_index"]
    if type(index) is not int or int(index) < 0:
        raise LLMContractError("phone_index must be a non-negative integer")
    utterance = local_identifier(
        "utt", row["speaker"], row["event"], namespace=identifier_namespace
    )
    phone = local_identifier(
        "phone", utterance, index, row["target"], namespace=identifier_namespace
    )
    return phone, utterance


def build_llm_record(
    row: Mapping[str, object],
    gate: SpecificOutputGate,
    *,
    identifier_namespace: str = SCHEMA_VERSION,
    error_threshold: float = 0.50,
) -> dict[str, object]:
    """Transform one prediction into the whitelisted runtime record."""

    p_error = _probability(row.get("p_error"), "p_error")
    _probability(error_threshold, "error_threshold")
    target = str(row.get("target", ""))
    if target not in ARPABET_39:
        raise LLMContractError(f"invalid canonical phone: {target}")
    phone_id, utterance_id = _base_identifiers(row, identifier_namespace)

    alternative: str | None = None
    p_alternative: float | None = None
    if p_error < error_threshold:
        decision = "correct"
    else:
        p_alt = _probability(row.get("p_alt_given_error"), "p_alt_given_error")
        margin = _probability(row.get("top2_margin"), "top2_margin")
        top_alt = row.get("top_alt")
        authorized = (
            gate.enabled
            and gate.probability_threshold is not None
            and p_alt >= gate.probability_threshold
            and margin >= gate.min_margin
            and isinstance(top_alt, str)
            and (top_alt == DELETE or top_alt in ARPABET_39)
            and top_alt != target
        )
        if authorized:
            decision = "deletion" if top_alt == DELETE else "substitution"
            alternative = str(top_alt)
            p_alternative = p_alt
        else:
            decision = "atypical_unspecified"

    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "record_id": phone_id,
        "utterance_record_id": utterance_id,
        "phone_index": int(row["phone_index"]),
        "canonical_phone": target,
        "decision": decision,
        "error_probability": p_error,
        "alternative_phone": alternative,
        "alternative_probability_given_error": p_alternative,
        "abstain": False,
        "abstain_reason": None,
        "permissions": _permissions(decision),
    }
    validate_llm_record(record)
    return record


def build_abstention_record(
    row: Mapping[str, object],
    reason: str,
    *,
    identifier_namespace: str = SCHEMA_VERSION,
) -> dict[str, object]:
    """Create a fail-closed row for missing/invalid model evidence."""

    if not isinstance(reason, str) or not reason.strip():
        raise LLMContractError("abstention reason must be non-empty")
    target = str(row.get("target", ""))
    if target not in ARPABET_39:
        raise LLMContractError(f"invalid canonical phone: {target}")
    phone_id, utterance_id = _base_identifiers(row, identifier_namespace)
    record: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "record_id": phone_id,
        "utterance_record_id": utterance_id,
        "phone_index": int(row["phone_index"]),
        "canonical_phone": target,
        "decision": "abstain",
        "error_probability": None,
        "alternative_phone": None,
        "alternative_probability_given_error": None,
        "abstain": True,
        "abstain_reason": reason.strip(),
        "permissions": _permissions("abstain"),
    }
    validate_llm_record(record)
    return record


def find_sensitive_fields(value: object, path: str = "$") -> list[str]:
    """Recursively locate forbidden gold, identity, raw-score, and path keys."""

    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            child = f"{path}.{key}"
            forbidden = (
                normalized in FORBIDDEN_EXACT_KEYS
                or "gold" in normalized
                or normalized.endswith("_path")
                or normalized.startswith("raw_")
            )
            if forbidden:
                found.append(child)
            found.extend(find_sensitive_fields(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(find_sensitive_fields(item, f"{path}[{index}]"))
    return found


def _contains_raw_path_or_torgo(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(_contains_raw_path_or_torgo(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_raw_path_or_torgo(item) for item in value)
    if isinstance(value, str):
        return bool(re.search(r"[A-Za-z]:[\\/]", value)) or "torgo" in value.casefold()
    return False


def validate_llm_record(record: Mapping[str, object]) -> None:
    if set(record) != RUNTIME_KEYS:
        raise LLMContractError("runtime record must use the exact frozen key set")
    if record["schema_version"] != SCHEMA_VERSION:
        raise LLMContractError("runtime schema version changed")
    if not re.fullmatch(r"phone_[0-9a-f]{20}", str(record["record_id"])):
        raise LLMContractError("record_id is not a deterministic local identifier")
    if not re.fullmatch(r"utt_[0-9a-f]{20}", str(record["utterance_record_id"])):
        raise LLMContractError(
            "utterance_record_id is not a deterministic local identifier"
        )
    if type(record["phone_index"]) is not int or int(record["phone_index"]) < 0:
        raise LLMContractError("phone_index must be a non-negative integer")
    if record["canonical_phone"] not in ARPABET_39:
        raise LLMContractError("canonical_phone is outside the frozen vocabulary")
    decision = str(record["decision"])
    if decision not in DECISIONS:
        raise LLMContractError("invalid runtime decision")

    if decision == "abstain":
        if record["error_probability"] is not None:
            raise LLMContractError("abstention must not expose an error probability")
    else:
        _probability(record["error_probability"], "error_probability")
    if type(record["abstain"]) is not bool or record["abstain"] is (decision != "abstain"):
        raise LLMContractError("abstain flag and decision disagree")
    if decision == "abstain":
        if not isinstance(record["abstain_reason"], str) or not record["abstain_reason"].strip():
            raise LLMContractError("abstention requires a reason")
    elif record["abstain_reason"] is not None:
        raise LLMContractError("non-abstention must not have an abstain reason")

    alternative = record["alternative_phone"]
    p_alt = record["alternative_probability_given_error"]
    if decision == "substitution":
        if alternative not in ARPABET_39 or alternative == record["canonical_phone"]:
            raise LLMContractError("substitution alternative is invalid")
        _probability(p_alt, "alternative_probability_given_error")
    elif decision == "deletion":
        if alternative != DELETE:
            raise LLMContractError("deletion must use <DEL> as its alternative")
        _probability(p_alt, "alternative_probability_given_error")
    elif alternative is not None or p_alt is not None:
        raise LLMContractError("non-specific output leaked an alternative")

    permissions = record["permissions"]
    if not isinstance(permissions, Mapping) or set(permissions) != PERMISSION_KEYS:
        raise LLMContractError("permissions must use the exact frozen key set")
    if any(type(value) is not bool for value in permissions.values()):
        raise LLMContractError("all permissions must be boolean")
    expected = _permissions(decision)
    if dict(permissions) != expected:
        raise LLMContractError("permissions are not fail-closed for this decision")
    sensitive = find_sensitive_fields(record)
    if sensitive:
        raise LLMContractError(f"runtime record contains sensitive keys: {sensitive}")
    if _contains_raw_path_or_torgo(record):
        raise LLMContractError("runtime record contains a raw path or dataset identifier")


def export_llm_records(
    rows: Sequence[Mapping[str, object]],
    destination: str | os.PathLike[str],
    gate: SpecificOutputGate,
    *,
    identifier_namespace: str = SCHEMA_VERSION,
    error_threshold: float = 0.50,
) -> list[dict[str, object]]:
    """Write deterministic exact-schema JSONL for local research use only."""

    records = [
        build_llm_record(
            row,
            gate,
            identifier_namespace=identifier_namespace,
            error_threshold=error_threshold,
        )
        for row in rows
    ]
    ids = [str(record["record_id"]) for record in records]
    if len(set(ids)) != len(ids):
        raise LLMContractError("duplicate runtime record identifiers")
    records.sort(key=lambda item: (str(item["utterance_record_id"]), int(item["phone_index"])))
    payload = b"".join(canonical_json_bytes(record) for record in records)
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()
    return records


def read_llm_records(path: str | os.PathLike[str]) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise LLMContractError(f"blank JSONL line at {line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise LLMContractError(f"invalid JSONL line {line_number}") from error
            if not isinstance(value, dict):
                raise LLMContractError(f"runtime row {line_number} is not an object")
            validate_llm_record(value)
            records.append(value)
    if not records:
        raise LLMContractError("runtime artifact is empty")
    ids = [str(record["record_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise LLMContractError("runtime artifact contains duplicate record ids")
    return records
