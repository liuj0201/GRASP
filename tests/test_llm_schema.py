from __future__ import annotations

import json

import pytest

from da_cf_gop.llm_schema import (
    LOCAL_IDENTIFIER_SCHEME,
    SCHEMA_VERSION,
    LLMContractError,
    SpecificOutputGate,
    build_abstention_record,
    build_llm_record,
    export_llm_records,
    find_sensitive_fields,
    fit_specific_output_gate,
    local_identifier,
    read_llm_records,
    validate_llm_record,
)
from da_cf_gop.metrics import DELETE


def inference_row(index: int = 0, *, p_error: float = 0.9, top_alt: str = "D") -> dict:
    return {
        "speaker": "F03",
        "event": "Session1/HeadMic/001",
        "phone_index": index,
        "target": "T",
        "p_error": p_error,
        "top_alt": top_alt,
        "p_alt_given_error": 0.9,
        "top2_margin": 0.2,
    }


def enabled_gate() -> SpecificOutputGate:
    return SpecificOutputGate(True, 0.8, 0.1, 0.9, 0.2, 10, 2, None)


def test_gate_uses_frozen_grid_and_meets_precision_coverage() -> None:
    rows = []
    for index in range(10):
        rows.append({
            "stable": True,
            "p_error": 0.9,
            "top_alt": "D",
            "p_alt_given_error": 0.87,
            "top2_margin": 0.2,
            "gold_event": "substitution" if index < 8 else "correct",
            "gold_realized": "D" if index < 8 else "T",
        })
    gate = fit_specific_output_gate(rows)
    assert gate.enabled
    assert gate.probability_threshold == 0.85
    assert gate.precision == 0.8 and gate.coverage == 1.0


def test_failed_gate_degrades_to_atypical_without_gold_leakage() -> None:
    gate = SpecificOutputGate(False, None, 0.1, None, 0.0, 10, 0, "failed")
    record = build_llm_record(inference_row(), gate)
    assert record["decision"] == "atypical_unspecified"
    assert record["alternative_phone"] is None
    assert record["permissions"]["mention_error"] is True
    assert record["permissions"]["mention_alternative"] is False
    assert find_sensitive_fields(record) == []


def test_specific_output_and_deletion_have_exact_fail_closed_schema() -> None:
    substitution = build_llm_record(inference_row(), enabled_gate())
    assert substitution["decision"] == "substitution"
    assert substitution["alternative_phone"] == "D"
    assert substitution["permissions"]["patient_facing"] is False
    assert substitution["permissions"]["articulatory_claim"] is False
    validate_llm_record(substitution)

    deletion = build_llm_record(
        inference_row(index=1, top_alt=DELETE), enabled_gate()
    )
    assert deletion["decision"] == "deletion"
    assert deletion["alternative_phone"] == DELETE
    validate_llm_record(deletion)


def test_abstention_is_fail_closed() -> None:
    record = build_abstention_record(inference_row(), "non_finite_model_score")
    assert record["decision"] == "abstain" and record["abstain"] is True
    assert record["error_probability"] is None
    assert all(
        value is False
        for name, value in record["permissions"].items()
        if name != "offline_research_llm"
    )


def test_local_export_needs_no_key_and_writes_deterministic_jsonl(tmp_path) -> None:
    path = tmp_path / "out.jsonl"
    rows = [inference_row(0), inference_row(1)]
    export_llm_records(rows, path, enabled_gate())
    first = path.read_bytes()
    export_llm_records(list(reversed(rows)), path, enabled_gate())
    assert path.read_bytes() == first
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert all(isinstance(json.loads(line), dict) for line in lines)
    assert len(read_llm_records(path)) == 2


def test_local_identifier_is_deterministic_namespaced_and_domain_separated() -> None:
    assert LOCAL_IDENTIFIER_SCHEME == "local_deterministic_sha256.v1"
    first = local_identifier("phone", "event", 1)
    assert first == local_identifier("phone", "event", 1)
    assert first != local_identifier("phone", "event", 1, namespace="another-local-run")
    assert first.removeprefix("phone_") != local_identifier(
        "utt", "event", 1
    ).removeprefix("utt_")
    assert first == "phone_ccee5ef974cc9123fe58"


def test_identifier_namespace_flows_through_builders_and_schema_is_unchanged() -> None:
    default = build_llm_record(inference_row(), enabled_gate())
    explicit = build_llm_record(
        inference_row(), enabled_gate(), identifier_namespace=SCHEMA_VERSION
    )
    isolated = build_llm_record(
        inference_row(), enabled_gate(), identifier_namespace="local-analysis-2"
    )
    assert default == explicit
    assert isolated["record_id"] != default["record_id"]
    assert isolated["utterance_record_id"] != default["utterance_record_id"]
    assert set(isolated) == set(default)


def test_validator_rejects_extra_sensitive_key() -> None:
    record = build_llm_record(inference_row(), enabled_gate())
    record["gold_event"] = "substitution"
    with pytest.raises(LLMContractError, match="exact frozen key set"):
        validate_llm_record(record)
    assert find_sensitive_fields({"nested": [{"gold_phone": "D"}]}) == [
        "$.nested[0].gold_phone"
    ]
