from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from da_cf_gop.artifacts import read_jsonl, write_jsonl, write_jsonl_gz
from da_cf_gop.cli import COHORTS, build_parser, export_llm_stage, main
from da_cf_gop.llm_schema import RUNTIME_KEYS, validate_llm_record
from da_cf_gop.provenance import sha256_file, write_json
from da_cf_gop.verify import VerificationError, _verify_llm_export


def _config(root: Path) -> dict[str, object]:
    return {
        "_config_hash": "a" * 64,
        "paths": {"artifacts": str(root / "artifacts")},
        "cohorts": {
            "primary5": ["F03", "F04", "M01", "M04", "M05"],
        },
        "llm_export": {
            "schema_version": "da-cf-gop.llm.v1",
            "hmac_env": "DA_CF_RELEASE_KEY",
            "precision_floor": 0.8,
            "coverage_floor": 0.1,
            "alternative_margin_floor": 0.1,
            "error_threshold": 0.5,
            "threshold_grid": [0.5, 0.6, 0.7, 0.8, 0.9],
            "patient_facing": False,
        },
    }


def _gate(*, enabled: bool) -> dict[str, object]:
    return {
        "enabled": enabled,
        "probability_threshold": 0.8 if enabled else None,
        "min_margin": 0.1,
        "precision": 1.0 if enabled else None,
        "coverage": 1.0 if enabled else 0.0,
        "n_eligible": 4,
        "n_authorized": 4 if enabled else 0,
        "reason": None if enabled else "inner_oof_precision_or_coverage_requirement_not_met",
    }


def _prepare_export(root: Path) -> tuple[dict[str, object], Path]:
    cfg = _config(root)
    artifacts = Path(cfg["paths"]["artifacts"])
    evaluation = artifacts / "evaluation" / "primary5"
    manifests = artifacts / "manifests"
    speakers = ("F03", "F04", "M01", "M04", "M05")
    rows = []
    for index, speaker in enumerate(speakers):
        rows.append({
            "cohort": "primary5",
            "fold": f"primary5__outer_{speaker}",
            "speaker": speaker,
            "event": f"{speaker}/Session1/{index:04d}",
            "phone_index": 0,
            "target": "T" if speaker == "F03" else "K",
            "p_error": 0.91,
            "top_alt": "D" if speaker == "F03" else "<DEL>",
            "p_alt_given_error": 0.81 if speaker == "F03" else 0.92,
            "top2_margin": 0.22,
        })
    write_jsonl_gz(evaluation / "runtime_source_predictions.jsonl.gz", rows)
    write_jsonl_gz(
        evaluation / "phone_predictions.jsonl.gz",
        [{"method": "da_cf_adapted", **row} for row in rows],
    )
    inner_rows = []
    for outer in speakers:
        for validation in (value for value in speakers if value != outer):
            correct = outer == "F03"
            inner_rows.append({
                "event": f"{validation}_Session1_inner",
                "fold": f"primary5__outer_{outer}",
                "gold_event": "substitution",
                "gold_realized": "D" if correct else "T",
                "inner_validation_speaker": validation,
                "p_alt_given_error": 0.81,
                "p_error": 0.91,
                "phone_index": 0,
                "speaker": validation,
                "stable": True,
                "target": "K",
                "top2_margin": 0.22,
                "top_alt": "D",
            })
    write_jsonl_gz(evaluation / "inner_oof_predictions.jsonl.gz", inner_rows)
    write_json(
        evaluation / "specific_output_gates.json",
        {
            "schema_version": "da-cf-gop.specific-output-gates.v1",
            "cohort": "primary5",
            "folds": {
                f"primary5__outer_{speaker}": _gate(enabled=speaker == "F03")
                for speaker in speakers
            },
        },
    )
    write_jsonl_gz(
        manifests / "primary5.jsonl.gz",
        [
            {
                "speaker_id": speaker,
                "speaker_group": "dysarthric",
                "reading_event_id": f"{speaker}/Session1/{index:04d}",
                "canonical_phones": ["T" if speaker == "F03" else "K"],
            }
            for index, speaker in enumerate(speakers)
        ],
    )
    return cfg, artifacts


def test_parser_exposes_workflow_commands_and_requires_loso_cohort() -> None:
    parser = build_parser()
    expected = {
        "build-manifests",
        "extract-logits",
        "run-phone-loso",
        "run-parikh2025",
        "run-dual-graph",
        "run-severity",
        "export-llm",
        "verify",
    }
    subparser_action = next(
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )
    assert set(subparser_action.choices) == expected
    args = parser.parse_args(["run-phone-loso", "--cohort", "primary5"])
    assert args.cohort in COHORTS
    with pytest.raises(SystemExit) as captured:
        parser.parse_args(["run-phone-loso"])
    assert captured.value.code == 2


def test_export_llm_uses_the_matching_fold_gate_and_is_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, artifacts = _prepare_export(tmp_path)
    monkeypatch.delenv("DA_CF_RELEASE_KEY", raising=False)

    summary = export_llm_stage(cfg)
    destination = artifacts / "runtime" / "llm_phone_evidence.jsonl"
    first_bytes = destination.read_bytes()
    first_hash = sha256_file(destination)
    records = read_jsonl(destination)

    assert summary["stage"] == "export-llm"
    assert summary["outputs"]["llm_phone_evidence"] == first_hash
    assert len(records) == 5
    assert all(set(record) == RUNTIME_KEYS for record in records)
    assert all(record["permissions"]["patient_facing"] is False for record in records)
    assert {record["decision"] for record in records} == {
        "substitution",
        "atypical_unspecified",
    }
    assert all(validate_llm_record(record) is None for record in records)
    serialized = first_bytes.decode("utf-8").casefold()
    for forbidden in ("f03", "f04", "session1", "speaker", "fold", "gold", "torgo"):
        assert forbidden not in serialized

    export_llm_stage(cfg)
    assert destination.read_bytes() == first_bytes
    assert sha256_file(destination) == first_hash


def test_export_llm_needs_no_key_for_local_deployment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, artifacts = _prepare_export(tmp_path)
    monkeypatch.delenv("DA_CF_RELEASE_KEY", raising=False)
    summary = export_llm_stage(cfg)
    assert (artifacts / "runtime" / "llm_phone_evidence.jsonl").is_file()
    assert summary["details"]["deployment"] == "local_offline"
    assert summary["details"]["identifier_scheme"] == "local_deterministic_sha256.v1"
    assert "release_key_fingerprint" not in summary["details"]


def test_export_llm_refuses_a_missing_fold_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, artifacts = _prepare_export(tmp_path)
    gate_path = artifacts / "evaluation" / "primary5" / "specific_output_gates.json"
    document = json.loads(gate_path.read_text(encoding="utf-8"))
    del document["folds"]["primary5__outer_F04"]
    write_json(gate_path, document)
    with pytest.raises(ValueError, match="exact primary5 folds"):
        export_llm_stage(cfg)


def test_export_llm_recomputes_gate_instead_of_trusting_stored_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, artifacts = _prepare_export(tmp_path)
    gate_path = artifacts / "evaluation" / "primary5" / "specific_output_gates.json"
    document = json.loads(gate_path.read_text(encoding="utf-8"))
    document["folds"]["primary5__outer_F03"]["probability_threshold"] = 0.9
    write_json(gate_path, document)
    with pytest.raises(ValueError, match="cannot be recomputed from inner OOF"):
        export_llm_stage(cfg)


def test_llm_verifier_rebuilds_local_identifiers_independent_of_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, artifacts = _prepare_export(tmp_path)
    monkeypatch.delenv("DA_CF_RELEASE_KEY", raising=False)
    export_llm_stage(cfg)
    predictions = read_jsonl(
        artifacts / "evaluation" / "primary5" / "phone_predictions.jsonl.gz"
    )
    assert _verify_llm_export(tmp_path, cfg, predictions) == 5

    monkeypatch.setenv("DA_CF_RELEASE_KEY", "ignored-for-local-deployment")
    assert _verify_llm_export(tmp_path, cfg, predictions) == 5


def test_llm_verifier_rejects_runtime_record_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg, artifacts = _prepare_export(tmp_path)
    export_llm_stage(cfg)
    runtime_path = artifacts / "runtime" / "llm_phone_evidence.jsonl"
    records = read_jsonl(runtime_path)
    records[0]["error_probability"] = 0.0
    write_jsonl(runtime_path, records)
    predictions = read_jsonl(
        artifacts / "evaluation" / "primary5" / "phone_predictions.jsonl.gz"
    )
    with pytest.raises(VerificationError, match="local identifiers|output hash"):
        _verify_llm_export(tmp_path, cfg, predictions)


def test_llm_verifier_rejects_well_formed_rebound_local_record_id(
    tmp_path: Path,
) -> None:
    cfg, artifacts = _prepare_export(tmp_path)
    export_llm_stage(cfg)
    runtime_path = artifacts / "runtime" / "llm_phone_evidence.jsonl"
    records = read_jsonl(runtime_path)
    original = records[0]["record_id"]
    candidate = "phone_" + "0" * 20
    records[0]["record_id"] = (
        candidate if candidate != original else "phone_" + "1" * 20
    )
    # Keep both the runtime schema and the declared output hash valid.  The
    # verifier must still reject an identifier not derived from local inputs.
    assert validate_llm_record(records[0]) is None
    write_jsonl(runtime_path, records)
    summary_path = artifacts / "summaries" / "export-llm.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["outputs"]["llm_phone_evidence"] = sha256_file(runtime_path)
    write_json(summary_path, summary)

    predictions = read_jsonl(
        artifacts / "evaluation" / "primary5" / "phone_predictions.jsonl.gz"
    )
    with pytest.raises(
        VerificationError,
        match="LLM runtime records/local identifiers",
    ):
        _verify_llm_export(tmp_path, cfg, predictions)


def test_verify_success_stdout_is_exactly_pass(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import da_cf_gop.verify as verify_module

    monkeypatch.setattr(verify_module, "verify_and_print", lambda _root: print("PASS"))
    assert main(["verify"]) == 0
    captured = capsys.readouterr()
    assert captured.out == "PASS\n"
    assert captured.err == ""
