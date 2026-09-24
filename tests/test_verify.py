from __future__ import annotations

import gzip
import hashlib
import io
import json
from copy import deepcopy

import numpy as np
import pytest

from da_cf_gop.artifacts import write_csv, write_stage_summary
from da_cf_gop.data import build_torgo_manifest
from da_cf_gop.folds import folds_document, nested_loso_folds
from da_cf_gop.llm_schema import ARPABET_39, SpecificOutputGate, export_llm_records
from da_cf_gop.metrics import DELETE, evaluate_predictions
from da_cf_gop.provenance import (
    build_cache_descriptor,
    companion_descriptor_path,
    write_deterministic_npz,
    write_json,
    sha256_file,
    sha256_json,
)
from da_cf_gop.verify import (
    FROZEN_METHODS,
    VerificationError,
    _logical_json_sha256,
    _recompute_paired_and_success,
    _verify_cohort_manifest_bundle,
    _verify_prediction_protocol,
    _verify_severity_inference_inventory,
    _verify_summary_outputs,
    _expected_metrics_csv_rows,
    verify_deterministic_gzip,
    verify_matched_method_coverage,
    verify_metrics_report,
    verify_metrics_csv,
    verify_project,
    verify_severity_artifacts,
)


def test_logical_manifest_hash_matches_compact_generator_bytes() -> None:
    value = [{"b": 2, "a": 1}]
    expected = hashlib.sha256(b'[{"a":1,"b":2}]').hexdigest()
    assert _logical_json_sha256(value) == expected
    # Artifact JSON serialization has a final newline; the manifest audit
    # deliberately hashes the logical compact JSON value without that byte.
    assert _logical_json_sha256(value) != sha256_json(value)


def prediction(method: str, cohort: str, index: int, error: bool) -> dict:
    target = "T"
    candidates = sorted((ARPABET_39 - {target}) | {DELETE})
    top = "D"
    second = next(candidate for candidate in candidates if candidate != top)
    probs = {candidate: 0.0 for candidate in candidates}
    probs[top], probs[second] = 0.5, 0.2
    rest = [candidate for candidate in candidates if candidate not in {top, second}]
    for candidate in rest:
        probs[candidate] = 0.3 / len(rest)
    return {
        "method": method,
        "cohort": cohort,
        "fold": "outer_F03",
        "speaker": "F03",
        "event": f"event-{index}",
        "phone_index": index,
        "target": target,
        "gold_event": "substitution" if error else "correct",
        "gold_realized": "D" if error else target,
        "stable": True,
        "gop": -1.0 if error else 1.0,
        "p_error": 0.9 if error else 0.1,
        "top_alt": top,
        "p_alt_given_error": 0.5,
        "candidate_probabilities": probs,
        "top2_margin": 0.3,
    }


def write_deterministic_jsonl_gz(path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed:
            for row in rows:
                compressed.write((json.dumps(row, sort_keys=True) + "\n").encode())


def build_project(tmp_path):
    artifact_root = tmp_path / "artifacts"
    manifests = artifact_root / "manifests"
    cache = artifact_root / "cache"
    evaluation = artifact_root / "evaluation"
    runtime = artifact_root / "runtime"
    for directory in (manifests, cache, evaluation, runtime):
        directory.mkdir(parents=True)

    write_json(
        manifests / "cohorts.json",
        {"records": [{"speaker": "F03", "session": "S1", "stem": "001"}]},
    )
    source = tmp_path / "source.wav"
    source.write_bytes(b"audio")
    npz = cache / "logits.npz"
    write_deterministic_npz(npz, {"logits": np.zeros((3, 40), dtype=np.float32)})
    descriptor = build_cache_descriptor(
        artifact_type="ctc_logits", artifact=npz, config={"frozen": True},
        sources={"audio": source}, root=tmp_path,
    )
    write_json(companion_descriptor_path(npz), descriptor)

    rows = []
    for cohort in ("primary5", "sensitivity7"):
        for method in ("all_phone_cf_calibrated", "da_cf_adapted"):
            rows.extend([
                prediction(method, cohort, 0, False),
                prediction(method, cohort, 1, True),
            ])
    write_deterministic_jsonl_gz(evaluation / "phone_predictions.jsonl.gz", rows)
    report = evaluate_predictions(rows)
    write_json(evaluation / "metrics.json", report)
    (evaluation / "method_metrics.csv").write_text(
        "cohort,method,macro_auprc\nprimary5,da_cf_adapted,1.0\n",
        encoding="utf-8",
    )
    disabled = SpecificOutputGate(False, None, 0.1, None, 0.0, 0, 0, "test")
    export_llm_records(
        [{"speaker": "F03", "event": "event-0", "phone_index": 0, "target": "T",
          "p_error": 0.1, "top_alt": "D", "p_alt_given_error": 0.5,
          "top2_margin": 0.3}],
        runtime / "llm_phone_evidence.jsonl",
        disabled,
        identifier_namespace="verifier-test-local",
    )
    return rows, report


def test_project_verifier_recomputes_metrics_and_checks_safety(tmp_path) -> None:
    rows, _ = build_project(tmp_path)
    audit = verify_project(tmp_path)
    assert audit["status"] == "PASS"
    assert audit["n_prediction_rows"] == len(rows)
    assert audit["cohorts"] == ["primary5", "sensitivity7"]


def test_metrics_verification_detects_tampering(tmp_path) -> None:
    rows, report = build_project(tmp_path)
    report["results"][0]["macro"]["auprc"]["value"] = 0.0
    with pytest.raises(VerificationError, match="cannot be recomputed"):
        verify_metrics_report(report, rows)


def _full_protocol_prediction_rows(cohort: str, speakers: tuple[str, ...]):
    rows = []
    for method in FROZEN_METHODS:
        for speaker_index, speaker in enumerate(speakers):
            for phone_index in range(4):
                row = prediction(method, cohort, phone_index, phone_index % 2 == 1)
                row.update({
                    "fold": f"{cohort}__outer_{speaker}",
                    "speaker": speaker,
                    "event": f"{speaker}_Session1_0001",
                })
                rows.append(row)
    return rows


def _claim_config() -> dict[str, object]:
    return {
        "_config_hash": "b" * 64,
        "seed": 20260829,
        "evaluation": {
            "primary_comparator": "all_phone_cf_calibrated",
            "bootstrap_replicates": 40,
            "required_absolute_improvement": 0.03,
            "required_primary_speakers_improved": 4,
            "required_sensitivity_speakers_improved": 5,
            "maximum_auroc_drop": 0.01,
            "maximum_brier_increase": 0.01,
            "maximum_identification_drop": 0.01,
        },
    }


def test_full_metric_tree_and_claims_are_recomputed_from_tokens() -> None:
    speakers = ("F03", "F04", "M01", "M04", "M05")
    rows = _full_protocol_prediction_rows("primary5", speakers)
    config = _claim_config()
    report = evaluate_predictions(rows)
    report["config_sha256"] = config["_config_hash"]
    report["headline_policy"] = {
        "exact_substitution_deletion_headline_allowed": True,
        "error_detection_status": "headline",
        "substitution_deletion_status": "headline",
        "minimum_stable_fraction": 0.8,
        "observed_patient_stable_fraction": 1.0,
    }
    paired, all_paired, success = _recompute_paired_and_success(
        report, config, "primary5", headline_allowed=True
    )
    report["paired_comparison"] = paired
    report["paired_comparisons"] = all_paired
    report["success_assessment"] = success
    audit = {
        "exact_substitution_deletion_headline_allowed": True,
        "minimum_stable_fraction": 0.8,
        "patient_stable_fraction": 1.0,
    }
    verify_metrics_report(
        report, rows, config=config, expected_cohort="primary5", manifest_audit=audit
    )

    for path in (
        ("results", 0, "per_speaker", 0, "auprc"),
        ("results", 0, "pooled_binary", "auprc"),
        ("paired_comparison", "n_improved"),
        ("success_assessment", "cohort_success"),
    ):
        changed = deepcopy(report)
        target = changed
        for key in path[:-1]:
            target = target[key]
        final = path[-1]
        target[final] = (not target[final]) if isinstance(target[final], bool) else 12345
        with pytest.raises(VerificationError, match="recomputed|mismatch"):
            verify_metrics_report(
                changed,
                rows,
                config=config,
                expected_cohort="primary5",
                manifest_audit=audit,
            )


def test_method_and_speaker_csvs_are_recomputed_from_metric_report(tmp_path) -> None:
    rows = _full_protocol_prediction_rows(
        "primary5", ("F03", "F04", "M01", "M04", "M05")
    )
    report = evaluate_predictions(rows)
    method_path = tmp_path / "method_metrics.csv"
    speaker_path = tmp_path / "speaker_metrics.csv"
    write_csv(method_path, _expected_metrics_csv_rows(report, "method"))
    write_csv(speaker_path, _expected_metrics_csv_rows(report, "speaker"))
    verify_metrics_csv(method_path, report=report, level="method")
    verify_metrics_csv(speaker_path, report=report, level="speaker")

    method_path.write_text(
        method_path.read_text(encoding="utf-8").replace("primary5", "tampered", 1),
        encoding="utf-8",
    )
    with pytest.raises(VerificationError, match="CSV"):
        verify_metrics_csv(method_path, report=report, level="method")


def test_protocol_requires_all_methods_speakers_folds_and_stable_tokens() -> None:
    primary = ("F03", "F04", "M01", "M04", "M05")
    sensitivity = ("F01", "F03", "F04", "M01", "M02", "M04", "M05")
    config = {
        "methods": list(FROZEN_METHODS[:8]),
        "ablations": list(FROZEN_METHODS[8:]),
        "cohorts": {"primary5": list(primary), "sensitivity7": list(sensitivity)},
    }
    manifests = {}
    rows = []
    for cohort, speakers in (("primary5", primary), ("sensitivity7", sensitivity)):
        manifests[cohort] = [
            {
                "speaker_id": speaker,
                "reading_event_id": f"{speaker}_Session1_0001",
                "stable_tokens": [{
                    "canonical_phone": "T",
                    "phone_index": 0,
                    "event_type": "match",
                    "realized_phone": "T",
                    "stable": True,
                }],
            }
            for speaker in speakers
        ]
        for method in FROZEN_METHODS:
            for speaker in speakers:
                row = prediction(method, cohort, 0, False)
                row.update({
                    "fold": f"{cohort}__outer_{speaker}",
                    "speaker": speaker,
                    "event": f"{speaker}_Session1_0001",
                })
                rows.append(row)
    verify_matched_method_coverage(rows)
    _verify_prediction_protocol(rows, config, manifests)

    changed = [dict(row) for row in rows]
    changed = [
        row for row in changed
        if not (
            row["cohort"] == "primary5"
            and row["method"] == FROZEN_METHODS[0]
            and row["speaker"] == "F03"
        )
    ]
    with pytest.raises(VerificationError, match="coverage differs from frozen stable tokens"):
        _verify_prediction_protocol(changed, config, manifests)


def test_manifest_bundle_schema_counts_and_nested_folds_are_recomputed(tmp_path) -> None:
    patients = ("F03", "F04", "M01", "M04", "M05")
    healthy = ("MC01", "MC02", "MC03", "MC04")
    speakers = (*patients, *healthy)
    manifest_dir = tmp_path / "manifests"
    rows = []
    for speaker in speakers:
        event = f"{speaker}_Session1_0001"
        rows.append({
            "agreed_insertions": 0,
            "audio_microphone": "headMic",
            "canonical_phones": ["T"],
            "category_insertions": 0,
            "duration_s": 1.0,
            "levenshtein_insertions": 0,
            "phn_microphone": "headMic",
            "phn_model_phones": ["T"],
            "phn_path": f"{speaker}.phn",
            "phn_raw_labels": ["t"],
            "phn_timit_phones": ["T"],
            "prompt": f"prompt {speaker}",
            "prompt_id": speaker,
            "prompt_path": f"{speaker}.txt",
            "reading_event_id": event,
            "sample_rate": 16000,
            "schema_version": "da-cf-gop.manifest.v1",
            "session": "Session1",
            "severity_label": None if speaker in healthy else "test",
            "severity_rank": None if speaker in healthy else 1.0,
            "sex": "female" if speaker.startswith("F") else "male",
            "speaker_group": "dysarthric" if speaker in patients else "healthy",
            "speaker_id": speaker,
            "stable_count": 1,
            "stable_fraction": 1.0,
            "stable_tokens": [{
                "canonical_phone": "T",
                "category_event_type": "match",
                "category_observed_index": 0,
                "category_realized_phone": "T",
                "event_type": "match",
                "levenshtein_event_type": "match",
                "levenshtein_observed_index": 0,
                "levenshtein_realized_phone": "T",
                "phone_index": 0,
                "realized_phone": "T",
                "stable": True,
            }],
            "stem": "0001",
            "uncertain_count": 0,
            "wav_path": f"{speaker}.wav",
        })
    manifest_path = manifest_dir / "primary5.jsonl.gz"
    exclusion_path = manifest_dir / "primary5.exclusions.jsonl.gz"
    write_deterministic_jsonl_gz(manifest_path, rows)
    write_deterministic_jsonl_gz(exclusion_path, [{
        "detail": "",
        "reading_event_id": "F03_Session1_0002",
        "reason": "missing_prompt",
        "session": "Session1",
        "speaker_id": "F03",
        "stem": "0002",
    }])
    audit = {
        "agreed_insertions": 0,
        "audio_policy": "headMic only",
        "canonical_tokens": 9,
        "category_insertions": 0,
        "cohort": "primary5",
        "data_root": "synthetic",
        "dysarthric_speakers_requested": list(patients),
        "exact_substitution_deletion_headline_allowed": True,
        "exclusion_reasons": {"missing_prompt": 1},
        "headline_stability_population": "dysarthric canonical tokens only",
        "healthy_speakers_requested": list(healthy),
        "levenshtein_insertions": 0,
        "manifest_sha256": sha256_json(rows),
        "minimum_stable_fraction": 0.8,
        "n_exclusions": 1,
        "n_patient_rows": 5,
        "n_rows": 9,
        "n_unique_canonical_sequences": 1,
        "n_unique_prompts": 9,
        "patient_canonical_tokens": 5,
        "patient_stable_fraction": 1.0,
        "patient_stable_tokens": 5,
        "patient_uncertain_tokens": 0,
        "phn_boundaries_serialized": False,
        "phn_microphone_counts": {"headMic": 9},
        "phn_policy": "same-event headMic; arrayMic sequence only for F01/M02",
        "reading_event_key": ["speaker_id", "session", "stem"],
        "rows_by_speaker": {speaker: 1 for speaker in sorted(speakers)},
        "schema_version": "da-cf-gop.manifest.v1",
        "stable_fraction": 1.0,
        "stable_tokens": 9,
        "uncertain_tokens": 0,
    }
    write_json(manifest_dir / "primary5.audit.json", audit)
    folds = []
    for held_out in patients:
        training = [speaker for speaker in patients if speaker != held_out]
        folds.append({
            "schema_version": "da-cf-gop.folds.v1",
            "cohort": "primary5",
            "fold_id": f"primary5__outer_{held_out}",
            "held_out_patient": held_out,
            "training_patients": training,
            "healthy_references": list(healthy),
            "inner_folds": [{
                "held_out_patient": inner,
                "training_patients": [speaker for speaker in training if speaker != inner],
                "healthy_references": list(healthy),
            } for inner in training],
        })
    write_json(
        manifest_dir / "primary5.folds.json",
        {"schema_version": "da-cf-gop.folds.v1", "folds": folds},
    )
    config = {
        "cohorts": {"primary5": list(patients), "healthy_phn": list(healthy)},
        "labels": {"stable_alignment_min_fraction": 0.8},
    }
    verified_rows, verified_audit = _verify_cohort_manifest_bundle(
        manifest_dir, config, "primary5"
    )
    assert len(verified_rows) == 9
    assert verified_audit["patient_stable_tokens"] == 5

    audit["patient_stable_tokens"] = 4
    write_json(manifest_dir / "primary5.audit.json", audit)
    with pytest.raises(VerificationError, match="patient_stable_tokens"):
        _verify_cohort_manifest_bundle(manifest_dir, config, "primary5")


def test_source_bound_manifest_uses_generator_logical_hash_and_verifies_sources(
    tmp_path,
) -> None:
    patients = ("F03", "F04", "M01", "M04", "M05")
    healthy = ("MC01", "MC02", "MC03", "MC04")
    data_root = tmp_path / "torgo"

    for speaker in (*patients, *healthy):
        speaker_set = "".join(character for character in speaker if character.isalpha())
        session = data_root / speaker_set / speaker / "Session1"
        (session / "wav_headMic").mkdir(parents=True)
        (session / "prompts").mkdir()
        (session / "phn_headMic").mkdir()
        (session / "wav_headMic" / "0001.wav").write_bytes(
            f"synthetic audio for {speaker}".encode("utf-8")
        )
        (session / "prompts" / "0001.txt").write_text("Tap.\n", encoding="utf-8")
        (session / "phn_headMic" / "0001.PHN").write_text(
            "0 100 t\n", encoding="utf-8"
        )
    # The bundle verifier intentionally rejects an empty JSONL artifact, so
    # include one real generator-produced exclusion alongside the valid rows.
    (data_root / "F" / "F03" / "Session1" / "wav_headMic" / "0002.wav").write_bytes(
        b"excluded synthetic audio"
    )

    def deterministic_g2p(texts):
        return [["T"] for _ in texts], [[] for _ in texts]

    generated_rows, exclusions, audit = build_torgo_manifest(
        data_root,
        cohort="primary5",
        healthy_speakers=healthy,
        g2p=deterministic_g2p,
        audio_probe=lambda _path: (1.0, 16000),
    )
    rows = [row.to_dict() for row in generated_rows]
    assert [row.reason for row in exclusions] == ["missing_prompt"]
    assert audit["manifest_sha256"] == _logical_json_sha256(rows)
    # write_json/sha256_json includes a final newline; the generator's audit
    # deliberately binds the compact logical value instead.
    assert audit["manifest_sha256"] != sha256_json(rows)

    descriptor = audit["g2p_provenance"]["descriptor_sha256"]
    assert {row["g2p_descriptor_sha256"] for row in rows} == {descriptor}
    for role, path_name, digest_name in (
        ("wav", "wav_path", "wav_sha256"),
        ("prompt", "prompt_path", "prompt_sha256"),
        ("phn", "phn_path", "phn_sha256"),
    ):
        entries = [
            {
                "reading_event_id": row["reading_event_id"],
                "path": row[path_name],
                "sha256": row[digest_name],
            }
            for row in rows
        ]
        assert audit["input_source_hashes"][role] == {
            "n_files": len(rows),
            "inventory_sha256": _logical_json_sha256(entries),
        }

    manifest_dir = tmp_path / "manifests"
    write_deterministic_jsonl_gz(manifest_dir / "primary5.jsonl.gz", rows)
    write_deterministic_jsonl_gz(
        manifest_dir / "primary5.exclusions.jsonl.gz",
        [row.to_dict() for row in exclusions],
    )
    write_json(manifest_dir / "primary5.audit.json", audit)
    write_json(
        manifest_dir / "primary5.folds.json",
        folds_document(nested_loso_folds("primary5", patients, healthy)),
    )
    config = {
        "cohorts": {"primary5": list(patients), "healthy_phn": list(healthy)},
        "labels": {"stable_alignment_min_fraction": 0.8},
    }
    verified_rows, _ = _verify_cohort_manifest_bundle(
        manifest_dir, config, "primary5"
    )
    assert [row["reading_event_id"] for row in verified_rows] == [
        row["reading_event_id"] for row in rows
    ]

    changed_audit = deepcopy(audit)
    changed_audit["input_source_hashes"]["wav"]["inventory_sha256"] = "0" * 64
    write_json(manifest_dir / "primary5.audit.json", changed_audit)
    with pytest.raises(VerificationError, match="primary5.wav.inventory_sha256"):
        _verify_cohort_manifest_bundle(manifest_dir, config, "primary5")

    changed_audit = deepcopy(audit)
    changed_audit["g2p_provenance"]["descriptor_sha256"] = "f" * 64
    write_json(manifest_dir / "primary5.audit.json", changed_audit)
    with pytest.raises(VerificationError, match="frozen G2P descriptor"):
        _verify_cohort_manifest_bundle(manifest_dir, config, "primary5")


def test_matched_coverage_rejects_method_specific_token_drop() -> None:
    rows = [
        prediction("a", "primary5", 0, False),
        prediction("a", "primary5", 1, True),
        prediction("b", "primary5", 0, False),
    ]
    with pytest.raises(VerificationError, match="coverage differs"):
        verify_matched_method_coverage(rows)


def test_gzip_mtime_must_be_zero(tmp_path) -> None:
    path = tmp_path / "bad.gz"
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=123) as stream:
        stream.write(b"x")
    path.write_bytes(buffer.getvalue())
    with pytest.raises(VerificationError, match="mtime"):
        verify_deterministic_gzip(path)


def test_stage_summary_accepts_real_bare_hash_shape_and_rechecks_output(tmp_path) -> None:
    output = tmp_path / "artifacts" / "runtime" / "llm_phone_evidence.jsonl"
    output.parent.mkdir(parents=True)
    output.write_text("{}\n", encoding="utf-8")
    summary_path = tmp_path / "artifacts" / "summaries" / "export-llm.json"
    write_stage_summary(
        summary_path,
        stage="export-llm",
        config_sha256="a" * 64,
        training_speakers=[],
        source_files={"frozen_config": "a" * 64},
        models={},
        manifests={"primary5": "b" * 64},
        upstream_artifacts={"runtime_source_predictions": "c" * 64},
        exclusions={},
        outputs={"llm_phone_evidence": sha256_file(output)},
        details={},
    )
    _verify_summary_outputs(
        summary_path, tmp_path, expected_config_sha256="a" * 64
    )
    output.write_text('{"tampered":true}\n', encoding="utf-8")
    with pytest.raises(VerificationError, match="output hash changed"):
        _verify_summary_outputs(
            summary_path, tmp_path, expected_config_sha256="a" * 64
        )


def test_severity_artifact_verifier_rejects_legacy_incomplete_rows(tmp_path) -> None:
    evaluation = tmp_path / "evaluation"
    evaluation.mkdir()
    score_path = evaluation / "severity_phone_scores.jsonl.gz"
    rows = [
        {
            "schema_version": "da-cf-gop.severity-phone-score.v1",
            "method": "DA-CF-GoP/adapted",
            "recording_id": f"recording-{index:03d}",
            "event": f"event-{index % 329:03d}",
            "phone_index": 0,
            "gop": float(index) / 415.0,
        }
        for index in range(415)
    ]
    write_deterministic_jsonl_gz(score_path, rows)
    write_json(
        evaluation / "severity_phone_scores.provenance.json",
        {
            "schema_version": "da-cf-gop.severity-scoring-provenance.v1",
            "output_sha256": sha256_file(score_path),
            "n_recordings": 415,
            "n_events": 329,
            "n_phone_rows": 415,
            "legacy_cached_canonical_ids_used": False,
            "phn_or_textgrid_used_at_inference": False,
            "severity_used_for_training": False,
        },
    )
    methods = {
        name: {
            "coverage": {"n_events": 329, "n_speakers": 8, "event_coverage": 1.0}
        }
        for name in ("DA-CF-GoP/adapted", "MixGoP", "UQ-GoP", "CaGOP", "CTC-SF")
    }
    write_json(
        evaluation / "severity_metrics.json",
        {
            "schema_version": "da-cf-gop.severity.v1",
            "comparison_level": "system_level_cross_backend",
            "component_causal_claims_permitted": False,
            "severity_used_for_model_training": False,
            "audit_scope": {
                "n_manifest_rows": 415,
                "n_unique_reading_events": 329,
                "n_speakers": 8,
                "strict_matched_coverage": True,
            },
            "methods": methods,
            "resampling": {"bootstrap_replicates": 10_000, "exact_test_assignments": 256},
        },
    )
    header = (
        "method,n_events,n_speakers,event_coverage,kendall_tau_b,"
        "abs_kendall_tau_b,spearman_rho,pearson_r\n"
    )
    body = "".join(f"{name},329,8,1,0,0,0,0\n" for name in methods)
    (evaluation / "severity_metrics.csv").write_text(header + body, encoding="utf-8")
    # A result table and matching aggregate counts are not enough: the
    # verifier must be able to reconstruct exact speaker/fold/event/token
    # coverage from the dedicated scoring schema.
    with pytest.raises(VerificationError, match="non-frozen key set"):
        verify_severity_artifacts(evaluation)


def test_severity_inference_verification_uses_persisted_event_order() -> None:
    # Discovery order is by recording id, where "array2" sorts before
    # "array_".  The persisted inference manifest intentionally sorts by
    # event identity first, so F01 precedes F03.
    regenerated = [
        {
            "recording_id": "array2_F03_0149",
            "event_id": "F03|Session2|0149",
            "speaker_id": "F03",
        },
        {
            "recording_id": "array_F01_0014",
            "event_id": "F01|Session1|0014",
            "speaker_id": "F01",
        },
        {
            "recording_id": "head_F01_0014",
            "event_id": "F01|Session1|0014",
            "speaker_id": "F01",
        },
    ]
    observed = [regenerated[1], regenerated[2], regenerated[0]]
    _verify_severity_inference_inventory(observed, regenerated)

    tampered = [dict(row) for row in observed]
    tampered[0]["recording_id"] = "array_F01_9999"
    with pytest.raises(VerificationError, match="recording_id cannot be recomputed"):
        _verify_severity_inference_inventory(tampered, regenerated)
