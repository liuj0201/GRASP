from __future__ import annotations

import json
from pathlib import Path

import pytest

from da_cf_gop.severity import (
    FROZEN_COMPARATOR_METHODS,
    NEW_METHOD,
    SCHEMA_VERSION,
    SeverityError,
    build_audited_event_index,
    build_severity_inference_manifest,
    canonical_phone_sequences_from_audited_prompts,
    canonical_event_id,
    event_scores_from_phone_rows,
    run_severity_experiment,
    severity_correlations,
    validate_severity_phone_scores,
    write_severity_inference_manifest,
    write_severity_outputs,
)


SPEAKERS = ("F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05")
SEVERITIES = {"F01": 1, "F03": 1, "F04": 1, "M01": 2,
              "M02": 2, "M03": 3, "M04": 3, "M05": 3}


def synthetic_inputs():
    manifest = []
    phones = []
    frozen = []
    for speaker_index, speaker in enumerate(SPEAKERS):
        severity = SEVERITIES[speaker]
        for event_index in range(2):
            stem = f"{speaker_index * 10 + event_index:04d}"
            new_event = f"{speaker}_Session1_{stem}"
            # Two audited recordings of the same reading event exercise the
            # event-first aggregation rule.
            for channel_index, channel in enumerate(("array", "head")):
                utterance_id = f"{channel}_{speaker}_{stem}"
                manifest.append({
                    "utterance_id": utterance_id,
                    "speaker_id": speaker,
                    "speaker_group": "dysarthric",
                    "session": "Session1",
                    "stem": stem,
                    "paper_severity": severity,
                })
                for method_index, method in enumerate(FROZEN_COMPARATOR_METHODS):
                    frozen.append({
                        "utterance_id": utterance_id,
                        "speaker_id": speaker,
                        "paper_severity": severity,
                        "method": method,
                        "score_direction": "higher_is_better",
                        "score": (
                            -(0.35 + 0.08 * method_index) * severity
                            + 0.017 * speaker_index * (method_index + 1)
                            + 0.009 * event_index
                            + 0.003 * channel_index
                        ),
                    })
            for channel_index, channel in enumerate(("array", "head")):
                recording_id = f"{channel}_{speaker}_{stem}"
                for phone_index in range(3):
                    phones.append({
                        "method": NEW_METHOD,
                        "speaker": speaker,
                        "event": new_event,
                        "recording_id": recording_id,
                        "phone_index": phone_index,
                        "stable": True,
                        "gop": -severity + 0.031 * speaker_index
                               + 0.004 * event_index + 0.001 * phone_index
                               + 0.0005 * channel_index,
                    })
                # Unstable tokens never contribute to the utterance mean.
                phones.append({
                    "method": NEW_METHOD,
                    "speaker": speaker,
                    "event": new_event,
                    "recording_id": recording_id,
                    "phone_index": 99,
                    "stable": False,
                    "gop": 10_000.0,
                })
    return manifest, phones, frozen


def test_event_key_and_signed_direction_are_frozen():
    assert canonical_event_id("F03", "Session2", "0149") == "F03|Session2|0149"
    result = severity_correlations([1, 1, 2, 2, 3, 3], [-1, -1.1, -2, -2.1, -3, -3.1])
    assert result["kendall_tau_b"] < 0
    assert result["abs_kendall_tau_b"] == pytest.approx(abs(result["kendall_tau_b"]))
    assert result["spearman_rho"] < 0
    assert result["pearson_r"] < 0


def test_event_first_speaker_first_complete_report_is_deterministic():
    manifest, phones, frozen = synthetic_inputs()
    kwargs = dict(
        new_scores_are_phone_level=True,
        n_bootstrap=80,
        seed=20260829,
        expected_manifest_rows=len(manifest),
        expected_events=16,
        expected_speakers=8,
    )
    report = run_severity_experiment(phones, manifest, frozen, **kwargs)
    repeated = run_severity_experiment(
        list(reversed(phones)), list(reversed(manifest)), list(reversed(frozen)), **kwargs
    )
    assert report == repeated
    assert report["schema_version"] == SCHEMA_VERSION
    assert report["comparison_level"] == "system_level_cross_backend"
    assert report["component_causal_claims_permitted"] is False
    assert report["severity_used_for_model_training"] is False
    assert report["audit_scope"]["n_manifest_rows"] == 32
    assert report["audit_scope"]["n_unique_reading_events"] == 16
    assert set(report["methods"]) == {NEW_METHOD, *FROZEN_COMPARATOR_METHODS}

    proposed = report["methods"][NEW_METHOD]
    assert proposed["coverage"]["event_coverage"] == 1.0
    assert proposed["coverage"]["n_speakers"] == 8
    assert proposed["correlations"]["kendall_tau_b"] < 0
    for comparison in report["paired_comparisons_vs_new_method"].values():
        bootstrap = comparison["paired_bootstrap"]
        assert bootstrap["n_bootstrap"] == 80
        assert bootstrap["unit"] == "paired_speaker"
        exact = comparison["exact_paired_sign_flip"]
        assert exact["n_speakers"] == 8
        assert exact["n_permutations"] == 256
        assert 0.0 <= exact["tests"]["kendall_tau_b"]["p_two_sided"] <= 1.0


def test_unstable_tokens_are_excluded_and_duplicate_token_is_rejected():
    _, phones, _ = synthetic_inputs()
    first_event = event_scores_from_phone_rows(phones)[0]
    assert first_event["n_phones"] == 6
    assert first_event["n_recordings"] == 2
    assert abs(first_event["score"]) < 10.0
    with pytest.raises(SeverityError, match="duplicate phone score"):
        event_scores_from_phone_rows([phones[0], dict(phones[0])])


def test_missing_event_fails_closed_instead_of_intersecting():
    manifest, phones, frozen = synthetic_inputs()
    missing_event = phones[0]["event"]
    incomplete = [row for row in phones if row["event"] != missing_event]
    with pytest.raises(SeverityError, match="exact audited-event coverage"):
        run_severity_experiment(
            incomplete,
            manifest,
            frozen,
            n_bootstrap=5,
            expected_manifest_rows=len(manifest),
            expected_events=16,
            expected_speakers=8,
        )


def test_frozen_row_coverage_and_duplicate_manifest_are_rejected():
    manifest, _, frozen = synthetic_inputs()
    with pytest.raises(SeverityError, match="duplicate audited utterance_id"):
        build_audited_event_index(
            [*manifest, dict(manifest[0])],
            expected_manifest_rows=None,
            expected_events=None,
            expected_speakers=None,
        )
    with pytest.raises(SeverityError, match="exact audited-row coverage"):
        run_severity_experiment(
            synthetic_inputs()[1],
            manifest,
            frozen[:-1],
            n_bootstrap=5,
            expected_manifest_rows=len(manifest),
            expected_events=16,
            expected_speakers=8,
        )


def test_output_json_and_csv_are_stable_and_machine_readable(tmp_path: Path):
    manifest, phones, frozen = synthetic_inputs()
    report = run_severity_experiment(
        phones,
        manifest,
        frozen,
        n_bootstrap=10,
        expected_manifest_rows=len(manifest),
        expected_events=16,
        expected_speakers=8,
    )
    json_a, csv_a = write_severity_outputs(
        report, tmp_path / "a" / "severity_metrics.json",
        tmp_path / "a" / "severity_metrics.csv",
    )
    json_b, csv_b = write_severity_outputs(
        report, tmp_path / "b" / "severity_metrics.json",
        tmp_path / "b" / "severity_metrics.csv",
    )
    assert json_a.read_bytes() == json_b.read_bytes()
    assert csv_a is not None and csv_b is not None
    assert csv_a.read_bytes() == csv_b.read_bytes()
    decoded = json.loads(json_a.read_text(encoding="utf-8"))
    assert decoded["schema_version"] == SCHEMA_VERSION
    assert len(csv_a.read_text(encoding="utf-8").splitlines()) == 6


def test_label_free_inference_manifest_deduplicates_physical_audio(tmp_path: Path):
    audio = tmp_path / "same.wav"
    audio.write_bytes(b"not-decoded-by-manifest-builder")
    other = tmp_path / "other.wav"
    other.write_bytes(b"not-decoded-by-manifest-builder")
    rows = [
        {
            "utterance_id": "array_F01_0001",
            "speaker_id": "F01",
            "speaker_group": "dysarthric",
            "session": "Session1",
            "stem": "0001",
            "paper_severity": 2,
            "wav_path": str(audio),
            "prompt": "A test sentence.",
            "microphone": "arrayMic",
        },
        {
            "utterance_id": "duplicate_F01_0001",
            "speaker_id": "F01",
            "speaker_group": "dysarthric",
            "session": "Session1",
            "stem": "0001",
            "paper_severity": 2,
            "wav_path": str(audio),
            "prompt": "A test sentence.",
            "microphone": "arrayMic",
        },
        {
            "utterance_id": "head_F01_0002",
            "speaker_id": "F01",
            "speaker_group": "dysarthric",
            "session": "Session1",
            "stem": "0002",
            "paper_severity": 2,
            "wav_path": str(other),
            "prompt": "Another sentence.",
            "microphone": "headMic",
        },
    ]
    inference = build_severity_inference_manifest(
        rows,
        expected_manifest_rows=3,
        expected_events=2,
        expected_speakers=1,
    )
    assert len(inference) == 3
    assert set(inference[0]) >= {"recording_id", "event_id", "wav_path", "prompt"}
    assert all("severity" not in row and "phn" not in row for row in inference)
    path = write_severity_inference_manifest(inference, tmp_path / "inference.jsonl")
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3


def _strict_severity_rows():
    audited = []
    scores = []
    canonical = {}
    for index, speaker in enumerate(SPEAKERS):
        recording = f"recording-{speaker}"
        event = f"{speaker}|Session1|{index:04d}"
        audited.append({
            "recording_id": recording,
            "event_id": event,
            "speaker_id": speaker,
            "prompt": "test prompt",
        })
        canonical[recording] = ("T", "EH")
        for phone_index, phone in enumerate(canonical[recording]):
            scores.append({
                "schema_version": "da-cf-gop.severity-phone-score.v1",
                "method": NEW_METHOD,
                "cohort": "severity8",
                "fold": f"severity8__outer_{speaker}",
                "speaker": speaker,
                "event": event,
                "recording_id": recording,
                "phone_index": phone_index,
                "canonical_phone": phone,
                "gop": -float(index + phone_index),
            })
    return audited, scores, canonical


def test_strict_severity_score_schema_and_prompt_token_coverage():
    audited, scores, canonical = _strict_severity_rows()
    stats = validate_severity_phone_scores(
        scores,
        audited_recordings=audited,
        canonical_phones_by_recording=canonical,
        expected_recordings=8,
        expected_events=8,
        expected_speakers=SPEAKERS,
    )
    assert stats == {
        "n_phone_rows": 16,
        "n_recordings": 8,
        "n_events": 8,
        "n_speakers": 8,
    }

    changed = [dict(row) for row in scores]
    changed[0]["canonical_phone"] = "K"
    with pytest.raises(SeverityError, match="disagree with prompt G2P"):
        validate_severity_phone_scores(
            changed,
            audited_recordings=audited,
            canonical_phones_by_recording=canonical,
            expected_recordings=8,
            expected_events=8,
            expected_speakers=SPEAKERS,
        )


def test_audited_prompt_g2p_is_rebuilt_for_every_recording():
    audited, _, _ = _strict_severity_rows()

    def fake_g2p(prompts):
        assert prompts == ["test prompt"]
        return [["T", "EH"]], [[]]

    canonical = canonical_phone_sequences_from_audited_prompts(
        audited, g2p=fake_g2p
    )
    assert set(canonical) == {row["recording_id"] for row in audited}
    assert set(canonical.values()) == {("T", "EH")}
