from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from da_cf_gop.cli import _dispatch
from da_cf_gop.experiment import ALL_METHODS, METHOD_ADAPTED, ExperimentUtterance, FittedFold
from da_cf_gop.phonology import PHONE_TO_CTC_ID
from da_cf_gop.provenance import sha256_file, write_json
from da_cf_gop.severity import EXPECTED_DYSARTHRIC_SPEAKERS, NEW_METHOD, SeverityError
from da_cf_gop.severity_scoring import (
    AuditedAcousticInput,
    generate_severity_phone_scores,
    load_audited_acoustic_inputs,
    sensitivity_fit_reuse_plan,
    severity_loso_folds,
    validate_audited_backend_descriptor,
)


def test_severity_folds_keep_m03_as_target_without_inventing_phn() -> None:
    labelled = tuple(
        speaker for speaker in EXPECTED_DYSARTHRIC_SPEAKERS if speaker != "M03"
    )
    healthy = ("MC01", "MC02", "MC03", "MC04")
    folds = severity_loso_folds(
        EXPECTED_DYSARTHRIC_SPEAKERS, labelled, healthy
    )

    assert tuple(fold.held_out_patient for fold in folds) == EXPECTED_DYSARTHRIC_SPEAKERS
    m03 = next(fold for fold in folds if fold.held_out_patient == "M03")
    f03 = next(fold for fold in folds if fold.held_out_patient == "F03")
    assert m03.training_patients == labelled
    assert len(m03.inner_folds) == 7
    assert "M03" not in f03.training_patients
    assert "F03" not in f03.training_patients
    assert len(f03.training_patients) == 6
    assert len(f03.inner_folds) == 6
    assert all(
        fold.held_out_patient not in set(fold.training_patients)
        for fold in folds
    )


def test_audited_loader_uses_prompt_g2p_and_ignores_legacy_canonical_ids(
    tmp_path: Path,
) -> None:
    source = tmp_path / "legacy.npz"
    np.savez(
        source,
        utterance_id=np.asarray("rec-1"),
        logits=np.zeros((8, 40), dtype=np.float32),
        canonical_ids=np.asarray([1, 1, 1], dtype=np.int16),
        duration_s=np.asarray(1.0, dtype=np.float64),
        # The loader validates the scalar string container but never parses or
        # uses this historical metadata as a model input.
        metadata_json=np.asarray('{"severity":"must-not-be-used"}'),
    )
    rows = [
        {
            "recording_id": "rec-1",
            "event_id": "F03|Session1|0001",
            "speaker_id": "F03",
            "session": "Session1",
            "stem": "0001",
            "prompt": "test prompt",
            "official_logits_npz": str(source),
        }
    ]

    loaded = load_audited_acoustic_inputs(
        rows,
        g2p=lambda prompts: ([['T', 'AA'] for _ in prompts], [[] for _ in prompts]),
    )

    assert len(loaded) == 1
    assert loaded[0].legacy_canonical_ids_match is False
    assert loaded[0].utterance.canonical_phones == ("T", "AA")
    assert loaded[0].utterance.canonical_ids == (
        PHONE_TO_CTC_ID["T"],
        PHONE_TO_CTC_ID["AA"],
    )
    assert loaded[0].utterance.realized_ids == ()
    assert loaded[0].utterance.token_labels == ()


def _audited(speaker: str, recording: str) -> AuditedAcousticInput:
    utterance = ExperimentUtterance(
        event_id=f"severity_recording::{recording}",
        speaker_id=speaker,
        speaker_group="dysarthric",
        canonical_phones=("T",),
        canonical_ids=(PHONE_TO_CTC_ID["T"],),
        logits=np.zeros((3, 40), dtype=np.float64),
    )
    return AuditedAcousticInput(
        recording_id=recording,
        event_id=f"{speaker}|Session1|0001",
        speaker_id=speaker,
        session="Session1",
        stem="0001",
        source_logits=f"{recording}.npz",
        source_sha256="a" * 64,
        legacy_canonical_ids_match=False,
        utterance=utterance,
    )


def test_generation_never_supplies_outer_or_unlabelled_patient_to_training(
    tmp_path: Path,
) -> None:
    # The helper intentionally requires at least three labelled patients.
    patients = ("F01", "F03", "F04", "M03")
    labelled = ("F01", "F03", "F04")
    healthy = ("MC01",)
    folds = severity_loso_folds(patients, labelled, healthy)
    training_rows = [
        SimpleNamespace(speaker_id=speaker) for speaker in (*labelled, *healthy)
    ]
    audited = [_audited(speaker, f"rec-{speaker}") for speaker in patients]
    supplied: dict[str, set[str]] = {}

    def loader(rows, _config):
        speakers = {row.speaker_id for row in rows}
        loader.last_speakers = speakers
        return []

    loader.last_speakers = set()

    def fit(fold, _utterances, _config, **kwargs):
        supplied[fold.held_out_patient] = set(loader.last_speakers)
        assert kwargs["methods"] == (METHOD_ADAPTED,)
        return SimpleNamespace(
            fit_hash=(fold.held_out_patient.lower() + "0" * 64)[:64],
            held_out_patient=fold.held_out_patient,
            methods=(METHOD_ADAPTED,),
        )

    def predict(fitted, utterances, _config, **_kwargs):
        assert all(row.speaker_id == fitted.held_out_patient for row in utterances)
        return [
            {
                "method": METHOD_ADAPTED,
                "event": row.event_id,
                "phone_index": 0,
                "target": "T",
                "gop": 0.25,
            }
            for row in utterances
        ]

    rows, hashes = generate_severity_phone_scores(
        {"unused": True},
        training_rows,  # type: ignore[arg-type]
        audited,
        folds,
        fold_cache_dir=tmp_path,
        progress=lambda _message: None,
        training_loader=loader,
        fit_fold=fit,
        predict_fold=predict,
    )

    assert len(rows) == len(patients)
    assert set(hashes) == set(patients)
    assert {row["method"] for row in rows} == {NEW_METHOD}
    for held_out, speakers in supplied.items():
        assert held_out not in speakers
        assert "M03" not in speakers
        assert "MC01" in speakers


def test_completed_sensitivity_run_reuses_exact_full_fits_and_only_fits_m03(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import da_cf_gop.severity_scoring as module

    patients = ("F01", "F03", "F04", "M03")
    labelled = ("F01", "F03", "F04")
    healthy = ("MC01",)
    folds = severity_loso_folds(patients, labelled, healthy)
    completed = tmp_path / "evaluation" / "sensitivity7" / "phone_predictions.jsonl.gz"
    completed.parent.mkdir(parents=True)
    completed.write_bytes(b"completed")
    overrides, methods, caches = sensitivity_fit_reuse_plan(
        tmp_path, folds, labelled, healthy
    )

    assert set(overrides) == set(labelled)
    assert "M03" not in overrides
    assert all(fold.cohort == "sensitivity7" for fold in overrides.values())
    assert all(tuple(methods[speaker]) == ALL_METHODS for speaker in labelled)
    assert all(Path(caches[speaker]) == tmp_path / "cache" for speaker in labelled)

    training_rows = [
        SimpleNamespace(speaker_id=speaker) for speaker in (*labelled, *healthy)
    ]
    audited = [_audited(speaker, f"reuse-{speaker}") for speaker in patients]
    cached_calls: list[tuple[str, str, tuple[str, ...], Path]] = []
    fitted_calls: list[tuple[str, str, tuple[str, ...], Path]] = []

    def loader(_rows, _config):
        return []

    def fitted_state(fold, fit_hash, method_inventory):
        return FittedFold(
            cohort=fold.cohort,
            fold_id=fold.fold_id,
            held_out_patient=fold.held_out_patient,
            training_speakers=(),
            fit_hash=fit_hash,
            selected_lambda=0.0,
            lambda_metrics={},
            recalibrator=None,  # type: ignore[arg-type]
            priors={},
            deletion_calibrators={},
            platt_calibrators={},
            temperature_calibrators={},
            gate=None,  # type: ignore[arg-type]
            inner_oof_rows=(),
            methods=tuple(method_inventory),
            settings={},
        )

    def fake_hash(fold, _rows, _config, *, methods):
        return (fold.held_out_patient.lower() + "1" * 64)[:64]

    def fake_load(*, fold, fit_hash, config, cache_dir, methods):
        del config
        cached_calls.append((fold.held_out_patient, fold.cohort, tuple(methods), Path(cache_dir)))
        return fitted_state(fold, fit_hash, methods)

    def fake_fit(fold, _rows, _config, **kwargs):
        fitted_calls.append(
            (fold.held_out_patient, fold.cohort, tuple(kwargs["methods"]), Path(kwargs["cache_dir"]))
        )
        return fitted_state(fold, "2" * 64, kwargs["methods"])

    def predict(fitted, utterances, _config, **_kwargs):
        return [
            {
                "method": METHOD_ADAPTED,
                "event": row.event_id,
                "phone_index": 0,
                "target": "T",
                "gop": 0.5,
            }
            for row in utterances
        ]

    monkeypatch.setattr(module, "_fold_fit_hash", fake_hash)
    monkeypatch.setattr(module, "_load_fitted_fold", fake_load)
    rows, _hashes = generate_severity_phone_scores(
        {"unused": True},
        training_rows,  # type: ignore[arg-type]
        audited,
        folds,
        fold_cache_dir=tmp_path / "cache" / "severity" / "implementation",
        progress=lambda _message: None,
        training_loader=loader,
        fit_fold=fake_fit,
        predict_fold=predict,
        fit_fold_overrides=overrides,
        fit_methods_by_speaker=methods,
        fit_cache_by_speaker=caches,
        require_cached_fit_targets=tuple(overrides),
    )

    assert len(rows) == 4
    assert {call[0] for call in cached_calls} == set(labelled)
    assert all(call[1] == "sensitivity7" for call in cached_calls)
    assert all(call[2] == ALL_METHODS for call in cached_calls)
    assert all(call[3] == tmp_path / "cache" for call in cached_calls)
    assert fitted_calls == [
        (
            "M03",
            "severity8",
            (METHOD_ADAPTED,),
            tmp_path / "cache" / "severity" / "implementation",
        )
    ]


def test_audited_backend_descriptor_is_bound_to_model_and_vocab_bytes(
    tmp_path: Path,
) -> None:
    cache = tmp_path / "legacy-cache"
    checkpoint = tmp_path / "checkpoint"
    processor = tmp_path / "processor"
    cache.mkdir()
    checkpoint.mkdir()
    processor.mkdir()
    model = checkpoint / "pytorch_model.bin"
    vocab = processor / "vocab.json"
    model.write_bytes(b"frozen model bytes")
    vocab.write_text('{"<pad>":0,"AA":1}', encoding="utf-8")
    descriptor = {
        "blank_id": 0,
        "vocab_size": 40,
        "config_sha256": "a" * 64,
        "vocab_sha256": sha256_file(vocab),
        "pytorch_model_bin_sha256": sha256_file(model),
        "n_requested": 571,
    }
    write_json(cache / "backend_descriptor.json", descriptor)
    config = {
        "backend": {"blank_id": 0, "vocab_size": 40},
        "paths": {"checkpoint": str(checkpoint), "processor": str(processor)},
        "severity": {"audited_recordings": 415},
    }

    audit = validate_audited_backend_descriptor(config, cache)
    assert audit["checkpoint_weight_sha256"] == sha256_file(model)
    assert audit["vocab_sha256"] == sha256_file(vocab)
    assert audit["per_npz_companion_descriptors"] is False

    vocab.write_text('{"<pad>":0,"AA":2}', encoding="utf-8")
    with pytest.raises(SeverityError, match="vocabulary hash"):
        validate_audited_backend_descriptor(config, cache)


def test_same_implementation_reuses_complete_severity_score_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import da_cf_gop.severity_scoring as module

    artifacts = tmp_path / "artifacts"
    manifests = artifacts / "manifests"
    cache = artifacts / "cache"
    manifests.mkdir(parents=True)
    cache.mkdir(parents=True)
    training_manifest = manifests / "severity_training8.jsonl.gz"
    training_index = cache / "severity_training8.logits.index.json"
    fold_document = manifests / "severity_training8.folds.json"
    audited_manifest = tmp_path / "audited.json"
    training_manifest.write_bytes(b"stable training manifest")
    training_index.write_bytes(b"stable training logits")
    fold_document.write_bytes(b"stable folds")
    audited_manifest.write_bytes(b"stable audited manifest")

    config = {
        "_config_hash": "a" * 64,
        "paths": {"artifacts": str(artifacts)},
        "backend": {"vocab_size": 40},
        "cohorts": {"healthy_phn": ["MC01"]},
        "severity": {
            "audited_manifest": str(audited_manifest),
            "audited_logits_cache": str(tmp_path / "legacy-cache"),
            "audited_recordings": 1,
            "unique_reading_events": 1,
            "dysarthric_speakers": ["F03"],
        },
    }
    audited = _audited("F03", "rec-F03")
    generated: list[Path] = []
    messages: list[str] = []

    monkeypatch.setattr(
        module, "build_severity_training_manifest_stage", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        module, "extract_severity_training_logits_stage", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        module,
        "build_audited_severity_events",
        lambda *_args, **_kwargs: {"recordings": [{}], "n_events": 1},
    )
    monkeypatch.setattr(
        module,
        "validate_audited_backend_descriptor",
        lambda *_args, **_kwargs: {
            "descriptor_sha256": "b" * 64,
            "checkpoint_weight_sha256": "c" * 64,
            "vocab_sha256": "d" * 64,
        },
    )
    monkeypatch.setattr(
        module,
        "write_severity_inference_manifest",
        lambda _rows, path: write_json(path, {"recording_id": "rec-F03"}),
    )
    monkeypatch.setattr(
        module, "load_audited_acoustic_inputs", lambda *_args, **_kwargs: [audited]
    )
    monkeypatch.setattr(
        module,
        "read_manifest",
        lambda _path: [SimpleNamespace(speaker_id="F03", speaker_group="dysarthric")],
    )
    monkeypatch.setattr(
        module,
        "severity_loso_folds",
        lambda *_args, **_kwargs: (SimpleNamespace(held_out_patient="F03"),),
    )
    monkeypatch.setattr(module, "_implementation_sha256", lambda: "1" * 64)

    def fake_generate(*_args, **kwargs):
        generated.append(Path(kwargs["fold_cache_dir"]))
        return ([{"stable": True}], {"F03": "f" * 64})

    monkeypatch.setattr(module, "generate_severity_phone_scores", fake_generate)

    first = module.run_severity_scoring_stage(config, progress=messages.append)
    second = module.run_severity_scoring_stage(config, progress=messages.append)

    assert first["details"]["reused"] is False
    assert second["details"]["reused"] is True
    assert generated == [artifacts / "cache" / "severity" / ("1" * 16)]
    assert messages.count(
        "severity phone scores: reused exact content-bound artifact"
    ) == 1


def test_run_severity_cli_scores_before_statistical_comparison(monkeypatch) -> None:
    import da_cf_gop.cli as cli
    import da_cf_gop.severity as severity
    import da_cf_gop.severity_scoring as scoring

    calls: list[str] = []
    monkeypatch.setattr(cli, "load_config", lambda _path: {"resolved": True})
    monkeypatch.setattr(
        scoring, "run_severity_scoring_stage", lambda _cfg: calls.append("score")
    )
    monkeypatch.setattr(
        severity, "run_severity_stage", lambda _cfg: calls.append("evaluate")
    )

    _dispatch(Namespace(command="run-severity", config=Path("unused.json")))
    assert calls == ["score", "evaluate"]
