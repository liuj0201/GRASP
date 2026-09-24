from __future__ import annotations

from dataclasses import replace
import copy
from types import SimpleNamespace

import numpy as np
import pytest

import da_cf_gop.experiment as experiment_module
from da_cf_gop.artifacts import ArtifactLayout
from da_cf_gop.calibration import (
    AlternativeTemperatureCalibrator,
    CTCSequenceRecalibrator,
    DeletionCalibrator,
    SpeakerWeightedPlattCalibrator,
)
from da_cf_gop.experiment import (
    ALL_METHODS,
    ExperimentUtterance,
    METHOD_ADAPTED,
    METHOD_ABLATION_NO_ABSTENTION,
    METHOD_FIXED,
    METHOD_UNIFORM_CALIBRATED,
    _MatrixMemo,
    _DeferredOuterPredictions,
    _SharedRawCache,
    _TokenEvidence,
    _all_paired_comparisons,
    _artifact_with_descriptor_hash,
    _build_priors,
    _deletion_likelihoods,
    _deletions_from_matrix,
    _counterfactual_cache_path,
    _fit_deletion_contexts,
    _finalize_deferred_outer_predictions,
    _flatten_speaker_metrics,
    _fold_cache_paths,
    _fold_state_bundle_hash,
    _load_fitted_fold,
    _matrix,
    _paired_and_success,
    _prediction_fields,
    _score_matrix,
    _specific_output_fields,
    _write_final_assessment_if_ready,
    fit_outer_fold,
    predict_with_fitted_fold,
    utterance_from_manifest,
)
from da_cf_gop.folds import nested_loso_folds
from da_cf_gop.llm_schema import SpecificOutputGate
from da_cf_gop.phonology import PHONE_TO_CTC_ID, StableTokenLabel
from da_cf_gop.provenance import read_json, write_deterministic_npz, write_json


PATIENTS = ("F03", "F04", "M01", "M04", "M05")
HEALTHY = ("MC01",)


def _label(
    position: int,
    target: str,
    event_type: str,
    realized: str | None,
) -> StableTokenLabel:
    observed_index = position if event_type != "deletion" else None
    return StableTokenLabel(
        phone_index=position,
        canonical_phone=target,
        event_type=event_type,
        realized_phone=realized,
        stable=True,
        levenshtein_observed_index=observed_index,
        category_observed_index=observed_index,
        levenshtein_event_type=event_type,
        category_event_type=event_type,
        levenshtein_realized_phone=realized,
        category_realized_phone=realized,
    )


def _utterance(speaker: str, index: int, *, group: str) -> ExperimentUtterance:
    rng = np.random.default_rng(4100 + index)
    logits = rng.normal(0.0, 0.35, size=(5, 40))
    # Give the canonical sequence a valid, finite CTC path while preserving
    # enough variation for nonconstant inner-OOF scores.
    logits[0, PHONE_TO_CTC_ID["AA"]] += 2.2 + 0.1 * index
    logits[1, 0] += 1.2
    logits[2, PHONE_TO_CTC_ID["B"]] += 1.8 - 0.05 * index
    logits[3:, 0] += 1.0
    if index % 2:
        labels = (
            _label(0, "AA", "match", "AA"),
            _label(1, "B", "substitution", "D"),
        )
        realized = (PHONE_TO_CTC_ID["AA"], PHONE_TO_CTC_ID["D"])
    else:
        labels = (
            _label(0, "AA", "match", "AA"),
            _label(1, "B", "deletion", None),
        )
        realized = (PHONE_TO_CTC_ID["AA"],)
    return ExperimentUtterance(
        event_id=f"{speaker}_Session1_0001",
        speaker_id=speaker,
        speaker_group=group,
        canonical_phones=("AA", "B"),
        canonical_ids=(PHONE_TO_CTC_ID["AA"], PHONE_TO_CTC_ID["B"]),
        logits=logits,
        realized_ids=realized,
        token_labels=labels,
    )


def _pool() -> list[ExperimentUtterance]:
    rows = [
        _utterance(speaker, index, group="dysarthric")
        for index, speaker in enumerate(PATIENTS, start=1)
    ]
    rows.append(_utterance("MC01", 9, group="healthy"))
    return rows


def _config() -> dict:
    return {
        "seed": 20260829,
        "backend": {
            "blank_id": 0,
            "vocab_size": 40,
            "batch_size": 512,
            "device": "cpu",
        },
        "recalibration": {
            "steps": 0,
            "learning_rate": 0.02,
            "identity_l2": 1.0,
            "max_utterances": 20,
            "batch_size": 8,
            "enforce_entropy_guard": True,
        },
        "prior": {
            "uniform_backoff": 0.05,
            "dirichlet_pseudocount": 20.0,
            "enrichment_cap": 4.0,
            "deletion_base_weight": 0.5,
            "lambda_grid": [0.0, 0.25, 0.5, 1.0],
        },
        "deletion": {
            "quantile": 0.9,
            "min_tokens_per_phone": 1,
            "min_speakers_per_phone": 1,
        },
        "calibration": {
            "platt_c": 1.0,
            "alternative_temperature_grid": [0.5, 1.0, 2.0],
        },
        "llm_export": {
            "precision_floor": 0.8,
            "coverage_floor": 0.1,
            "alternative_margin_floor": 0.1,
            "threshold_grid": [0.5, 0.75, 0.95],
        },
    }


def test_outer_fit_is_heldout_label_and_audio_independent_and_cached(tmp_path) -> None:
    fold = nested_loso_folds("primary5", PATIENTS, HEALTHY)[0]
    pool = _pool()
    fitted = fit_outer_fold(
        fold,
        pool,
        _config(),
        methods=ALL_METHODS,
        cache_dir=tmp_path,
    )

    heldout_index = next(
        index for index, row in enumerate(pool) if row.speaker_id == fold.held_out_patient
    )
    heldout = pool[heldout_index]
    changed_labels = (
        _label(0, "AA", "deletion", None),
        _label(1, "B", "match", "B"),
    )
    changed_logits = np.array(heldout.logits, copy=True)
    changed_logits[:, PHONE_TO_CTC_ID["ZH"]] += np.linspace(0.0, 11.0, 5)
    changed = replace(
        heldout,
        logits=changed_logits,
        realized_ids=(PHONE_TO_CTC_ID["B"],),
        token_labels=changed_labels,
    )
    changed_pool = list(pool)
    changed_pool[heldout_index] = changed
    refitted = fit_outer_fold(
        fold,
        changed_pool,
        _config(),
        methods=ALL_METHODS,
        cache_dir=tmp_path,
    )

    assert fitted.fit_hash == refitted.fit_hash
    assert fitted.selected_lambda == refitted.selected_lambda
    np.testing.assert_array_equal(fitted.recalibrator.weight, refitted.recalibrator.weight)
    assert fitted.priors == refitted.priors
    assert fitted.inner_oof_rows == refitted.inner_oof_rows
    assert fitted.platt_calibrators[METHOD_FIXED].coefficient == 1.0
    assert fitted.platt_calibrators[METHOD_FIXED].intercept == 0.0
    assert fitted.temperature_calibrators[METHOD_FIXED].temperature == 1.0

    # JSON canonicalization sorts prior keys.  Cache reload must not let that
    # non-semantic mapping order alter any outer prediction, even by one ULP.
    original_predictions = predict_with_fitted_fold(
        fitted, [heldout.acoustic_only()], _config()
    )
    cached_predictions = predict_with_fitted_fold(
        refitted, [heldout.acoustic_only()], _config()
    )
    assert original_predictions == cached_predictions


def test_fold_cache_rejects_tampered_complete_fit_state_and_sources(tmp_path) -> None:
    fold = nested_loso_folds("primary5", PATIENTS, HEALTHY)[0]
    methods = (METHOD_ADAPTED,)
    config = _config()
    fitted = fit_outer_fold(
        fold, _pool(), config, methods=methods, cache_dir=tmp_path
    )
    recal_path, state_path, source_path = _fold_cache_paths(
        tmp_path, fold.cohort, fitted.fit_hash
    )
    original_state = state_path.read_bytes()
    base_state = read_json(state_path)
    assert _fold_state_bundle_hash(tmp_path, fitted)

    mutations = (
        "priors",
        "deletion_calibrators",
        "platt_calibrators",
        "temperature_calibrators",
        "gate",
        "inner_oof_rows",
    )
    for field in mutations:
        changed = copy.deepcopy(base_state)
        if field == "priors":
            changed[field]["adapted"]["AA"]["<DEL>"] += 0.001
        elif field == "deletion_calibrators":
            changed[field]["recalibrated_topology"]["global_penalty"] += 0.001
        elif field == "platt_calibrators":
            changed[field][METHOD_ADAPTED]["coefficient"] += 0.001
        elif field == "temperature_calibrators":
            changed[field][METHOD_ADAPTED]["temperature"] = 2.0
        elif field == "gate":
            changed[field]["enabled"] = not changed[field]["enabled"]
        else:
            changed[field][0]["p_error"] = 0.123456
        write_json(state_path, changed)
        assert _load_fitted_fold(
            fold=fold,
            fit_hash=fitted.fit_hash,
            config=config,
            cache_dir=tmp_path,
            methods=methods,
        ) is None
        state_path.write_bytes(original_state)

    original_source = source_path.read_bytes()
    changed_source = read_json(source_path)
    changed_source["records"][0]["speaker"] = "tampered"
    write_json(source_path, changed_source)
    assert _load_fitted_fold(
        fold=fold,
        fit_hash=fitted.fit_hash,
        config=config,
        cache_dir=tmp_path,
        methods=methods,
    ) is None
    source_path.write_bytes(original_source)

    original_recal = recal_path.read_bytes()
    recal_path.write_bytes(original_recal + b"tamper")
    assert _load_fitted_fold(
        fold=fold,
        fit_hash=fitted.fit_hash,
        config=config,
        cache_dir=tmp_path,
        methods=methods,
    ) is None
    recal_path.write_bytes(original_recal)
    assert _load_fitted_fold(
        fold=fold,
        fit_hash=fitted.fit_hash,
        config=config,
        cache_dir=tmp_path,
        methods=methods,
    ) is not None

def test_score_matrix_is_bitwise_invariant_to_prior_candidate_order() -> None:
    config = _config()
    utterance = _utterance("F03", 1, group="dysarthric")
    matrix = _matrix(
        utterance, CTCSequenceRecalibrator.identity(40), config,
        topology_correction=True,
    )
    fixed, _empirical, _adapted = _build_priors(_pool(), config)
    reversed_candidates = {
        target: dict(reversed(tuple(weights.items())))
        for target, weights in fixed.items()
    }
    expected = _score_matrix(matrix, utterance.canonical_phones, fixed)
    observed = _score_matrix(
        matrix, utterance.canonical_phones, reversed_candidates
    )
    assert observed == expected


def test_fitted_fold_inference_requires_no_phn_or_boundaries(tmp_path) -> None:
    fold = nested_loso_folds("primary5", PATIENTS, HEALTHY)[0]
    pool = _pool()
    fitted = fit_outer_fold(
        fold,
        pool,
        _config(),
        methods=(METHOD_ADAPTED,),
        cache_dir=tmp_path,
    )
    heldout = next(row for row in pool if row.speaker_id == fold.held_out_patient)
    raw_source = tmp_path / "synthetic-raw-logits.npz"
    write_deterministic_npz(raw_source, {"logits": heldout.logits})
    heldout = replace(heldout, logits_source=str(raw_source))
    acoustic_only = heldout.acoustic_only()
    assert acoustic_only.realized_ids == ()
    assert acoustic_only.token_labels == ()

    rows = predict_with_fitted_fold(
        fitted, [acoustic_only], _config(), cache_dir=tmp_path
    )
    assert len(rows) == len(heldout.canonical_phones)
    assert all(row["method"] == METHOD_ADAPTED for row in rows)
    assert {(row["event"], row["phone_index"]) for row in rows} == {
        (heldout.event_id, 0),
        (heldout.event_id, 1),
    }
    for row in rows:
        assert np.isfinite(row["gop"])
        assert 0.0 <= row["p_error"] <= 1.0
        assert len(row["candidate_probabilities"]) == 39
        assert len(row["candidate_log_scores"]) == 39
        assert set(row["candidate_log_scores"]) == set(row["candidate_probabilities"])
        assert type(row["specific_output_authorized"]) is bool
        assert row["specific_prediction_authorized"] == row["specific_output_authorized"]
        assert np.isclose(sum(row["candidate_probabilities"].values()), 1.0)
        assert "gold_event" not in row
        assert "gold_realized" not in row
    matrix_path = _counterfactual_cache_path(
        tmp_path, fitted.cohort, fitted.fit_hash, heldout.event_id
    )
    assert len(_artifact_with_descriptor_hash(matrix_path)) == 64


def test_acoustic_manifest_loading_does_not_read_phn_sidecar(monkeypatch, tmp_path) -> None:
    class GoldGuard:
        reading_event_id = "F03_Session1_0001"
        speaker_id = "F03"
        speaker_group = "dysarthric"
        canonical_phones = ("AA", "B")

        @property
        def phn_model_phones(self):
            raise AssertionError("PHN sequence was read during acoustic loading")

        @property
        def stable_tokens(self):
            raise AssertionError("gold token labels were read during acoustic loading")

    cached = SimpleNamespace(
        utterance_id=GoldGuard.reading_event_id,
        canonical_ids=(PHONE_TO_CTC_ID["AA"], PHONE_TO_CTC_ID["B"]),
        logits=np.zeros((5, 40), dtype=np.float64),
    )
    monkeypatch.setattr(experiment_module, "load_logits_cache", lambda _path: cached)
    utterance = utterance_from_manifest(
        GoldGuard(), tmp_path / "not-read.npz", include_phn_sidecar=False
    )
    assert utterance.realized_ids == ()
    assert utterance.token_labels == ()


def test_outer_gold_is_finalized_only_from_deferred_gold_free_predictions(
    monkeypatch,
) -> None:
    acoustics = {
        row.event_id: row.acoustic_only()
        for row in (
            _utterance("F03", 1, group="dysarthric"),
            _utterance("F04", 2, group="dysarthric"),
        )
    }
    manifests = {
        event: SimpleNamespace(
            reading_event_id=event,
            speaker_id=acoustic.speaker_id,
            phn_model_phones=("AA", "D"),
            stable_tokens=(
                _label(0, "AA", "match", "AA"),
                _label(1, "B", "substitution", "D"),
            ),
        )
        for event, acoustic in acoustics.items()
    }
    metadata = {
        event: {
            "audio_microphone": "headMic",
            "phn_microphone": "headMic",
            "severity_rank": 2.0,
            "severity_label": "test",
        }
        for event in acoustics
    }
    deferred = []
    for index, (event, acoustic) in enumerate(acoustics.items()):
        fold_id = f"outer-{acoustic.speaker_id}"
        deferred.append(
            _DeferredOuterPredictions(
                fold_id,
                acoustic.speaker_id,
                (event,),
                (
                    {
                        "fold": fold_id,
                        "speaker": acoustic.speaker_id,
                        "event": event,
                        "phone_index": 0,
                    },
                ),
            )
        )
    with pytest.raises(ValueError, match="gold was attached before"):
        _DeferredOuterPredictions(
            "bad",
            "F03",
            (next(iter(acoustics)),),
            (
                {
                    "fold": "bad",
                    "speaker": "F03",
                    "event": next(iter(acoustics)),
                    "phone_index": 0,
                    "gold_event": "correct",
                },
            ),
        )

    first_event = next(iter(acoustics))
    with pytest.raises(ValueError, match="held-out PHN"):
        experiment_module._attach_phn_sidecar(
            acoustics[first_event],
            manifests[first_event],
            role="training",
            held_out_patient=acoustics[first_event].speaker_id,
        )

    all_predictions_generated = True
    accesses = []
    original_attach = experiment_module._attach_phn_sidecar

    def guarded_attach(acoustic, row, *, role, held_out_patient):
        assert all_predictions_generated
        assert role == "evaluation"
        accesses.append((acoustic.event_id, role))
        return original_attach(
            acoustic,
            row,
            role=role,
            held_out_patient=held_out_patient,
        )

    monkeypatch.setattr(experiment_module, "_attach_phn_sidecar", guarded_attach)
    joined = _finalize_deferred_outer_predictions(
        deferred, acoustics, manifests, metadata
    )
    assert len(accesses) == len(deferred)
    assert len(joined) == len(deferred)
    assert all(row["gold_event"] == "correct" for row in joined)
    assert all(not acoustic.token_labels for acoustic in acoustics.values())


def test_deletion_only_ctc_is_exact_projection_of_full_matrix() -> None:
    config = _config()
    recalibrator = CTCSequenceRecalibrator.identity(40)
    for utterance in _pool()[:3]:
        full = _matrix(
            utterance, recalibrator, config, topology_correction=True
        )
        projected = _deletions_from_matrix(full)
        narrow = _deletion_likelihoods(utterance, recalibrator, config)
        assert narrow.canonical_log_probability == projected.canonical_log_probability
        assert (
            narrow.canonical_corrected_log_probability
            == projected.canonical_corrected_log_probability
        )
        np.testing.assert_allclose(
            narrow.deletion_log_probabilities,
            projected.deletion_log_probabilities,
            rtol=0.0,
            atol=5e-14,
        )
        np.testing.assert_allclose(
            narrow.deletion_corrected_log_probabilities,
            projected.deletion_corrected_log_probabilities,
            rtol=0.0,
            atol=5e-14,
        )


def test_deletion_calibration_matches_complete_matrix_reference() -> None:
    config = _config()
    training = _pool()
    recalibrator = CTCSequenceRecalibrator.identity(40)
    full = {
        row.event_id: _matrix(
            row, recalibrator, config, topology_correction=True
        )
        for row in training
    }
    reference_rows = []
    for utterance in training:
        matrix = full[utterance.event_id]
        projected = _deletions_from_matrix(matrix)
        for label in utterance.token_labels:
            if label.stable and label.event_type == "match":
                reference_rows.append(
                    {
                        "speaker_id": utterance.speaker_id,
                        "target_phone": label.canonical_phone,
                        "canonical_log_probability": (
                            matrix.canonical_corrected_log_probability
                        ),
                        "deletion_log_probability": float(
                            projected.deletion_corrected_log_probabilities[
                                label.phone_index
                            ]
                        ),
                        "is_error": False,
                    }
                )
    reference = DeletionCalibrator.fit(
        reference_rows, quantile=0.9, min_tokens=1, min_speakers=1
    )
    optimized = _fit_deletion_contexts(
        training, recalibrator, config, _MatrixMemo()
    )["recalibrated_topology"]
    assert optimized == reference


def test_raw_content_cache_reuses_only_fold_independent_matrix(monkeypatch) -> None:
    config = _config()
    utterance = _utterance("F03", 1, group="dysarthric")
    shared = _SharedRawCache()
    first = _MatrixMemo(shared)
    second = _MatrixMemo(shared)
    recalibrator = CTCSequenceRecalibrator.identity(40)
    calls = []
    original = experiment_module._matrix

    def counted(*args, **kwargs):
        calls.append(args[1])
        return original(*args, **kwargs)

    monkeypatch.setattr(experiment_module, "_matrix", counted)
    raw_a = first.get(utterance, None, config)
    raw_b = second.get(utterance.acoustic_only(), None, config)
    assert raw_a is raw_b
    first.get(utterance, recalibrator, config)
    second.get(utterance, recalibrator, config)
    assert len(calls) == 3
    assert sum(value is None for value in calls) == 1


def test_candidate_log_scores_and_no_abstention_policy_are_explicit() -> None:
    alternatives = {
        phone: -8.0
        for phone in experiment_module.alternatives_for(
            "AA", experiment_module.ARPABET_39, include_deletion=True
        )
    }
    alternatives["D"] = 0.0
    alternatives["ZH"] = -float("inf")
    evidence = _TokenEvidence(0, "AA", -4.0, alternatives)
    prediction = _prediction_fields(
        evidence,
        SpeakerWeightedPlattCalibrator(1.0, 0.0, 1.0),
        AlternativeTemperatureCalibrator(1.0, {1.0: 0.0}),
    )
    assert len(prediction["candidate_log_scores"]) == 39
    assert prediction["candidate_log_scores"]["ZH"] is None
    gate = SpecificOutputGate(True, 0.99, 0.99, 1.0, 0.1, 10, 1, None)
    adapted = _specific_output_fields(METHOD_ADAPTED, prediction, gate, _config())
    unfiltered = _specific_output_fields(
        METHOD_ABLATION_NO_ABSTENTION, prediction, gate, _config()
    )
    assert adapted["specific_output_authorized"] is False
    assert adapted["specific_abstain"] is True
    assert adapted["decision"] == "atypical_unspecified"
    assert unfiltered["specific_output_authorized"] is True
    assert unfiltered["specific_abstain"] is False
    assert unfiltered["decision"] == "substitution"


def test_paired_report_uses_speaker_differences_for_exact_test() -> None:
    def group(method: str, values: list[float]) -> dict:
        macro = {
            "auprc": {"value": float(np.mean(values))},
            "auroc": {"value": 0.8},
            "brier": {"value": 0.2},
            "substitution_top1": {"value": 0.4},
            "deletion_f1": {"value": 0.3},
        }
        return {
            "method": method,
            "per_speaker": [
                {"speaker": speaker, "auprc": value}
                for speaker, value in zip(PATIENTS, values)
            ],
            "macro": macro,
        }

    report = {
        "results": [
            group(METHOD_ADAPTED, [0.7, 0.8, 0.6, 0.9, 0.75]),
            group(METHOD_UNIFORM_CALIBRATED, [0.6, 0.7, 0.55, 0.8, 0.7]),
        ]
    }
    config = {
        "seed": 20260829,
        "evaluation": {
            "primary_comparator": METHOD_UNIFORM_CALIBRATED,
            "bootstrap_replicates": 100,
            "required_absolute_improvement": 0.03,
            "required_primary_speakers_improved": 4,
            "maximum_auroc_drop": 0.01,
            "maximum_brier_increase": 0.01,
            "maximum_identification_drop": 0.01,
        },
    }
    paired, success = _paired_and_success(report, config, "primary5")
    comparisons = _all_paired_comparisons(report, config)
    assert paired["n_improved"] == 5
    assert paired["exact_sign_flip"]["n_speakers"] == 5
    assert paired["exact_sign_flip"]["difference"] > 0
    assert success["auprc_difference"] > 0.03
    assert len(comparisons) == 1
    assert comparisons[0]["comparator"] == METHOD_UNIFORM_CALIBRATED
    assert comparisons[0]["bootstrap"]["n_bootstrap"] == 100


def test_speaker_csv_and_cross_cohort_claim_are_machine_readable(tmp_path) -> None:
    rows = _flatten_speaker_metrics(
        {
            "results": [
                {
                    "cohort": "primary5",
                    "method": METHOD_ADAPTED,
                    "per_speaker": [
                        {"speaker": "F03", "auprc": 0.8, "detail": {"n": 2}}
                    ],
                }
            ]
        }
    )
    assert rows == [
        {
            "cohort": "primary5",
            "method": METHOD_ADAPTED,
            "speaker": "F03",
            "auprc": 0.8,
            "detail": '{"n": 2}',
        }
    ]

    layout = ArtifactLayout.from_path(tmp_path)
    layout.create()
    for cohort, success, difference, improved, ci_low, p_value in (
        ("primary5", True, 0.04, 5, 0.01, 0.03125),
        ("sensitivity7", True, 0.03, 6, 0.005, 0.03125),
    ):
        destination = layout.evaluation / cohort / "metrics.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        write_json(
            destination,
            {
                "config_sha256": "a" * 64,
                "success_assessment": {
                    "cohort_success": success,
                    "auprc_difference": difference,
                },
                "paired_comparison": {
                    "n_improved": improved,
                    "bootstrap": {"ci_low": ci_low},
                    "exact_sign_flip": {"p_two_sided": p_value},
                },
            },
        )
    path = _write_final_assessment_if_ready(layout)
    assert path == layout.evaluation / "final_assessment.json"
    assessment = read_json(path)
    assert assessment["strictly_supported_superiority"] is True
    assert assessment["permitted_wording"] == "statistically_supported_superiority"
