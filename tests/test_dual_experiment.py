"""Protocol tests; no large model, corpus audio, or CUDA is required."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from da_cf_gop import dual_experiment as experiment
from da_cf_gop.dual_lexicon import evaluate_alignment


def configuration():
    config = json.loads(experiment.DEFAULT_CONFIG.read_text(encoding="utf-8"))
    config.update(alpha_grid=[20.0], edit_scale_grid=[1.0], bootstrap_replicates=20)
    return config


def record(speaker, event=None):
    return {"event": event or f"{speaker}_event", "speaker": speaker, "prompt": "test",
            "canonical_phones": ["T", "D"], "acceptable_phones": [["T"], ["D"]],
            "diagnostic_mask": [True, True], "quality_exclusion": None,
            "labels": evaluate_alignment([["T"], ["D"]], ["T", "T"]),
            "phn_path": "must_not_enter_inference", "severity": "must_not_enter_inference"}


def scored_tokens():
    return [{"phone_index": i, "target": p, "gop": score,
             "candidate_event": "substitution", "candidate_phone": "T",
             "candidate_confidence": .9, "candidate_margin": .8}
            for i, (p, score) in enumerate((("T", 2.0), ("D", -2.0)))]


def evaluated_fixture():
    config = configuration()
    rows = [record(s) for s in [*config["cohorts"]["primary5"], "MC04"]]
    predictions = []
    for method in (*experiment.GRAPH_METHODS, *experiment.BASELINE_METHODS):
        for source in rows:
            for token in scored_tokens():
                predictions.append({**token, "method": method, "event": source["event"],
                                    "speaker": source["speaker"], "fold": "synthetic",
                                    "p_error": .1 if token["phone_index"] == 0 else .9,
                                    "predicted_error": token["phone_index"] == 1,
                                    "specific_authorized": False})
    return config, rows, predictions


def test_folds_are_patient_disjoint_and_health_roles_are_independent():
    config = configuration()
    for cohort in ("primary5", "sensitivity7"):
        folds = experiment.make_folds(config, cohort)
        assert sorted(f["test_patient"] for f in folds) == sorted(config["cohorts"][cohort])
        for fold in folds:
            assert fold["test_patient"] != fold["development_patient"]
            assert not set(fold["training_patients"]) & {fold["test_patient"], fold["development_patient"]}
            assert set(fold["training_patients"]) | {fold["test_patient"], fold["development_patient"]} == set(config["cohorts"][cohort])
            assert not set(fold["healthy_development"]) & set(fold["healthy_test"])
            assert not set(fold["healthy_train_reserved"]) & set(fold["healthy_development"] + fold["healthy_test"])


def test_acoustic_record_is_an_exact_whitelist_without_gold():
    source = record("F03")
    output = experiment.acoustic_record(source)
    assert set(output) == {"event", "speaker", "prompt", "canonical_phones", "acceptable_phones"}
    source["labels"] = {"corrupted_test_gold": True}
    source["phn_path"] = "a_changed_path"
    assert experiment.acoustic_record(source) == output


def test_test_gold_cannot_change_prior_calibrators_or_predictions(monkeypatch):
    config = configuration()
    fold = experiment.make_folds(config, "primary5")[0]
    speakers = [*config["cohorts"]["primary5"], "MC03", "MC04"]
    rows = [record(s) for s in speakers]
    excluded = record(fold["training_patients"][0], "excluded_training")
    excluded.update(quality_exclusion="unpaired_or_nonmatching_closure_release", labels={"invalid_if_accessed": True})
    rows.append(excluded)
    sources = {r["event"]: None for r in rows}
    cache = {r["event"]: {m: scored_tokens() for m in experiment.BASELINE_METHODS} for r in rows}
    gaps = {r["event"]: {} for r in rows}
    monkeypatch.setattr(experiment, "log_probs_for", lambda _: np.zeros((2, 40)))
    monkeypatch.setattr(experiment, "graph_predictions", lambda *a, **k: (scored_tokens(), [], {}))
    monkeypatch.setattr(experiment, "viterbi_predictions", lambda *a, **k: scored_tokens())
    monkeypatch.setattr(experiment, "viterbi_gap_predictions", lambda *a, **k: [])
    saved = []
    monkeypatch.setattr(experiment, "write_json", lambda path, value: saved.append(value))
    tmp_path = Path("unused_write_is_monkeypatched")
    first = experiment.run_fold(fold, rows, sources, cache, gaps, config, tmp_path)
    changed = copy.deepcopy(rows)
    for source in changed:
        if source["speaker"] in [fold["test_patient"], *fold["healthy_test"]]:
            source["labels"] = {"would_raise_if_test_gold_were_read": True}
            source["phn_path"] = "deleted_test_annotation"
    second = experiment.run_fold(fold, changed, sources, cache, gaps, config, tmp_path)
    assert first == second
    assert saved[0] == saved[1]
    assert saved[0]["prior"]["training_speakers"] == sorted(fold["training_patients"])
    for calibrator in saved[0]["calibrators"].values():
        assert calibrator["training_speakers"] == sorted([fold["development_patient"], *fold["healthy_development"]])
    assert all("gold_error" not in row for row in first[0])


def test_comparison_uses_identical_keys_and_rejects_missing_or_duplicate_tokens():
    config, records, predictions = evaluated_fixture()
    _, metrics = experiment.evaluate_results(predictions, records, "primary5", config)
    assert metrics["same_token_keys"] is True
    with pytest.raises(ValueError, match="identical scored token keys"):
        experiment.evaluate_results(predictions[1:], records, "primary5", config)
    with pytest.raises(ValueError, match="duplicate token"):
        experiment.evaluate_results([*predictions, predictions[0]], records, "primary5", config)


def test_quality_and_alignment_masks_are_identical_for_every_method():
    config, records, predictions = evaluated_fixture()
    records[0]["quality_exclusion"] = "unpaired_or_nonmatching_closure_release"
    records[1]["labels"]["tokens"][0]["diagnostic"] = False
    evaluated = experiment.add_gold(predictions, records)
    keys = {}
    for method in (*experiment.GRAPH_METHODS, *experiment.BASELINE_METHODS):
        keys[method] = {(r["event"], r["phone_index"]) for r in evaluated if r["method"] == method}
    assert len({frozenset(value) for value in keys.values()}) == 1
    assert not any(r["event"] == records[0]["event"] for r in evaluated)


def test_baseline_scalar_scores_are_not_mislabeled_probabilities(monkeypatch):
    monkeypatch.setattr(experiment, "conventional_forced_alignment_gop", lambda *a: [SimpleNamespace(score=-50), SimpleNamespace(score=10)])
    monkeypatch.setattr(experiment, "ctc_sf_sd_norm", lambda *a: [SimpleNamespace(normalized_score=-10), SimpleNamespace(normalized_score=0)])
    candidate_rows = [{"phone_index": i, "target": p, "gop": -3.0,
                       "candidate_log_probabilities": {"T": -5.0, "<DEL>": -6.0}}
                      for i, p in enumerate(("T", "D"))]
    monkeypatch.setattr(experiment, "score_event", lambda *a, **k: {"rps": candidate_rows, "ups": candidate_rows})
    predictions, _ = experiment.baseline_predictions(experiment.acoustic_record(record("F03")), np.zeros((2, 40)))
    assert predictions["conventional_gop_raw"][0]["gop"] == -50
    for method in experiment.BASELINE_METHODS:
        assert all("p_error" not in row for row in predictions[method])
    for method in ("conventional_gop_raw", "ctc_sf_sd_norm_raw"):
        assert all(row["candidate_event"] == "unspecified" for row in predictions[method])


def test_scalar_gop_diagnosis_is_unavailable_not_artificial_zero_f1():
    config, records, predictions = evaluated_fixture()
    for row in predictions:
        if row["method"] in ("conventional_gop_raw", "ctc_sf_sd_norm_raw"):
            row.update(candidate_event="unspecified", candidate_phone=None, candidate_confidence=0, candidate_margin=0)
    _, metrics = experiment.evaluate_results(predictions, records, "primary5", config)
    for method in ("conventional_gop_raw", "ctc_sf_sd_norm_raw"):
        row = next(row for row in metrics["results"] if row["method"] == method)
        assert row["diagnosis"] is None or row["diagnosis"].get("supported") is False


def test_uncertain_greedy_insertion_is_not_a_none_equals_none_true_positive():
    source = record("F03")
    source["labels"]["gaps"] = [{"diagnostic": True, "stable": True, "inserted_phones": []}]
    prediction = {"event": source["event"], "speaker": "F03", "method": "M0_greedy_single",
                  "gap_index": 0, "ordinal": 0, "insertion_probability": .5,
                  "candidate_phone": None, "candidate_confidence": 0.0}
    results = experiment.evaluate_insertions([prediction], [source], {"F03"})
    assert results[0]["precision"] == 0.0
    assert results[0]["recall"] == 0.0
    assert results[0]["f1"] == 0.0
