"""Input-quality exclusions reach fitting, calibration, priors and outer output."""
import copy
import json

import numpy as np

from da_cf_gop import final_experiment as experiment
from da_cf_gop.dual_prior import generic_prior
from test_final_experiment import row


def test_retained_evidence_is_unchanged():
    rows = [row(), {**row(), "event": "bad_prompt"}]
    before = copy.deepcopy(rows)
    retained = experiment.retained_rows(iter(rows), {"excluded_events": ["bad_prompt"]})
    assert retained == before[:1]
    assert retained[0] is rows[0]
    assert rows == before
    assert experiment.retained_rows(iter(rows), {}) == rows


def test_exclusion_precedes_every_fold_role_and_prior(monkeypatch, tmp_path):
    speakers = ["A", "B", "C"]
    evidence = [{**row(), "speaker": speaker, "event": speaker + "_ok", "phone_index": i}
                for speaker in speakers for i in range(2)]
    evidence.append({**row(), "speaker": "A", "event": "bad_prompt"})
    records = [{"event": event, "speaker": speaker, "quality_exclusion": None}
               for event, speaker in [(s + "_ok", s) for s in speakers] + [("bad_prompt", "A")]]
    folds = [{"fold": "test_" + s, "test_patient": s,
              "development_patient": speakers[(i + 1) % 3],
              "training_patients": [speakers[(i + 2) % 3]]} for i, s in enumerate(speakers)]
    config = json.loads(experiment.CONFIG.read_text())
    config.update(cohorts={"primary5": speakers}, excluded_events=["bad_prompt"],
                  classifier_candidates=[{"family": "logistic", "C": .1}])
    config_path = tmp_path / "protocol.json"
    config_path.write_text(json.dumps(config))
    calls = {"fit": [], "prior": [], "test": []}

    def read(path):
        return iter(records if path.name == "evaluation_labels.jsonl.gz" else evidence)

    def attach_gold(raw, source):
        assert all(r["event"] != "bad_prompt" for r in [*raw, *source])
        assert {r["speaker"] for r in raw} == {r["speaker"] for r in source}
        return [{**r, "gold_error": bool(r["phone_index"]), "stable_event": True,
                 "gold_event": "deletion" if r["phone_index"] else "match", "gold_realized": None}
                for r in raw]

    def estimate(selected, *, training_speakers, **kwargs):
        assert all(r["event"] != "bad_prompt" for r in selected)
        assert {r["speaker"] for r in selected} == set(training_speakers)
        calls["prior"].append(tuple(training_speakers))
        prior = generic_prior()
        prior["training_speakers"] = training_speakers
        return prior

    def fit(train, dev, xt, xd, candidate):
        fold = folds[len(calls["fit"])]
        assert {r["speaker"] for r in train} == set(fold["training_patients"])
        assert {r["speaker"] for r in dev} == {fold["development_patient"]}
        assert all(r["event"] != "bad_prompt" for r in [*train, *dev])
        calls["fit"].append(fold)
        return {}, np.arange(len(dev), dtype=float)

    def score(model, test, x):
        fold = calls["fit"][-1]
        assert {r["speaker"] for r in test} == {fold["test_patient"]}
        assert all(r["event"] != "bad_prompt" and "gold_error" not in r for r in test)
        calls["test"].append(fold["test_patient"])
        return np.arange(len(test), dtype=float)

    monkeypatch.setattr(experiment, "read_jsonl", read)
    monkeypatch.setattr(experiment, "read_jsonl_folds", lambda directory: folds)
    monkeypatch.setattr(experiment, "add_gold", attach_gold)
    monkeypatch.setattr(experiment, "estimate_error_prior", estimate)
    monkeypatch.setattr(experiment, "fit_predictor", fit)
    monkeypatch.setattr(experiment, "model_score", score)
    monkeypatch.setattr(experiment, "event_model", lambda train, x: {})
    monkeypatch.setattr(experiment, "diagnostic_values", lambda fitted, x, cf:
                        (np.zeros(len(x)), np.zeros(len(x), dtype=int), np.ones(len(x))))
    monkeypatch.setattr(experiment, "METHODS", ("da_cf_gop",))
    experiment.predict("primary5", tmp_path / "results", config_path)
    assert calls["test"] == speakers
    assert calls["prior"] == [("C",), (), ("A",), (), ("B",), ()]
    result = tmp_path / "results/primary5"
    for speaker in speakers:
        fitted = json.loads((result / f"{speaker}.fit.json").read_text())
        assert (fitted["n_train"], fitted["n_development"], fitted["n_test_predictions"]) == (2, 2, 2)
