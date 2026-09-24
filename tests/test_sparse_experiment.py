"""Synthetic driver checks; no corpus fitting, decoding, or test metrics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from da_cf_gop import final_experiment as original
from da_cf_gop import sparse_experiment as sparse
from da_cf_gop.dual_prior import generic_prior
from da_cf_gop.phonology import PHONE_TO_CTC_ID


class UnreadableLabels(dict):
    def __getitem__(self, key):
        raise AssertionError("Held-out labels must not be accessed")


@pytest.fixture
def driver_data(tmp_path, monkeypatch):
    root, old, output = tmp_path / "code", tmp_path / "old", tmp_path / "output"
    cohorts = {"primary5": list("ABCDE"), "sensitivity7": list("ABCDEFG")}
    candidates = [{"family": "logistic", "C": .1}, {"family": "logistic", "C": 1.}]
    config = {"cohorts": cohorts, "excluded_events": [], "training_neighbor_budget": 4,
              "classifier_candidates": candidates, "strategies": [
                  {"name": "full_graph_da"},
                  {"name": "top1", "policy": "frame_topk", "top_k": 1},
                  {"name": "top1_train", "policy": "frame_topk", "top_k": 1,
                   "training_neighbors": True},
              ]}
    config_path = tmp_path / "config.json"
    sparse.write_json(config_path, config)
    sparse.write_json(root / "configs/final_v3_prompt_clean.json", {"classifier_candidates": candidates})
    folds = {}
    for cohort, speakers in cohorts.items():
        fold = {"fold": f"{cohort}_test_A", "test_patient": "A",
                "development_patient": "B", "training_patients": speakers[2:]}
        folds[cohort] = fold
        sparse.write_json(old / cohort / "folds.json", [fold])
    records, base, acoustic = [], [], {}
    for speaker in cohorts["sensitivity7"]:
        event = f"{speaker}_utterance"
        labels = [{"phone_index": i, "diagnostic": True, "stable_error": True,
                   "stable_event": True, "is_error": bool(i),
                   "event_type": "substitution" if i else "match",
                   "realized_phone": "T" if i else "K"} for i in range(2)]
        records.append({"event": event, "speaker": speaker, "quality_exclusion": None,
                        "canonical_phones": ["K", "K"], "labels": {"tokens": labels}})
        acoustic[event] = {"event": event, "speaker": speaker, "canonical_phones": ["K", "K"],
                           "acceptable_phones": [["K"], ["K"]]}
        for i, masses in enumerate(([.8, .1, .1], [.3, .5, .2])):
            sub = np.zeros(40)
            sub[PHONE_TO_CTC_ID["T"]] = masses[1]
            base.append({"event": event, "speaker": speaker, "phone_index": i, "target": "K",
                         "acceptable": ["K"], "gop": 2. - 3. * i, "viterbi_gop": 1.5 - 2. * i,
                         "event_posteriors": list(masses), "substitution_posteriors": sub.tolist(),
                         "baseline": {name: .5 for name in original.RAW}})
    real_reader = sparse.read_jsonl

    def read_records(path):
        if Path(path).name == "evaluation_labels.jsonl.gz":
            return deepcopy(records)
        return real_reader(path)

    monkeypatch.setattr(sparse, "ROOT", root)
    monkeypatch.setattr(sparse, "OLD", old)
    monkeypatch.setattr(sparse, "inputs", lambda unused: (deepcopy(base), deepcopy(acoustic)))
    monkeypatch.setattr(sparse, "read_jsonl", read_records)
    probabilities = np.full((3, 40), .1 / 38)
    probabilities[:, 0], probabilities[:, PHONE_TO_CTC_ID["T"]] = .2, .7
    monkeypatch.setattr(sparse, "acoustic_log_probs", lambda event: np.log(probabilities))
    return output, config_path, config, folds, records, base, acoustic


def test_prepare_crossfits_speakers_but_shares_identical_graphs(driver_data):
    output, config_path, config, folds, records, _, acoustic = driver_data
    # Neither cohort uses A or B for its training map.
    for row in records:
        if row["speaker"] in {"A", "B"}:
            row["labels"] = UnreadableLabels()
    sparse.prepare(output, config_path)
    jobs = list(sparse.read_jsonl(output / "graph_jobs.jsonl.gz"))
    assert len(jobs) == len(acoustic)  # Acoustic T already covers every learned edge.
    by_event = {job["event"]: job["identifier"] for job in jobs}
    for cohort, fold in folds.items():
        index = json.loads((output / "fold_indices" / f"{fold['fold']}.json").read_text())
        for speaker in config["cohorts"][cohort]:
            contributors = sorted(set(fold["training_patients"]) - {speaker})
            context = index["speaker_context"][speaker]
            mapping = json.loads((output / "training_maps" / f"{context}.json").read_text())
            assert mapping["training_speakers"] == contributors
            assert speaker not in contributors
            assert {e["speaker"] for row in mapping["sources"]
                    for e in row["supporting_training_events"]} == set(contributors)
            event = f"{speaker}_utterance"
            selected = index["strategies"]["top1_train"][event]
            assert selected["context"] == context
            assert selected["graph"] == by_event[event] == index["strategies"]["top1"][event]["graph"]
            assert "training_confusions" in selected["selection_sources"]
            assert "training_confusions" not in index["strategies"]["top1"][event]["selection_sources"]
    sparse.write_json(output / "graph_cache/existing.json", {})
    with pytest.raises(AssertionError, match="Cached graphs"):
        sparse.prepare(output, config_path)
    preparation_path = output / "preparation.json"
    preparation = json.loads(preparation_path.read_text())
    preparation["source_sha256"]["dual_decode.py"] = "wrong-decoder"
    sparse.write_json(preparation_path, preparation)
    with pytest.raises(AssertionError, match="Decoder changed"):
        sparse.infer(output, workers=1)


def test_predict_routes_fold_roles_and_preserves_original_graph_features(driver_data, monkeypatch):
    output, config_path, config, folds, records, base, _ = driver_data
    # The test patient's labels stay unreadable throughout prepare and predict.
    records[0]["labels"] = UnreadableLabels()
    sparse.prepare(output, config_path)
    for job in sparse.read_jsonl(output / "graph_jobs.jsonl.gz"):
        tokens = deepcopy([r for r in base if r["event"] == job["event"]])
        for row in tokens:
            row["gop"] += .25  # Distinguish restricted evidence from the full cached evidence.
            row["allowed_substitution_phone_ids"] = job["substitution_candidates"][row["phone_index"]]
        sparse.write_json(output / "graph_cache" / f"{job['identifier']}.json",
                          {"event": job["event"], "tokens": tokens})
    sparse.write_json(output / "inference_complete.json", {})
    seen_fits, seen_tests, seen_calibrations = [], [], []

    def fit(train, dev, xt, xd, candidate):
        train_speakers = {row["speaker"] for row in train}
        assert train_speakers in [set(f["training_patients"]) for f in folds.values()]
        assert {row["speaker"] for row in dev} == {"B"}
        for rows, features in ((train, xt), (dev, xd)):
            expected, _ = original.continuous_features(rows, generic_prior(), "graph_da")
            np.testing.assert_allclose(features, expected)
        seen_fits.append(train_speakers)
        score = np.array([-1., 1.]) * (1 if candidate["C"] == .1 else -1)
        return {"C": candidate["C"]}, score

    def calibrate(score, labels):
        np.testing.assert_array_equal(score, [-1., 1.])
        assert labels == [False, True]
        seen_calibrations.append(labels)
        return {"slope": 1., "intercept": 0., "threshold": .5}

    def score(fitted, test, xx):
        assert fitted["C"] == .1
        assert {row["speaker"] for row in test} == {"A"}
        assert all("gold_error" not in row for row in test)
        expected, _ = original.continuous_features(test, generic_prior(), "graph_da")
        np.testing.assert_allclose(xx, expected)
        seen_tests.append([sparse.key(row) for row in test])
        return np.array([-1., 1.])

    monkeypatch.setattr(sparse, "fit_predictor", fit)
    monkeypatch.setattr(sparse, "calibration", calibrate)
    monkeypatch.setattr(sparse, "model_score", score)
    sparse.predict(output)
    methods = {strategy["name"] for strategy in config["strategies"]}
    expected_calls = len(config["cohorts"]) * len(methods)
    assert len(seen_fits) == expected_calls * len(config["classifier_candidates"])
    assert len(seen_tests) == len(seen_calibrations) == expected_calls
    for cohort in config["cohorts"]:
        predictions = list(sparse.read_jsonl(output / cohort / "predictions_without_gold.jsonl.gz"))
        assert {row["method"] for row in predictions} == methods
        for method in methods:
            own = [row for row in predictions if row["method"] == method]
            assert [sparse.key(row) for row in own] == [("A_utterance", 0), ("A_utterance", 1)]
            assert all("gold_error" not in row and row["candidate_event"] == "unspecified" for row in own)
            expected_candidates = (set(range(1, 40)) - {PHONE_TO_CTC_ID["K"]}
                                   if method == "full_graph_da" else {PHONE_TO_CTC_ID["T"]})
            assert all(set(row["allowed_substitution_phone_ids"]) == expected_candidates for row in own)
    marker = json.loads((output / "predictions_complete.json").read_text())
    assert marker["test_labels_joined"] is False
    assert marker["prediction_source_sha256"] == hashlib.sha256(Path(sparse.__file__).read_bytes()).hexdigest()
