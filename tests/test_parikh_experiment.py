"""Small end-to-end checks of the PP-AF runner's inference/evaluation split."""

from copy import deepcopy
import json

import numpy as np
import pytest
import torch

from da_cf_gop.calibration import CTCSequenceRecalibrator
from da_cf_gop import parikh_experiment as runner
from da_cf_gop.phonology import PHONE_TO_CTC_ID


def test_acoustic_manifest_is_independent_of_gold_fields(tmp_path):
    original = [{
        "speaker_id": "F03", "reading_event_id": "F03/session/word",
        "canonical_phones": ["K", "AE", "T"],
        "phn_path": "actual.phn", "phn_phones": ["K", "AE", "D"],
        "stable": True, "severity": "severe", "gold_event": "substitution",
    }]
    altered = deepcopy(original)
    altered[0].update({"phn_path": "missing.phn", "phn_phones": ["M"],
                       "stable": False, "severity": "healthy", "gold_event": "correct"})
    first, second = tmp_path / "first.jsonl.gz", tmp_path / "second.jsonl.gz"
    runner.write_jsonl(first, original)
    runner.write_jsonl(second, altered)
    expected = {"F03/session/word": {"speaker": "F03", "phones": ("K", "AE", "T")}}
    assert runner.acoustic_manifest(first, {"F03"}) == expected
    assert runner.acoustic_manifest(second, {"F03"}) == expected
    # Absence of every annotation field also leaves inference identical.
    runner.write_jsonl(second, [{key: original[0][key] for key in (
        "speaker_id", "reading_event_id", "canonical_phones"
    )}])
    assert runner.acoustic_manifest(second, {"F03"}) == expected
    assert runner.acoustic_manifest(first, {"M01"}) == {}


def test_duplicate_reading_events_are_rejected(tmp_path):
    row = {"speaker_id": "F03", "reading_event_id": "same-event", "canonical_phones": ["T"]}
    path = tmp_path / "duplicates.jsonl.gz"
    runner.write_jsonl(path, [row, row])
    with pytest.raises(ValueError, match="duplicate reading event"):
        runner.acoustic_manifest(path, {"F03"})


@pytest.mark.parametrize("contaminated", [None, "fit_summary", "model_summary"])
def test_recalibrator_checks_both_saved_training_speaker_lists(tmp_path, contaminated):
    fold = {"fold_id": "primary5__outer_F03", "held_out_patient": "F03",
            "training_patients": ["F04"], "healthy_references": ["MC01"]}
    training = ["F04", "MC01"]
    prefix = tmp_path / "cache" / "folds" / "primary5" / "saved-fit"
    fit_summary = {"fold_id": fold["fold_id"], "held_out_patient": "F03",
                   "training_speakers": training + (["F03"] if contaminated == "fit_summary" else [])}
    runner.write_json(prefix.with_suffix(".fit.json"), fit_summary)
    model_summary = {"training_speakers": training + (["F03"] if contaminated == "model_summary" else [])}
    model = CTCSequenceRecalibrator(np.eye(40), np.zeros(40), 0, model_summary)
    model.save(prefix.with_suffix(".recalibrator.npz"))
    if contaminated:
        with pytest.raises(ValueError, match="training speakers|metadata"):
            runner.load_recalibrator(tmp_path, "primary5", fold, {"fit_hash": "saved-fit"})
    else:
        loaded, info = runner.load_recalibrator(tmp_path, "primary5", fold, {"fit_hash": "saved-fit"})
        np.testing.assert_array_equal(loaded.weight, np.eye(40))
        assert info["training_speakers"] == training
        assert info["held_out"] not in info["training_speakers"]


def test_score_event_uses_only_logits_prompt_and_exact_paper_difference():
    logits = np.random.default_rng(111).normal(size=(6, 40))
    canonical = [PHONE_TO_CTC_ID[phone] for phone in ("T", "D", "T")]
    output = runner.score_event(logits, canonical, device="cpu")
    log_probs = torch.log_softmax(torch.tensor(logits, dtype=torch.float64), dim=-1).unsqueeze(1)

    def exact(labels):
        return -torch.nn.functional.ctc_loss(
            log_probs, torch.tensor(labels, dtype=torch.long), torch.tensor([len(logits)]),
            torch.tensor([len(labels)]), blank=0, reduction="sum", zero_infinity=False,
        ).item()

    target_lp = exact(canonical)
    assert set(output) == {"rps", "ups"}
    for mode, rows in output.items():
        assert len(rows) == len(canonical)
        for position, row in enumerate(rows):
            assert not ({"gold_event", "gold_realized", "phn", "severity", "speaker", "stable"} & row.keys())
            candidate_lps = []
            for phone, stored in row["candidate_log_probabilities"].items():
                replacement = [] if phone == "<DEL>" else [PHONE_TO_CTC_ID[phone]]
                altered = canonical[:position] + replacement + canonical[position + 1:]
                lp = exact(altered)
                assert stored == pytest.approx(lp, abs=1e-10)
                candidate_lps.append(lp)
            assert row["gop"] == pytest.approx(target_lp - max(candidate_lps), abs=1e-10)
            assert row["error_score"] == -row["gop"]
            assert np.isfinite(row["gop"])
            assert row["candidate_count"] == (39 if mode == "ups" else 3)
    # RPS is a subset of UPS, so its best error likelihood cannot be greater.
    assert all(a["gop"] >= b["gop"] - 1e-10 for a, b in zip(output["rps"], output["ups"]))


def _evaluation_fixture(tmp_path):
    baseline, predictions = [], []
    for speaker in ("F03", "F04"):
        for position, (target, gold, gop) in enumerate((("T", "correct", 1.0), ("D", "substitution", -1.0))):
            common = {"speaker": speaker, "event": f"{speaker}/session/word",
                      "phone_index": position, "target": target, "gop": gop}
            for method in runner.COMPARATORS:
                baseline.append({**common, "method": method, "gold_event": gold,
                                 "gold_realized": target if gold == "correct" else "T", "stable": True})
            for method in runner.METHODS:
                predictions.append({**common, "method": method, "error_score": -gop})
    path = tmp_path / "baseline.jsonl.gz"
    runner.write_jsonl(path, baseline)
    return predictions, baseline, path


def test_evaluation_joins_gold_on_identical_frozen_keys(tmp_path):
    predictions, _baseline, path = _evaluation_fixture(tmp_path)
    before = deepcopy(predictions)
    # Predictions are generated for all canonical tokens. A token absent from
    # the stable evaluation set must not expand any method's scored coverage.
    for method in runner.METHODS:
        predictions.append({"method": method, "speaker": "F03", "event": "F03/uncertain",
                            "phone_index": 0, "target": "T", "gop": 999., "error_score": -999.})
    evaluated, report = runner.evaluate(predictions, path)
    assert report["n_stable_tokens_per_method"] == 4
    assert len(evaluated) == 4 * len(runner.METHODS)
    expected = {runner.token_key(row) for row in before}
    for method in runner.METHODS:
        assert {runner.token_key(row) for row in evaluated if row["method"] == method} == expected
    assert all("gold_event" not in row for row in predictions)
    for result in report["results"]:
        assert result["macro"] == {"auprc": 1.0, "auroc": 1.0}
        assert result["pooled"]["n"] == 4
    # The entire metric product is finite machine-readable JSON.
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("problem", ["prediction_duplicate", "prediction_missing", "baseline_duplicate", "gold_disagreement"])
def test_evaluation_rejects_duplicate_missing_or_inconsistent_gold(tmp_path, problem):
    predictions, baseline, path = _evaluation_fixture(tmp_path)
    if problem == "prediction_duplicate":
        predictions.append(deepcopy(predictions[0]))
    elif problem == "prediction_missing":
        predictions.pop(0)
    elif problem == "baseline_duplicate":
        baseline.append(deepcopy(baseline[0]))
    else:
        baseline[0]["gold_event"] = "deletion"
    runner.write_jsonl(path, baseline)
    with pytest.raises(ValueError, match="duplicate/missing|disagree on frozen gold"):
        runner.evaluate(predictions, path)
