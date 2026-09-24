import importlib.util
from pathlib import Path

import numpy as np

from da_cf_gop.dual_prior import estimate_error_prior, serializable_prior
from da_cf_gop.phonology import PHONE_TO_CTC_ID


def exporter():
    path = Path(__file__).resolve().parents[1] / "scripts/export_dual_prior_tables.py"
    spec = importlib.util.spec_from_file_location("export_dual_prior_tables", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def training_row():
    return {
        "speaker": "M01", "quality_exclusion": None, "canonical_phones": ["T", "T", "K"],
        "labels": {
            "tokens": [
                {"phone_index": 0, "event_type": "match", "realized_phone": "T", "stable_event": True, "diagnostic": True},
                {"phone_index": 1, "event_type": "substitution", "realized_phone": "D", "stable_event": True, "diagnostic": True},
                {"phone_index": 2, "event_type": "deletion", "realized_phone": None, "stable_event": True, "diagnostic": True},
            ],
            "gaps": [{"stable": True, "diagnostic": True, "inserted_phones": ["AH", "D", "T"]}],
        },
    }


def fit_from(row):
    prior = estimate_error_prior([row], training_speakers=["M01"], alpha=10)
    return {"fold": {"fold": "test_F03", "training_patients": ["M01"], "test_patient": "F03", "development_patient": "F04"},
            "prior": serializable_prior(prior)}


def test_directed_table_preserves_saved_probabilities_and_true_edge_counts():
    module = exporter()
    row = training_row()
    fit = fit_from(row)
    counts = module.training_counts([row], ["M01"])
    edges, insertion, rate = module.fit_tables(fit, counts)
    assert len(edges) == 39 * 40
    assert len(insertion) == 39
    assert all(e["operation_probability"] > 0 and e["conditional_alternative_probability"] > 0 for e in edges)
    assert all(e["canonical_phone"] != e["alternative_phone"] for e in edges if e["operation"] == "SUB")
    for phone in PHONE_TO_CTC_ID:
        selected = [e for e in edges if e["canonical_phone"] == phone]
        assert len([e for e in selected if e["operation"] == "SUB"]) == 38
        np.testing.assert_allclose(sum(e["joint_prior_probability"] for e in selected), 1)
    lookup = {(e["canonical_phone"], e["operation"], e["alternative_phone"]): e for e in edges}
    td = lookup[("T", "SUB", "D")]
    p, q = PHONE_TO_CTC_ID["T"], PHONE_TO_CTC_ID["D"]
    assert td["joint_prior_probability"] == fit["prior"]["op_probs"][p][1] * fit["prior"]["sub_probs"][p][q]
    assert td["raw_train_count"] == 1
    assert lookup[("T", "SUB", "K")]["raw_train_count"] == 0
    assert lookup[("K", "DEL", "<DEL>")]["raw_train_count"] == 1
    assert {r["alternative_phone"]: r["raw_train_count"] for r in insertion}["T"] == 0
    assert rate["raw_capped_insertion_successes"] == 2 and rate["raw_gaps_over_cap"] == 1


def test_export_counts_ignore_nontraining_quality_and_ambiguous_events():
    module = exporter()
    training = training_row()
    ignored_patient = {**training, "speaker": "F03", "labels": None}
    ignored_quality = {**training, "quality_exclusion": "excluded", "labels": None}
    ignored_event = {
        **training, "labels": {"tokens": [{**training["labels"]["tokens"][0], "stable_event": False}],
                               "gaps": [{"stable": False, "diagnostic": True, "inserted_phones": ["T"]}]},
    }
    base = module.training_counts([training], ["M01"])
    changed = module.training_counts([training, ignored_patient, ignored_quality, ignored_event], ["M01"])
    assert base == changed
    fit = fit_from(training)
    assert module.fit_tables(fit, base) == module.fit_tables(fit, changed)
