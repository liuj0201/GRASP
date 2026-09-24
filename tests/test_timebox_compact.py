import copy

import numpy as np

from da_cf_gop.dual_prior import generic_prior
from da_cf_gop.phonology import PHONE_TO_CTC_ID
from da_cf_gop.timebox_compact import FeatureBuilder, compact_features, compact_prior


def evidence(speaker="A"):
    mass = np.zeros(40)
    mass[PHONE_TO_CTC_ID["D"]] = .15
    mass[PHONE_TO_CTC_ID["G"]] = .05
    return {"speaker": speaker, "target": "T", "acceptable": ["T"],
        "gop": -1., "viterbi_gop": -2., "event_posteriors": [.7, .2, .1],
        "substitution_posteriors": mass.tolist(), "baseline": {
            "ctc_sf_sd_norm_raw": -.1, "ppaf_rps_raw": -.2, "ppaf_ups_raw": -.3}}


def record(speaker, realized="D"):
    return {"speaker": speaker, "event": speaker + "_event", "quality_exclusion": None,
        "canonical_phones": ["T"], "labels": {"tokens": [{"phone_index": 0,
            "stable_event": True, "diagnostic": True, "event_type": "substitution",
            "realized_phone": realized}], "gaps": []}}


def test_generic_cf_is_zero_and_auxiliary_inputs_do_not_change_graph():
    prior = generic_prior()
    prior["identity_reliability"] = np.ones(40)
    row = evidence()
    baseline = compact_features([row], prior)
    assert baseline.shape == (1, 9)
    np.testing.assert_allclose(baseline[:, 4:6], 0., atol=1e-14)
    altered = copy.deepcopy(row)
    altered["baseline"]["ppaf_rps_raw"] = -100.
    result = compact_features([altered], prior)
    np.testing.assert_array_equal(result[:, :6], baseline[:, :6])
    assert result[0, 7] != baseline[0, 7]


def test_cf_identity_support_uses_acoustics_and_preserves_acceptable_set():
    prior = generic_prior()
    prior["identity_reliability"] = np.ones(40)
    p, d = PHONE_TO_CTC_ID["T"], PHONE_TO_CTC_ID["D"]
    prior["sub_probs"][p, d] = .9
    row = evidence()
    before = copy.deepcopy(row)
    x = compact_features([row], prior)
    assert x[0, 5] > 0
    assert row == before
    prior["identity_reliability"][p] = 0
    assert abs(compact_features([row], prior)[0, 5]) < 1e-14


def test_prior_and_training_crossfit_ignore_nontraining_labels():
    records = [record("A"), record("B"), record("C")]
    original = compact_prior(records, ["A", "B"])
    poisoned = copy.deepcopy(records)
    poisoned[-1]["labels"]["tokens"][0]["realized_phone"] = "G"
    after = compact_prior(poisoned, ["A", "B"])
    for field in ("op_probs", "sub_probs", "identity_reliability"):
        np.testing.assert_array_equal(original[field], after[field])
    builder = FeatureBuilder(records)
    _, sources = builder.training([evidence("A"), evidence("B")], ["A", "B"])
    assert sources == {"A": ["B"], "B": ["A"]}
    assert all("C" not in value for value in sources.values())
