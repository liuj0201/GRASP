"""Numerical identities and information-flow checks for the new experiment."""
import copy

import numpy as np

from da_cf_gop.dual_prior import generic_prior
from da_cf_gop.final_experiment import corrected_events, continuous_features, calibration, output_rows
from da_cf_gop.phonology import CTC_ID_TO_PHONE


def row(acceptable=(1,)):
    sub = np.arange(40, dtype=float)
    sub[0] = 0
    sub[list(acceptable)] = 0
    sub = .15 * sub / sub.sum()
    return {"event": "F03_Session1_test", "speaker": "F03", "phone_index": 0,
            "target": CTC_ID_TO_PHONE[1], "acceptable": [CTC_ID_TO_PHONE[i] for i in acceptable],
            "event_posteriors": [.7, .15, .15], "substitution_posteriors": sub.tolist(),
            "gop": float(np.log(.7/.3)), "viterbi_gop": .8,
            "baseline": {"conventional_gop_raw": -3., "ctc_sf_sd_norm_raw": -.4,
                         "ppaf_rps_raw": -.8, "ppaf_ups_raw": -1.1}}


def test_local_prior_replacement_identity_and_acceptable_mass():
    for acceptable in ((1,), (1, 2)):
        r = row(acceptable)
        actual = corrected_events(r, generic_prior())
        expected = np.r_[.7, r["substitution_posteriors"][1:], .15]
        np.testing.assert_allclose(actual, expected, atol=1e-14)
        assert all(actual[i] == 0 for i in acceptable)
        assert np.isclose(actual.sum(), 1)


def test_local_importance_ratio_matches_direct_reweighting():
    r, prior = row(), generic_prior()
    prior["op_probs"][1] = [.5, .4, .1]
    prior["sub_probs"][1, 2] *= 4
    actual = corrected_events(r, prior)
    conditional = prior["sub_probs"][1].copy()
    conditional[0] = conditional[1] = 0
    conditional /= conditional.sum()
    expected = np.r_[.7 * .5/.9, np.asarray(r["substitution_posteriors"])[1:] * .4*conditional[1:]/(.07/38), .15*.1/.03]
    expected /= expected.sum()
    np.testing.assert_allclose(actual, expected)


def test_features_ignore_gold_fields_and_forced_alignment_baseline():
    original = row()
    contaminated = copy.deepcopy(original)
    contaminated.update(gold_error=True, gold_event="deletion", gold_realized="D", severity=999,
                        phn_boundaries=[0, 1000], speaker_id="arbitrary")
    del contaminated["baseline"]["conventional_gop_raw"]
    for method in ("sf_da", "ppaf_da", "isolated_da", "graph_da", "graph_cf_da", "da_cf_gop"):
        first = continuous_features([original], generic_prior(), method)
        second = continuous_features([contaminated], generic_prior(), method)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])


def test_calibration_is_monotone_and_output_has_no_gold():
    fitted = calibration([-2., -1., 0., 1., 2., 3.], [0, 0, 0, 0, 1, 1])
    assert fitted["slope"] >= 0
    raw = row()
    raw["gold_error"] = True
    prediction = output_rows([raw], "ppaf_rps_raw", "test", [1.], fitted)[0]
    assert not any(k.startswith("gold") for k in prediction)
    assert prediction["candidate_event"] == "unspecified"
