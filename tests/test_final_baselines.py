import numpy as np
from scipy.special import logsumexp

from da_cf_gop.final_baselines import automatic_segments, fit_development_calibration, plain_score


def test_automatic_spans_follow_canonical_ctc_states_including_repeats():
    # AA, blank, AA are two anchors, not one merged target segment.
    logits = np.full((5, 40), -12.)
    logits[np.arange(5), [0, 1, 0, 1, 0]] = 0.
    spans = automatic_segments(logits - logsumexp(logits, axis=1, keepdims=True), ["AA", "AA"])
    np.testing.assert_array_equal(spans, [[1, 2], [3, 4]])


def test_prediction_projection_does_not_copy_gold_or_manual_boundaries():
    inputs = {"event": "x", "speaker": "F03", "canonical_phones": ["AA"],
              "phn_path": "never-read.phn", "gold_error": True, "start": 123.}
    row = plain_score(inputs, 0, "baseline", -3.5)
    assert row["target"] == "AA" and row["gop"] == -3.5
    assert not {"phn_path", "gold_error", "start"}.intersection(row)
    assert row["candidate_event"] == "unspecified"


def test_development_calibration_is_stable_for_native_gmm_density_units():
    rows = [{"speaker": "dev", "gop": float(x), "gold_error": bool(i < 6),
             "stable_event": False, "candidate_confidence": 0., "candidate_margin": 0.,
             "candidate_event": "unspecified"} for i, x in enumerate(np.linspace(-2, 2, 12))]
    config = {"decision_threshold_grid": [.1, .5, .9], "specific_confidence_grid": [.5],
              "specific_margin_floor": .1, "specific_precision_floor": .8, "specific_coverage_floor": .1}
    ordinary = fit_development_calibration(rows, config)
    large = fit_development_calibration([{**r, "gop": r["gop"] * 1e8} for r in rows], config)
    np.testing.assert_allclose(ordinary["slope"], large["slope"] * 1e8, rtol=1e-10)
    np.testing.assert_allclose(ordinary["intercept"], large["intercept"], atol=1e-10)
    assert large["training_speakers"] == ["dev"]
