"""Numerical identities and held-out information flow for residual fusion."""
import numpy as np

from da_cf_gop import timebox_residual as residual
from da_cf_gop.dual_prior import generic_prior


def test_zero_correction_preserves_backbone_exactly():
    graph = np.asarray([-7.1, -1.0, 0.4, 8.5])
    actual = residual.mix_scores(graph, [99, -40, 10, 1], [-2, 7, 8, -9],
                                 {"cf": 0.0, "isolated": 0.0})
    np.testing.assert_array_equal(actual, graph)


def test_generic_prior_is_zero_cf_correction():
    substitutions = np.zeros(40)
    substitutions[2:] = 0.2 / 38
    row = {"event": "A_0", "speaker": "A", "phone_index": 0, "target": "AA",
           "acceptable": ["AA"], "event_posteriors": [0.7, 0.2, 0.1],
           "substitution_posteriors": substitutions.tolist(), "gop": 0.8, "viterbi_gop": 0.7,
           "baseline": {name: -0.5 for name in residual.fe.RAW}}
    correction, _ = residual.cf_log_odds_delta([row], generic_prior())
    np.testing.assert_allclose(correction, 0, atol=1e-14)


def test_matched_ablations_only_disable_registered_branches():
    assert len(residual.candidate_weights("residual_full")) == 9
    assert residual.candidate_weights("residual_graph_da") == [{"cf": 0.0, "isolated": 0.0}]
    assert all(w["isolated"] == 0 for w in residual.candidate_weights("residual_graph_cf_da"))
    assert all(w["cf"] == 0 for w in residual.candidate_weights("residual_graph_isolated_da"))


def test_training_oof_never_fits_heldout_speaker(monkeypatch):
    speakers = ["A", "B", "C"]
    evidence = [{"event": f"{s}_{i}", "speaker": s, "phone_index": i,
                 "target": "AA", "gold_error": bool(i)} for s in speakers for i in range(2)]
    records = [{"event": f"{s}_{i}", "speaker": s} for s in speakers for i in range(2)]
    fit_calls = []
    active_heldout = {"speaker": None}

    def attach(raw, own_records):
        assert {r["speaker"] for r in raw} == {r["speaker"] for r in own_records}
        return list(raw)

    def prior(own_records, fit_speakers, config):
        assert set(fit_speakers) == {r["speaker"] for r in own_records}
        heldout = (set(speakers) - set(fit_speakers)).pop()
        active_heldout["speaker"] = heldout
        return {"training_speakers": fit_speakers}

    def features(rows, prior, method):
        return np.zeros((len(rows), 4)), np.zeros((len(rows), 41))

    def backbone(train, heldout, xt, xh, candidate):
        heldout_speakers = {r["speaker"] for r in heldout}
        train_speakers = {r["speaker"] for r in train}
        assert heldout_speakers == {active_heldout["speaker"]}
        assert not train_speakers & heldout_speakers
        fit_calls.append((train_speakers, heldout_speakers))
        return {}, np.zeros(len(heldout))

    def isolated(train):
        assert active_heldout["speaker"] not in {r["speaker"] for r in train}
        return {}

    def correction(rows, prior):
        assert not {r["speaker"] for r in rows} & set(prior["training_speakers"])
        return np.zeros(len(rows)), None

    monkeypatch.setattr(residual.fe, "add_gold", attach)
    monkeypatch.setattr(residual.fe, "prior_for", prior)
    monkeypatch.setattr(residual.fe, "continuous_features", features)
    monkeypatch.setattr(residual.fe, "fit_predictor", backbone)
    monkeypatch.setattr(residual, "fit_isolated", isolated)
    monkeypatch.setattr(residual, "score_isolated", lambda model, rows: np.zeros(len(rows)))
    monkeypatch.setattr(residual, "cf_log_odds_delta", correction)
    result = residual.collect_oof(evidence, records, speakers, {})
    joined, graph_scores, _, _, roles = result
    assert len(fit_calls) == 6
    assert len(joined) == 6
    assert graph_scores.shape == (2, 6)
    for role in roles:
        assert role["held_out"] not in role["cf_prior_speakers"]
        assert role["held_out"] not in role["scaler_fit_speakers"]
