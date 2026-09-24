import numpy as np
from scipy.special import expit, logit

from da_cf_gop import timebox_ensemble


def test_probability_ensemble_weights_families_equally(monkeypatch):
    probabilities = [.1, .2, .3, .4, .5, .6, .8, .9]
    candidates = [{"family": "logistic"}] * 6 + [{"family": "histogram"}] * 2
    monkeypatch.setattr(timebox_ensemble, "score_model", lambda model, rows, x: np.array([logit(model)]))
    score = timebox_ensemble.ensemble_score(probabilities, candidates, [{}], np.zeros((1, 4)))
    expected = .5 * np.mean(probabilities[:6]) + .5 * np.mean(probabilities[6:])
    np.testing.assert_allclose(expit(score), [expected], atol=1e-15)
    assert abs(expit(score[0]) - np.mean(probabilities)) > .01
