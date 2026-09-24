import numpy as np
from scipy.optimize import check_grad

from da_cf_gop.timebox_ranking import design, objective, sampled_pairs


def test_ranking_objective_gradient_matches_finite_difference():
    rng = np.random.default_rng(17)
    x = rng.normal(size=(7, 4))
    y = np.array([0, 1, 0, 1, 1, 0, 1.])
    differences = x[[1, 3, 4]] - x[[0, 2, 5]]
    parameters = rng.normal(size=5) * .2
    args = (x, y, np.ones(7) / 7, differences, np.ones(3) / 3, .01)
    error = check_grad(lambda p: objective(p, *args)[0], lambda p: objective(p, *args)[1], parameters)
    assert error < 1e-6


def test_pairs_never_cross_patients_and_have_equal_patient_weight():
    rows = [{"speaker": speaker, "gold_error": i % 3 == 0}
            for speaker, count in (("A", 12), ("B", 30)) for i in range(count)]
    positive, negative, weights = sampled_pairs(rows, maximum=20)
    for a, b in zip(positive, negative):
        assert rows[a]["speaker"] == rows[b]["speaker"]
        assert rows[a]["gold_error"] and not rows[b]["gold_error"]
    for speaker in ("A", "B"):
        assert np.isclose(sum(w for a, w in zip(positive, weights) if rows[a]["speaker"] == speaker), .5)


def test_auxiliary_columns_do_not_expand_phone_interactions():
    rows = [{"target": "AA"}, {"target": "T"}]
    numeric = np.arange(18, dtype=float).reshape(2, 9)
    graph = design(rows, numeric[:, :4])
    full = design(rows, numeric)
    assert graph.shape == (2, 199)
    assert full.shape == (2, 204)
    np.testing.assert_array_equal(full[:, :199], graph)
    np.testing.assert_array_equal(full[:, 199:], numeric[:, 4:])
