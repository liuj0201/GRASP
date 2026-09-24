from itertools import product
import numpy as np
import pytest

from da_cf_gop.dual_decode import decode


def make_priors(vocab=3, insertion=0.15):
    sub = np.ones((vocab, vocab), dtype=float)
    sub[:, 0] = 0
    np.fill_diagonal(sub, 0)
    sub /= sub.sum(axis=1, keepdims=True)
    ins = np.ones(vocab, dtype=float)
    ins[0] = 0
    ins /= ins.sum()
    return {
        "op_probs": np.tile([0.65, 0.25, 0.10], (vocab, 1)),
        "sub_probs": sub,
        "insertion_prob": insertion,
        "insert_phone_probs": ins,
    }


def collapse(frames):
    return tuple(a for i, a in enumerate(frames) if a != 0 and (i == 0 or a != frames[i - 1]))


def brute(log_probs, acceptable, priors, kmax, edit_scale=1.0, exclude_accepted_errors=True):
    """Enumerate complete edit paths independently of frame label paths."""
    n, vocab = len(acceptable), log_probs.shape[1]
    acoustic = {}
    acoustic_best = {}
    for labels in product(range(vocab), repeat=len(log_probs)):
        output = collapse(labels)
        lp = sum(log_probs[t, a] for t, a in enumerate(labels))
        acoustic[output] = acoustic.get(output, 0) + np.exp(lp)
        acoustic_best[output] = max(acoustic_best.get(output, -np.inf), lp)
    paths = []
    ip = np.broadcast_to(priors["insertion_prob"], n + 1)

    def walk(i, k, output, events, insertions, probability):
        if k < kmax:
            for a in range(1, vocab):
                p = (ip[i] * priors["insert_phone_probs"][a]) ** edit_scale
                if p:
                    walk(i, k + 1, output + (a,), events, insertions + ((i, k, a),), probability * p)
        stop = (1 - ip[i]) ** edit_scale if k < kmax else 1
        if i == n:
            paths.append((output, events, insertions, probability * stop))
            return
        canonical = acceptable[i][0]
        choices = set(acceptable[i])
        op = priors["op_probs"][canonical]
        for a in choices:
            walk(i + 1, 0, output + (a,), events + ((0, a),), insertions,
                 probability * stop * op[0] ** edit_scale / len(choices))
        other = [a for a in range(1, vocab) if a not in choices]
        denom = sum(priors["sub_probs"][canonical, a] for a in other)
        for a in other:
            p = op[1] ** edit_scale * (priors["sub_probs"][canonical, a] / denom) ** edit_scale
            walk(i + 1, 0, output + (a,), events + ((1, a),), insertions, probability * stop * p)
        walk(i + 1, 0, output, events + ((2, -1),), insertions, probability * stop * op[2] ** edit_scale)

    walk(0, 0, (), (), (), 1.0)
    token = np.zeros((n, 3))
    sub = np.zeros((n, vocab))
    insertion = np.zeros((n + 1, kmax, vocab))
    z, best = 0.0, -np.inf
    best_events = None
    max_event_scores = np.full((n, 3), -np.inf)
    for output, events, insertions, prior in paths:
        accepted = len(output) == n and all(a in acceptable[i] for i, a in enumerate(output))
        if exclude_accepted_errors and accepted and (insertions or any(e != 0 for e, _ in events)):
            continue
        mass = prior * acoustic.get(output, 0)
        z += mass
        pathbest = np.log(prior) + acoustic_best.get(output, -np.inf) if prior else -np.inf
        if pathbest > best:
            best, best_events = pathbest, events
        for i, (event, a) in enumerate(events):
            token[i, event] += mass
            max_event_scores[i, event] = max(max_event_scores[i, event], pathbest)
            if event == 1:
                sub[i, a] += mass
        for i, k, a in insertions:
            insertion[i, k, a] += mass
    return (np.log(z), token / z, sub / z, insertion / z, best, best_events,
            max_event_scores, sum(p[-1] for p in paths))


@pytest.mark.parametrize("acceptable,kmax,scale", [
    ([[1]], 0, 1.0),
    ([[1], [1]], 0, 1.0),
    ([[1], [2]], 1, 1.0),
    ([[1], [2]], 2, 0.5),
    ([[1], [1]], 1, 0.5),
    ([[1, 2]], 1, 1.0),
    ([], 2, 1.0),
])
def test_full_sum_matches_independent_edit_and_ctc_enumeration(acceptable, kmax, scale):
    vocab = 4 if acceptable == [[1, 2]] else 3
    rng = np.random.default_rng(41)
    probs = rng.dirichlet(np.ones(vocab), size=4)
    priors = make_priors(vocab)
    actual = decode(np.log(probs), acceptable, priors, max_insertions=kmax, edit_scale=scale)
    expected = brute(np.log(probs), acceptable, priors, kmax, scale)
    np.testing.assert_allclose(actual["log_partition"], expected[0], atol=1e-12)
    np.testing.assert_allclose(actual["token_posteriors"], expected[1], atol=1e-12)
    np.testing.assert_allclose(actual["substitution_posteriors"], expected[2], atol=1e-12)
    np.testing.assert_allclose(actual["insertion_posteriors"], expected[3], atol=1e-12)
    np.testing.assert_allclose(actual["viterbi_score"], expected[4], atol=1e-12)
    np.testing.assert_allclose(actual["max_event_log_scores"], expected[6], atol=1e-12)
    if scale == 1:
        assert expected[7] == pytest.approx(1.0)
    output = []
    for i in range(len(acceptable) + 1):
        output.extend(a for a in actual["viterbi_insertions"][i] if a >= 0)
        if i < len(acceptable) and actual["viterbi_token_events"][i] != 2:
            output.append(actual["viterbi_token_phones"][i])
    assert collapse(actual["viterbi_frame_labels"]) == tuple(output)


def test_deletions_on_blank_path_are_counted_once_not_once_per_timing():
    priors = make_priors(insertion=0)
    # Only blank labels are possible: a unique CTC path, all anchors deleted.
    lp = np.full((5, 3), -np.inf)
    lp[:, 0] = 0
    got = decode(lp, [[1], [2]], priors, max_insertions=2)
    assert got["log_partition"] == pytest.approx(np.log(0.1 ** 2))
    np.testing.assert_allclose(got["token_posteriors"], [[0, 0, 1], [0, 0, 1]])


def test_empty_audio_requires_all_remaining_anchors_deleted():
    got = decode(np.zeros((0, 3)), [[1], [2]], make_priors(insertion=0), max_insertions=0)
    assert got["log_partition"] == pytest.approx(np.log(0.01))
    np.testing.assert_array_equal(got["viterbi_token_events"], [2, 2])


def test_acceptable_variant_not_repeated_as_substitution_and_duplicates_have_no_weight():
    lp = np.log([[0.01, 0.01, 0.97, 0.01]])
    priors = make_priors(4, insertion=0)
    one = decode(lp, [[1, 2]], priors, max_insertions=0)
    duplicates = decode(lp, [[1, 2, 2, 1]], priors, max_insertions=0)
    np.testing.assert_array_equal(one["token_posteriors"], duplicates["token_posteriors"])
    assert one["substitution_posteriors"][0, 2] == 0
    assert one["viterbi_token_events"][0] == 0
    assert one["viterbi_token_phones"][0] == 2


@pytest.mark.parametrize("acceptable,output", [([[1]], [1]), ([[1, 2]], [2]), ([[1], [2]], [1, 2])])
def test_fully_acceptable_surface_cannot_be_deleted_and_reinserted_as_error(acceptable, output):
    vocab = 4
    lp = np.full((len(output), vocab), -np.inf)
    lp[np.arange(len(output)), output] = 0.0
    priors = make_priors(vocab, insertion=0.8)
    priors["op_probs"][:] = [0.01, 0.01, 0.98]
    got = decode(lp, acceptable, priors)
    np.testing.assert_allclose(got["token_posteriors"], np.tile([1, 0, 0], (len(acceptable), 1)))
    np.testing.assert_array_equal(got["viterbi_token_events"], np.zeros(len(acceptable)))
    assert got["insertion_posteriors"].sum() == 0
    assert got["accepted_surface_errors_excluded"] is True


def test_can_explain_multiple_substitutions_and_two_insertions():
    priors = make_priors(4, insertion=0.45)
    priors["op_probs"][:] = [0.01, 0.98, 0.01]
    lp = np.full((5, 4), np.log(1e-9))
    lp[np.arange(5), [2, 3, 2, 3, 2]] = np.log(1 - 3e-9)
    got = decode(lp, [[1], [1]], priors, max_insertions=2)
    assert np.all(got["token_posteriors"][:, 1] > 0.9)
    assert got["insertion_posteriors"].sum() > 2.8
    assert got["insertion_posteriors"].shape == (3, 2, 4)
    assert np.all(got["insertion_posteriors"].sum(axis=2) <= 1 + 1e-12)


def test_acoustic_evidence_changes_local_posteriors():
    priors = make_priors(insertion=0)
    correct = decode(np.log([[0.01, 0.98, 0.01]]), [[1]], priors, max_insertions=0)
    wrong = decode(np.log([[0.01, 0.01, 0.98]]), [[1]], priors, max_insertions=0)
    assert correct["token_log_odds"][0] > 0
    assert wrong["token_log_odds"][0] < 0


def test_tiny_blank_mass_is_not_lost_by_total_minus_largest_cancellation():
    priors = make_priors(insertion=0)
    priors["op_probs"][:] = [1.0, 0.0, 0.0]
    lp = np.log([[1e-30, 1.0, 1e-40], [1e-30, 1.0, 1e-40], [1e-30, 1.0, 1e-40]])
    got = decode(lp, [[1], [1]], priors, max_insertions=0)
    assert got["log_partition"] == pytest.approx(np.log(1e-30))
    np.testing.assert_allclose(got["token_posteriors"], [[1, 0, 0], [1, 0, 0]])


def test_framewise_log_offsets_only_shift_partition_and_path_score():
    priors = make_priors()
    lp = np.log([[0.6, 0.3, 0.1], [0.5, 0.2, 0.3], [0.7, 0.2, 0.1]])
    a = decode(lp, [[1]], priors)
    b = decode(lp - 2000, [[1]], priors)
    np.testing.assert_allclose(a["token_posteriors"], b["token_posteriors"], atol=1e-12)
    assert b["log_partition"] - a["log_partition"] == pytest.approx(-6000)
    assert b["viterbi_score"] - a["viterbi_score"] == pytest.approx(-6000)


def test_long_peaky_recording_remains_finite_in_log_domain():
    rng = np.random.default_rng(1)
    lp = rng.normal(0, 8, size=(1000, 40))
    lp -= np.log(np.exp(lp).sum(axis=1, keepdims=True))
    priors = make_priors(40, insertion=0.05)
    priors["op_probs"][:] = [0.95, 0.049, 0.001]
    got = decode(lp, [[1 + i % 39] for i in range(50)], priors)
    np.testing.assert_allclose(got["token_posteriors"].sum(axis=1), 1, atol=1e-8)
    assert np.all(np.isfinite(got["token_log_odds"]))
    assert np.all(np.isfinite(got["max_marginal_token_log_odds"]))
    assert got["viterbi_score"] <= got["log_partition"]


def test_uniform_variant_scale_changes_only_global_weight():
    lp = np.log([[0.1, 0.2, 0.6, 0.1], [0.3, 0.2, 0.4, 0.1]])
    priors = make_priors(4)
    a = decode(lp, [[1, 2]], priors, variant_scale=1)
    b = decode(lp, [[1, 2]], priors, variant_scale=0.5)
    np.testing.assert_allclose(a["token_posteriors"], b["token_posteriors"], atol=1e-12)
    assert b["log_partition"] - a["log_partition"] == pytest.approx(0.5 * np.log(2))


def test_reject_invalid_anchor_and_prior_inputs():
    lp = np.log([[0.5, 0.2, 0.3]])
    with pytest.raises(ValueError, match="nonblank"):
        decode(lp, [[0]], make_priors())
    with pytest.raises(ValueError, match="canonical"):
        decode(lp, [[2]], make_priors(), canonical_ids=[1])
    prior = make_priors()
    prior["op_probs"][1] *= 2
    with pytest.raises(ValueError, match="normalized"):
        decode(lp, [[1]], prior)
