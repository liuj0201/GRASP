from itertools import product

import numpy as np
import pytest

from da_cf_gop.dual_decode import decode
from test_dual_decode import collapse, make_priors


def restricted_brute(lp, acceptable, priors, kmax, substitutions, insertions, scale=1.0):
    """Independent edit-path and frame-path enumeration, with original weights."""
    n, vocab = len(acceptable), lp.shape[1]
    acoustic, acoustic_max = {}, {}
    for frames in product(range(vocab), repeat=len(lp)):
        surface = collapse(frames)
        score = sum(lp[t, a] for t, a in enumerate(frames))
        acoustic[surface] = acoustic.get(surface, 0.0) + np.exp(score)
        acoustic_max[surface] = max(acoustic_max.get(surface, -np.inf), score)
    paths = []
    ip = np.broadcast_to(priors["insertion_prob"], (n + 1,))

    def walk(i, slot, output, events, inserted, weight):
        if slot < kmax:
            phones = range(1, vocab) if insertions[i] is None else insertions[i]
            for a in phones:
                p = (ip[i] * priors["insert_phone_probs"][a]) ** scale
                if p:
                    walk(i, slot + 1, output + (a,), events,
                         inserted + ((i, slot, a),), weight * p)
        stop = (1 - ip[i]) ** scale if slot < kmax else 1.0
        if i == n:
            accepted = len(output) == n and all(a in acceptable[j] for j, a in enumerate(output))
            if not (accepted and (inserted or any(e != 0 for e, _ in events))):
                paths.append((output, events, inserted, weight * stop))
            return
        choices = acceptable[i]
        op = priors["op_probs"][choices[0]]
        for a in choices:
            walk(i + 1, 0, output + (a,), events + ((0, a),), inserted,
                 weight * stop * op[0] ** scale / len(choices))
        other = [a for a in range(1, vocab) if a not in choices]
        denom = sum(priors["sub_probs"][choices[0], a] for a in other)
        for a in other:
            if substitutions[i] is None or a in substitutions[i]:
                p = (op[1] * priors["sub_probs"][choices[0], a] / denom) ** scale
                walk(i + 1, 0, output + (a,), events + ((1, a),), inserted, weight * stop * p)
        walk(i + 1, 0, output, events + ((2, -1),), inserted, weight * stop * op[2] ** scale)

    walk(0, 0, (), (), (), 1.0)
    token = np.zeros((n, 3))
    sub = np.zeros((n, vocab))
    ins = np.zeros((n + 1, kmax, vocab))
    maxevents = np.full((n, 3), -np.inf)
    z, best = 0.0, -np.inf
    for output, events, inserted, weight in paths:
        mass = weight * acoustic.get(output, 0.0)
        score = np.log(weight) + acoustic_max.get(output, -np.inf) if weight else -np.inf
        z += mass
        best = max(best, score)
        for i, (event, a) in enumerate(events):
            token[i, event] += mass
            maxevents[i, event] = max(maxevents[i, event], score)
            if event == 1:
                sub[i, a] += mass
        for i, slot, a in inserted:
            ins[i, slot, a] += mass
    return np.log(z), token / z, sub / z, ins / z, best, maxevents


@pytest.mark.parametrize("acceptable,kmax,subs,ins,scale", [
    ([[1]], 0, [[2]], [[], []], 1.0),
    ([[1], [1]], 1, [[2], []], [[1], [], [2]], 1.0),
    ([[1], [2]], 2, [[], [3]], [[2], None, []], 0.5),
    ([[1, 2]], 1, [[]], [[1], [2]], 1.0),
    ([[1], [2]], 0, [None, [3]], [None, None, None], 1.0),
    ([], 2, [], [[2]], 1.0),
    ([], 2, [], [[]], 1.0),
])
def test_pruned_sum_and_max_match_exhaustive_enumeration(acceptable, kmax, subs, ins, scale):
    lp = np.log(np.random.default_rng(31).dirichlet(np.ones(4), size=4))
    prior = make_priors(4)
    got = decode(lp, acceptable, prior, max_insertions=kmax, edit_scale=scale,
                 substitution_candidates=subs, insertion_candidates=ins)
    expected = restricted_brute(lp, acceptable, prior, kmax, subs, ins, scale)
    for key, value in zip(("log_partition", "token_posteriors", "substitution_posteriors",
                           "insertion_posteriors", "viterbi_score", "max_event_log_scores"), expected):
        np.testing.assert_allclose(got[key], value, atol=2e-12)
    output = []
    for i in range(len(acceptable) + 1):
        output.extend(a for a in got["viterbi_insertions"][i] if a >= 0)
        if i < len(acceptable) and got["viterbi_token_events"][i] != 2:
            output.append(got["viterbi_token_phones"][i])
    assert collapse(got["viterbi_frame_labels"]) == tuple(output)


def test_full_candidate_sets_equal_unrestricted_inference():
    lp = np.log(np.random.default_rng(4).dirichlet(np.ones(4), size=5))
    prior = make_priors(4)
    full = decode(lp, [[1, 2], [2]], prior)
    explicit = decode(lp, [[1, 2], [2]], prior,
                      substitution_candidates=[[1, 2, 3], [1, 2, 3]],
                      insertion_candidates=[[1, 2, 3]] * 3)
    for key in ("log_partition", "token_posteriors", "substitution_posteriors", "insertion_posteriors",
                "viterbi_score", "max_event_log_scores", "viterbi_frame_labels"):
        np.testing.assert_allclose(full[key], explicit[key], atol=2e-12)


def test_pruning_removes_states_arcs_and_keeps_original_weights():
    lp = np.log(np.random.default_rng(7).dirichlet(np.ones(8), size=8))
    prior = make_priors(8)
    full = decode(lp, [[1], [2], [3]], prior)
    sparse = decode(lp, [[1], [2], [3]], prior,
                    substitution_candidates=[[2], [3], [1]], insertion_candidates=[[]] * 4)
    assert sparse["log_partition"] < full["log_partition"]
    assert sparse["viterbi_score"] <= full["viterbi_score"]
    for key in ("graph_arc_count", "graph_state_count", "ctc_state_count", "graph_array_bytes",
                "sum_forward_bytes", "max_history_trace_bytes"):
        assert sparse[key] < full[key]
    for i, allowed in enumerate(([2], [3], [1])):
        assert not np.any(sparse["substitution_posteriors"][i, [a for a in range(8) if a not in allowed]])
    assert sparse["insertion_posteriors"].sum() == 0
    # A single surviving substitution keeps its old 1/7 phone share.
    exact = np.full((1, 8), -np.inf)
    exact[0, 2] = 0
    one = decode(exact, [[1]], prior, max_insertions=0, substitution_candidates=[[2]])
    assert one["log_partition"] == pytest.approx(np.log(0.25 / 6))


def test_accepted_paths_blank_and_deletion_remain_distinct():
    prior = make_priors(4, insertion=0.8)
    prior["op_probs"][:] = [0.01, 0.01, 0.98]
    exact = np.full((3, 4), -np.inf)
    exact[np.arange(3), [2, 0, 2]] = 0
    got = decode(exact, [[1, 2], [2]], prior,
                 substitution_candidates=[[], []], insertion_candidates=[[2]] * 3)
    np.testing.assert_allclose(got["token_posteriors"], [[1, 0, 0], [1, 0, 0]])
    assert got["insertion_posteriors"].sum() == 0
    blank = np.full((5, 4), -np.inf)
    blank[:, 0] = 0
    deleted = decode(blank, [[1], [2]], make_priors(4, insertion=0),
                     substitution_candidates=[[], []], insertion_candidates=[[]] * 3)
    assert deleted["log_partition"] == pytest.approx(np.log(0.1 ** 2))
    np.testing.assert_allclose(deleted["token_posteriors"], [[0, 0, 1], [0, 0, 1]])


def test_empty_audio_and_impossible_repeated_output():
    got = decode(np.zeros((0, 4)), [[1], [2]], make_priors(4, insertion=0),
                 substitution_candidates=[[], []], insertion_candidates=[[]] * 3,
                 include_viterbi=False)
    assert got["log_partition"] == pytest.approx(np.log(0.01))
    assert got["max_history_trace_bytes"] == 0
    prior = make_priors(4, insertion=0)
    prior["op_probs"][:] = [1, 0, 0]
    exact = np.full((2, 4), -np.inf)
    exact[:, 1] = 0
    with pytest.raises(ValueError, match="No accepting"):
        decode(exact, [[1], [1]], prior, substitution_candidates=[[], []],
               insertion_candidates=[[]] * 3)


@pytest.mark.parametrize("kwargs", [
    {"substitution_candidates": [[0]]},
    {"substitution_candidates": [[4]]},
    {"insertion_candidates": [[1], [1.5]]},
    {"substitution_candidates": []},
])
def test_invalid_candidate_ids_and_counts(kwargs):
    with pytest.raises(ValueError, match="nonblank|rows"):
        decode(np.log([[0.1, 0.6, 0.2, 0.1]]), [[1]], make_priors(4), **kwargs)
