from dataclasses import replace
import itertools

import numpy as np
import pytest
import torch

from da_cf_gop.ctc import counterfactual_log_probability_matrix, log_softmax
from da_cf_gop.parikh2025 import candidate_mask, graph_audit, ppaf_scores, restricted_phone_graph
from da_cf_gop.phonology import ARPABET_39, CTC_ID_TO_PHONE, PHONE_TO_CTC_ID


def _torch_log_probability(log_probs, labels, blank=0):
    """Independent direct Torch call: one unreduced, full-sequence loss."""
    values = torch.tensor(log_probs, dtype=torch.float64).unsqueeze(1)
    targets = torch.tensor(labels, dtype=torch.long)
    return -torch.nn.CTCLoss(blank=blank, reduction="sum", zero_infinity=False)(
        values, targets, torch.tensor([len(log_probs)]), torch.tensor([len(labels)])
    ).item()


@pytest.mark.parametrize("labels", [[1], [1, 2], [1, 1], [1, 2, 1], [1, 2, 2]])
@pytest.mark.parametrize("restricted", [False, True])
def test_ppaf_matches_torch_sequence_losses(labels, restricted):
    # T, D, N form a small real subset of the author's directed graph.
    phone_map = {1: "T", 2: "D", 3: "N"}
    values = log_softmax(np.random.default_rng(79).normal(size=(7, 4)))
    matrix = counterfactual_log_probability_matrix(values, labels, device="cpu")
    actual = ppaf_scores(matrix, labels, phone_map, restricted=restricted)
    mask = candidate_mask(matrix.candidate_ids, labels, phone_map, restricted=restricted)
    original = _torch_log_probability(values, labels)
    expected = []
    for position, candidates in enumerate(matrix.candidate_ids):
        alternatives = []
        for column, phone in enumerate(candidates):
            if not mask[position, column]:
                continue
            changed = labels[:position] + ([] if phone == -1 else [int(phone)]) + labels[position + 1:]
            alternatives.append(_torch_log_probability(values, changed))
        expected.append(original - max(alternatives))
    np.testing.assert_allclose(actual, expected, atol=1e-10)


def test_graph_projection_is_directed_and_reports_collapsed_edges():
    graph = restricted_phone_graph()
    assert set(graph) == set(ARPABET_39)
    assert graph["T"] == ("D", "N")
    assert graph["D"] == ("N", "T")
    assert graph["W"] == ("V",)
    assert "W" not in graph["V"]  # No automatic symmetry.
    assert graph["IY"] == ("IH",)  # i->iː is not a new ARPABET phone.
    assert graph["AH"] == ("ER",)
    assert graph["L"] == ("OW", "R", "UH")  # union l and dark-l rows.
    assert graph["HH"] == graph["EY"] == ()
    assert all(phone not in edges and len(edges) == len(set(edges)) for phone, edges in graph.items())
    audit = graph_audit()
    assert audit["missing_source_rows"] == ["EY", "HH"]
    assert set(audit["unmapped_symbols"]) == {"ʔ", "ən"}
    assert audit["mapping"]["e"] == "EH"
    assert audit["mapping"]["o"] == "OW"


def test_mask_full_inventory_and_candidate_order_invariance():
    labels = [PHONE_TO_CTC_ID[p] for p in ("T", "HH", "EY")]
    values = log_softmax(np.random.default_rng(4).normal(size=(8, 40)))
    matrix = counterfactual_log_probability_matrix(values, labels, device="cpu")
    ups_mask = candidate_mask(matrix.candidate_ids, labels, CTC_ID_TO_PHONE, restricted=False)
    rps_mask = candidate_mask(matrix.candidate_ids, labels, CTC_ID_TO_PHONE)
    assert ups_mask.all()
    assert rps_mask.sum(axis=1).tolist() == [3, 1, 1]
    assert rps_mask[:, -1].all()
    permutation = np.random.default_rng(3).permutation(39)
    permuted = replace(matrix, **{
        field: getattr(matrix, field)[:, permutation]
        for field in ("candidate_ids", "log_probabilities", "log_path_counts", "corrected_log_probabilities")
    })
    for restricted in (False, True):
        np.testing.assert_array_equal(
            ppaf_scores(matrix, labels, CTC_ID_TO_PHONE, restricted=restricted),
            ppaf_scores(permuted, labels, CTC_ID_TO_PHONE, restricted=restricted),
        )


def test_single_phone_deletion_and_upstream_guard_are_explicit():
    labels = [1]
    values = log_softmax(np.array([[3., 0., 0.], [3., 0., 0.]]))
    matrix = counterfactual_log_probability_matrix(values, labels, device="cpu")
    phone_map = {1: "HH", 2: "D"}  # HH has no source graph row.
    scores = ppaf_scores(matrix, labels, phone_map, restricted=True)
    expected = _torch_log_probability(values, labels) - _torch_log_probability(values, [])
    assert scores[0] == pytest.approx(expected, abs=1e-10)
    guarded = ppaf_scores(matrix, labels, phone_map, restricted=True, include_single_phone_deletion=False)
    assert np.isnan(guarded[0])


def test_scorer_ignores_topology_and_deletion_corrected_arrays():
    values = log_softmax(np.random.default_rng(21).normal(size=(6, 4)))
    labels = [1, 2, 1]
    phone_map = {1: "T", 2: "D", 3: "N"}
    matrix = counterfactual_log_probability_matrix(values, labels, device="cpu")
    altered = replace(matrix, canonical_corrected_log_probability=1e6,
                      corrected_log_probabilities=np.full_like(matrix.log_probabilities, -1e6),
                      log_path_counts=np.zeros_like(matrix.log_probabilities))
    for restricted in (False, True):
        np.testing.assert_array_equal(
            ppaf_scores(matrix, labels, phone_map, restricted=restricted),
            ppaf_scores(altered, labels, phone_map, restricted=restricted),
        )


def test_ctc_counterfactuals_match_exhaustive_raw_paths():
    # Check CTC collapsing and deletion independently, especially a repeated
    # pair produced by deleting the middle phone of T,D,T.
    values = log_softmax(np.random.default_rng(93).normal(size=(4, 3)))
    labels = [1, 2, 1]
    matrix = counterfactual_log_probability_matrix(values, labels, device="cpu")
    totals = {}
    for path in itertools.product(range(3), repeat=4):
        collapsed = tuple(phone for i, phone in enumerate(path) if phone and (i == 0 or phone != path[i - 1]))
        totals[collapsed] = totals.get(collapsed, 0.0) + np.exp(sum(values[i, phone] for i, phone in enumerate(path)))
    for position, candidates in enumerate(matrix.candidate_ids):
        for column, phone in enumerate(candidates):
            changed = tuple(labels[:position] + ([] if phone == -1 else [int(phone)]) + labels[position + 1:])
            expected = np.log(totals[changed]) if totals.get(changed, 0.0) else -np.inf
            assert matrix.log_probabilities[position, column] == pytest.approx(expected, abs=1e-10)
