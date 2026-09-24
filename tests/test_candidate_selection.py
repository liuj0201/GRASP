import numpy as np
import pytest

from da_cf_gop.candidate_selection import select_candidates
from da_cf_gop.phonology import PHONE_TO_CTC_ID


def _logs(probabilities):
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


def test_frame_selection_uses_raw_nonblank_support_and_keeps_anchor_roles_separate():
    # Blank can dominate a frame; it must not consume the top-k phone budget.
    # The weak phone on the final frame must still pass the absolute threshold.
    probabilities = np.zeros((3, 40))
    probabilities[0, [0, 1, 2]] = [.80, .15, .05]
    probabilities[1, [0, 2, 3]] = [.70, .20, .10]
    probabilities[2, [0, 4]] = [.995, .005]
    selection = select_candidates(
        _logs(probabilities), [[1, 3], [5]], top_k=1,
        posterior_threshold=.01, add_phonological_neighbors=False,
    )
    assert selection.active_phone_ids == (1, 2)
    assert selection.substitution_phone_ids == ((2,), (1, 2))
    assert selection.insertion_phone_ids == ((1, 2),) * 3
    assert selection.substitution_counts == (1, 2)
    assert selection.insertion_counts == (2, 2, 2)
    assert selection.retained_acoustic_mass == pytest.approx(.40 / .505)
    assert selection.selection_seconds >= 0


def test_phonological_union_is_directed_and_never_turns_an_accepted_phone_into_sub():
    probabilities = np.zeros((1, 40))
    probabilities[0, 0] = 1.0
    w, v, f = (PHONE_TO_CTC_ID[p] for p in ("W", "V", "F"))
    selection = select_candidates(
        _logs(probabilities), [[w], [v], [w, v]], add_phonological_neighbors=True,
    )
    assert selection.active_phone_ids == ()
    assert selection.substitution_phone_ids == ((v,), (f,), (f,))
    assert selection.insertion_phone_ids == ((),) * 4
    assert selection.retained_acoustic_mass == 1.0
    assert selection.selection_sources == ("acoustic_frame_topk", "ppaf_generic_phonology")


def test_supplied_confusions_add_only_directed_substitution_support():
    probabilities = np.zeros((1, 40))
    probabilities[0, [0, 1]] = [.5, .5]
    supplied = [[] for _ in range(40)]
    supplied[2] = [3, 4]
    selection = select_candidates(_logs(probabilities), [[2, 3], [4]], learned_neighbors=supplied)
    assert selection.substitution_phone_ids == ((1, 4), (1,))
    assert selection.insertion_phone_ids == ((1,),) * 3
    assert supplied[2] == [3, 4]
    assert selection.selection_sources == ("acoustic_frame_topk", "training_confusions")


def test_mass_cover_is_a_minimal_global_set_and_preserves_low_peak_accumulated_mass():
    probabilities = np.zeros((3, 40))
    probabilities[0, [0, 1, 3]] = [.59, .35, .06]
    probabilities[1, [0, 2, 3]] = [.84, .10, .06]
    probabilities[2, [0, 3]] = [.94, .06]
    selection = select_candidates(
        _logs(probabilities), [[4]], policy="mass_cover", mass_coverage=.80,
        add_phonological_neighbors=False,
    )
    # Phone 3 is weak at every frame but carries more total mass than phone 2.
    assert selection.active_phone_ids == (1, 3)
    assert selection.retained_acoustic_mass == pytest.approx(.53 / .63)
    assert .35 / .63 < .80  # The next smaller set cannot meet the target.
    reversed_frames = select_candidates(
        _logs(probabilities[::-1]), [[4]], policy="mass_cover", mass_coverage=.80,
        add_phonological_neighbors=False,
    )
    assert reversed_frames.active_phone_ids == selection.active_phone_ids


def test_uncertain_audio_expands_mass_cover_instead_of_forcing_a_small_cap():
    probabilities = np.full((2, 40), .5 / 39)
    probabilities[:, 0] = .5
    selection = select_candidates(
        _logs(probabilities), [[20]], policy="mass_cover", mass_coverage=.99,
        add_phonological_neighbors=False,
    )
    assert selection.active_phone_ids == tuple(range(1, 40))
    assert selection.substitution_counts == (38,)
    assert selection.retained_acoustic_mass == pytest.approx(1.0)


@pytest.mark.parametrize("policy", ["frame_topk", "mass_cover"])
def test_equal_mass_ties_use_phone_id_order(policy):
    probabilities = np.zeros((1, 40))
    probabilities[0, [0, 1, 2, 3]] = [.4, .2, .2, .2]
    selection = select_candidates(
        _logs(probabilities), [[4]], policy=policy, top_k=2,
        mass_coverage=.60, add_phonological_neighbors=False,
    )
    assert selection.active_phone_ids == (1, 2)
