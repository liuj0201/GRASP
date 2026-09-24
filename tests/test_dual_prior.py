from copy import deepcopy

import numpy as np
import pytest

from da_cf_gop.dual_prior import estimate_error_prior, generic_prior
from da_cf_gop.phonology import PHONE_TO_CTC_ID


def row(speaker, event="substitution", actual="D"):
    return {"speaker": speaker, "canonical_phones": ["T"], "labels": {
        "tokens": [{"phone_index": 0, "event_type": event, "realized_phone": actual,
                    "stable_event": True, "diagnostic": True}],
        "gaps": [{"stable": True, "diagnostic": True, "inserted_phones": []}]}}


def test_generic_prior_full_support_and_normalization():
    p = generic_prior()
    np.testing.assert_allclose(p["op_probs"].sum(1), 1)
    np.testing.assert_allclose(p["sub_probs"].sum(1), 1)
    for source in range(1, 40):
        assert p["sub_probs"][source, source] == 0
        assert (np.delete(p["sub_probs"][source, 1:], source - 1) > 0).all()


def test_speaker_equal_counts_and_directional_confusions():
    first = [row("A"), row("B", "match", "T")]
    duplicate = [row("A")] * 100 + [row("B", "match", "T")]
    kw = {"training_speakers": ["A", "B"], "min_phone_tokens": 1, "min_phone_speakers": 1}
    a, b = estimate_error_prior(first, **kw), estimate_error_prior(duplicate, **kw)
    for field in ("op_probs", "sub_probs", "insert_phone_probs"):
        np.testing.assert_allclose(a[field], b[field])
    t, d, k = [PHONE_TO_CTC_ID[p] for p in ("T", "D", "K")]
    assert a["sub_probs"][t, d] > a["sub_probs"][t, k]


def test_nontraining_data_rejected_and_ambiguous_labels_unused():
    with pytest.raises(ValueError, match="non-training"):
        estimate_error_prior([row("TEST")], training_speakers=["TRAIN"])
    rows = [row("TRAIN")]
    altered = deepcopy(rows[0])
    altered["labels"]["tokens"][0]["diagnostic"] = False
    altered["labels"]["gaps"][0]["diagnostic"] = False
    one = estimate_error_prior(rows, training_speakers=["TRAIN"])
    two = estimate_error_prior(rows + [altered], training_speakers=["TRAIN"])
    np.testing.assert_equal(one["sub_probs"], two["sub_probs"])
