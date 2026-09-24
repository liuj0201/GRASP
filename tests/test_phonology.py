from __future__ import annotations

import math
import os
from pathlib import Path

import pytest

from da_cf_gop.phonology import (
    ARPABET_39,
    align_category_aware,
    align_levenshtein,
    category_substitution_cost,
    derive_stable_token_labels,
    g2p_provenance,
    ipa_rows_to_phones,
    normalize_phn_label,
    project_model_phone,
    read_phn,
    texts_to_phones,
    validate_prompt,
)


def test_model_inventory_and_required_projection() -> None:
    assert len(ARPABET_39) == 39
    assert len(set(ARPABET_39)) == 39
    assert project_model_phone("AX") == "AH"
    assert project_model_phone("dx") == "T"
    assert "AX" not in ARPABET_39
    assert "DX" not in ARPABET_39


def test_espeak_ipa_rows_project_to_arpabet39() -> None:
    rows, unknown = ipa_rows_to_phones(["ð ə | k æ t", "ɾ iː"])
    assert rows == [["DH", "AH", "K", "AE", "T"], ["T", "IY"]]
    assert unknown == [[], []]


def test_real_espeak_g2p_is_frozen_deterministic_and_ignores_ambient_data(
    tmp_path, monkeypatch
) -> None:
    import espeakng_loader

    monkeypatch.setenv("ESPEAK_DATA_PATH", str(tmp_path / "ambient-wrong-data"))
    prompts = ["Bat", "A cat sat.", "She chose three blue toys."]
    expected = [
        ["B", "AE", "T"],
        ["AH", "K", "AE", "T", "S", "AE", "T"],
        ["SH", "IY", "CH", "OW", "Z", "TH", "R", "IY", "B", "L", "UW", "T", "OY", "Z"],
    ]
    first, first_unknown = texts_to_phones(prompts)
    second, second_unknown = texts_to_phones(prompts)
    assert first == second == expected
    assert first_unknown == second_unknown == [[], [], []]
    assert Path(os.environ["ESPEAK_DATA_PATH"]).resolve() == Path(
        espeakng_loader.get_data_path()
    ).resolve()

    provenance = g2p_provenance()
    assert provenance["backend"] == "phonemizer_espeak_bundled"
    assert provenance["espeak_runtime_version"] == "1.52.0"
    for key in (
        "descriptor_sha256",
        "espeak_library_sha256",
        "espeak_data_sha256",
    ):
        assert len(str(provenance[key])) == 64


@pytest.mark.parametrize(
    ("prompt", "valid", "reason"),
    [
        (" The   quick brown fox. ", True, ""),
        ("[say ah repeatedly]", False, "elicitation_instruction"),
        ("picture.jpg", False, "picture_prompt"),
        ("xxx", False, "empty_or_noise"),
        ("1234", False, "no_english_letters"),
    ],
)
def test_prompt_validation(prompt: str, valid: bool, reason: str) -> None:
    assert validate_prompt(prompt) == (valid, reason)


def test_phn_three_layers_and_known_drops(tmp_path) -> None:
    path = tmp_path / "0001.PHN"
    path.write_text(
        "0 10 h#\n10 20 pcl\n20 30 p\n30 40 ix\n40 50 ax-h\n"
        "50 60 dx\n60 70 eng\n70 80 uw1\n80 90 q\n",
        encoding="utf-8",
    )
    phn = read_phn(path)

    assert phn.raw_labels == (
        "h#", "pcl", "p", "ix", "ax-h", "dx", "eng", "uw1", "q"
    )
    assert phn.timit_phones == ("P", "IH", "AX", "DX", "NG", "UW")
    assert phn.model_phones == ("P", "IH", "AH", "T", "NG", "UW")
    assert phn.dropped_count == 3
    assert all(segment.start_sample < segment.end_sample for segment in phn.segments)


def test_unknown_phn_label_fails_closed(tmp_path) -> None:
    path = tmp_path / "bad.PHN"
    path.write_text("0 10 definitely_unknown\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unmapped PHN label"):
        read_phn(path)
    assert normalize_phn_label("definitely_unknown", strict=False) is None


def test_category_costs_are_exact_code5_migration() -> None:
    assert category_substitution_cost("AX", "AH") == pytest.approx(0.15)
    assert category_substitution_cost("DX", "D") == pytest.approx(0.15)
    assert category_substitution_cost("Y", "IY") == pytest.approx(0.6)
    assert category_substitution_cost("AA", "IY") == pytest.approx(1.0)
    assert category_substitution_cost("T", "K") == pytest.approx(1.0)
    assert category_substitution_cost("T", "AA") == pytest.approx(2.2)


def test_aligners_are_deterministic_and_report_indices() -> None:
    expected = [(0, 0), (1, None), (2, 1)]
    assert align_levenshtein(["B", "T", "AA"], ["B", "AA"]) == expected
    assert align_category_aware(["B", "T", "AA"], ["B", "AA"]) == expected

    # A unit-cost mismatch ties two gaps; diagonal wins by the frozen tie-break.
    assert align_levenshtein(["T"], ["AA"]) == [(0, 0)]
    # Category-aware cross-class substitution costs 2.2, so two gaps win.
    assert align_category_aware(["T"], ["AA"]) == [(None, 0), (0, None)]


def test_stable_token_consensus_and_uncertain_count() -> None:
    stable = derive_stable_token_labels(["B", "T", "AA"], ["B", "AA"])
    assert [token.event_type for token in stable.tokens] == ["match", "deletion", "match"]
    assert stable.stable_count == 3
    assert stable.uncertain_count == 0
    assert stable.stable_fraction == 1.0
    assert stable.exact_identification_headline_allowed

    disputed = derive_stable_token_labels(["T"], ["AA"])
    assert disputed.tokens[0].event_type == "alignment_uncertain"
    assert not disputed.tokens[0].stable
    assert disputed.uncertain_count == 1
    assert disputed.levenshtein_insertions == 0
    assert disputed.category_insertions == 1
    assert not disputed.exact_identification_headline_allowed
    assert math.isclose(disputed.stable_fraction, 0.0)


def test_insertions_are_counted_but_not_canonical_tokens() -> None:
    result = derive_stable_token_labels(["B", "AA"], ["B", "T", "AA"])
    assert len(result.tokens) == 2
    assert result.levenshtein_insertions == 1
    assert result.category_insertions == 1
    assert result.agreed_insertions == 1
    assert all(token.event_type == "match" for token in result.tokens)
