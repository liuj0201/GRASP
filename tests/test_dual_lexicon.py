from itertools import product

import pytest

from da_cf_gop.dual_lexicon import (
    build_acceptable_graph, evaluate_alignment, load_dictionary, load_policy, model_phones,
)


def test_dictionary_preserves_stress_and_source_order(tmp_path):
    path = tmp_path / "cmu.dict"
    path.write_text("roses R OW1 Z IH0 Z\nroses(2) R OW1 Z AH0 Z # unstressed variant\n", encoding="utf-8")
    dictionary = load_dictionary(path)
    assert dictionary["roses"] == [("R", "OW1", "Z", "IH0", "Z"), ("R", "OW1", "Z", "AH0", "Z")]
    assert model_phones(dictionary["roses"][0]) == ["R", "OW", "Z", "IH", "Z"]


def test_only_attested_local_weak_variant_is_accepted():
    dictionary = {"roses": [("R", "OW1", "Z", "IH0", "Z"), ("R", "OW1", "Z", "AH0", "Z")]}
    graph = build_acceptable_graph("roses", ["N"], dictionary)
    assert graph["canonical_phones"] == ["R", "OW", "Z", "IH", "Z"]
    assert graph["acceptable_phones"][3] == ["IH", "AH"]
    assert all(graph["diagnostic_mask"])
    assert graph["variant_position_count"] == 1
    assert graph["words"][0]["canonical_source_phones"][1] == "OW1"
    labels = evaluate_alignment(graph["acceptable_phones"], ["R", "OW", "Z", "AH", "Z"])
    assert all(not token["is_error"] for token in labels["tokens"])


@pytest.mark.parametrize("word,variants,reason", [
    ("read", [("R", "IY1", "D"), ("R", "EH1", "D")], "context_ambiguous_homograph"),
    ("family", [("F", "AE1", "M", "AH0", "L", "IY0"), ("F", "AE1", "M", "L", "IY0")], "length_changing_variant_unavailable"),
    ("water", [("W", "AO1", "T", "ER0"), ("W", "AA1", "T", "ER0")], "unsupported_lexical_variant_unavailable"),
])
def test_unsupported_variants_make_block_nondiagnostic(word, variants, reason):
    graph = build_acceptable_graph(word, dictionary={word: variants})
    assert not any(graph["diagnostic_mask"])
    assert reason in graph["exclusions"][0]["reasons"]
    labels = evaluate_alignment(graph["acceptable_phones"], graph["canonical_phones"], graph["diagnostic_mask"])
    assert labels["stable_error_count"] == labels["stable_event_count"] == 0


def test_no_unattested_cartesian_combinations_within_word():
    variants = [("AH0", "N", "IH0"), ("IH0", "N", "IH0"), ("AH0", "N", "AH0")]
    graph = build_acceptable_graph("sample", dictionary={"sample": variants})
    assert graph["acceptable_phones"] == [["AH"], ["N"], ["IH"]]
    assert not any(graph["diagnostic_mask"])


def test_oov_fallback_only_used_for_oov_words(monkeypatch):
    seen = []
    def fallback(word):
        seen.append(word)
        return ("Z", "IH", "P")
    monkeypatch.setattr("da_cf_gop.dual_lexicon._oov_pronunciation", fallback)
    graph = build_acceptable_graph("cat zzip", dictionary={"cat": [("K", "AE1", "T")]})
    assert seen == ["zzip"]
    assert graph["canonical_phones"] == ["K", "AE", "T", "Z", "IH", "P"]
    assert graph["words"][1]["start"] == 3
    assert graph["words"][1]["source"] == "frozen_local_espeak_en_us_oov"


def test_graph_weights_are_deduplicated_and_input_not_mutated():
    variants = [("AH0",), ("IH0",), ("IH0",)]
    dictionary = {"a": variants}
    graph = build_acceptable_graph("a", dictionary=dictionary)
    assert graph["acceptable_phones"] == [["AH", "IH"]]
    assert dictionary["a"] == variants


def test_repeated_phone_alignment_keeps_only_consensus():
    result = evaluate_alignment([["T"], ["T"]], ["T"])
    assert result["edit_distance"] == 1
    assert result["stable_error_count"] == 0
    assert all(token["event_type"] == "alignment_uncertain" for token in result["tokens"])


def test_binary_error_mask_can_exceed_exact_event_mask():
    result = evaluate_alignment([["T"], ["K"]], ["D"])
    assert all(token["stable_error"] for token in result["tokens"])
    assert not any(token["stable_event"] for token in result["tokens"])
    assert all(token["is_error"] for token in result["tokens"])


def test_insertions_have_separate_reference_gap_and_ordinal():
    result = evaluate_alignment([["K"], ["AE"], ["T"]], ["K", "AH", "IH", "AE", "T"])
    assert not any(token["is_error"] for token in result["tokens"])
    assert result["gaps"][1]["inserted_phones"] == ["AH", "IH"]
    assert result["gaps"][1]["ordinals"][0]["phone"] == "AH"
    assert result["gaps"][1]["ordinals"][1]["phone"] == "IH"
    assert result["gaps"][1]["ordinals"][1]["ordinal"] == 1


def test_no_global_t_deletion_or_arbitrary_vowel_weakening():
    graph = build_acceptable_graph("cat", dictionary={"cat": [("K", "AE1", "T")]})
    assert graph["acceptable_phones"] == [["K"], ["AE"], ["T"]]
    result = evaluate_alignment(graph["acceptable_phones"], ["K", "AH", "T"])
    assert result["tokens"][1]["event_type"] == "substitution"
    result = evaluate_alignment(graph["acceptable_phones"], ["K", "AE"])
    assert result["tokens"][2]["event_type"] == "deletion"


def _brute_optima(reference, observed):
    n, m = len(reference), len(observed)
    paths = []
    def walk(i, j, cost, events, gaps):
        if i == n and j == m:
            paths.append((cost, events, gaps))
            return
        if i < n:
            walk(i + 1, j, cost + 1, events + [("deletion", None)], gaps)
        if j < m:
            next_gaps = [list(g) for g in gaps]
            next_gaps[i].append(observed[j])
            walk(i, j + 1, cost + 1, events, next_gaps)
        if i < n and j < m:
            error = observed[j] not in reference[i]
            walk(i + 1, j + 1, cost + error,
                 events + [("substitution" if error else "match", observed[j])], gaps)
    walk(0, 0, 0, [], [[] for _ in range(n + 1)])
    minimum = min(path[0] for path in paths)
    return minimum, [p for p in paths if p[0] == minimum]


def test_all_optimal_dp_matches_exhaustive_paths_and_gap_intervals():
    # Includes empty references/observations, repeats, acceptable alternatives,
    # multiple errors, ambiguous insertion positions, and DEL/SUB ambiguity.
    for n in range(4):
        for m in range(4):
            for reference in product((("T",), ("D",), ("T", "D")), repeat=n):
                for observed in product(("T", "D"), repeat=m):
                    cost, paths = _brute_optima(reference, observed)
                    result = evaluate_alignment(reference, observed)
                    assert result["edit_distance"] == cost
                    for i, token in enumerate(result["tokens"]):
                        expected = {p[1][i] for p in paths}
                        actual = {(e["event_type"], e["realized_phone"]) for e in token["event_candidates"]}
                        assert actual == expected
                        assert token["stable_event"] == (len(expected) == 1)
                        assert token["stable_error"] == (len({e != "match" for e, _ in expected}) == 1)
                    for i, gap in enumerate(result["gaps"]):
                        expected = {tuple(p[2][i]) for p in paths}
                        assert {tuple(g) for g in gap["possible_sequences"]} == expected
                        assert gap["stable"] == (len(expected) == 1)


def test_bundled_policy_has_no_patient_source_or_blanket_rules():
    policy = load_policy()
    assert policy["weak_vowel_pairs"] == [["AH0", "IH0"]]
    assert "patient_derived_acceptable_variants" in policy["disabled_rules"]
    assert "length_changing_normal_variants" in policy["disabled_rules"]
