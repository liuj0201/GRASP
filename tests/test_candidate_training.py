from copy import deepcopy

import pytest

from da_cf_gop.candidate_training import estimate_neighbors
from da_cf_gop.phonology import PHONE_TO_CTC_ID


def _record(speaker, event, targets, *, source="K"):
    return {
        "speaker": speaker, "event": event, "quality_exclusion": None,
        "canonical_phones": [source] * len(targets),
        "labels": {"tokens": [
            {"phone_index": index, "event_type": "match" if target == source else "substitution",
             "realized_phone": target, "stable_event": True, "diagnostic": True}
            for index, target in enumerate(targets)
        ]},
    }


def test_heldout_only_confusions_and_ambiguous_labels_never_enter_training_neighbors():
    train = _record("TRAIN", "train1", ["T", "D", "N", "G"])
    train["labels"]["tokens"][1]["stable_event"] = False
    train["labels"]["tokens"][2]["diagnostic"] = False
    excluded = _record("TRAIN", "excluded", ["B"])
    excluded["quality_exclusion"] = "invalid_prompt"
    test = _record("TEST", "test1", ["F"] * 100)
    result = estimate_neighbors([train, test, excluded], ["TRAIN"])
    k, t, g = (PHONE_TO_CTC_ID[p] for p in ("K", "T", "G"))
    assert set(result["neighbors"][k]) == {t, g}
    assert result["training_speakers"] == ["TRAIN"]
    assert result["diagnostic_tokens_by_speaker"] == {"TRAIN": 2}
    assert result["selected_training_records"] == 1
    assert all(
        event["speaker"] == "TRAIN" and event["event"] == "train1"
        for source in result["sources"] for event in source["supporting_training_events"]
    )
    # No reverse relation or unobserved source is inferred.
    assert result["neighbors"][t] == []
    assert result["neighbors"][PHONE_TO_CTC_ID["HH"]] == []


def test_equal_patient_weight_can_rank_a_rare_patients_edge_above_raw_majority():
    large = _record("A", "many", ["D"] * 8 + ["K"] * 92)
    small = _record("B", "few", ["T", "K"])
    result = estimate_neighbors([large, small], ["A", "B"], max_neighbors=1)
    k, t = (PHONE_TO_CTC_ID[p] for p in ("K", "T"))
    assert result["neighbors"][k] == [t]
    by_target = {row["target_phone"]: row for row in result["sources"]}
    assert by_target["T"]["balanced_rate"] == pytest.approx(.25)
    assert by_target["D"]["balanced_rate"] == pytest.approx(.04)
    assert by_target["T"]["raw_count"] == 1
    assert by_target["D"]["raw_count"] == 8
    assert by_target["T"]["speaker_count"] == 1
    replicated = deepcopy(large)
    replicated["event"] = "many2"
    again = estimate_neighbors([large, replicated, small], ["A", "B"], max_neighbors=1)
    assert again["neighbors"] == result["neighbors"]
    assert [row["balanced_rate"] for row in again["sources"]] == pytest.approx(
        [row["balanced_rate"] for row in result["sources"]],
    )


def test_rank_ties_and_minimum_count_are_deterministic_without_synthetic_edges():
    records = [_record("A", "a", ["T", "D", "G"]), _record("B", "b", ["T", "D", "K"])]
    result = estimate_neighbors(records, ["B", "A"], max_neighbors=1, min_count=2)
    k, d = (PHONE_TO_CTC_ID[p] for p in ("K", "D"))
    assert result["neighbors"][k] == [d]  # Equal rates/counts; D has the lower ID.
    assert len(result["sources"]) == 3  # Includes unselected observations for audit.
    assert sum(row["selected"] for row in result["sources"]) == 1
    assert estimate_neighbors(records[::-1], ["A", "B"], max_neighbors=1, min_count=2) == result
    none = estimate_neighbors(records, ["A", "B"], min_count=3)
    assert all(not row for row in none["neighbors"])


def test_no_observed_substitutions_returns_an_empty_relation():
    record = _record("A", "a", ["K"])
    # A malformed self-substitution must not become an error candidate either.
    record["labels"]["tokens"][0]["event_type"] = "substitution"
    result = estimate_neighbors([record], ["A"])
    assert result["sources"] == []
    assert result["neighbors"] == [[] for _ in range(40)]
