"""Controlled verbalization of one pre-grouped phoneme's current-attempt facts.

Synthetic position records are the oracle. Code groups/counts; the LLM is NOT
being tested for independent phonetic discovery, diagnosis, or free prose.
Twenty test cases are changes of twenty other test cases, not independent data.
"""

from collections import Counter
from copy import deepcopy
import json
import re


WORDS = {
    "k": ["book", "cat", "quick", "key", "back", "school"],
    "t": ["tea", "hat", "table", "coat", "boat", "time"],
    "s": ["sun", "bus", "seat", "rice", "soap", "soup"],
    "p": ["pen", "cup", "paper", "map", "park", "shop"],
    "f": ["fish", "leaf", "food", "roof", "fan", "safe"],
    "m": ["moon", "home", "milk", "room", "map", "name"],
    "n": ["nose", "rain", "night", "green", "name", "phone"],
    "b": ["book", "cab", "ball", "robe", "bed", "boat"],
    "d": ["door", "bed", "day", "road", "dog", "food"],
    "g": ["go", "bag", "game", "dog", "gate", "leg"],
    "v": ["van", "five", "voice", "wave", "vest", "save"],
    "z": ["zoo", "nose", "zero", "rose", "zone", "cheese"],
}

CONDITIONS = (
    "zero_flags", "singleton_no_recurrence", "all_flagged",
    "partial_flags", "repeated_word", "unassessed_excluded",
    "no_assessed_positions", "mixed_word_positions", "spelling_variation",
    "partial_repeated_word",
)


def _records(index):
    """Build deterministic synthetic observations, including other phonemes."""
    condition = CONDITIONS[index % len(CONDITIONS)]
    phoneme = list(WORDS)[(index // 10 + index % 10) % len(WORDS)]
    if condition == "spelling_variation":
        phoneme = "k"
    words = WORDS[phoneme]
    offset = (index // 10) % len(words)
    ordered = words[offset:] + words[:offset]
    if condition == "spelling_variation":
        ordered = words[:]
    # Indices denote occurrences, so a repeated spelling is not deduplicated.
    shapes = {
        "zero_flags": ([True, True, True, False], [False] * 4),
        "singleton_no_recurrence": ([True] * 4, [True, False, False, False]),
        "all_flagged": ([True] * 3, [True] * 3),
        "partial_flags": ([True] * 5, [True, False, True, False, False]),
        "repeated_word": ([True] * 4, [True, True, False, True]),
        "unassessed_excluded": ([True, False, True, False, True],
                                [True, False, True, False, False]),
        "no_assessed_positions": ([False] * 3, [False] * 3),
        "mixed_word_positions": ([True] * 4, [True, True, False, True]),
        "spelling_variation": ([True] * 5, [True, True, True, False, False]),
        "partial_repeated_word": ([True, True, False, True, True],
                                  [True, False, False, True, False]),
    }
    assessed, flagged = shapes[condition]
    records = []
    for pos, (is_assessed, is_flagged) in enumerate(zip(assessed, flagged)):
        word = ordered[pos % len(ordered)]
        if "repeated_word" in condition and pos in (1, 3):
            word = ordered[0]
        records.append({
            "position_id": pos, "phoneme": phoneme, "word": word,
            "assessed": is_assessed, "flagged": is_flagged,
        })
    # Vary the ignored denominator distractor across batches, including cases
    # with identical flagged words. It must never inflate the assessed count.
    for extra in range(index // 10):
        records.append({
            "position_id": len(records), "phoneme": phoneme,
            "word": ordered[extra % len(ordered)],
            "assessed": False, "flagged": False,
        })
    # These distractors are excluded by the preprocessing group, not the LLM.
    other = "s" if phoneme != "s" else "k"
    records.extend([
        {"position_id": len(records), "phoneme": other, "word": WORDS[other][0],
         "assessed": True, "flagged": True},
        {"position_id": len(records) + 1, "phoneme": other, "word": WORDS[other][1],
         "assessed": False, "flagged": False},
    ])
    return phoneme, records, condition


def _aggregate_input(phoneme, records):
    group = [r for r in records if r["phoneme"] == phoneme]
    assessed = [r for r in group if r["assessed"]]
    flagged = [r for r in assessed if r["flagged"]]
    return {
        "phoneme": phoneme,
        "flagged_count": len(flagged),
        "assessed_count": len(assessed),
        "unassessed_count": len(group) - len(assessed),
        "flagged_word_occurrences": [r["word"] for r in flagged],
    }


def _derive_gold(phoneme, records):
    # Deliberately compute from source records, never from the model input text
    # or the rendered template that the scoring function will inspect.
    expected = {"phoneme": phoneme, "flagged": 0, "assessed": 0, "words": []}
    for row in records:
        if row["phoneme"] != phoneme or not row["assessed"]:
            continue
        expected["assessed"] += 1
        if row["flagged"]:
            expected["flagged"] += 1
            expected["words"].append(row["word"])
    return expected


def _case(case_id, split, phoneme, records, condition, group_id, **metadata):
    return {
        "id": case_id, "task": "summary", "split": split,
        "input": _aggregate_input(phoneme, records),
        "gold": _derive_gold(phoneme, records),
        "metadata": {
            "condition": condition, "group_id": group_id,
            "source": "synthetic_position_records",
            "raw_positions": records, "scope": "controlled_verbalization",
            **metadata,
        },
    }


def prepare():
    cases = []
    for split, count, start in (("dev", 30, 0), ("test", 80, 30)):
        for i in range(count):
            phoneme, records, condition = _records(start + i)
            case_id = f"summary_{split}_{i:03d}"
            fixture_metadata = {}
            if split == "test":
                # Before formal inference, vary actual assessed counts (and
                # flagged members for all-flagged cases), not invisible IDs or
                # unassessed distractors. Preserve the zero-assessed condition.
                additions = 0 if condition == "no_assessed_positions" else i // 10 + 1
                for extra in range(additions):
                    records.append({
                        "position_id": len(records), "phoneme": phoneme,
                        "word": WORDS[phoneme][(i + extra) % len(WORDS[phoneme])],
                        "assessed": True, "flagged": condition == "all_flagged",
                    })
                fixture_metadata = {
                    "fixture_revision": "pre_formal_semantic_dedup_v2",
                    "additional_assessed_positions": additions,
                }
            cases.append(_case(case_id, split, phoneme, records, condition,
                               case_id, variant="base", **fixture_metadata))
    bases = [case for case in cases if case["split"] == "test"][:20]
    for i, base in enumerate(bases):
        records = deepcopy(base["metadata"]["raw_positions"])
        phoneme = base["input"]["phoneme"]
        target_rows = [r for r in records if r["phoneme"] == phoneme]
        assessed = [r for r in target_rows if r["assessed"]]
        if not assessed:
            target_rows[0]["assessed"] = True
            target_rows[0]["flagged"] = True
            change = "assessment_add"
        elif i % 2 == 0:
            assessed[0]["flagged"] = not assessed[0]["flagged"]
            change = "flag_flip"
        else:
            assessed[0]["assessed"] = False
            assessed[0]["flagged"] = False
            change = "assessment_drop"
        cases.append(_case(
            f"summary_test_{80 + i:03d}", "test", phoneme, records,
            base["metadata"]["condition"], base["id"], variant=change,
            paired_with=base["id"],
            fixture_revision="pre_formal_semantic_dedup_v2",
            additional_assessed_positions=base["metadata"]["additional_assessed_positions"],
        ))
    return cases


def messages(case):
    return [
        {"role": "system", "content": (
            "Summarize one pre-grouped phoneme from this attempt. Output exactly "
            "one sentence using this format: In this attempt, /PHONEME/ was flagged "
            "in FLAGGED of ASSESSED assessed positions; flagged words: WORDS. "
            "Replace PHONEME and the two counts with the supplied values. WORDS is "
            "the comma-separated flagged_word_occurrences, preserving repetitions, "
            "or the literal word none when the list is empty. Use the same format "
            "when a count is zero or one. Unassessed positions are not part of the "
            "assessed count. Do not add any other statement or diagnosis.\n"
            "Example: phoneme=r, flagged=2, assessed=4, words=red, car\n"
            "In this attempt, /r/ was flagged in 2 of 4 assessed positions; flagged words: red, car.\n"
            "Example: phoneme=r, flagged=0, assessed=3, words=none\n"
            "In this attempt, /r/ was flagged in 0 of 3 assessed positions; flagged words: none."
        )},
        {"role": "user", "content": (
            f"phoneme={case['input']['phoneme']}, flagged={case['input']['flagged_count']}, "
            f"assessed={case['input']['assessed_count']}, "
            f"words={', '.join(case['input']['flagged_word_occurrences']) or 'none'}"
        )},
    ]


PATTERN = re.compile(
    r"In this attempt,\s*/(?P<phoneme>[^/\s]+)/\s+was flagged in\s+"
    r"(?P<flagged>\d+)\s+of\s+(?P<assessed>\d+)\s+assessed positions;\s*"
    r"flagged words:\s*(?P<words>[A-Za-z ,]+)\.?", re.IGNORECASE,
)


def evaluate(case, raw_text):
    match = PATTERN.fullmatch(" ".join(raw_text.strip().split()))
    checks = {"parseable": match is not None, "phoneme_correct": False,
              "flagged_count_correct": False, "assessed_count_correct": False,
              "word_occurrences_correct": False}
    if match is None:
        return {"success": False, "checks": checks, "parsed": None}
    parsed = match.groupdict()
    parsed["flagged"] = int(parsed["flagged"])
    parsed["assessed"] = int(parsed["assessed"])
    word_text = parsed["words"].strip().lower()
    parsed["words"] = [] if word_text == "none" else [
        word.strip() for word in word_text.split(",")
    ]
    gold = case["gold"]
    checks.update({
        "phoneme_correct": parsed["phoneme"] == gold["phoneme"],
        "flagged_count_correct": parsed["flagged"] == gold["flagged"],
        "assessed_count_correct": parsed["assessed"] == gold["assessed"],
        "word_occurrences_correct": Counter(parsed["words"]) == Counter(gold["words"]),
    })
    return {"success": all(checks.values()), "checks": checks, "parsed": parsed}


def baselines(case):
    # Baseline uses the supplied aggregate, whereas the oracle uses raw records.
    stats = case["input"]
    words = ", ".join(stats["flagged_word_occurrences"]) or "none"
    return {"deterministic_template": (
        f"In this attempt, /{stats['phoneme']}/ was flagged in "
        f"{stats['flagged_count']} of {stats['assessed_count']} assessed positions; "
        f"flagged words: {words}."
    )}
