"""Small-model pilot: controlled English sentences containing target nouns.

The grammar is an explicit task restriction, not a general English quality judge.
All cases are synthetic. Target nouns have one entry in the repository CMU lexicon.
"""

import re
from pathlib import Path


TEST_NOUNS = "cat dog book cup bag ball box bed chair desk door drum duck fish flag frog gate glass hat hill horse house kite lamp leaf map moon mouse park pen pig plate pond pot ring road rock rope bell ship shoe shop sock spoon star stone street sun table tree".split()
DEV_NOUNS = "bird boat bowl bread bus cake coat egg farm fork goat key nest train wall".split()
CONTEXT_NOUNS = "table room floor garden path school".split()
DETERMINERS = set("the a an my your our this that".split())
ADJECTIVES = set("red blue green small big old new soft warm clean quiet bright good white black brown".split())
PREPOSITIONS = set("in on near beside behind under above below".split())
TRANSITIVE = set("sees finds holds carries likes watches touches".split())
INTRANSITIVE = set("sleeps runs walks sits stands rests lies floats flies waits".split())
MASS_NOUNS = {"bread"}

TEST_PAIRS = [
    ("cat", "book", "beside"), ("dog", "ball", "beside"),
    ("book", "table", "on"), ("cup", "plate", "beside"),
    ("bag", "chair", "beside"), ("ball", "box", "in"),
    ("box", "bed", "beside"), ("chair", "desk", "beside"),
    ("desk", "pen", "near"), ("door", "house", "in"),
    ("drum", "flag", "near"), ("duck", "pond", "near"),
    ("fish", "pond", "in"), ("flag", "gate", "beside"),
    ("frog", "rock", "on"), ("gate", "park", "in"),
    ("glass", "table", "on"), ("hat", "bed", "on"),
    ("hill", "house", "behind"), ("horse", "gate", "near"),
    ("house", "road", "beside"), ("kite", "tree", "near"),
    ("lamp", "desk", "on"), ("leaf", "stone", "on"),
    ("map", "table", "on"), ("moon", "hill", "above"),
    ("mouse", "box", "behind"), ("park", "street", "beside"),
    ("pen", "book", "beside"), ("pig", "pond", "near"),
    ("plate", "spoon", "beside"), ("pond", "tree", "beside"),
    ("pot", "table", "on"), ("ring", "box", "in"),
    ("road", "hill", "below"), ("rock", "tree", "beside"),
    ("rope", "bag", "in"), ("bell", "door", "beside"),
    ("ship", "moon", "below"), ("shoe", "bed", "under"),
    ("shop", "street", "beside"), ("sock", "shoe", "near"),
    ("spoon", "cup", "beside"), ("star", "house", "above"),
    ("stone", "gate", "near"), ("street", "house", "beside"),
    ("sun", "tree", "above"), ("table", "chair", "beside"),
    ("tree", "hill", "below"), ("cat", "dog", "beside"),
]
DEV_PAIRS = [
    ("bird", "nest", "in"), ("boat", "wall", "beside"),
    ("bowl", "fork", "beside"), ("bread", "cake", "beside"),
    ("bus", "train", "near"), ("cake", "bowl", "beside"),
    ("key", "coat", "near"), ("egg", "bowl", "in"),
    ("farm", "goat", "near"), ("fork", "bread", "beside"),
    ("goat", "wall", "near"), ("key", "bowl", "in"),
    ("nest", "wall", "on"), ("train", "farm", "near"),
    ("wall", "farm", "near"),
]


def _single_context(noun):
    if noun in set("cat dog duck frog horse mouse pig bird goat".split()):
        return "garden", "in"
    if noun in {"moon", "star", "sun", "kite"}:
        return "garden", "above"
    if noun in set("bed chair desk door gate hill house park pond road ship shop street tree boat bus farm train wall".split()):
        return "school", "near"
    if noun == "table":
        return "room", "in"
    return "table", "on"


def prepare():
    dictionary = Path(__file__).resolve().parents[2] / "resources" / "dual_graph" / "cmudict.dict"
    target_vocab = set(TEST_NOUNS + DEV_NOUNS)
    pronunciations = {word: [] for word in target_vocab}
    for line in dictionary.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        word = re.sub(r"\(\d+\)$", "", fields[0])
        if word in pronunciations:
            pronunciations[word].append(fields[1:])
    assert all(len(phones) == 1 for phones in pronunciations.values())
    cases = []
    for split, nouns, pairs in [("dev", DEV_NOUNS, DEV_PAIRS), ("test", TEST_NOUNS, TEST_PAIRS)]:
        specs = [(n,) for n in nouns] + [(a, b) for a, b, _ in pairs]
        for index, targets in enumerate(specs):
            if len(targets) == 1:
                context, relation = _single_context(targets[0])
                example = f"The {targets[0]} is {relation} the {context}."
            else:
                relation = pairs[index - len(nouns)][2]
                example = f"The {targets[0]} is {relation} the {targets[1]}."
            original = f"Please look at the {targets[0]}." if len(targets) == 1 else f"Please look at the {targets[0]} and the {targets[1]}."
            cases.append({
                "id": f"sentence_{split}_{index + 1:03d}",
                "task": "sentence", "split": split,
                "input": {"targets": list(targets), "original_sentence": original},
                "gold": {"example_valid_sentence": example, "relation": relation,
                         "target_pronunciations": {word: pronunciations[word][0] for word in targets}},
                "metadata": {
                    "target_count": len(targets), "source": "synthetic everyday noun combinations",
                    "scope": "controlled English; singular common target nouns; one pronunciation per target in repository CMU lexicon",
                    "pronunciation_source": "code/resources/dual_graph/cmudict.dict",
                    "grammar_scope": "NP + is + PP/adjective/NP, or NP + present singular verb (+ NP/PP)",
                    "not_measured": ["clinical benefit", "general English fluency", "semantic plausibility"],
                },
            })
    return cases


def messages(case):
    targets = case["input"]["targets"]
    system = (
        "Write one simple, sensible English practice sentence of 5 to 10 words. "
        "Include every target word exactly as spelled; do not pluralize or change it. "
        "When two words are listed, BOTH words must appear in your sentence. "
        "Use an everyday situation. Do not copy the original sentence. "
        "Output only the new sentence, without a heading or explanation.\n"
        "Example required words: violin, telescope\n"
        "The violin is beside the telescope."
    )
    user = (
        f"Required words (use ALL unchanged): {', '.join(targets)}\n"
        f"Original sentence: {case['input']['original_sentence']}\n"
        "Write a different sentence now."
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _tokens(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text.lower())


def _grammar_parse(words, nouns):
    """A small compositional recognizer; passing is a subset certificate only."""
    def np(start):
        if start >= len(words) or words[start] not in DETERMINERS:
            return None
        det = words[start]
        pos = start + 1
        while pos < len(words) and words[pos] in ADJECTIVES:
            pos += 1
        if pos >= len(words) or words[pos] not in nouns:
            return None
        if det in {"a", "an"}:
            if words[pos] in MASS_NOUNS:
                return None
            vowel = words[start + 1][0] in "aeiou"
            if (det == "an") != vowel:
                return None
        return pos + 1

    def pp(start):
        if start < len(words) and words[start] in PREPOSITIONS:
            return np(start + 1)
        return None

    def end_or_pp(start):
        return start == len(words) or pp(start) == len(words)

    pos = np(0)
    if pos is None or pos >= len(words):
        return False
    verb = words[pos]
    pos += 1
    if verb == "is":
        if pp(pos) == len(words):
            return True
        complement = np(pos)
        if complement is not None and end_or_pp(complement):
            return True
        initial = pos
        while pos < len(words) and words[pos] in ADJECTIVES:
            pos += 1
        return pos > initial and end_or_pp(pos)
    if verb in TRANSITIVE:
        obj_end = np(pos)
        return obj_end is not None and end_or_pp(obj_end)
    if verb in INTRANSITIVE:
        return end_or_pp(pos)
    return False


def evaluate(case, raw_text):
    sentence = raw_text.strip().strip('"\u201c\u201d')
    words = _tokens(sentence)
    targets = case["input"]["targets"]
    nouns = set(targets + CONTEXT_NOUNS)
    allowed = nouns | DETERMINERS | ADJECTIVES | PREPOSITIONS | TRANSITIVE | INTRANSITIVE | {"is"}
    content = sentence[:-1] if sentence.endswith((".", "!", "?")) else sentence
    one_sentence = bool(content) and bool(re.fullmatch(r"[A-Za-z ]+", content))
    checks = {
        "target_coverage": all(word in words for word in targets),
        "word_count_5_to_10": 5 <= len(words) <= 10,
        "one_plain_sentence": one_sentence,
        "different_from_original": words != _tokens(case["input"]["original_sentence"]),
        "allowed_vocabulary": all(word in allowed for word in words),
        "controlled_grammar": _grammar_parse(words, nouns),
    }
    hard_names = ["target_coverage", "word_count_5_to_10", "one_plain_sentence", "different_from_original"]
    return {
        "success": all(checks[key] for key in hard_names), "checks": checks,
        "hard_constraint_success": all(checks[key] for key in hard_names),
        "word_count": len(words), "parsed_sentence": sentence,
        "outside_vocabulary": sorted(set(words) - allowed),
        "interpretation": "Primary success checks lexical/length/single-output constraints only. The old controlled grammar is an auxiliary coverage probe, not an English correctness judge.",
    }


def baselines(case):
    targets = case["input"]["targets"]
    other = targets[1] if len(targets) == 2 else ("room" if targets[0] == "table" else "table")
    return {
        "fixed_template": f"The {targets[0]} is near the {other}.",
        "rule_generator": case["gold"]["example_valid_sentence"],
    }


if __name__ == "__main__":
    cases = prepare()
    assert len(cases) == 130
    assert sum(c["split"] == "dev" for c in cases) == 30
    test = [c for c in cases if c["split"] == "test"]
    assert sum(c["metadata"]["target_count"] == 1 for c in test) == 50
    assert sum(c["metadata"]["target_count"] == 2 for c in test) == 50
    dev_keys = {tuple(sorted(c["input"]["targets"])) for c in cases if c["split"] == "dev"}
    test_keys = {tuple(sorted(c["input"]["targets"])) for c in test}
    assert len(test_keys) == 100
    assert not dev_keys & test_keys
    for case in cases:
        for name, output in baselines(case).items():
            result = evaluate(case, output)
            assert result["success"], (case["id"], name, result)
    fixture = next(c for c in test if c["input"]["targets"] == ["cat", "book"])
    assert evaluate(fixture, "The cat is beside my book.")["success"]
    assert not evaluate(fixture, "The cats is beside my book.")["success"]
    result = evaluate(fixture, "The cat are beside my book.")
    assert result["success"] and not result["checks"]["controlled_grammar"]
    assert not evaluate(fixture, "The cat is beside a book. Another sentence.")["success"]
    print("sentence_task smokecheck: 130 cases; 30 dev/100 test; both baselines pass all cases; negative checks passed")
