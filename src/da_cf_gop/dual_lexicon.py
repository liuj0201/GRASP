"""Conservative text-only acceptable graph and prediction-independent labels.

The first version supports position-anchored lexical variants only. Unsupported
normal variants are recorded, and their entire word block is non-diagnostic;
they are NOT silently treated as patient errors or invented graph paths. No PHN,
speaker identity, learned prior, or acoustic prediction enters graph building.
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

from .phonology import ARPABET_39, project_model_phone, text_to_phones


CODE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY = CODE_ROOT / "configs" / "dual_acceptable_v1.json"
DEFAULT_DICTIONARY = CODE_ROOT / "resources" / "dual_graph" / "cmudict.dict"


def load_policy(path: str | Path = DEFAULT_POLICY) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


@lru_cache(maxsize=2)
def load_dictionary(path: str | Path = DEFAULT_DICTIONARY) -> dict[str, list[tuple[str, ...]]]:
    """Read CMUdict in source order, preserving stress; no network or hashes."""
    result: dict[str, list[tuple[str, ...]]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        fields = line.split("#", 1)[0].split()
        if len(fields) < 2 or fields[0].startswith(";;;"):
            continue
        word = re.sub(r"\(\d+\)$", "", fields[0]).lower()
        phones = tuple(fields[1:])
        if phones not in result.setdefault(word, []):
            result[word].append(phones)
    return result


def model_phones(source: Sequence[str]) -> list[str]:
    """Keep source stress elsewhere, project labels to the frozen model space."""
    return [project_model_phone(re.sub(r"[012]$", "", p.upper())) for p in source]


def _words(prompt: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['.-][a-z0-9]+)*", prompt.lower().replace("’", "'"))


@lru_cache(maxsize=2048)
def _oov_pronunciation(word: str) -> tuple[str, ...]:
    phones, unknown = text_to_phones(word)
    if unknown or not phones:
        raise ValueError(f"OOV G2P cannot map {word!r}: {unknown}")
    return tuple(phones)


def build_acceptable_graph(
    prompt: str,
    fallback_phones: Sequence[str] | None = None,
    dictionary: Mapping[str, Sequence[Sequence[str]]] | None = None,
    *,
    policy: dict | None = None,
) -> dict:
    """Build canonical phones, independent accepted sets, spans and provenance.

    Dictionary-only normal variants require unchanged stress and a documented
    AH0/IH0 alternative. Differing positions must be identical across variants,
    so set expansion cannot generate an unattested within-word combination.
    ``fallback_phones`` is only an emergency whole-prompt representation; if
    needed, that utterance is non-diagnostic because word anchors are unknown.
    """
    dictionary = load_dictionary() if dictionary is None else dictionary
    policy = load_policy() if policy is None else policy
    words = _words(prompt)
    canonical: list[str] = []
    acceptable: list[list[str]] = []
    diagnostic: list[bool] = []
    blocks: list[dict] = []
    exclusions: list[dict] = []
    weak_pairs = {frozenset(pair) for pair in policy["weak_vowel_pairs"]}
    blocked = set(policy["homograph_blocklist"])

    for word_index, word in enumerate(words):
        source = dictionary.get(word)
        source_kind = "cmudict"
        if source:
            pronunciations = [tuple(p) for p in source]
        else:
            source_kind = "frozen_local_espeak_en_us_oov"
            try:
                pronunciations = [_oov_pronunciation(word)]
            except ValueError:
                if not fallback_phones:
                    raise
                fallback = model_phones(fallback_phones)
                return {
                    "policy_version": policy["policy_version"], "prompt": prompt,
                    "canonical_phones": fallback,
                    "acceptable_phones": [[p] for p in fallback],
                    "diagnostic_mask": [False] * len(fallback), "words": [],
                    "exclusions": [{"word": word, "reason": "unanchored_whole_prompt_g2p_fallback"}],
                    "variant_position_count": 0, "source": "cached_frozen_g2p_fallback",
                }
        base_source = pronunciations[0]
        base = model_phones(base_source)
        local = [[p] for p in base]
        accepted = [{"source_phones": list(base_source), "model_phones": base,
                     "rule_id": f"{source_kind}.primary"}]
        unavailable: list[dict] = []
        changed_positions: set[int] = set()
        for variant_source in pronunciations[1:]:
            variant = model_phones(variant_source)
            if variant == base:
                accepted.append({"source_phones": list(variant_source), "model_phones": variant,
                                 "rule_id": "cmudict.model_equivalent_stress_variant"})
                continue
            positions = [i for i, (a, b) in enumerate(zip(base_source, variant_source)) if a != b]
            allowed = (
                word not in blocked and len(base_source) == len(variant_source)
                and len(positions) == 1
                and frozenset((base_source[positions[0]], variant_source[positions[0]])) in weak_pairs
            )
            if allowed:
                position = positions[0]
                changed_positions.add(position)
                if variant[position] not in local[position]:
                    local[position].append(variant[position])
                accepted.append({"source_phones": list(variant_source), "model_phones": variant,
                                 "rule_id": "cmudict.same_word_unstressed_AH_IH",
                                 "word_phone_index": position})
            else:
                reason = "context_ambiguous_homograph" if word in blocked else (
                    "length_changing_variant_unavailable" if len(base_source) != len(variant_source)
                    else "unsupported_lexical_variant_unavailable"
                )
                unavailable.append({"source_phones": list(variant_source), "model_phones": variant,
                                    "reason": reason})
        if len(changed_positions) > 1:
            unavailable.append({"reason": "multiple_variant_positions_would_invent_combinations"})
            local = [[p] for p in base]
            accepted = accepted[:1]
        if word in blocked:
            unavailable.append({"reason": "context_ambiguous_homograph"})
        start = len(canonical)
        canonical.extend(base)
        acceptable.extend(local)
        diagnostic.extend([not unavailable] * len(base))
        block = {
            "word": word, "word_index": word_index, "start": start, "end": len(canonical),
            "source": source_kind, "source_pronunciations": [list(p) for p in pronunciations],
            "canonical_source_phones": list(base_source), "accepted_variants": accepted,
            "unavailable_variants": unavailable, "diagnostic": not unavailable,
            "left_word": words[word_index - 1] if word_index else None,
            "right_word": words[word_index + 1] if word_index + 1 < len(words) else None,
            "context_condition": "same lexical word; identical stress; no cross-word rules",
            "source_url": "https://github.com/cmusphinx/cmudict" if source_kind == "cmudict" else None,
        }
        blocks.append(block)
        if unavailable:
            exclusions.append({"word": word, "start": start, "end": len(canonical),
                               "reasons": sorted({v["reason"] for v in unavailable})})
    if not canonical:
        raise ValueError("reference prompt contains no supported lexical tokens")
    return {
        "policy_version": policy["policy_version"], "prompt": prompt,
        "canonical_phones": canonical, "acceptable_phones": acceptable,
        "diagnostic_mask": diagnostic, "words": blocks, "exclusions": exclusions,
        "variant_position_count": sum(len(choices) > 1 for choices in acceptable),
        "source": "CMUdict with frozen local eSpeak OOV fallback",
    }


def evaluate_alignment(
    acceptable_phones: Sequence[Sequence[str]],
    realized_phones: Sequence[str],
    diagnostic_mask: Sequence[bool] | None = None,
) -> dict:
    """Consensus across EVERY minimum-cost alignment to a fixed accepted graph.

    Forward/backward distances identify all globally optimal edges. A canonical
    token is binary-stable when every edge consuming it agrees about error vs
    OK; exact stability additionally requires the same event and realized phone.
    Insertions are anchored to gaps, with ordinals, not mixed into token labels.
    """
    allowed = [tuple(dict.fromkeys(model_phones(p))) for p in acceptable_phones]
    if any(not p for p in allowed):
        raise ValueError("each reference position needs an acceptable phone")
    observed = model_phones(realized_phones)
    n, m = len(allowed), len(observed)
    diagnostic = list(diagnostic_mask) if diagnostic_mask is not None else [True] * n
    if len(diagnostic) != n:
        raise ValueError("diagnostic mask length differs from reference")
    forward = [[n + m + 1] * (m + 1) for _ in range(n + 1)]
    backward = [[n + m + 1] * (m + 1) for _ in range(n + 1)]
    forward[0][0] = 0
    for i in range(n + 1):
        for j in range(m + 1):
            value = forward[i][j]
            if i < n:
                forward[i + 1][j] = min(forward[i + 1][j], value + 1)
            if j < m:
                forward[i][j + 1] = min(forward[i][j + 1], value + 1)
            if i < n and j < m:
                forward[i + 1][j + 1] = min(
                    forward[i + 1][j + 1], value + int(observed[j] not in allowed[i]))
    backward[n][m] = 0
    for i in range(n, -1, -1):
        for j in range(m, -1, -1):
            if i < n:
                backward[i][j] = min(backward[i][j], 1 + backward[i + 1][j])
            if j < m:
                backward[i][j] = min(backward[i][j], 1 + backward[i][j + 1])
            if i < n and j < m:
                backward[i][j] = min(backward[i][j],
                    int(observed[j] not in allowed[i]) + backward[i + 1][j + 1])
    optimum = forward[n][m]

    def optimal(i: int, j: int, ni: int, nj: int, cost: int) -> bool:
        return forward[i][j] + cost + backward[ni][nj] == optimum

    possibilities: list[set[tuple[str, str | None]]] = [set() for _ in range(n)]
    entries: list[set[int]] = [set() for _ in range(n + 1)]
    exits: list[set[int]] = [set() for _ in range(n + 1)]
    insertion_edges: list[set[int]] = [set() for _ in range(n + 1)]
    entries[0].add(0)
    exits[n].add(m)
    for i in range(n + 1):
        for j in range(m + 1):
            if j < m and optimal(i, j, i, j + 1, 1):
                insertion_edges[i].add(j)
            if i < n and optimal(i, j, i + 1, j, 1):
                possibilities[i].add(("deletion", None))
                exits[i].add(j)
                entries[i + 1].add(j)
            if i < n and j < m:
                error = observed[j] not in allowed[i]
                if optimal(i, j, i + 1, j + 1, int(error)):
                    possibilities[i].add(("substitution" if error else "match", observed[j]))
                    exits[i].add(j)
                    entries[i + 1].add(j + 1)

    tokens: list[dict] = []
    for i, events in enumerate(possibilities):
        error_values = {event != "match" for event, _ in events}
        stable_error = diagnostic[i] and len(error_values) == 1
        stable_event = diagnostic[i] and len(events) == 1
        event, phone = next(iter(events)) if stable_event else ("alignment_uncertain", None)
        tokens.append({
            "phone_index": i, "canonical_phone": allowed[i][0],
            "acceptable_phones": list(allowed[i]), "diagnostic": bool(diagnostic[i]),
            "is_error": next(iter(error_values)) if stable_error else None,
            "stable_error": stable_error, "stable_event": stable_event,
            "event_type": event, "realized_phone": phone,
            "event_candidates": [{"event_type": e, "realized_phone": p}
                                 for e, p in sorted(events, key=lambda x: (x[0], x[1] or ""))],
        })
    gaps: list[dict] = []
    for i in range(n + 1):
        sequences: set[tuple[str, ...]] = set()
        for start in entries[i]:
            for end in exits[i]:
                if end >= start and all(j in insertion_edges[i] for j in range(start, end)):
                    sequences.add(tuple(observed[start:end]))
        if not sequences:
            raise RuntimeError("optimal alignment lacks a gap realization")
        gap_diagnostic = (i == 0 or diagnostic[i - 1]) and (i == n or diagnostic[i])
        ordinals: list[dict] = []
        for ordinal in range(max(map(len, sequences), default=0)):
            phones = {seq[ordinal] if ordinal < len(seq) else None for seq in sequences}
            stable = gap_diagnostic and len(phones) == 1
            phone = next(iter(phones)) if stable else None
            present_values = {p is not None for p in phones}
            ordinals.append({"ordinal": ordinal, "stable": stable,
                             "present": next(iter(present_values)) if gap_diagnostic and len(present_values) == 1 else None,
                             "phone": phone, "possible_phones": sorted(phones, key=lambda p: p or "")})
        stable = gap_diagnostic and len(sequences) == 1
        gaps.append({"gap_index": i, "diagnostic": bool(gap_diagnostic), "stable": stable,
                     "inserted_phones": list(next(iter(sequences))) if stable else None,
                     "possible_sequences": [list(s) for s in sorted(sequences)], "ordinals": ordinals})
    return {"tokens": tokens, "gaps": gaps, "edit_distance": optimum,
            "stable_error_count": sum(t["stable_error"] for t in tokens),
            "stable_event_count": sum(t["stable_event"] for t in tokens)}
