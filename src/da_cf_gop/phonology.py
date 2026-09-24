"""Phone normalization and independent canonical--PHN alignment.

This module is intentionally self-contained.  In particular, it never imports
any of the historical ``phoneme_project/code*`` packages at runtime.  The
category-aware alignment costs are a documented migration of code5's frozen
Needleman--Wunsch implementation.
"""

from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Iterable, Sequence


# The official CTC-SF checkpoint has blank at id 0 and these 39 phones at
# ids 1..39.  Keeping the order here makes all downstream matrices auditable.
ARPABET_39: tuple[str, ...] = (
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G", "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW", "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
)
PHONE_TO_CTC_ID = {phone: index + 1 for index, phone in enumerate(ARPABET_39)}
CTC_ID_TO_PHONE = {index: phone for phone, index in PHONE_TO_CTC_ID.items()}
BLANK_ID = 0

# The intermediate layer deliberately retains AX and DX.  Only the final model
# layer applies the frozen 41-ish -> 39-phone projection required by the plan.
MODEL_PROJECTION = {"AX": "AH", "DX": "T"}
TIMIT_NORMALIZED_INVENTORY = frozenset((*ARPABET_39, "AX", "DX"))
VOWELS = frozenset({
    "AA", "AE", "AH", "AO", "AW", "AX", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
})


# eSpeak IPA -> intermediate phone layer.  Mapping to model space happens in
# ``project_model_phone`` so AX/DX behavior is explicit and testable.
IPA_TO_TIMIT: dict[str, str] = {
    "ɑ": "AA", "ɑː": "AA", "ɒ": "AA", "æ": "AE",
    "ʌ": "AH", "ɐ": "AH", "ɔ": "AO", "ɔː": "AO",
    "aʊ": "AW", "ə": "AX", "ɚ": "ER", "ɝ": "ER",
    "ɜ": "ER", "ɜː": "ER", "aɪ": "AY", "ɛ": "EH", "e": "EH",
    "eɪ": "EY", "ɪ": "IH", "ᵻ": "IH", "i": "IY", "iː": "IY",
    "oʊ": "OW", "əʊ": "OW", "ɔɪ": "OY", "ʊ": "UH",
    "u": "UW", "uː": "UW", "b": "B", "tʃ": "CH", "d": "D",
    "ð": "DH", "ɾ": "DX", "f": "F", "g": "G", "ɡ": "G",
    "h": "HH", "dʒ": "JH", "k": "K", "l": "L", "ɫ": "L",
    "m": "M", "n": "N", "ŋ": "NG", "p": "P", "r": "R",
    "ɹ": "R", "s": "S", "ʃ": "SH", "t": "T", "θ": "TH",
    "v": "V", "w": "W", "j": "Y", "z": "Z", "ʒ": "ZH",
}
IPA_TO_TIMIT_SEQUENCE: dict[str, tuple[str, ...]] = {
    "ɔːɹ": ("AO", "R"), "ɛɹ": ("EH", "R"),
    "əl": ("AX", "L"), "ʊɹ": ("UH", "R"),
    "ɑːɹ": ("AA", "R"), "ɪɹ": ("IH", "R"),
    "aɪɚ": ("AY", "ER"), "iə": ("IY", "AX"),
    "aɪə": ("AY", "AX"), "ʔ": ("T",),
}
_COMBINING_DROP = frozenset({"̩", "̯", "̥", "̬", "ʰ", "ˈ", "ˌ"})
_INVALID_PROMPTS = frozenset({"", "xxx", "noise", "silence"})


# TIMIT/TORGO allophone normalization.  Standard non-speech, closure, noise,
# and glottal-stop symbols are known drops; an unknown symbol is an error rather
# than being silently treated as silence.
PHN_DROP = frozenset({
    "h#", "pau", "sil", "sp", "spn", "epi", "q", "noi", "noise",
    "bcl", "dcl", "gcl", "kcl", "pcl", "tcl", "vcl", "eng-cl",
    "xx", "**", "#", "@",
})
PHN_ALLOPHONE_MAP: dict[str, str] = {
    "ax-h": "AX", "axr": "ER", "ix": "IH", "ux": "UW",
    "el": "L", "em": "M", "en": "N", "eng": "NG", "nx": "N",
    "hv": "HH",
}


@dataclass(frozen=True)
class PhnSegment:
    """One syntactically valid PHN line, including known dropped symbols."""

    index: int
    start_sample: int
    end_sample: int
    raw_label: str
    timit_phone: str | None
    model_phone: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PhnTranscript:
    """The three required views of a TORGO PHN transcription."""

    segments: tuple[PhnSegment, ...]

    @property
    def raw_labels(self) -> tuple[str, ...]:
        return tuple(segment.raw_label for segment in self.segments)

    @property
    def timit_phones(self) -> tuple[str, ...]:
        return tuple(
            segment.timit_phone for segment in self.segments
            if segment.timit_phone is not None
        )

    @property
    def model_phones(self) -> tuple[str, ...]:
        return tuple(
            segment.model_phone for segment in self.segments
            if segment.model_phone is not None
        )

    @property
    def dropped_count(self) -> int:
        return sum(segment.timit_phone is None for segment in self.segments)


@dataclass(frozen=True)
class StableTokenLabel:
    """Consensus label for one canonical position from two independent DPs."""

    phone_index: int
    canonical_phone: str
    event_type: str
    realized_phone: str | None
    stable: bool
    levenshtein_observed_index: int | None
    category_observed_index: int | None
    levenshtein_event_type: str
    category_event_type: str
    levenshtein_realized_phone: str | None
    category_realized_phone: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "StableTokenLabel":
        return cls(**value)  # type: ignore[arg-type]


@dataclass(frozen=True)
class StableAlignment:
    tokens: tuple[StableTokenLabel, ...]
    levenshtein_alignment: tuple[tuple[int | None, int | None], ...]
    category_alignment: tuple[tuple[int | None, int | None], ...]
    levenshtein_insertions: int
    category_insertions: int
    agreed_insertions: int
    uncertain_count: int
    stable_count: int
    stable_fraction: float
    exact_identification_headline_allowed: bool

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def normalize_prompt(prompt: str) -> str:
    return " ".join((prompt or "").strip().split())


def validate_prompt(prompt: str) -> tuple[bool, str]:
    text = normalize_prompt(prompt)
    lower = text.lower()
    if lower in _INVALID_PROMPTS:
        return False, "empty_or_noise"
    if text.startswith("["):
        return False, "elicitation_instruction"
    if any(suffix in lower for suffix in (".jpg", ".jpeg", ".png")):
        return False, "picture_prompt"
    if not re.search(r"[A-Za-z]", text):
        return False, "no_english_letters"
    return True, ""


def clean_ipa_token(token: str) -> str:
    normalized = unicodedata.normalize("NFC", token.strip())
    return "".join(char for char in normalized if char not in _COMBINING_DROP)


def project_model_phone(phone: str) -> str:
    upper = phone.upper()
    projected = MODEL_PROJECTION.get(upper, upper)
    if projected not in ARPABET_39:
        raise ValueError(f"phone {phone!r} does not project into ARPABET-39")
    return projected


def ipa_token_to_timit(token: str) -> tuple[str, ...] | None:
    cleaned = clean_ipa_token(token)
    if cleaned in IPA_TO_TIMIT_SEQUENCE:
        return IPA_TO_TIMIT_SEQUENCE[cleaned]
    if cleaned in IPA_TO_TIMIT:
        return (IPA_TO_TIMIT[cleaned],)
    upper = cleaned.upper()
    if upper in TIMIT_NORMALIZED_INVENTORY:
        return (upper,)
    if cleaned.endswith("ː") and cleaned[:-1] in IPA_TO_TIMIT:
        return (IPA_TO_TIMIT[cleaned[:-1]],)
    return None


def ipa_rows_to_phones(ipa_rows: Sequence[str]) -> tuple[list[list[str]], list[list[str]]]:
    """Map already-phonemized IPA rows to the frozen model phone space."""

    phone_rows: list[list[str]] = []
    unknown_rows: list[list[str]] = []
    for ipa in ipa_rows:
        phones: list[str] = []
        unknown: list[str] = []
        for raw in ipa.replace("|", " ").split():
            mapped = ipa_token_to_timit(raw)
            if mapped is None:
                unknown.append(clean_ipa_token(raw))
            else:
                phones.extend(project_model_phone(phone) for phone in mapped)
        phone_rows.append(phones)
        unknown_rows.append(unknown)
    return phone_rows, unknown_rows


def _bundled_espeak_paths() -> tuple[Path, Path]:
    import espeakng_loader

    return (
        Path(espeakng_loader.get_library_path()).resolve(),
        Path(espeakng_loader.get_data_path()).resolve(),
    )


def _setup_espeak() -> None:
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    library_path, data_path = _bundled_espeak_paths()
    # An ambient eSpeak data directory must never alter the canonical phones.
    # Reassert both bundled paths on every call so an environment mutation
    # after an earlier invocation cannot silently change the G2P backend.
    os.environ["ESPEAK_DATA_PATH"] = str(data_path)
    EspeakWrapper.set_library(str(library_path))


@lru_cache(maxsize=1)
def g2p_provenance() -> dict[str, object]:
    """Describe and content-bind the frozen local eSpeak G2P runtime."""

    _setup_espeak()
    from phonemizer.backend.espeak.wrapper import EspeakWrapper

    # Imported lazily so PHN-only utilities do not pay for directory hashing.
    from .backend import sha256_path

    library_path, data_path = _bundled_espeak_paths()
    runtime_version = ".".join(str(value) for value in EspeakWrapper().version)
    body: dict[str, object] = {
        "backend": "phonemizer_espeak_bundled",
        "language": "en-us",
        "phonemizer_version": distribution_version("phonemizer"),
        "espeakng_loader_version": distribution_version("espeakng-loader"),
        "espeak_runtime_version": runtime_version,
        "espeak_library_path": library_path.as_posix(),
        "espeak_library_sha256": sha256_path(library_path),
        "espeak_data_path": data_path.as_posix(),
        "espeak_data_sha256": sha256_path(data_path),
    }
    payload = json.dumps(
        body,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    body["descriptor_sha256"] = sha256(payload).hexdigest()
    return body


def _phonemize(texts: Sequence[str]) -> list[str]:
    _setup_espeak()
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    result = phonemize(
        list(texts), language="en-us", backend="espeak",
        separator=Separator(phone=" ", word=" | ", syllable=""),
        strip=True, preserve_punctuation=False, with_stress=False, njobs=1,
    )
    return [result] if isinstance(result, str) else list(result)


def texts_to_phones(texts: Sequence[str]) -> tuple[list[list[str]], list[list[str]]]:
    """Run frozen local eSpeak and return ARPABET-39 rows plus unknown IPA."""

    if not texts:
        return [], []
    return ipa_rows_to_phones(_phonemize(texts))


def text_to_phones(text: str) -> tuple[list[str], list[str]]:
    rows, unknown = texts_to_phones([text])
    return rows[0], unknown[0]


def normalize_phn_label(label: str, *, strict: bool = True) -> str | None:
    """Return the intermediate TIMIT-normalized phone for a raw PHN label."""

    raw = label.strip().lower()
    unstressed = re.sub(r"[0-9]+$", "", raw)
    if unstressed in PHN_DROP:
        return None
    normalized = PHN_ALLOPHONE_MAP.get(unstressed, unstressed.upper())
    if normalized in TIMIT_NORMALIZED_INVENTORY:
        return normalized
    if strict:
        raise ValueError(f"unmapped PHN label {label!r}")
    return None


def read_phn(path: str | Path, *, strict: bool = True) -> PhnTranscript:
    """Read PHN without exposing its boundaries to the acoustic scorer."""

    segments: list[PhnSegment] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8", errors="replace").splitlines(), 1
    ):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) < 3:
            if strict:
                raise ValueError(f"invalid PHN line {line_number} in {path}")
            continue
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError as exc:
            if strict:
                raise ValueError(f"invalid PHN bounds on line {line_number} in {path}") from exc
            continue
        if start < 0 or end <= start:
            if strict:
                raise ValueError(f"non-positive PHN interval on line {line_number} in {path}")
            continue
        raw_label = parts[2].strip()
        timit = normalize_phn_label(raw_label, strict=strict)
        model = project_model_phone(timit) if timit is not None else None
        segments.append(PhnSegment(
            index=len(segments), start_sample=start, end_sample=end,
            raw_label=raw_label, timit_phone=timit, model_phone=model,
        ))
    if strict and not segments:
        raise ValueError(f"empty PHN transcription: {path}")
    return PhnTranscript(tuple(segments))


def _validate_phone_sequence(sequence: Sequence[str], name: str) -> tuple[str, ...]:
    phones = tuple(str(phone).upper() for phone in sequence)
    unknown = sorted(set(phones) - TIMIT_NORMALIZED_INVENTORY)
    if unknown:
        raise ValueError(f"{name} contains unknown phones: {unknown}")
    return phones


def align_levenshtein(
    reference: Sequence[str], observed: Sequence[str]
) -> list[tuple[int | None, int | None]]:
    """Unit-cost global alignment with diagonal/delete/insert tie-breaking."""

    ref = _validate_phone_sequence(reference, "reference")
    obs = _validate_phone_sequence(observed, "observed")
    return _global_align(ref, obs, lambda left, right: 0.0 if left == right else 1.0)


def category_substitution_cost(target: str, actual: str) -> float:
    """Frozen code5 substitution cost, including its original branch order."""

    target, actual = target.upper(), actual.upper()
    if target == actual:
        return 0.0
    if frozenset((target, actual)) in {
        frozenset(("AX", "AH")), frozenset(("DX", "T")),
        frozenset(("DX", "D")),
    }:
        return 0.15
    if (target in VOWELS) == (actual in VOWELS):
        return 1.0
    if (target, actual) in {
        ("Y", "IY"), ("IY", "Y"), ("W", "UW"), ("UW", "W"),
    }:
        return 0.6
    return 2.2


def align_category_aware(
    reference: Sequence[str], observed: Sequence[str]
) -> list[tuple[int | None, int | None]]:
    """Code5-compatible category-aware Needleman--Wunsch alignment."""

    ref = _validate_phone_sequence(reference, "reference")
    obs = _validate_phone_sequence(observed, "observed")
    return _global_align(ref, obs, category_substitution_cost)


# Compatibility name used in historical reports; implementation remains local.
align_sequences = align_category_aware


def _global_align(
    reference: tuple[str, ...],
    observed: tuple[str, ...],
    substitution_cost,
) -> list[tuple[int | None, int | None]]:
    n, m = len(reference), len(observed)
    cost = [[math.inf] * (m + 1) for _ in range(n + 1)]
    back = [[0] * (m + 1) for _ in range(n + 1)]
    cost[0][0] = 0.0
    for i in range(1, n + 1):
        cost[i][0], back[i][0] = float(i), 2
    for j in range(1, m + 1):
        cost[0][j], back[0][j] = float(j), 3
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                cost[i - 1][j - 1] + substitution_cost(reference[i - 1], observed[j - 1]),
                cost[i - 1][j] + 1.0,
                cost[i][j - 1] + 1.0,
            )
            # ``min`` chooses the first minimum: diagonal, deletion, insertion.
            direction = min(range(3), key=choices.__getitem__) + 1
            cost[i][j] = choices[direction - 1]
            back[i][j] = direction
    alignment: list[tuple[int | None, int | None]] = []
    i, j = n, m
    while i or j:
        direction = back[i][j]
        if direction == 1:
            alignment.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif direction == 2:
            alignment.append((i - 1, None))
            i -= 1
        elif direction == 3:
            alignment.append((None, j - 1))
            j -= 1
        else:  # pragma: no cover - defensive corruption guard
            raise RuntimeError(f"invalid alignment backpointer at {(i, j)}")
    alignment.reverse()
    return alignment


def _canonical_events(
    reference: tuple[str, ...],
    observed: tuple[str, ...],
    alignment: Sequence[tuple[int | None, int | None]],
) -> tuple[dict[int, tuple[str, str | None, int | None]], set[int]]:
    events: dict[int, tuple[str, str | None, int | None]] = {}
    insertions: set[int] = set()
    for reference_index, observed_index in alignment:
        if reference_index is None:
            if observed_index is not None:
                insertions.add(observed_index)
            continue
        if reference_index in events:
            raise ValueError(f"canonical position aligned twice: {reference_index}")
        if observed_index is None:
            events[reference_index] = ("deletion", None, None)
        else:
            realized = observed[observed_index]
            event = "match" if realized == reference[reference_index] else "substitution"
            events[reference_index] = (event, realized, observed_index)
    if set(events) != set(range(len(reference))):
        raise ValueError("alignment does not cover every canonical position")
    return events, insertions


def derive_stable_token_labels(
    canonical_phones: Sequence[str],
    realized_phones: Sequence[str],
    *,
    minimum_stable_fraction: float = 0.80,
) -> StableAlignment:
    """Consensus token labels from unit and category-aware global alignment.

    A token is stable only when both algorithms agree on event type and mapped
    realized phone.  Insertion positions are reported separately and never
    become canonical-token headline labels.
    """

    if not 0.0 <= minimum_stable_fraction <= 1.0:
        raise ValueError("minimum_stable_fraction must lie in [0, 1]")
    reference = tuple(project_model_phone(phone) for phone in canonical_phones)
    observed = tuple(project_model_phone(phone) for phone in realized_phones)
    lev = tuple(align_levenshtein(reference, observed))
    category = tuple(align_category_aware(reference, observed))
    lev_events, lev_insertions = _canonical_events(reference, observed, lev)
    cat_events, cat_insertions = _canonical_events(reference, observed, category)

    tokens: list[StableTokenLabel] = []
    for index, canonical in enumerate(reference):
        lev_event, lev_phone, lev_observed_index = lev_events[index]
        cat_event, cat_phone, cat_observed_index = cat_events[index]
        stable = lev_event == cat_event and lev_phone == cat_phone
        tokens.append(StableTokenLabel(
            phone_index=index,
            canonical_phone=canonical,
            event_type=lev_event if stable else "alignment_uncertain",
            realized_phone=lev_phone if stable else None,
            stable=stable,
            levenshtein_observed_index=lev_observed_index,
            category_observed_index=cat_observed_index,
            levenshtein_event_type=lev_event,
            category_event_type=cat_event,
            levenshtein_realized_phone=lev_phone,
            category_realized_phone=cat_phone,
        ))
    stable_count = sum(token.stable for token in tokens)
    stable_fraction = stable_count / len(tokens) if tokens else 0.0
    return StableAlignment(
        tokens=tuple(tokens),
        levenshtein_alignment=lev,
        category_alignment=category,
        levenshtein_insertions=len(lev_insertions),
        category_insertions=len(cat_insertions),
        agreed_insertions=len(lev_insertions & cat_insertions),
        uncertain_count=len(tokens) - stable_count,
        stable_count=stable_count,
        stable_fraction=stable_fraction,
        exact_identification_headline_allowed=(
            bool(tokens) and stable_fraction >= minimum_stable_fraction
        ),
    )


def phone_counts(rows: Iterable[Sequence[str]]) -> dict[str, int]:
    counts = {phone: 0 for phone in ARPABET_39}
    for row in rows:
        for phone in row:
            model_phone = project_model_phone(phone)
            counts[model_phone] += 1
    return counts
