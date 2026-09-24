"""Parikh et al. (Interspeech 2025) PP-AF, on the shared ARPABET backend.

Paper: https://www.isca-archive.org/interspeech_2025/parikh25_interspeech.pdf
Author code: https://github.com/Aditya3107/GOP_MDD_Phonological

Equation 4 is log P(canonical | audio) minus the maximum log P of a
single-position replacement/deletion. These are full, unnormalized sequence
CTC probabilities, NOT topology-corrected likelihoods or length-normalized
losses. The original RPS model configuration uses ctc_loss_reduction='sum'.

This is a scoring-method transfer to the existing 39-phone backend, not a
claim to reproduce the authors' XLSR/MPC and SpeechOcean762 system numbers. UPS uses every
nonblank phone in this backend. RPS projects the authors' directed IPA graph
through the framework's frozen IPA/TIMIT mapping, unions collapsed source
rows, and removes duplicate and self edges. The missing HH/EY source rows
receive deletion only; no TORGO data are used to complete the graph.

The paper's deletion hypothesis permits an empty sequence for a one-phone
utterance. This implementation supports it by default. Set
include_single_phone_deletion=False to reproduce the public scripts' N>1
guard; an empty candidate set then produces NaN with no invented fallback.

PA-AF is deliberately not named or implemented here: the public PA-AF RPS
recurrence omits normal CTC skips and terminal-state restriction and permits
phone switches within a wildcard. Replacing it with an SD sum would not be
a faithful reproduction of that public implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .ctc import CounterfactualMatrix, DELETION_ID
from .phonology import ARPABET_39, IPA_TO_TIMIT, MODEL_PROJECTION


# Transcribed from PHONEME_CONFUSION_MAP in the authors' PPAF_RPS.py.
# It is directed: do not automatically add reciprocal or transitive edges.
AUTHOR_IPA_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "p": ("b", "m"), "b": ("p", "m"),
    "t": ("d", "ɾ", "n"), "d": ("t", "ɾ", "n"),
    "k": ("g", "ŋ"), "g": ("k", "ŋ"), "ʔ": (),
    "f": ("v",), "v": ("f",),
    "θ": ("ð", "f"), "ð": ("θ", "v", "d"),
    "s": ("z", "ʃ"), "z": ("s", "ʒ"),
    "ʃ": ("ʒ", "tʃ", "s"), "ʒ": ("ʃ", "dʒ"),
    "tʃ": ("ʃ", "dʒ"), "dʒ": ("ʒ", "tʃ"),
    "m": ("n", "p", "b"), "n": ("m", "ŋ", "ɾ"),
    "ŋ": ("n", "k", "g"),
    "ɹ": ("l", "w"), "l": ("ɹ", "ɫ"), "ɫ": ("o", "ʊ"),
    "j": ("dʒ",), "w": ("v",),
    "i": ("ɪ", "iː"), "ɪ": ("i", "e"),
    "e": ("ɛ", "ɪ"), "ɛ": ("æ", "e"),
    "æ": ("ɑ", "ɛ"), "ɑ": ("ɔ", "ʌ"),
    "ɔ": ("ɑ", "o"), "o": ("ɔ", "ʊ"),
    "u": ("ʊ", "o"), "ʊ": ("u", "o"),
    "ʌ": ("ə", "ɜ"), "ə": ("ʌ", "ɜ", "ɚ"),
    "ɜ": ("ə", "ɚ"), "ɚ": ("ɜ", "ə"),
    "aɪ": ("ɑ", "e"), "aʊ": ("æ", "ʌ"),
    "ɔɪ": ("ɔ", "ɪ"), "oʊ": ("o", "ʊ"),
    "ɾ": ("t", "d"), "n̩": ("n", "ən"),
}


def _project_ipa(symbol: str) -> str | None:
    # /o/ has no separate ARPABET39 monophthong: project to OW explicitly.
    # Syllabic n is folded like the existing PHN en -> N normalization.
    # The sequence /ən/ is not one replacement phone and is not truncated.
    special = {"o": "OW", "n̩": "N"}
    phone = special.get(symbol, IPA_TO_TIMIT.get(symbol))
    return MODEL_PROJECTION.get(phone, phone) if phone is not None else None


def restricted_phone_graph() -> dict[str, tuple[str, ...]]:
    """Return the frozen, projected directed graph (without deletion)."""

    edges: dict[str, set[str]] = {phone: set() for phone in ARPABET_39}
    for ipa_source, ipa_targets in AUTHOR_IPA_CONFUSIONS.items():
        source = _project_ipa(ipa_source)
        if source not in edges:
            continue
        for ipa_target in ipa_targets:
            target = _project_ipa(ipa_target)
            if target in edges and target != source:
                edges[source].add(target)
    return {phone: tuple(sorted(targets)) for phone, targets in edges.items()}


def graph_audit() -> dict[str, object]:
    """Small human-readable mapping audit; contains no data-derived state."""

    symbols = sorted(set(AUTHOR_IPA_CONFUSIONS).union(
        target for targets in AUTHOR_IPA_CONFUSIONS.values() for target in targets
    ))
    source_rows = {
        phone: [source for source in AUTHOR_IPA_CONFUSIONS if _project_ipa(source) == phone]
        for phone in ARPABET_39
    }
    self_edges = [
        {"source_ipa": source, "target_ipa": target, "model_phone": _project_ipa(source)}
        for source, targets in AUTHOR_IPA_CONFUSIONS.items()
        for target in targets
        if _project_ipa(source) is not None and _project_ipa(source) == _project_ipa(target)
    ]
    graph = restricted_phone_graph()
    return {
        "source": "https://github.com/Aditya3107/GOP_MDD_Phonological/blob/main/PPAF_RPS.py",
        "mapping": {symbol: _project_ipa(symbol) for symbol in symbols},
        "graph": {phone: list(targets) for phone, targets in graph.items()},
        "source_rows": source_rows,
        "missing_source_rows": [phone for phone, rows in source_rows.items() if not rows],
        "removed_self_edges": self_edges,
        "unmapped_symbols": [symbol for symbol in symbols if _project_ipa(symbol) is None],
        "projection_notes": [
            "Uses frozen IPA_TO_TIMIT then AX->AH and DX->T; e->EH is unchanged.",
            "o->OW is an explicit approximate projection; n̩->N folds syllabicity.",
            "Long/short IPA vowel variants share the existing ARPABET class.",
            "Merge source aliases by union; deduplicate targets and remove self edges.",
            "Drop glottal-stop source ʔ and multi-phone target ən; add no inferred edges.",
            "HH/EY have no author source row and therefore have deletion-only RPS candidates.",
            "UPS uses all 39 backend phones rather than the public script's incomplete IPA list.",
        ],
    }


def candidate_mask(
    candidate_ids: np.ndarray,
    canonical_ids: Sequence[int],
    id_to_phone: Mapping[int, str],
    *,
    restricted: bool = True,
    include_single_phone_deletion: bool = True,
) -> np.ndarray:
    """Mask a counterfactual matrix by fixed RPS or unrestricted UPS rules."""

    candidates = np.asarray(candidate_ids, dtype=np.int64)
    canonical = tuple(int(phone) for phone in canonical_ids)
    if candidates.ndim != 2 or candidates.shape[0] != len(canonical):
        raise ValueError("candidate rows must match the canonical sequence")
    graph = restricted_phone_graph()
    mask = np.zeros(candidates.shape, dtype=bool)
    for position, source_id in enumerate(canonical):
        source = id_to_phone[source_id]
        if source not in graph:
            raise ValueError(f"canonical phone {source!r} is outside ARPABET39")
        allowed = graph[source] if restricted else graph.keys()
        for column, target_id in enumerate(candidates[position]):
            if target_id == DELETION_ID:
                mask[position, column] = len(canonical) > 1 or include_single_phone_deletion
            elif target_id != source_id:
                mask[position, column] = id_to_phone[int(target_id)] in allowed
    return mask


def ppaf_scores(
    matrix: CounterfactualMatrix,
    canonical_ids: Sequence[int],
    id_to_phone: Mapping[int, str],
    *,
    restricted: bool,
    include_single_phone_deletion: bool = True,
) -> np.ndarray:
    """PP-AF Eq. 4, higher is better; receives no gold or patient metadata.

    Raw and externally sequence-recalibrated inputs use this identical rule.
    No deletion penalty, topology correction, prior weighting, or score
    calibration is applied. An impossible canonical sequence is an explicit
    input error. Rows with no feasible candidate are returned as NaN so a
    caller can report an exclusion instead of inventing a finite score.
    """

    if not np.isfinite(matrix.canonical_log_probability):
        raise ValueError("canonical sequence must have finite CTC probability")
    mask = candidate_mask(
        matrix.candidate_ids, canonical_ids, id_to_phone,
        restricted=restricted,
        include_single_phone_deletion=include_single_phone_deletion,
    )
    if not len(canonical_ids):
        return np.empty(0, dtype=np.float64)
    likelihoods = np.asarray(matrix.log_probabilities, dtype=np.float64)
    if np.isnan(likelihoods).any() or np.isposinf(likelihoods).any():
        raise ValueError("candidate CTC probabilities may contain -inf but not NaN or +inf")
    best = np.max(np.where(mask, likelihoods, -np.inf), axis=1)
    scores = np.full(len(canonical_ids), np.nan, dtype=np.float64)
    feasible = np.isfinite(best)
    scores[feasible] = matrix.canonical_log_probability - best[feasible]
    return scores
