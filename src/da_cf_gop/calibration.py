"""Speaker-disjoint sequence, deletion and probability calibration."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .ctc import ctc_minimum_frames
from .provenance import write_deterministic_npz


@dataclass(frozen=True)
class CalibrationUtterance:
    utterance_id: str
    speaker_id: str
    logits: np.ndarray
    realized_ids: tuple[int, ...]


def _coerce_utterance(value: Any, index: int) -> CalibrationUtterance:
    if isinstance(value, CalibrationUtterance):
        return value
    if isinstance(value, Mapping):
        return CalibrationUtterance(
            str(value.get("utterance_id", f"utt_{index}")),
            str(value["speaker_id"]),
            np.asarray(value["logits"]),
            tuple(int(item) for item in value.get("realized_ids", value.get("target_ids", ()))),
        )
    if isinstance(value, (tuple, list)) and len(value) in {3, 4}:
        if len(value) == 3:
            logits, labels, speaker = value
            utterance_id = f"utt_{index}"
        else:
            utterance_id, speaker, logits, labels = value
        return CalibrationUtterance(
            str(utterance_id), str(speaker), np.asarray(logits), tuple(int(item) for item in labels)
        )
    raise TypeError("calibration utterances must be dataclasses, mappings, or 3/4-tuples")


def speaker_equal_sample(
    utterances: Sequence[CalibrationUtterance],
    max_utterances: int = 500,
    *,
    seed: int = 20260829,
) -> list[CalibrationUtterance]:
    """Deterministically cap data by round-robin sampling speakers."""

    if max_utterances <= 0:
        raise ValueError("max_utterances must be positive")
    grouped: defaultdict[str, list[CalibrationUtterance]] = defaultdict(list)
    for utterance in utterances:
        if not utterance.speaker_id:
            raise ValueError("speaker_id is required for speaker-equal calibration")
        grouped[utterance.speaker_id].append(utterance)
    rng = np.random.default_rng(seed)
    queues = {
        speaker: [rows[index] for index in rng.permutation(len(rows))]
        for speaker, rows in sorted(grouped.items())
    }
    selected: list[CalibrationUtterance] = []
    offsets = {speaker: 0 for speaker in queues}
    while len(selected) < min(max_utterances, len(utterances)):
        advanced = False
        for speaker in sorted(queues):
            offset = offsets[speaker]
            if offset < len(queues[speaker]) and len(selected) < max_utterances:
                selected.append(queues[speaker][offset])
                offsets[speaker] += 1
                advanced = True
        if not advanced:
            break
    return selected


def posterior_entropy(logits: np.ndarray) -> float:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or not np.isfinite(values).all():
        raise ValueError("logits must be a non-empty finite matrix")
    shifted = values - values.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return float(np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-300, 1.0)), axis=1)))


def assert_safe_entropy_shift(
    before: float,
    after: float,
    *,
    max_increase: float = 0.25,
    max_ratio: float = 4.0,
) -> None:
    """Fail the fold if recalibration repeats the old entropy-flattening bug."""

    if not np.isfinite(before) or not np.isfinite(after) or before < 0 or after < 0:
        raise ValueError("entropy values must be finite and non-negative")
    ratio = after / max(before, np.finfo(float).eps)
    if after - before > max_increase or ratio >= max_ratio:
        raise RuntimeError(
            f"unsafe posterior entropy shift: {before:.6f} -> {after:.6f} "
            f"(increase={after-before:.6f}, ratio={ratio:.3f})"
        )


@dataclass(frozen=True)
class CTCSequenceRecalibrator:
    """Identity-initialized linear map fitted with the sequence CTC objective."""

    weight: np.ndarray
    bias: np.ndarray
    blank_id: int
    training_summary: dict[str, Any] = field(default_factory=dict)

    @property
    def vocab_size(self) -> int:
        return int(self.weight.shape[0])

    @classmethod
    def identity(cls, vocab_size: int, blank_id: int = 0) -> "CTCSequenceRecalibrator":
        if vocab_size <= 1 or not 0 <= blank_id < vocab_size:
            raise ValueError("invalid vocabulary size or blank id")
        return cls(np.eye(vocab_size), np.zeros(vocab_size), int(blank_id), {"status": "identity"})

    @classmethod
    def fit(
        cls,
        utterances: Iterable[Any],
        *,
        blank_id: int = 0,
        steps: int = 100,
        learning_rate: float = 0.02,
        l2_identity: float = 1.0,
        max_utterances: int = 500,
        batch_size: int = 32,
        seed: int = 20260829,
        device: str = "auto",
        enforce_entropy_guard: bool = True,
    ) -> "CTCSequenceRecalibrator":
        """Fit on training speakers only, excluding impossible CTC sequences.

        Each speaker's utterance losses sum to the same weight.  The optimizer
        uses ``zero_infinity=False`` and therefore cannot silently accept an
        impossible sequence.
        """

        import torch
        import torch.nn.functional as functional

        if steps < 0 or learning_rate <= 0 or l2_identity < 0 or batch_size <= 0:
            raise ValueError("invalid recalibration hyperparameters")
        rows = [_coerce_utterance(value, index) for index, value in enumerate(utterances)]
        if not rows:
            raise ValueError("at least one calibration utterance is required")
        vocab_sizes = {
            int(np.asarray(row.logits).shape[1])
            for row in rows
            if np.asarray(row.logits).ndim == 2
        }
        if len(vocab_sizes) != 1:
            raise ValueError("all calibration logits must share one two-dimensional vocabulary")
        vocab_size = vocab_sizes.pop()
        if vocab_size <= 1 or not 0 <= blank_id < vocab_size:
            raise ValueError("invalid dynamic vocabulary or blank id")

        valid: list[CalibrationUtterance] = []
        excluded: defaultdict[str, int] = defaultdict(int)
        for row in rows:
            logits = np.asarray(row.logits, dtype=np.float32)
            labels = list(row.realized_ids)
            if logits.ndim != 2 or logits.shape[1] != vocab_size or logits.shape[0] == 0:
                raise ValueError(f"invalid logits for {row.utterance_id}")
            if not np.isfinite(logits).all():
                raise ValueError(f"non-finite logits for {row.utterance_id}")
            if not labels:
                excluded["empty_realized_sequence"] += 1
                continue
            if blank_id in labels or any(label < 0 or label >= vocab_size for label in labels):
                raise ValueError(f"invalid realized sequence for {row.utterance_id}")
            if logits.shape[0] < ctc_minimum_frames(labels):
                excluded["ctc_impossible_sequence"] += 1
                continue
            valid.append(
                CalibrationUtterance(row.utterance_id, row.speaker_id, logits, tuple(labels))
            )
        if not valid:
            raise ValueError("no CTC-feasible calibration utterance remains")
        selected = speaker_equal_sample(valid, max_utterances, seed=seed)
        speaker_counts = Counter(row.speaker_id for row in selected)
        n_speakers = len(speaker_counts)

        selected_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else ("cpu" if device == "auto" else device)
        )
        torch_device = torch.device(selected_device)
        weight = torch.eye(vocab_size, device=torch_device, dtype=torch.float32, requires_grad=True)
        bias = torch.zeros(vocab_size, device=torch_device, dtype=torch.float32, requires_grad=True)
        identity = torch.eye(vocab_size, device=torch_device, dtype=torch.float32)
        optimizer = torch.optim.Adam((weight, bias), lr=learning_rate)
        rng = np.random.default_rng(seed)

        for _step in range(steps):
            optimizer.zero_grad()
            order = rng.permutation(len(selected))
            epoch_loss_value = 0.0
            for offset in range(0, len(order), batch_size):
                indices = order[offset : offset + batch_size]
                subset = [selected[int(index)] for index in indices]
                maximum_frames = max(row.logits.shape[0] for row in subset)
                recalibrated = []
                input_lengths = []
                targets: list[int] = []
                target_lengths = []
                sample_weights = []
                for row in subset:
                    tensor = torch.as_tensor(row.logits, device=torch_device)
                    transformed = tensor @ weight.T + bias
                    transformed = functional.log_softmax(transformed, dim=1)
                    padding = maximum_frames - transformed.shape[0]
                    if padding:
                        transformed = functional.pad(transformed, (0, 0, 0, padding))
                    recalibrated.append(transformed)
                    input_lengths.append(row.logits.shape[0])
                    targets.extend(row.realized_ids)
                    target_lengths.append(len(row.realized_ids))
                    sample_weights.append(1.0 / (n_speakers * speaker_counts[row.speaker_id]))
                batch_log_probs = torch.stack(recalibrated, dim=1)
                losses = functional.ctc_loss(
                    batch_log_probs,
                    torch.tensor(targets, dtype=torch.long, device=torch_device),
                    torch.tensor(input_lengths, dtype=torch.long, device=torch_device),
                    torch.tensor(target_lengths, dtype=torch.long, device=torch_device),
                    blank=int(blank_id),
                    reduction="none",
                    zero_infinity=False,
                )
                if not bool(torch.isfinite(losses).all()):
                    raise RuntimeError("non-finite CTC recalibration loss; fold is invalid")
                batch_weight = torch.tensor(sample_weights, dtype=losses.dtype, device=torch_device)
                loss = torch.sum(losses * batch_weight)
                loss.backward()
                epoch_loss_value += float(loss.detach())
            regularizer = l2_identity * torch.sum((weight - identity) ** 2)
            regularizer.backward()
            optimizer.step()

        fitted = cls(
            weight.detach().cpu().numpy().astype(np.float64),
            bias.detach().cpu().numpy().astype(np.float64),
            int(blank_id),
            {
                "status": "fitted",
                "vocab_size": vocab_size,
                "steps": steps,
                "learning_rate": learning_rate,
                "l2_identity": l2_identity,
                "selected_utterances": len(selected),
                "training_speakers": sorted(speaker_counts),
                "selected_per_speaker": dict(sorted(speaker_counts.items())),
                "excluded": dict(sorted(excluded.items())),
                "zero_infinity": False,
                "seed": seed,
            },
        )
        before_entropy = float(np.mean([posterior_entropy(row.logits) for row in selected]))
        after_entropy = float(np.mean([posterior_entropy(fitted.apply(row.logits)) for row in selected]))
        if enforce_entropy_guard:
            assert_safe_entropy_shift(before_entropy, after_entropy)
        summary = dict(fitted.training_summary)
        summary.update(
            {
                "entropy_before": before_entropy,
                "entropy_after": after_entropy,
                "entropy_guard_passed": True if enforce_entropy_guard else None,
            }
        )
        return cls(fitted.weight, fitted.bias, fitted.blank_id, summary)

    def apply(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.vocab_size or not np.isfinite(values).all():
            raise ValueError("logits do not match this recalibrator")
        return values @ self.weight.T + self.bias

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        import json

        write_deterministic_npz(
            destination,
            {
            "weight": np.asarray(self.weight, dtype=np.float64),
            "bias": np.asarray(self.bias, dtype=np.float64),
            "blank_id": np.asarray(self.blank_id, dtype=np.int64),
            "training_summary_json": np.asarray(json.dumps(self.training_summary, sort_keys=True)),
            },
        )
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "CTCSequenceRecalibrator":
        import json

        with np.load(path, allow_pickle=False) as archive:
            return cls(
                archive["weight"].astype(np.float64, copy=True),
                archive["bias"].astype(np.float64, copy=True),
                int(archive["blank_id"].item()),
                json.loads(str(archive["training_summary_json"].item())),
            )


def _weighted_quantile(values: Sequence[float], weights: Sequence[float], quantile: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    if array.ndim != 1 or weight.shape != array.shape or not len(array):
        raise ValueError("weighted quantile needs same-length non-empty vectors")
    if not np.isfinite(array).all() or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError("weighted quantile inputs must be finite with positive weights")
    order = np.argsort(array, kind="stable")
    array, weight = array[order], weight[order]
    cumulative = np.cumsum(weight) / np.sum(weight)
    return float(array[min(int(np.searchsorted(cumulative, quantile, side="left")), len(array) - 1)])


def _row_field(row: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(row, Mapping) and name in row:
            return row[name]
        if hasattr(row, name):
            return getattr(row, name)
    return default


@dataclass(frozen=True)
class DeletionCalibrator:
    per_phone: dict[str, float]
    global_penalty: float
    quantile: float = 0.9
    min_tokens: int = 20
    min_speakers: int = 2

    @classmethod
    def fit(
        cls,
        tokens: Iterable[Any],
        *,
        quantile: float = 0.9,
        min_tokens: int = 20,
        min_speakers: int = 2,
    ) -> "DeletionCalibrator":
        """Fit q90 of deletion advantage on PHN-correct training tokens."""

        if not 0.0 <= quantile <= 1.0 or min_tokens <= 0 or min_speakers <= 0:
            raise ValueError("invalid deletion calibration hyperparameters")
        rows: list[tuple[str, str, float]] = []
        for row in tokens:
            target = str(_row_field(row, "target_phone", "target", default="")).upper()
            speaker = str(_row_field(row, "speaker_id", "speaker", default=""))
            canonical = _row_field(row, "canonical_log_probability", "canonical_lp")
            deletion = _row_field(row, "deletion_log_probability", "del_lp")
            is_error = _row_field(row, "is_error", "error", default=None)
            if is_error is None:
                realized = str(_row_field(row, "realized_phone", "realized", default=target)).upper()
                is_error = realized != target
            if bool(is_error):
                continue
            if not target or not speaker or canonical is None or deletion is None:
                continue
            advantage = float(deletion) - float(canonical)
            if np.isfinite(advantage):
                rows.append((target, speaker, advantage))
        if not rows:
            raise ValueError("no finite PHN-correct token is available for deletion calibration")

        global_counts = Counter(speaker for _target, speaker, _value in rows)
        global_values = [value for _target, _speaker, value in rows]
        global_weights = [1.0 / global_counts[speaker] for _target, speaker, _value in rows]
        global_penalty = _weighted_quantile(global_values, global_weights, quantile)

        by_phone: defaultdict[str, list[tuple[str, float]]] = defaultdict(list)
        for target, speaker, advantage in rows:
            by_phone[target].append((speaker, advantage))
        per_phone: dict[str, float] = {}
        for target, local in by_phone.items():
            local_speakers = Counter(speaker for speaker, _value in local)
            if len(local) < min_tokens or len(local_speakers) < min_speakers:
                continue
            per_phone[target] = _weighted_quantile(
                [value for _speaker, value in local],
                [1.0 / local_speakers[speaker] for speaker, _value in local],
                quantile,
            )
        return cls(per_phone, global_penalty, quantile, min_tokens, min_speakers)

    def penalty(self, target_phone: str) -> float:
        return float(self.per_phone.get(str(target_phone).upper(), self.global_penalty))

    def calibrate(self, target_phone: str, deletion_log_probability: float) -> float:
        return float(deletion_log_probability) - self.penalty(target_phone)


def speaker_equal_weights(speakers: Sequence[str]) -> np.ndarray:
    speaker_array = np.asarray(speakers, dtype=str)
    if speaker_array.ndim != 1 or len(speaker_array) == 0 or np.any(speaker_array == ""):
        raise ValueError("speakers must be a non-empty vector of ids")
    counts = Counter(speaker_array.tolist())
    weights = np.asarray([1.0 / counts[speaker] for speaker in speaker_array], dtype=np.float64)
    # Scaling does not change the speaker equality but makes C=1 comparable to
    # the conventional sklearn interpretation with mean sample weight one.
    return weights * (len(weights) / weights.sum())


@dataclass(frozen=True)
class SpeakerWeightedPlattCalibrator:
    coefficient: float
    intercept: float
    c_value: float = 1.0

    @classmethod
    def fit(
        cls,
        error_evidence: Sequence[float],
        labels: Sequence[int],
        speakers: Sequence[str],
        *,
        c_value: float = 1.0,
    ) -> "SpeakerWeightedPlattCalibrator":
        """Fit p(error) from ``-GoP`` with speaker-equal sample weights."""

        from sklearn.linear_model import LogisticRegression

        evidence = np.asarray(error_evidence, dtype=np.float64)
        truth = np.asarray(labels, dtype=np.int64)
        if evidence.ndim != 1 or truth.shape != evidence.shape or len(speakers) != len(evidence):
            raise ValueError("evidence, labels and speakers must be same-length vectors")
        if not np.isfinite(evidence).all() or set(np.unique(truth)) != {0, 1}:
            raise ValueError("Platt fitting requires finite evidence and both binary classes")
        if c_value <= 0:
            raise ValueError("C must be positive")
        model = LogisticRegression(
            C=float(c_value), penalty="l2", solver="lbfgs", class_weight=None, random_state=0
        )
        model.fit(evidence[:, None], truth, sample_weight=speaker_equal_weights(speakers))
        return cls(float(model.coef_[0, 0]), float(model.intercept_[0]), float(c_value))

    def predict_error_probability(self, error_evidence: Sequence[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(error_evidence, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("error evidence must be finite")
        logits = np.clip(self.coefficient * values + self.intercept, -709.0, 709.0)
        return 1.0 / (1.0 + np.exp(-logits))


def conditional_softmax(log_scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    values = np.asarray(log_scores, dtype=np.float64)
    if values.ndim not in {1, 2} or values.shape[-1] == 0:
        raise ValueError("alternative scores must be a non-empty vector or matrix")
    if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any(axis=-1).all():
        raise ValueError("each row needs finite or -inf scores and no NaN/+inf")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    scaled = values / float(temperature)
    maximum = np.max(scaled, axis=-1, keepdims=True)
    probabilities = np.exp(scaled - maximum)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities


@dataclass(frozen=True)
class AlternativeTemperatureCalibrator:
    temperature: float
    nll_by_temperature: dict[float, float]

    @classmethod
    def fit(
        cls,
        alternative_log_scores: np.ndarray,
        gold_indices: Sequence[int],
        speakers: Sequence[str],
        *,
        grid: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    ) -> "AlternativeTemperatureCalibrator":
        """Choose temperature by inner-OOF speaker-equal conditional NLL."""

        scores = np.asarray(alternative_log_scores, dtype=np.float64)
        gold = np.asarray(gold_indices, dtype=np.int64)
        if scores.ndim != 2 or scores.shape[0] != len(gold) or len(speakers) != len(gold):
            raise ValueError("scores, gold indices and speakers have inconsistent shapes")
        if np.any(gold < 0) or np.any(gold >= scores.shape[1]):
            raise ValueError("gold alternative index is outside the candidate matrix")
        temperatures = tuple(float(value) for value in grid)
        if not temperatures or len(set(temperatures)) != len(temperatures) or any(value <= 0 for value in temperatures):
            raise ValueError("temperature grid must contain unique positive values")
        weights = speaker_equal_weights(speakers)
        nll: dict[float, float] = {}
        for temperature in temperatures:
            probabilities = conditional_softmax(scores, temperature)
            selected = probabilities[np.arange(len(gold)), gold]
            if np.any(selected <= 0):
                nll[temperature] = float("inf")
            else:
                nll[temperature] = float(
                    np.sum(weights * -np.log(selected)) / np.sum(weights)
                )
        best_value = min(nll.values())
        tied = [temperature for temperature, value in nll.items() if np.isclose(value, best_value, atol=1e-12, rtol=0.0)]
        best = min(tied, key=lambda value: (0 if value == 1.0 else 1, abs(value - 1.0), value))
        return cls(float(best), nll)

    def probabilities(self, alternative_log_scores: np.ndarray) -> np.ndarray:
        return conditional_softmax(alternative_log_scores, self.temperature)
