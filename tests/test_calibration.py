from __future__ import annotations

import numpy as np
import pytest

from da_cf_gop.backend import sha256_file

from da_cf_gop.calibration import (
    AlternativeTemperatureCalibrator,
    CTCSequenceRecalibrator,
    CalibrationUtterance,
    DeletionCalibrator,
    SpeakerWeightedPlattCalibrator,
    assert_safe_entropy_shift,
    speaker_equal_sample,
    speaker_equal_weights,
)


def _utterance(name: str, speaker: str, seed: int, frames: int = 7) -> CalibrationUtterance:
    logits = np.random.default_rng(seed).normal(size=(frames, 4)).astype(np.float32)
    return CalibrationUtterance(name, speaker, logits, (1, 2))


def test_speaker_equal_sampling_and_weights() -> None:
    rows = [_utterance(f"a{i}", "A", i) for i in range(20)] + [
        _utterance(f"b{i}", "B", 100 + i) for i in range(2)
    ]
    selected = speaker_equal_sample(rows, max_utterances=4, seed=3)
    assert sum(row.speaker_id == "A" for row in selected) == 2
    assert sum(row.speaker_id == "B" for row in selected) == 2
    weights = speaker_equal_weights([row.speaker_id for row in rows])
    assert weights[:20].sum() == pytest.approx(weights[20:].sum())


def test_dynamic_sequence_recalibrator_and_impossible_exclusion(tmp_path) -> None:
    rows = [
        _utterance("a", "A", 1),
        _utterance("b", "B", 2),
        CalibrationUtterance("impossible", "B", np.zeros((2, 4), dtype=np.float32), (1, 1)),
    ]
    model = CTCSequenceRecalibrator.fit(
        rows, blank_id=0, steps=1, batch_size=2, device="cpu", enforce_entropy_guard=False
    )
    assert model.weight.shape == (4, 4)
    assert model.bias.shape == (4,)
    assert model.training_summary["zero_infinity"] is False
    assert model.training_summary["excluded"]["ctc_impossible_sequence"] == 1
    transformed = model.apply(rows[0].logits)
    assert transformed.shape == rows[0].logits.shape
    path = model.save(tmp_path / "recalibrator.npz")
    first_hash = sha256_file(path)
    model.save(path)
    assert sha256_file(path) == first_hash
    restored = CTCSequenceRecalibrator.load(path)
    assert np.allclose(restored.apply(rows[0].logits), transformed)


def test_entropy_guard_fails_closed() -> None:
    assert_safe_entropy_shift(0.5, 0.7)
    with pytest.raises(RuntimeError, match="unsafe"):
        assert_safe_entropy_shift(0.1, 0.5)
    with pytest.raises(RuntimeError, match="unsafe"):
        assert_safe_entropy_shift(1.0, 1.3)


def test_deletion_q90_requires_phone_support_and_two_speakers() -> None:
    rows = []
    for speaker, offset in (("A", 0.0), ("B", 0.2)):
        rows.extend(
            {
                "speaker_id": speaker,
                "target": "T",
                "canonical_lp": -2.0,
                "del_lp": -2.0 + offset + value,
                "error": 0,
            }
            for value in np.linspace(-0.5, 0.5, 10)
        )
    rows.append(
        {"speaker_id": "A", "target": "K", "canonical_lp": -2.0, "del_lp": 8.0, "error": 0}
    )
    fitted = DeletionCalibrator.fit(rows, min_tokens=20, min_speakers=2)
    assert "T" in fitted.per_phone
    assert "K" not in fitted.per_phone
    assert fitted.penalty("K") == fitted.global_penalty
    assert fitted.calibrate("T", 2.0) == pytest.approx(2.0 - fitted.penalty("T"))


def test_platt_is_speaker_weighted_and_monotone() -> None:
    evidence = [-3, -2, 2, 3, -3, -2, 2, 3]
    labels = [0, 0, 1, 1, 0, 0, 1, 1]
    speakers = ["A"] * 4 + ["B"] * 4
    model = SpeakerWeightedPlattCalibrator.fit(evidence, labels, speakers)
    probabilities = model.predict_error_probability([-4, 0, 4])
    assert np.all(np.diff(probabilities) > 0)
    assert np.all((probabilities > 0) & (probabilities < 1))


def test_temperature_selection_prefers_one_on_exact_tie() -> None:
    scores = np.zeros((4, 3))
    model = AlternativeTemperatureCalibrator.fit(
        scores, [0, 1, 2, 0], ["A", "A", "B", "B"]
    )
    assert model.temperature == 1.0
    assert np.allclose(model.probabilities(scores).sum(axis=1), 1.0)
