"""Live, text-anchored homework inference using the prompt-clean frozen scorer.

Only audio, assigned text and frozen fitted artifacts enter inference. Corpus
manifests, cached predictions and PHN/transcript labels are never loaded here.
The default is an explicitly identified research fold, not a newly validated
production model. Lexical policy exclusions remain unassessed in the UI.
"""
from __future__ import annotations

import io
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.signal import resample_poly
from scipy.special import logsumexp
import soundfile as sf

from .backend import OfficialCTCSFBackend, sha256_file
from .baselines import ctc_sf_sd_norm
from .dual_experiment import graph_predictions, phone_ids
from .dual_lexicon import build_acceptable_graph
from .dual_prior import generic_prior
from .final_experiment import continuous_features, diagnostic_values, model_score, output_rows
from .parikh_experiment import score_event

ROOT = Path(__file__).resolve().parents[2]
MODEL_VERSION = "final_v3_prompt_clean"
SAMPLE_RATE = 16_000
MAX_AUDIO_SECONDS = 20
MAX_REFERENCE_PHONES = 80
POLICY_SPEAKERS = ("F01", "F03", "F04", "M01", "M02", "M04", "M05")
IPA = dict(zip(
    "AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH".split(),
    "ɑ æ ʌ/ə ɔ aʊ aɪ b tʃ d ð ɛ ɝ eɪ f ɡ h ɪ i dʒ k l m n ŋ oʊ ɔɪ p ɹ s ʃ t θ ʊ u v w j z ʒ".split(),
))


def decode_audio_bytes(body: bytes, filename: str = "recording.wav") -> np.ndarray:
    """Decode WAV/FLAC/OGG supported by libsndfile, downmix, resample to 16 kHz."""
    try:
        with sf.SoundFile(io.BytesIO(body)) as source:
            if source.frames / source.samplerate > MAX_AUDIO_SECONDS:
                raise ValueError(f"Please use a recording of at most {MAX_AUDIO_SECONDS} seconds.")
            audio = source.read(dtype="float32", always_2d=True)
            sample_rate = source.samplerate
    except (sf.LibsndfileError, sf.SoundFileRuntimeError) as error:
        raise ValueError("Cannot decode this recording. Please upload a WAV, FLAC or OGG audio file.") from error
    wave = audio.mean(axis=1)
    if not wave.size or not np.isfinite(wave).all():
        raise ValueError("The recording must contain finite audio samples.")
    if sample_rate != SAMPLE_RATE:
        divisor = int(np.gcd(sample_rate, SAMPLE_RATE))
        wave = resample_poly(wave, SAMPLE_RATE // divisor, sample_rate // divisor)
    return np.ascontiguousarray(wave, dtype=np.float32)


def present_predictions(reference_text: str, graph: dict, predictions: list[dict]) -> dict:
    """Map reference positions to words; these indices are never audio times."""
    phones, words = [], []
    for word in graph["words"]:
        indices = list(range(word["start"], word["end"]))
        for i in indices:
            prediction = predictions[i]
            assessable = bool(graph["diagnostic_mask"][i])
            status = ("review" if prediction["predicted_error"] else "no_flag") if assessable else "unassessed"
            phones.append({
                "phone_index": i, "word_index": word["word_index"],
                "canonical": graph["canonical_phones"][i], "ipa": IPA[graph["canonical_phones"][i]],
                "acceptable": graph["acceptable_phones"][i], "assessable": assessable,
                "status": status, "error_probability": prediction["p_error"] if assessable else None,
                "error_flag": prediction["predicted_error"] if assessable else None,
                "diagnostic_candidate": None,
                "diagnostic_status": "withheld_unvalidated_specific_diagnosis" if assessable else "lexical_policy_exclusion",
            })
        states = [phones[i]["status"] for i in indices]
        words.append({"word_index": word["word_index"], "text": word["word"],
                      "start_phone": word["start"], "end_phone": word["end"], "phones": indices,
                      "status": "review" if "review" in states else ("unassessed" if "unassessed" in states else "no_flag"),
                      "unassessed_reasons": sorted({v["reason"] for v in word["unavailable_variants"]})})
    return {"reference_text": reference_text, "words": words, "phones": phones,
            "summary": {"assessed_phones": sum(p["assessable"] for p in phones),
                        "review_phones": sum(p["status"] == "review" for p in phones),
                        "unassessed_phones": sum(p["status"] == "unassessed" for p in phones)},
            "lexical_policy": {"version": graph["policy_version"], "exclusions": graph["exclusions"]}}


class HomeworkEngine:
    """Lazy inference engine. Callers serialize access to the shared backend."""

    def __init__(self, *, policy_speaker: str = "F01", backend=None, device: str = "auto"):
        if policy_speaker not in POLICY_SPEAKERS:
            raise ValueError(f"Unknown research policy: {policy_speaker}")
        self.policy_speaker = policy_speaker
        self.backend = backend
        self.device = device
        self.fit = None
        self.prior = None
        self.model = None
        self.policy_config = None
        self.artifact_hashes = None

    def _load(self):
        if self.fit is not None:
            return
        import torch
        from threadpoolctl import threadpool_limits

        # Match the measured final pipeline and avoid oversubscribing short jobs.
        torch.set_num_threads(2)
        self._thread_limits = threadpool_limits(limits=2)
        directory = ROOT / "artifacts/evaluation" / MODEL_VERSION / "sensitivity7"
        fit_path = directory / f"{self.policy_speaker}.fit.json"
        model_path = directory / "models" / f"{self.policy_speaker}.da_cf_gop.joblib"
        config_path = directory / "config.json"
        fit = json.loads(fit_path.read_text(encoding="utf-8"))
        prior = {**fit["cf_prior"]}
        for name in ("op_probs", "sub_probs", "insert_phone_probs"):
            prior[name] = np.asarray(prior[name])
        model = joblib.load(model_path)
        if self.backend is None:
            paths = json.loads((ROOT / "configs/frozen_v1.json").read_text())["paths"]
            self.backend = OfficialCTCSFBackend(
                (ROOT / paths["checkpoint"]).resolve(), (ROOT / paths["processor"]).resolve(), device=self.device)
        self.device = str(self.backend.device)
        self.fit, self.prior, self.model = fit, prior, model
        self.policy_config = json.loads(config_path.read_text(encoding="utf-8"))
        self.artifact_hashes = {"fit_sha256": sha256_file(fit_path), "model_sha256": sha256_file(model_path),
                                "config_sha256": sha256_file(config_path)}

    def status(self) -> dict:
        return {"engine_loaded": self.fit is not None, "method": "DA-CF-GOP", "model_version": MODEL_VERSION,
                "policy_id": f"sensitivity7_test_{self.policy_speaker}",
                "policy_scope": "Frozen research policy; new-user performance has not been validated.",
                "default_policy_selection": "F01: lexicographically first frozen fold, independent of test scores",
                "device": self.device, "sample_rate": SAMPLE_RATE,
                "max_audio_seconds": MAX_AUDIO_SECONDS, "max_reference_phones": MAX_REFERENCE_PHONES,
                "feedback_modes": ["template", "qwen"], "replayed": False}

    def assess(self, audio: np.ndarray, reference_text: str) -> dict:
        start = time.perf_counter()
        reference_text = reference_text.strip()
        if not reference_text or len(reference_text) > 500:
            raise ValueError("Enter the assigned English text (1–500 characters).")
        if any(char.isalnum() and not char.isascii() for char in reference_text):
            raise ValueError("Use English letters and digits in the assigned text; accented or non-English words are not supported.")
        if any((ord(char) < 32 and char not in "\t\r\n") or ord(char) == 127 for char in reference_text):
            raise ValueError("The assigned text contains control characters. Paste plain English text.")
        wave = np.asarray(audio, dtype=np.float32)
        if wave.ndim != 1 or not wave.size or not np.isfinite(wave).all():
            raise ValueError("The recording must be a finite, mono waveform.")
        if not .1 <= len(wave) / SAMPLE_RATE <= MAX_AUDIO_SECONDS:
            raise ValueError(f"Please use a recording between 0.1 and {MAX_AUDIO_SECONDS} seconds.")
        if np.max(np.abs(wave)) < 1e-5:
            raise ValueError("The recording is silent. Check the microphone and record again.")
        graph = build_acceptable_graph(reference_text)
        canonical = phone_ids(graph["canonical_phones"])
        if len(canonical) > MAX_REFERENCE_PHONES:
            raise ValueError(f"Use a shorter assignment (at most {MAX_REFERENCE_PHONES} target phones).")
        text_end = time.perf_counter()
        self._load()
        ready = time.perf_counter()
        logits = self.backend.infer(wave)
        acoustic_end = time.perf_counter()
        minimum_frames = len(canonical) + sum(a == b for a, b in zip(canonical, canonical[1:]))
        if len(logits) < minimum_frames:
            raise ValueError("The recording is too short for the assigned text. Record the full assignment.")
        lp = logits.astype(float) - logsumexp(logits.astype(float), axis=1, keepdims=True)
        sf_scores = ctc_sf_sd_norm(lp, canonical)
        ppaf = score_event(lp, canonical, device=self.device)
        isolated_end = time.perf_counter()
        tokens, _, decoded = graph_predictions(graph, lp, generic_prior(), 1.)
        graph_end = time.perf_counter()
        evidence = [{"event": "live", "speaker": "live", "phone_index": i,
                     "target": graph["canonical_phones"][i], "acceptable": graph["acceptable_phones"][i],
                     "baseline": {"ctc_sf_sd_norm_raw": sf_scores[i].normalized_score,
                                  "ppaf_rps_raw": ppaf["rps"][i]["gop"], "ppaf_ups_raw": ppaf["ups"][i]["gop"]},
                     "gop": token["gop"], "event_posteriors": token["event_posteriors"],
                     "substitution_posteriors": token["substitution_posteriors"],
                     "viterbi_gop": float(decoded["max_marginal_token_log_odds"][i])}
                    for i, token in enumerate(tokens)]
        x, cf = continuous_features(evidence, self.prior, "da_cf_gop")
        scores = model_score(self.model["binary"], evidence, x)
        diagnostic = diagnostic_values(self.model["event"], x, cf)
        fitted = self.fit["methods"]["da_cf_gop"]
        predictions = output_rows(evidence, "da_cf_gop", self.fit["fold"]["fold"], scores,
                                  fitted["calibration"], cf=cf, diagnostic=diagnostic,
                                  type_threshold=fitted["type_threshold"], specific_gate=fitted["specific_gate"])
        result = present_predictions(reference_text, graph, predictions)
        fold = self.fit["fold"]
        result["provenance"] = {
            "model_version": MODEL_VERSION, "policy_id": fold["fold"],
            "training_speakers": fold["training_patients"], "calibration_speakers": [fold["development_patient"]],
            "held_out_speaker": fold["test_patient"], "replayed": False,
            "forced_alignment": False, "reference_source": "assigned_text", "labels_read": False,
            "input_quality_excluded_events": self.policy_config["excluded_events"],
            "policy_scope": "Frozen research policy; new-user performance has not been validated.",
            **self.artifact_hashes,
        }
        # Research diagnostics are separate from patient-facing phones/feedback.
        result["research_predictions"] = predictions
        result["timings"] = {"text_graph_s": text_end - start, "model_load_s": ready - text_end,
                             "acoustic_s": acoustic_end - ready, "isolated_ctc_s": isolated_end - acoustic_end,
                             "candidate_graph_s": graph_end - isolated_end,
                             "scoring_s": time.perf_counter() - graph_end,
                             "total_s": time.perf_counter() - start, "audio_s": len(wave) / SAMPLE_RATE}
        return result
