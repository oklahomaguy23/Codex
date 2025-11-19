"""Simple background classifier based on spectral features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class ClassificationResult:
    """Holds diagnostic information for a classified chunk."""

    label: str
    energy_db: float
    spectral_centroid: float
    zero_crossing_rate: float
    confidence: float
    stress_level: str | None = None
    stress_score: float | None = None

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "energy_db": self.energy_db,
            "spectral_centroid": self.spectral_centroid,
            "zero_crossing_rate": self.zero_crossing_rate,
            "confidence": self.confidence,
            "stress_level": self.stress_level,
            "stress_score": self.stress_score,
        }


class BackgroundClassifier:
    """Light‑weight heuristics for identifying background sources.

    The classifier relies on simple signal features (RMS energy, spectral centroid
    and zero-crossing rate).  The heuristics are intentionally conservative so the
    script can run on low-powered machines without requiring a large model.
    """

    def __init__(
        self,
        silence_threshold_db: float = -45.0,
        speech_centroid_hz: float = 1500.0,
        music_centroid_hz: float = 3000.0,
    ) -> None:
        self.silence_threshold_db = silence_threshold_db
        self.speech_centroid_hz = speech_centroid_hz
        self.music_centroid_hz = music_centroid_hz

    @staticmethod
    def _to_mono(samples: np.ndarray) -> np.ndarray:
        if samples.ndim == 1:
            return samples
        return np.mean(samples, axis=1)

    @staticmethod
    def _energy_db(samples: np.ndarray) -> float:
        rms = np.sqrt(np.mean(np.square(samples) + 1e-12))
        return 20 * np.log10(rms + 1e-12)

    @staticmethod
    def _zero_crossing_rate(samples: np.ndarray) -> float:
        zero_crossings = np.where(np.diff(np.signbit(samples)))[0]
        return zero_crossings.size / max(len(samples) - 1, 1)

    @staticmethod
    def _spectral_centroid(samples: np.ndarray, sample_rate: int) -> float:
        spectrum = np.fft.rfft(samples * np.hanning(len(samples)))
        magnitude = np.abs(spectrum)
        freqs = np.fft.rfftfreq(len(samples), 1 / sample_rate)
        if np.sum(magnitude) == 0:
            return 0.0
        return float(np.sum(freqs * magnitude) / np.sum(magnitude))

    @staticmethod
    def _pitch_variability(samples: np.ndarray) -> float:
        if samples.size < 2:
            return 0.0
        return float(np.std(np.diff(samples)))

    def _estimate_stress(
        self,
        mono_samples: np.ndarray,
        energy_db: float,
        centroid: float,
        zcr: float,
    ) -> tuple[str, float]:
        """Return a heuristic stress level and score (0-1)."""

        intensity = np.clip((energy_db + 60.0) / 40.0, 0.0, 1.0)
        pitch_factor = np.clip((centroid - 1200.0) / 2200.0, 0.0, 1.0)
        articulation = np.clip((zcr - 0.08) / 0.25, 0.0, 1.0)
        jitter = np.clip(self._pitch_variability(mono_samples) / 0.05, 0.0, 1.0)
        score = 0.4 * intensity + 0.25 * pitch_factor + 0.2 * articulation + 0.15 * jitter
        if score < 0.35:
            level = "calm"
        elif score < 0.55:
            level = "neutral"
        elif score < 0.75:
            level = "tense"
        else:
            level = "stressed"
        return level, float(score)

    def classify(self, samples: Sequence[float], sample_rate: int) -> ClassificationResult:
        np_samples = np.asarray(samples, dtype=np.float64)
        if np_samples.size == 0:
            raise ValueError("Received an empty audio chunk.")
        mono = self._to_mono(np_samples)

        energy_db = self._energy_db(mono)
        zcr = self._zero_crossing_rate(mono)
        centroid = self._spectral_centroid(mono, sample_rate)
        stress_level = None
        stress_score = None

        if energy_db < self.silence_threshold_db:
            label = "ambient/silence"
            confidence = min(1.0, abs(energy_db - self.silence_threshold_db) / 20)
        elif centroid < self.speech_centroid_hz and zcr < 0.15:
            label = "voice/call"
            confidence = min(1.0, (self.speech_centroid_hz - centroid) / self.speech_centroid_hz)
            stress_level, stress_score = self._estimate_stress(mono, energy_db, centroid, zcr)
        elif centroid > self.music_centroid_hz or zcr > 0.25:
            label = "music/high-activity"
            confidence = min(1.0, centroid / (self.music_centroid_hz * 2))
        else:
            label = "broadband noise"
            confidence = 0.4

        return ClassificationResult(
            label=label,
            energy_db=energy_db,
            spectral_centroid=centroid,
            zero_crossing_rate=zcr,
            confidence=confidence,
            stress_level=stress_level,
            stress_score=stress_score,
        )
