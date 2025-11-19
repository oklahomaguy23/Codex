"""Analyze existing audio files in chunks."""
from __future__ import annotations

import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

from .classifier import BackgroundClassifier, ClassificationResult
from .stream import MonitorEvent


@dataclass
class FileChunk:
    index: int
    samples: np.ndarray


class FileAnalyzer:
    """Loads WAV files and yields classification events."""

    def __init__(
        self,
        path: str | Path,
        classifier: Optional[BackgroundClassifier] = None,
        chunk_duration: float = 1.0,
    ) -> None:
        self.path = Path(path)
        self.classifier = classifier or BackgroundClassifier()
        self.chunk_duration = chunk_duration
        self._sample_rate: Optional[int] = None

    @property
    def sample_rate(self) -> int:
        if self._sample_rate is None:
            raise RuntimeError("No audio has been analyzed yet, sample rate unknown.")
        return self._sample_rate

    def _read_chunks(self) -> Iterable[FileChunk]:
        with wave.open(str(self.path), "rb") as wf:
            sample_rate = wf.getframerate()
            channels = wf.getnchannels()
            frames_per_chunk = int(self.chunk_duration * sample_rate)
            if frames_per_chunk <= 0:
                raise ValueError("chunk_duration is too small for the current sample rate")
            self._sample_rate = sample_rate
            index = 0
            while True:
                frames = wf.readframes(frames_per_chunk)
                if not frames:
                    break
                data = np.frombuffer(frames, dtype=np.int16)
                data = data.astype(np.float32) / 32768.0
                if channels > 1:
                    data = data.reshape(-1, channels)
                yield FileChunk(index=index, samples=data)
                index += 1

    def analyze(
        self, include_samples: bool = False
    ) -> List[MonitorEvent] | List[Tuple[MonitorEvent, np.ndarray]]:
        events: List[MonitorEvent] | List[Tuple[MonitorEvent, np.ndarray]] = []
        for chunk in self._read_chunks():
            result = self.classifier.classify(chunk.samples, self.sample_rate)
            event = MonitorEvent(timestamp=time.time(), result=result)
            if include_samples:
                events.append((event, chunk.samples.copy()))
            else:
                events.append(event)
        return events

    def summary(self) -> List[ClassificationResult]:
        return [event.result for event in self.analyze(include_samples=False)]
