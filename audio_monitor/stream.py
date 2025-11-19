"""Live audio monitoring utilities."""
from __future__ import annotations

import contextlib
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import numpy as np

try:  # optional import so tests can run without PortAudio
    import sounddevice as sd
except Exception:  # pragma: no cover - import guard
    sd = None

from .classifier import BackgroundClassifier, ClassificationResult


@dataclass
class MonitorEvent:
    timestamp: float
    result: ClassificationResult


class LiveAudioMonitor:
    """Monitors the default headset microphone and classifies audio chunks."""

    def __init__(
        self,
        classifier: Optional[BackgroundClassifier] = None,
        sample_rate: int = 16000,
        block_duration: float = 1.0,
        device: Optional[int] = None,
        on_event: Optional[Callable[[MonitorEvent], None]] = None,
    ) -> None:
        if sd is None:
            raise RuntimeError(
                "sounddevice is required for live monitoring. Install it with 'pip install sounddevice'."
            )
        self.classifier = classifier or BackgroundClassifier()
        self.sample_rate = sample_rate
        self.block_duration = block_duration
        self.device = device
        self.on_event = on_event or self._default_logger
        self._queue: "queue.Queue[np.ndarray]" = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def _default_logger(self, event: MonitorEvent) -> None:
        ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
        res = event.result
        print(
            f"[{ts}] {res.label:<18} | {res.energy_db:6.1f} dB | centroid {res.spectral_centroid:7.0f} Hz |"
            f" zcr {res.zero_crossing_rate:.2f} | conf {res.confidence:.2f}"
        )

    def _callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - relies on audio device
        if status:
            print(status)
        self._queue.put(indata.copy())

    def _run(self) -> None:  # pragma: no cover - relies on audio device
        blocksize = int(self.block_duration * self.sample_rate)
        with contextlib.ExitStack() as stack:
            stream = stack.enter_context(
                sd.InputStream(
                    samplerate=self.sample_rate,
                    blocksize=blocksize,
                    channels=1,
                    callback=self._callback,
                    device=self.device,
                )
            )
            stream.start()
            while not self._stop.is_set():
                try:
                    chunk = self._queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                result = self.classifier.classify(np.squeeze(chunk), self.sample_rate)
                event = MonitorEvent(timestamp=time.time(), result=result)
                self.on_event(event)

    def start(self) -> None:  # pragma: no cover - relies on audio device
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:  # pragma: no cover - relies on audio device
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)


def stream_events(
    chunks: Iterable[np.ndarray],
    sample_rate: int,
    classifier: Optional[BackgroundClassifier] = None,
) -> Iterable[MonitorEvent]:
    """Utility for iterating over existing chunks (used by file analysis and tests)."""

    classifier = classifier or BackgroundClassifier()
    for chunk in chunks:
        result = classifier.classify(chunk, sample_rate)
        yield MonitorEvent(timestamp=time.time(), result=result)


def list_input_devices(keyword: Optional[str] = None) -> List[dict]:  # pragma: no cover - relies on audio device
    """Return available microphone devices with optional name filtering.

    Parameters
    ----------
    keyword:
        Optional case-insensitive string used to filter device names. Useful for
        isolating Bluetooth headsets (e.g. ``keyword="bluetooth"`` or a brand
        name).
    """

    if sd is None:
        raise RuntimeError(
            "sounddevice is required for listing devices. Install it with 'pip install sounddevice'."
        )

    devices = sd.query_devices()
    filtered: List[dict] = []
    for device_id, info in enumerate(devices):
        if info.get("max_input_channels", 0) <= 0:
            continue
        if keyword and keyword.lower() not in info.get("name", "").lower():
            continue
        filtered.append(
            {
                "id": device_id,
                "name": info.get("name", ""),
                "max_input_channels": info.get("max_input_channels", 0),
                "default_samplerate": info.get("default_samplerate"),
            }
        )
    return filtered
