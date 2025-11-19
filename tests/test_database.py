from pathlib import Path

import numpy as np

from audio_monitor.classifier import ClassificationResult
from audio_monitor.database import AudioEventStore


def _dummy_result(label: str = "voice/call") -> ClassificationResult:
    return ClassificationResult(
        label=label,
        energy_db=-10.0,
        spectral_centroid=1500.0,
        zero_crossing_rate=0.1,
        confidence=0.9,
        stress_level="neutral" if label == "voice/call" else None,
        stress_score=0.42 if label == "voice/call" else None,
    )


def test_add_and_retrieve_events(tmp_path):
    db_path = tmp_path / "events.db"
    store = AudioEventStore(db_path)
    store.initialize()

    samples = np.zeros(1600, dtype=np.float32)
    for idx, label in enumerate(["voice/call", "music/high-activity"]):
        store.add_event(
            source="sample.wav",
            chunk_index=idx,
            timestamp=idx * 1.5,
            result=_dummy_result(label=label),
            sample_rate=16000,
            transcript="meeting notes" if idx == 0 else "party mix",
            samples=samples,
        )

    events = store.list_events(order_by="label", descending=False)
    assert [event.label for event in events] == ["music/high-activity", "voice/call"]
    assert events[-1].stress_level == "neutral"
    assert events[-1].stress_score == 0.42

    search_hits = store.search_transcripts("meeting")
    assert len(search_hits) == 1
    assert search_hits[0].transcript == "meeting notes"

    exported = tmp_path / "chunk.wav"
    assert store.export_audio(events[0].id, exported)
    assert exported.exists()
    assert exported.read_bytes().startswith(b"RIFF")


def test_get_event_returns_none_for_missing_id(tmp_path):
    store = AudioEventStore(tmp_path / "events.db")
    store.initialize()
    assert store.get_event(9999) is None
