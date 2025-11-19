"""Embedded SQLite storage for classified audio chunks."""
from __future__ import annotations

import io
import sqlite3
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

from .classifier import ClassificationResult


@dataclass
class AudioEventRecord:
    """Metadata describing a classified audio chunk stored in the database."""

    id: int
    source: str
    chunk_index: int
    timestamp: float
    label: str
    energy_db: float
    spectral_centroid: float
    zero_crossing_rate: float
    confidence: float
    sample_rate: int
    transcript: Optional[str]
    stress_level: Optional[str]
    stress_score: Optional[float]


class AudioEventStore:
    """Lightweight wrapper around an on-disk SQLite database."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: Optional[sqlite3.Connection] = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.path)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize(self) -> None:
        conn = self.connect()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                label TEXT NOT NULL,
                energy_db REAL NOT NULL,
                spectral_centroid REAL NOT NULL,
                zero_crossing_rate REAL NOT NULL,
                confidence REAL NOT NULL,
                sample_rate INTEGER NOT NULL,
                transcript TEXT,
                stress_level TEXT,
                stress_score REAL,
                audio BLOB
            )
            """
        )
        self._ensure_columns(conn)
        conn.commit()

    def _ensure_columns(self, conn: sqlite3.Connection) -> None:
        columns = {row["name"] for row in conn.execute("PRAGMA table_info(events)")}
        if "stress_level" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN stress_level TEXT")
        if "stress_score" not in columns:
            conn.execute("ALTER TABLE events ADD COLUMN stress_score REAL")

    @staticmethod
    def _serialize_wav(samples: np.ndarray, sample_rate: int) -> bytes:
        """Convert floating-point samples to a WAV byte payload."""

        data = np.asarray(samples, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        clipped = np.clip(data, -1.0, 1.0)
        int_samples = (clipped * 32767).astype(np.int16)
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(int_samples.shape[1])
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(int_samples.tobytes())
        return buffer.getvalue()

    def add_event(
        self,
        source: str,
        chunk_index: int,
        timestamp: float,
        result: ClassificationResult,
        sample_rate: int,
        transcript: Optional[str] = None,
        samples: Optional[np.ndarray] = None,
    ) -> int:
        conn = self.connect()
        audio_blob = None
        if samples is not None:
            audio_blob = sqlite3.Binary(self._serialize_wav(samples, sample_rate))
        cursor = conn.execute(
            """
            INSERT INTO events (
                source,
                chunk_index,
                timestamp,
                label,
                energy_db,
                spectral_centroid,
                zero_crossing_rate,
                confidence,
                sample_rate,
                transcript,
                stress_level,
                stress_score,
                audio
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                source,
                chunk_index,
                timestamp,
                result.label,
                result.energy_db,
                result.spectral_centroid,
                result.zero_crossing_rate,
                result.confidence,
                sample_rate,
                transcript,
                result.stress_level,
                result.stress_score,
                audio_blob,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)

    def list_events(
        self,
        *,
        order_by: str = "timestamp",
        descending: bool = True,
        label: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[AudioEventRecord]:
        column_map = {
            "timestamp": "timestamp",
            "label": "label",
            "energy": "energy_db",
            "confidence": "confidence",
        }
        column = column_map.get(order_by, "timestamp")
        direction = "DESC" if descending else "ASC"
        query = "SELECT * FROM events"
        params: List[object] = []
        if label:
            query += " WHERE label = ?"
            params.append(label)
        query += f" ORDER BY {column} {direction}"
        if limit:
            query += " LIMIT ?"
            params.append(limit)
        rows = self.connect().execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_event(self, event_id: int) -> Optional[AudioEventRecord]:
        row = self.connect().execute("SELECT * FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def export_audio(self, event_id: int, destination: Path) -> bool:
        row = self.connect().execute("SELECT audio FROM events WHERE id = ?", (event_id,)).fetchone()
        if row is None or row["audio"] is None:
            return False
        destination.write_bytes(row["audio"])
        return True

    def search_transcripts(self, keyword: str) -> List[AudioEventRecord]:
        pattern = f"%{keyword}%"
        rows = self.connect().execute(
            "SELECT * FROM events WHERE transcript LIKE ? ORDER BY timestamp DESC",
            (pattern,),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> AudioEventRecord:
        return AudioEventRecord(
            id=row["id"],
            source=row["source"],
            chunk_index=row["chunk_index"],
            timestamp=row["timestamp"],
            label=row["label"],
            energy_db=row["energy_db"],
            spectral_centroid=row["spectral_centroid"],
            zero_crossing_rate=row["zero_crossing_rate"],
            confidence=row["confidence"],
            sample_rate=row["sample_rate"],
            transcript=row["transcript"],
            stress_level=row["stress_level"],
            stress_score=row["stress_score"],
        )

    def __enter__(self) -> "AudioEventStore":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
