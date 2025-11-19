"""Simple interactive library browser for stored audio events."""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .database import AudioEventStore, AudioEventRecord


@dataclass
class MenuOption:
    key: str
    description: str


class LibraryInterface:
    """Text-based menu to browse stored audio events."""

    def __init__(self, store: AudioEventStore) -> None:
        self.store = store
        self.options = [
            MenuOption("list", "List the most recent events"),
            MenuOption("filter", "Filter events by label"),
            MenuOption("search", "Search transcripts for a keyword"),
            MenuOption("export", "Export an event's audio to a WAV file"),
            MenuOption("view", "Show the metadata for a specific event"),
            MenuOption("help", "Show this menu again"),
            MenuOption("quit", "Exit the library"),
        ]

    def run(self) -> None:
        self._print_header()
        self._print_menu()
        while True:
            command = input("library> ").strip().lower()
            if command in {"quit", "q", "exit"}:
                print("Goodbye!")
                break
            if command in {"help", "?"}:
                self._print_menu()
                continue
            if command.startswith("list"):
                self._handle_list()
            elif command.startswith("filter"):
                self._handle_filter(command)
            elif command.startswith("search"):
                self._handle_search(command)
            elif command.startswith("export"):
                self._handle_export(command)
            elif command.startswith("view"):
                self._handle_view(command)
            else:
                print("Unknown command. Type 'help' to see available actions.")

    def _print_header(self) -> None:
        print("=" * 60)
        print("Audio Library Browser")
        print("=" * 60)

    def _print_menu(self) -> None:
        print("Available commands:")
        for option in self.options:
            print(f"  {option.key:<7} - {option.description}")

    def _format_event(self, event: AudioEventRecord) -> str:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))
        transcript = (event.transcript or "").strip()
        if len(transcript) > 40:
            transcript = transcript[:37] + "..."
        stress = event.stress_level or "-"
        stress_score = f" ({event.stress_score:.2f})" if event.stress_score is not None else ""
        return (
            f"[{event.id:04d}] {ts} | {event.label:<18} | "
            f"{event.energy_db:6.1f} dB | conf {event.confidence:.2f} | "
            f"stress {stress}{stress_score} | src: {event.source} | note: {transcript}"
        )

    def _handle_list(self, limit: int = 10) -> None:
        events = self.store.list_events(limit=limit)
        if not events:
            print("No events found. Use the file analyzer with --store-db to add entries.")
            return
        for event in events:
            print(self._format_event(event))

    def _handle_filter(self, command: str) -> None:
        parts = command.split(maxsplit=1)
        if len(parts) == 1:
            label = input("Label to filter by: ").strip()
        else:
            label = parts[1]
        events = self.store.list_events(order_by="timestamp", label=label, descending=True)
        if not events:
            print(f"No events found for label '{label}'.")
            return
        for event in events:
            print(self._format_event(event))

    def _handle_search(self, command: str) -> None:
        parts = command.split(maxsplit=1)
        if len(parts) == 1:
            keyword = input("Keyword to search in transcripts: ").strip()
        else:
            keyword = parts[1]
        events = self.store.search_transcripts(keyword)
        if not events:
            print(f"No transcripts matched '{keyword}'.")
            return
        for event in events:
            print(self._format_event(event))

    def _handle_view(self, command: str) -> None:
        event_id = self._extract_id(command)
        if event_id is None:
            return
        record = self.store.get_event(event_id)
        if record is None:
            print(f"No event found with id {event_id}.")
            return
        self._print_event_details(record)

    def _handle_export(self, command: str) -> None:
        event_id = self._extract_id(command)
        if event_id is None:
            return
        path = input("Destination WAV path: ").strip()
        success = self.store.export_audio(event_id, Path(path))
        if not success:
            print("Unable to export audio for that event.")
        else:
            print(f"Saved WAV snippet to {path}.")

    def _extract_id(self, command: str) -> Optional[int]:
        parts = command.split(maxsplit=1)
        if len(parts) == 1:
            raw = input("Event id: ").strip()
        else:
            raw = parts[1]
        if not raw.isdigit():
            print("Please provide a numeric event id.")
            return None
        return int(raw)

    def _print_event_details(self, record: AudioEventRecord) -> None:
        ts = time.strftime("%c", time.localtime(record.timestamp))
        print("-" * 60)
        print(f"Event {record.id} | {ts}")
        print(f"Source: {record.source} (chunk {record.chunk_index})")
        print(f"Label: {record.label} | Confidence: {record.confidence:.2f}")
        print(f"Energy: {record.energy_db:.1f} dB | Spectral centroid: {record.spectral_centroid:.0f} Hz")
        print(f"Zero crossing rate: {record.zero_crossing_rate:.3f}")
        if record.stress_level:
            score = f" ({record.stress_score:.2f})" if record.stress_score is not None else ""
            print(f"Stress estimate: {record.stress_level}{score}")
        else:
            print("Stress estimate: n/a")
        print(f"Transcript: {record.transcript or '(none)'}")
        print("-" * 60)


__all__ = ["LibraryInterface"]
