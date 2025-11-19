"""Command line interface for the audio monitor."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Sequence, Tuple, cast

from .classifier import BackgroundClassifier
from .database import AudioEventStore
from .file_analyzer import FileAnalyzer
from .stream import LiveAudioMonitor, MonitorEvent, list_input_devices
from .ui import LibraryInterface


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor headset audio and classify background sources.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    live_parser = subparsers.add_parser("live", help="Run live monitoring using the default headset mic")
    live_parser.add_argument("--sample-rate", type=int, default=16000)
    live_parser.add_argument("--block-duration", type=float, default=1.0)
    live_parser.add_argument("--device", type=int, default=None, help="PortAudio device id")

    file_parser = subparsers.add_parser("file", help="Analyze a WAV file")
    file_parser.add_argument("path", type=Path)
    file_parser.add_argument("--chunk-duration", type=float, default=1.0)
    file_parser.add_argument("--json", type=Path, help="Optional path to export JSON summary")
    file_parser.add_argument(
        "--sort-by",
        choices=["index", "label", "energy"],
        default="index",
        help="Sort console output by chunk index, label, or energy level.",
    )
    file_parser.add_argument(
        "--store-db",
        type=Path,
        help="Optional SQLite database file for storing classified chunks.",
    )
    file_parser.add_argument(
        "--transcript",
        type=str,
        help="Path to a text transcript or inline notes to attach to stored chunks.",
    )

    devices_parser = subparsers.add_parser("devices", help="List available recording devices")
    devices_parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="Optional case-insensitive substring to match device names (e.g. 'bluetooth').",
    )

    library_parser = subparsers.add_parser("library", help="Browse stored audio events")
    library_parser.add_argument("--db", type=Path, default=Path("audio_events.db"))
    library_parser.add_argument(
        "--sort-by",
        choices=["timestamp", "label", "energy", "confidence"],
        default="timestamp",
    )
    library_parser.add_argument("--label", type=str, help="Filter events by label")
    library_parser.add_argument("--limit", type=int, default=15)
    library_parser.add_argument("--ascending", action="store_true", help="Sort in ascending order")
    library_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Launch the interactive text UI instead of printing a table.",
    )

    return parser.parse_args(argv)


def run_live(args: argparse.Namespace) -> None:
    monitor = LiveAudioMonitor(
        classifier=BackgroundClassifier(),
        sample_rate=args.sample_rate,
        block_duration=args.block_duration,
        device=args.device,
    )
    print("Starting live monitoring. Press Ctrl+C to stop.")
    try:
        monitor.start()
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        monitor.stop()


def _sort_entries(
    entries: List[Tuple[int, MonitorEvent, object]], sort_by: str
) -> List[Tuple[int, MonitorEvent, object]]:
    if sort_by == "label":
        return sorted(entries, key=lambda item: (item[1].result.label, item[0]))
    if sort_by == "energy":
        return sorted(entries, key=lambda item: item[1].result.energy_db, reverse=True)
    return entries


def _load_transcript(transcript_arg: str | None) -> str | None:
    if not transcript_arg:
        return None
    path = Path(transcript_arg)
    if path.exists():
        return path.read_text().strip()
    return transcript_arg.strip()


def run_file(args: argparse.Namespace) -> None:
    include_samples = args.store_db is not None
    analyzer = FileAnalyzer(path=args.path, chunk_duration=args.chunk_duration)
    analyzed = analyzer.analyze(include_samples=include_samples)

    entries: List[Tuple[int, MonitorEvent, object]] = []
    TypedSequence = Sequence[Tuple[MonitorEvent, object]]
    if include_samples:
        typed = cast(TypedSequence, analyzed)
    else:
        base = [(event, None) for event in cast(Sequence[MonitorEvent], analyzed)]
        typed = cast(TypedSequence, base)

    for idx, (event, samples) in enumerate(typed):
        entries.append((idx, event, samples))

    sorted_entries = _sort_entries(entries, args.sort_by)
    for chunk_index, entry, _ in sorted_entries:
        res = entry.result
        stress = ""
        if res.stress_level:
            stress = f" | stress {res.stress_level} ({res.stress_score:.2f})"
        print(
            f"Chunk {chunk_index:03d}: {res.label:<18} | {res.energy_db:6.1f} dB | "
            f"centroid {res.spectral_centroid:7.0f} Hz | zcr {res.zero_crossing_rate:.2f}" + stress
        )

    results = [entry.result for _, entry, _ in entries]
    if args.json:
        payload = [result.as_dict() for result in results]
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"Saved summary to {args.json}")

    if args.store_db:
        store = AudioEventStore(args.store_db)
        store.initialize()
        transcript = _load_transcript(args.transcript)
        sample_rate = analyzer.sample_rate
        stored = 0
        for chunk_index, event, samples in entries:
            store.add_event(
                source=str(args.path),
                chunk_index=chunk_index,
                timestamp=event.timestamp,
                result=event.result,
                sample_rate=sample_rate,
                transcript=transcript,
                samples=samples,
            )
            stored += 1
        print(f"Stored {stored} classified chunks in {args.store_db}.")
        if transcript:
            print("Attached transcript/notes to each entry.")


def run_devices(args: argparse.Namespace) -> None:
    devices = list_input_devices(keyword=args.filter)
    if not devices:
        msg = "No input devices found"
        if args.filter:
            msg += f" matching '{args.filter}'"
        print(msg)
        return
    print("Available input devices:")
    for device in devices:
        rate = device.get("default_samplerate") or "n/a"
        print(
            f"  id {device['id']:>2}: {device['name']} | max channels: {device['max_input_channels']} | "
            f"default samplerate: {rate}"
        )


def run_library(args: argparse.Namespace) -> None:
    store = AudioEventStore(args.db)
    store.initialize()
    if args.interactive:
        LibraryInterface(store).run()
        return

    events = store.list_events(
        order_by=args.sort_by,
        descending=not args.ascending,
        label=args.label,
        limit=args.limit,
    )
    if not events:
        print("No events found. Run the 'file' command with --store-db to populate the library.")
        return
    print(
        f"Showing {len(events)} event(s) sorted by {args.sort_by} "
        f"in {'ascending' if args.ascending else 'descending'} order"
    )
    for event in events:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(event.timestamp))
        transcript = (event.transcript or "").replace("\n", " ")
        if len(transcript) > 40:
            transcript = transcript[:37] + "..."
        stress = event.stress_level or "-"
        stress_score = f" ({event.stress_score:.2f})" if event.stress_score is not None else ""
        print(
            f"[{event.id:04d}] {ts} | chunk {event.chunk_index:03d} | {event.label:<18} | "
            f"{event.energy_db:6.1f} dB | conf {event.confidence:.2f} | stress {stress}{stress_score} | {transcript}"
        )


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.command == "live":
        run_live(args)
    elif args.command == "file":
        run_file(args)
    elif args.command == "devices":
        run_devices(args)
    else:
        run_library(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
