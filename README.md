# Codex Audio Monitor

This repository contains a small Python utility that listens to your headset microphone and
classifies the background information into broad categories such as voice calls, music/high
activity, broadband noise, and silence.  The heuristics are intentionally lightweight so the
program can run without a GPU or cloud connection.

## Features

- **Live monitoring** using the default PortAudio device via `sounddevice`.
- **File analysis** mode for processing existing WAV recordings in chunks.
- JSON export of the classification summary for later inspection.
- **Embedded SQLite library** for storing chunk-level audio, metrics, and transcripts.
- **Conversation-aware stress analysis** that flags tense or stressed voices when a
  voice/call chunk is detected and stores the estimate alongside each recording.
- **Text-based library browser** that lets you sort, filter, and export stored snippets.
- Simple numpy-based classifier with unit tests to make the heuristics easy to tune.
- Device discovery helper to list available microphones (including Bluetooth headsets).

## Requirements

- Python 3.10+
- `numpy`
- `sounddevice` (only required for live monitoring)
- `pytest` for running the tests

You can install the dependencies manually with `pip install -r requirements.txt`, or
follow the laptop quickstart guide below to create an isolated environment and get the
`audio-monitor` command on your `PATH`.

## Laptop quickstart

1. **Clone or download** this repository onto your laptop.
2. **Create a virtual environment** (recommended so the tools stay isolated):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
   ```

3. **Install the package in editable mode**. This installs dependencies and exposes the
   `audio-monitor` console command so you can run the tool from anywhere inside the
   repository:

   ```bash
   pip install -e .
   ```

4. **Verify your headset is visible**. If you use a Bluetooth headset, pair it with the
   OS first, then run:

   ```bash
   audio-monitor devices --filter bluetooth
   ```

   Copy the `id` for your headset (you can omit `--filter` to list everything) and feed
   it into the live monitor via `--device`.

5. **Start live monitoring** (press `Ctrl+C` to exit):

   ```bash
   audio-monitor live --device <id> --sample-rate 16000 --block-duration 1.0
   ```

6. **Analyze and store recordings** with the same command names used below, e.g.

   ```bash
   audio-monitor file meeting.wav --chunk-duration 1.0 --store-db audio_events.db
   ```

If PortAudio is not installed on your OS, install it first (macOS: `brew install
portaudio`, Ubuntu/Debian: `sudo apt install libportaudio2`, Windows: install the
official binaries or let `pip` install the bundled wheel). After these steps, the tool
is ready for day-to-day use on your laptop.

## Usage

### Live monitoring

```bash
python -m audio_monitor live --sample-rate 16000 --block-duration 1.0
```

Use `--device` to select a different input device as reported by the `devices` command
described below. Press `Ctrl+C` to stop the session.

### Discover microphones / Bluetooth headsets

```bash
python -m audio_monitor devices --filter bluetooth
```

The command prints the PortAudio device ID, name, channel count, and default sample rate
for each available microphone. Provide a substring via `--filter` (e.g. `sony`, `jabra`,
`airpods`, or `bluetooth`) to quickly locate your headset entry. Use the reported `id`
value with `--device` when launching `live` monitoring.

#### Pairing tips for Bluetooth headsets

1. Pair the headset with your OS as usual and ensure the microphone profile is enabled.
2. On macOS, confirm the device is listed under **System Preferences → Sound → Input**.
   On Windows, check **Sound settings → Input devices**. On Linux (PulseAudio/PipeWire), use
   `pavucontrol` or your desktop environment's sound control panel.
3. Once the OS exposes the headset as a recording device, rerun the `devices` command to
   capture the PortAudio ID and feed it to `python -m audio_monitor live --device <id>`.

### Analyze a WAV file

```bash
python -m audio_monitor file path/to/recording.wav --chunk-duration 1.0 --json summary.json
```

Each chunk is printed to the console and, if requested, written to a JSON file.

Add `--store-db audio_events.db` to persist the results and raw audio snippets inside an
embedded SQLite database. Voice/call chunks automatically receive a stress estimate so you
can scan for tense or stressed segments later. You can optionally attach a transcript or
free-form notes that describe the context of the recording:

```bash
python -m audio_monitor file meeting.wav --store-db audio_events.db --transcript notes.txt
```

Use `--sort-by label` or `--sort-by energy` to quickly surface the most interesting chunks.

### Browse the embedded audio library

Once you have stored a few sessions you can inspect them with the new `library` command.
The command prints a sortable table by default:

```bash
python -m audio_monitor library --db audio_events.db --sort-by energy --limit 20
```

Pass `--interactive` to launch the menu-driven interface. From there you can list events
by label, search transcripts, view per-event metadata, or export a chunk back to a WAV
file for sharing.

## Tests

Run the unit tests with:

```bash
pytest
```
