"""
main.py — Meeting Transcriber entrypoint.

Running model: persistent-background. The process stays alive with the hotkey
always registered. Press the hotkey to toggle recording on/off.

State machine: IDLE → RECORDING → PROCESSING → IDLE
  IDLE        hotkey → start capture, → RECORDING
  RECORDING   hotkey → stop capture, spawn process thread → PROCESSING
  PROCESSING  hotkey is ignored; wait for transcription + write to finish → IDLE

Indicator: system tray icon (color-coded) + console state messages.
"""

from __future__ import annotations

import os
import sys
import threading
from datetime import datetime
from enum import Enum, auto
import faulthandler
import traceback

import keyboard
import pystray
from PIL import Image, ImageDraw

from activity_overlay import ActivityOverlay
from capture import AudioCapture
from note_writer import write_failure_note, write_note
from transcribe import transcribe

# -------------------------------------------------------------------- config

try:
    import tomllib  # Python 3.11+
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ImportError:
        raise ImportError(
            "TOML support not found. On Python < 3.11 install: pip install tomli"
        )

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CONFIG_PATH = os.path.join(_SCRIPT_DIR, "config.toml")
_LOG_HANDLE = None


class _Tee:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _setup_logging() -> str:
    global _LOG_HANDLE
    log_dir = os.path.join(_SCRIPT_DIR, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir,
        f"meeting-transcriber-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log",
    )
    _LOG_HANDLE = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, _LOG_HANDLE)
    sys.stderr = _Tee(sys.__stderr__, _LOG_HANDLE)
    faulthandler.enable(file=_LOG_HANDLE, all_threads=True)
    print(f"[main] Log file: {log_path}", flush=True)
    return log_path


def _load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        raise FileNotFoundError(
            f"config.toml not found at {_CONFIG_PATH}\n"
            "Copy config.example.toml to config.toml and fill in your paths."
        )
    with open(_CONFIG_PATH, "rb") as fh:
        return tomllib.load(fh)


# -------------------------------------------------------------------- state

class State(Enum):
    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()


_state = State.IDLE
_state_lock = threading.Lock()
_capture: AudioCapture | None = None
_start_time: datetime | None = None
_tray: pystray.Icon | None = None
_config: dict | None = None
_overlay: ActivityOverlay | None = None


def _get_config() -> dict:
    global _config
    if _config is None:
        _config = _load_config()
    return _config


# ------------------------------------------------------------------- indicator

def _make_icon(color: str) -> Image.Image:
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.ellipse([4, 4, 60, 60], fill=color)
    return img


_ICONS = {
    State.IDLE:       _make_icon("#808080"),  # gray
    State.RECORDING:  _make_icon("#FF2020"),  # red
    State.PROCESSING: _make_icon("#FFA500"),  # orange
}

_LABELS = {
    State.IDLE:       "Meeting Transcriber — idle",
    State.RECORDING:  "Meeting Transcriber — RECORDING",
    State.PROCESSING: "Meeting Transcriber — processing…",
}


def _set_state(new_state: State) -> None:
    global _state
    with _state_lock:
        _state = new_state
    print(f"[main] {_LABELS[new_state]}", flush=True)
    if _tray is not None:
        _tray.icon = _ICONS[new_state]
        _tray.title = _LABELS[new_state]


# ----------------------------------------------------------------- hotkey handler

def _on_hotkey() -> None:
    """Called from keyboard's background listener thread."""
    global _capture, _start_time, _state

    # Read + mutate state atomically; decide what action to take.
    action: str | None = None
    cap_to_stop: AudioCapture | None = None
    started_at: datetime | None = None
    meeting_notes: dict[str, str] = {"attendees": "", "operator_notes": ""}

    with _state_lock:
        current = _state
        if current == State.IDLE:
            _state = State.RECORDING
            action = "start"
        elif current == State.RECORDING:
            _state = State.PROCESSING
            action = "stop"
            cap_to_stop = _capture
            started_at = _start_time
        # PROCESSING: ignore hotkey

    if action == "start":
        cfg = _get_config()
        device_name = cfg.get("audio", {}).get("output_device_name") or None
        cap = AudioCapture(
            output_device_name=device_name,
            level_callback=_overlay.set_level if _overlay is not None else None,
        )
        if _overlay is not None:
            _overlay.show()
        cap.start()
        with _state_lock:
            _capture = cap
            _start_time = datetime.now()
        print(f"[main] {_LABELS[State.RECORDING]}", flush=True)
        if _tray is not None:
            _tray.icon = _ICONS[State.RECORDING]
            _tray.title = _LABELS[State.RECORDING]

    elif action == "stop" and cap_to_stop is not None:
        if _overlay is not None:
            meeting_notes = _overlay.get_notes()
            _overlay.hide()
        print(f"[main] {_LABELS[State.PROCESSING]}", flush=True)
        if _tray is not None:
            _tray.icon = _ICONS[State.PROCESSING]
            _tray.title = _LABELS[State.PROCESSING]
        t = threading.Thread(
            target=_process,
            args=(cap_to_stop, started_at, meeting_notes),
            daemon=True,
            name="process",
        )
        t.start()


# --------------------------------------------------------------- process pipeline

def _process(
    cap: AudioCapture,
    started_at: datetime | None,
    meeting_notes: dict[str, str] | None = None,
) -> None:
    """Stop capture → transcribe → write note → dispose temp audio. Runs in a thread."""
    if started_at is None:
        started_at = datetime.now()

    cfg = _get_config()
    vault_path = cfg.get("paths", {}).get("vault_meetings_path", "")
    fallback = cfg.get("paths", {}).get("fallback_folder", "")
    meeting_notes = meeting_notes or {"attendees": "", "operator_notes": ""}

    print("[main] Processing: stopping audio capture.", flush=True)
    audio_array, temp_file = cap.stop()
    if temp_file is not None:
        print(f"[main] Processing: audio saved to temp file: {temp_file}", flush=True)
    elif audio_array is not None:
        print(
            f"[main] Processing: in-memory audio samples: {len(audio_array)} "
            f"({len(audio_array) / 16000.0:.1f}s).",
            flush=True,
        )

    temp_to_delete: str | None = None

    try:
        audio_input = temp_file if temp_file is not None else audio_array
        print("[main] Processing: transcription starting.", flush=True)
        transcript, duration = transcribe(audio_input)
        print(
            f"[main] Processing: transcription finished; duration={duration:.1f}s, "
            f"chars={len(transcript)}.",
            flush=True,
        )
        write_note(
            transcript,
            started_at,
            duration,
            vault_path,
            fallback,
            attendees=meeting_notes.get("attendees", ""),
            operator_notes=meeting_notes.get("operator_notes", ""),
        )
        # Normal success: schedule temp audio for deletion.
        temp_to_delete = temp_file

    except Exception as exc:
        print(f"[main] ERROR: Transcription failed: {exc}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        try:
            write_failure_note(
                started_at,
                str(exc),
                vault_path,
                fallback,
                attendees=meeting_notes.get("attendees", ""),
                operator_notes=meeting_notes.get("operator_notes", ""),
            )
        except Exception as write_exc:
            print(f"[main] ERROR: Could not write failure note: {write_exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
        # Failure path: retain temp_file so the session is recoverable.
        if temp_file:
            print(
                f"[main] Audio retained for recovery: {temp_file}",
                file=sys.stderr,
            )
        else:
            print("[main] No temp audio file — in-memory buffer was lost.", file=sys.stderr)

    finally:
        # Explicit cleanup — log the outcome so a missed delete is visible.
        if temp_to_delete:
            if os.path.exists(temp_to_delete):
                try:
                    os.remove(temp_to_delete)
                    print(f"[main] Temp audio disposed: {temp_to_delete}")
                except Exception as del_exc:
                    print(
                        f"[main] WARNING: Failed to delete temp audio {temp_to_delete}: {del_exc}",
                        file=sys.stderr,
                    )
            else:
                print(
                    f"[main] WARNING: Expected temp audio not found at {temp_to_delete}",
                    file=sys.stderr,
                )
        _set_state(State.IDLE)


# ----------------------------------------------------------------------- quit

def _quit_handler(icon: pystray.Icon, item: pystray.MenuItem) -> None:
    with _state_lock:
        s = _state
    if s != State.IDLE:
        print(
            f"[main] WARNING: Quitting while in {s.name} state. "
            "In-progress recording or transcription will be discarded.",
            file=sys.stderr,
        )
    if _overlay is not None:
        _overlay.stop()
    icon.stop()


# ----------------------------------------------------------------------- main

def main() -> None:
    global _tray, _overlay

    _setup_logging()
    _overlay = ActivityOverlay()
    _overlay.start()

    cfg = _get_config()
    hotkey_combo: str = cfg.get("hotkey", {}).get("hotkey", "ctrl+shift+r")

    print(f"[main] Meeting Transcriber starting.")
    print(f"[main] Hotkey: {hotkey_combo}  (first press: start | second press: stop)")
    print("[main] Right-click the tray icon to quit.")

    keyboard.add_hotkey(hotkey_combo, _on_hotkey, suppress=True)

    menu = pystray.Menu(
        pystray.MenuItem("Quit", _quit_handler),
    )
    _tray = pystray.Icon(
        name="meeting-transcriber",
        icon=_ICONS[State.IDLE],
        title=_LABELS[State.IDLE],
        menu=menu,
    )

    # pystray.Icon.run() blocks the main thread on Windows (required by Win32 message loop).
    _tray.run()

    keyboard.unhook_all()
    print("[main] Meeting Transcriber stopped.")


if __name__ == "__main__":
    main()
