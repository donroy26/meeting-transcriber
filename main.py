"""
main.py — Meeting Transcriber entrypoint.

Running model: persistent-background. The process stays alive with the hotkey
always registered. Press the hotkey to toggle recording on/off.

State machine: IDLE → RECORDING → PROCESSING → IDLE
  IDLE        hotkey → start capture, → RECORDING
  RECORDING   hotkey → stop capture, spawn process thread → PROCESSING
  PROCESSING  hotkey is ignored; wait for transcription + write to finish → IDLE

Indicator: system tray icon (color-coded) + console state messages.

Startup: nothing in this module imports the ML stack (torch, faster-whisper,
whisperx, pyannote). That import chain pulls ~6 GB of DLLs off disk and took
75 s on a cold cache, during which the hotkey did not yet exist. It is now
imported lazily on the model-warmup thread, so the hotkey and tray are live in
a few seconds and a meeting started immediately just waits on the model lock.
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time
import wave
from datetime import datetime
from enum import Enum, auto
import faulthandler
import traceback

import numpy as np
import pystray
from PIL import Image, ImageDraw

import hotkey
from activity_overlay import ActivityOverlay
from capture import (
    SAMPLE_RATE,
    AudioCapture,
    retain_session_audio,
    save_recovery_wav,
)
from hotkey import HotkeyListener
from note_writer import write_failure_note, write_note

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


_log_lock = threading.Lock()


def log(msg: str, *, err: bool = False) -> None:
    """Write one timestamped log line atomically.

    Timestamp and single-write matter for incident analysis: `print` emits the
    text and the newline as two separate writes, so concurrent threads produced
    glued lines like `— processing…[main] Processing:` in the logs, and without
    timestamps there was no way to correlate a lost recording with anything
    else on the machine.
    """
    line = f"{datetime.now():%H:%M:%S} {msg}\n"
    stream = sys.stderr if err else sys.stdout
    with _log_lock:
        try:
            stream.write(line)
        except Exception:
            pass


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
    log(f"[main] Log file: {log_path}")
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


# Recording and transcription are decoupled: stopping a recording enqueues a
# processing job and immediately frees the recorder, so a new meeting can be
# recorded while earlier ones transcribe. A single worker drains the queue
# (serialized — the GPU only fits one transcription at a time).
_state_lock = threading.Lock()
_recording = False           # guarded by _state_lock
_jobs_pending = 0            # guarded by _state_lock; queued + in-flight jobs
_capture: AudioCapture | None = None
_start_time: datetime | None = None
_jobs: queue.Queue = queue.Queue()
_tray: pystray.Icon | None = None
_config: dict | None = None
_overlay: ActivityOverlay | None = None
_listener: HotkeyListener | None = None
_hotkey_combo: str = "ctrl+shift+r"


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


def _refresh_indicator() -> None:
    """Recompute tray icon/label from recording + queue state."""
    with _state_lock:
        rec = _recording
        pending = _jobs_pending
    if rec:
        state = State.RECORDING
        label = _LABELS[state]
        if pending:
            label += f"  (+{pending} transcribing)"
    elif pending:
        state = State.PROCESSING
        label = _LABELS[state]
        if pending > 1:
            label += f"  ({pending} queued)"
    else:
        state = State.IDLE
        label = _LABELS[state]
    log(f"[main] {label}")
    if _tray is not None:
        _tray.icon = _ICONS[state]
        _tray.title = label


def _notify(message: str, title: str = "Meeting Transcriber") -> None:
    """Balloon notification from the tray icon. Never raises into a caller."""
    if _tray is None:
        return
    try:
        _tray.notify(message, title)
    except Exception as exc:
        log(f"[main] WARNING: tray notification failed: {exc}", err=True)


# ----------------------------------------------------------------- hotkey handler

# Toggle requests and capture events flow through this queue to a dedicated
# worker thread. An entry is either the string "toggle"/"quit" or a
# (kind, message) tuple from the capture thread.
_actions: queue.Queue = queue.Queue()


def _on_hotkey() -> None:
    """Called from the hotkey listener's message-pump thread. No real work here —
    a slow callback delays the next press."""
    _actions.put("toggle")


def _action_worker() -> None:
    """Serializes start/stop toggles so rapid double-presses cannot race."""
    while True:
        action = _actions.get()
        if action == "quit":
            return
        try:
            if isinstance(action, tuple):
                _handle_capture_event(*action)
            else:
                _handle_toggle()
        except Exception as exc:
            log(f"[main] ERROR: action handling failed: {exc}", err=True)
            traceback.print_exc(file=sys.stderr)


def _handle_capture_event(kind: str, message: str) -> None:
    """React to something the capture thread reported while recording."""
    if kind == "error":
        log(f"[main] ERROR: audio capture failed: {message}", err=True)
        with _state_lock:
            rec = _recording
        if not rec:
            return
        _notify(f"Recording stopped — audio capture failed:\n{message}")
        _stop_recording(reason=f"audio capture failed: {message}")
    else:
        log(f"[main] WARNING: {message}", err=True)
        _notify(message)


def _handle_toggle() -> None:
    with _state_lock:
        currently_recording = _recording
    if currently_recording:
        _stop_recording()
    else:
        _start_recording()


def _start_recording() -> None:
    global _capture, _start_time, _recording

    with _state_lock:
        _recording = True
    _refresh_indicator()
    try:
        cfg = _get_config()
        device_name = cfg.get("audio", {}).get("output_device_name") or None
        cap = AudioCapture(
            output_device_name=device_name,
            level_callback=_overlay.set_level if _overlay is not None else None,
            on_event=lambda kind, msg: _actions.put((kind, msg)),
        )
        if _overlay is not None:
            _overlay.show()
        cap.start()
    except Exception as exc:
        log(f"[main] ERROR: Could not start recording: {exc}", err=True)
        traceback.print_exc(file=sys.stderr)
        _notify(f"Could not start recording:\n{exc}")
        if _overlay is not None:
            _overlay.hide()
        with _state_lock:
            _recording = False
        _refresh_indicator()
        return
    with _state_lock:
        _capture = cap
        _start_time = datetime.now()


def _stop_recording(reason: str | None = None) -> None:
    """Stop capture and queue the audio for transcription.

    *reason* is set when something other than the user ended the recording; it
    is carried into the note as a capture warning.
    """
    global _capture, _start_time, _recording, _jobs_pending

    with _state_lock:
        cap_to_stop = _capture
        started_at = _start_time
        _capture = None
        _recording = False
    if cap_to_stop is None:
        # Start never completed (or failed); nothing to process.
        _refresh_indicator()
        return
    meeting_notes: dict[str, str] = {"attendees": "", "operator_notes": ""}
    if _overlay is not None:
        meeting_notes = _overlay.get_notes()
        _overlay.hide()
    # Stop the capture NOW (releases mic and finalizes buffers), then hand
    # the finished audio to the processing worker — a new meeting can be
    # recorded while this one waits its turn to transcribe.
    log("[main] Stopping audio capture." + (f" Reason: {reason}" if reason else ""))
    audio_array, temp_file, warnings = cap_to_stop.stop()
    if reason:
        warnings = list(warnings) + [reason]
    with _state_lock:
        _jobs_pending += 1
    _jobs.put((audio_array, temp_file, started_at, meeting_notes, warnings))
    _refresh_indicator()


def _process_worker() -> None:
    """Drains the transcription queue one job at a time (GPU fits only one)."""
    global _jobs_pending
    while True:
        job = _jobs.get()
        if job is None:
            return
        audio_array, temp_file, started_at, meeting_notes, warnings = job
        try:
            _process(audio_array, temp_file, started_at, meeting_notes, warnings)
        except Exception as exc:
            log(f"[main] ERROR: processing job failed: {exc}", err=True)
            traceback.print_exc(file=sys.stderr)
        finally:
            with _state_lock:
                _jobs_pending = max(0, _jobs_pending - 1)
            _refresh_indicator()


# --------------------------------------------------------------- process pipeline

def _wav_peak_and_duration(path: str) -> tuple[float, float]:
    """Peak amplitude and duration of a 16-bit mono WAV, scanned in blocks.

    Blockwise so an hour-long session (~67 MB on disk, ~230 MB as float32) is
    never materialized just to ask whether it is silent.
    """
    with wave.open(path, "rb") as wf:
        rate = wf.getframerate() or SAMPLE_RATE
        duration = wf.getnframes() / float(rate)
        if wf.getsampwidth() != 2:
            # Unknown format — say "loud" rather than falsely declare silence.
            return 1.0, duration
        peak = 0
        while True:
            block = wf.readframes(1 << 16)
            if not block:
                break
            samples = np.frombuffer(block, dtype=np.int16).astype(np.int32)
            if len(samples):
                peak = max(peak, int(np.max(np.abs(samples))))
    return peak / 32_768.0, duration


def _audio_peak_and_duration(audio_array, temp_file: str | None) -> tuple[float, float]:
    """Loudest sample (0.0–1.0) and length in seconds of a captured session."""
    if temp_file is not None:
        return _wav_peak_and_duration(temp_file)
    if audio_array is None or len(audio_array) == 0:
        return 0.0, 0.0
    return float(np.max(np.abs(audio_array))), len(audio_array) / float(SAMPLE_RATE)


def _retain_audio(
    cfg: dict,
    audio_array,
    temp_file: str | None,
    started_at: datetime,
) -> tuple[str | None, str | None]:
    """Persist the session audio per ``[recovery]`` config.

    Returns ``(retained_path, temp_to_delete)``. Retention moves the temp file
    when one exists, so on success there is nothing left to delete; when
    retention is off or fails, the temp file comes back for disposal.
    """
    retain_days = int(cfg.get("recovery", {}).get("retain_audio_days", 0) or 0)
    if retain_days <= 0:
        return None, temp_file

    name_stem = (
        f"{started_at.strftime('%Y-%m-%d')}_meeting-{started_at.strftime('%H%M')}"
    )
    retained: str | None = None
    to_delete: str | None = None
    try:
        retained = retain_session_audio(
            audio_array, temp_file, _recovery_audio_dir(cfg), name_stem
        )
        if retained:
            log(f"[main] Audio retained ({retain_days}d): {retained}")
    except Exception as exc:
        log(f"[main] WARNING: Could not retain audio: {exc}", err=True)
        to_delete = temp_file  # retention failed — clean up the temp
    _prune_old_audio(cfg)
    return retained, to_delete


def _process(
    audio_array,
    temp_file: str | None,
    started_at: datetime | None,
    meeting_notes: dict[str, str] | None = None,
    capture_warnings: list[str] | None = None,
) -> None:
    """Transcribe → write note → dispose temp audio. Runs on the process worker."""
    if started_at is None:
        started_at = datetime.now()

    cfg = _get_config()
    vault_path = cfg.get("paths", {}).get("vault_meetings_path", "")
    fallback = cfg.get("paths", {}).get("fallback_folder", "")
    meeting_notes = meeting_notes or {"attendees": "", "operator_notes": ""}
    capture_warnings = list(capture_warnings or [])

    if temp_file is not None:
        log(f"[main] Processing: audio saved to temp file: {temp_file}")
    elif audio_array is not None:
        log(
            f"[main] Processing: in-memory audio samples: {len(audio_array)} "
            f"({len(audio_array) / float(SAMPLE_RATE):.1f}s)."
        )

    temp_to_delete: str | None = None

    try:
        # Check the audio before spending a model on it. A dead mic and silent
        # loopbacks used to produce a normal-looking note with an empty
        # transcript — four of them on 2026-09-11 — with nothing to say the
        # meeting had been lost.
        peak, measured = _audio_peak_and_duration(audio_array, temp_file)
        reason: str | None = None
        if measured < 1.0:
            reason = (
                f"Nothing to transcribe: the recording is only {measured:.1f}s "
                f"long (peak {peak:.3f})."
            )
        elif peak < AudioCapture._SILENCE_PEAK:
            reason = (
                f"No audio captured (peak {peak:.3f}, duration {measured:.1f}s). "
                "Microphone and system audio both delivered silence."
            )
        if reason is not None:
            log(f"[main] WARNING: {reason} Writing a failure note.", err=True)
            retained, temp_to_delete = _retain_audio(
                cfg, audio_array, temp_file, started_at
            )
            _notify(f"Meeting NOT transcribed — {reason}")
            write_failure_note(
                started_at,
                "\n".join([reason] + capture_warnings),
                vault_path,
                fallback,
                attendees=meeting_notes.get("attendees", ""),
                operator_notes=meeting_notes.get("operator_notes", ""),
                audio_path=retained or temp_file,
            )
            return

        # Imported here, not at module scope: this is the 6 GB / 75 s cold
        # import chain, and paying it at startup left the hotkey dead until it
        # finished. Python caches the module, so it costs nothing after the
        # warm-up thread (or the first meeting) has done it once.
        from transcribe import transcribe

        audio_input = temp_file if temp_file is not None else audio_array
        log("[main] Processing: transcription starting.")
        transcript, duration = transcribe(
            audio_input,
            diarization_cfg=cfg.get("diarization", {}),
            engine=cfg.get("transcription", {}).get("engine", "faster-whisper"),
            language=cfg.get("transcription", {}).get("language", ""),
        )
        log(
            f"[main] Processing: transcription finished; duration={duration:.1f}s, "
            f"chars={len(transcript)}."
        )

        # Surface capture problems in the note itself. This only prepends lines
        # to the transcript text — the note template is untouched.
        notices = list(capture_warnings)
        if not transcript.strip():
            log(
                "[main] WARNING: engine returned an empty transcript for "
                f"non-silent audio (peak {peak:.3f}).",
                err=True,
            )
            notices.insert(0, "transcript came back empty.")
        if notices:
            banner = "\n".join(f"> Capture warning: {note}" for note in notices)
            transcript = f"{banner}\n\n{transcript}" if transcript else banner
            _notify("Meeting transcribed with capture warnings — check the note.")

        write_note(
            transcript,
            started_at,
            duration,
            vault_path,
            fallback,
            attendees=meeting_notes.get("attendees", ""),
            operator_notes=meeting_notes.get("operator_notes", ""),
        )

        # Retain the source audio for a rolling window so a garbled transcript
        # (e.g. a wrong-language hallucination) can be re-run. When retention is
        # off (retain_audio_days = 0) the temp file is deleted as before.
        _, temp_to_delete = _retain_audio(cfg, audio_array, temp_file, started_at)

    except Exception as exc:
        log(f"[main] ERROR: Transcription failed: {exc}", err=True)
        traceback.print_exc(file=sys.stderr)

        # Make the session recoverable: keep the temp file if there is one,
        # otherwise dump the in-memory buffer to a WAV before anything else.
        recovery_path = temp_file
        if recovery_path is None and audio_array is not None and len(audio_array) > 0:
            try:
                recovery_path = save_recovery_wav(audio_array)
                log(f"[main] In-memory audio dumped for recovery: {recovery_path}",
                    err=True)
            except Exception as save_exc:
                log(f"[main] ERROR: Could not save recovery audio: {save_exc}",
                    err=True)
        elif recovery_path:
            log(f"[main] Audio retained for recovery: {recovery_path}", err=True)

        _notify(f"Transcription failed:\n{exc}")
        try:
            write_failure_note(
                started_at,
                "\n".join([str(exc)] + capture_warnings),
                vault_path,
                fallback,
                attendees=meeting_notes.get("attendees", ""),
                operator_notes=meeting_notes.get("operator_notes", ""),
                audio_path=recovery_path,
            )
        except Exception as write_exc:
            log(f"[main] ERROR: Could not write failure note: {write_exc}", err=True)
            traceback.print_exc(file=sys.stderr)

    finally:
        # Explicit cleanup — log the outcome so a missed delete is visible.
        if temp_to_delete:
            if os.path.exists(temp_to_delete):
                try:
                    os.remove(temp_to_delete)
                    log(f"[main] Temp audio disposed: {temp_to_delete}")
                except Exception as del_exc:
                    log(
                        f"[main] WARNING: Failed to delete temp audio "
                        f"{temp_to_delete}: {del_exc}",
                        err=True,
                    )
            else:
                log(
                    f"[main] WARNING: Expected temp audio not found at {temp_to_delete}",
                    err=True,
                )


# ------------------------------------------------------------------ tray menu

def _toggle_menu_text(item: pystray.MenuItem) -> str:
    with _state_lock:
        return "Stop recording" if _recording else "Start recording"


def _toggle_handler(icon: pystray.Icon, item: pystray.MenuItem) -> None:
    """Tray fallback for the hotkey — same path, same serialization."""
    _actions.put("toggle")


def _reregister_handler(icon: pystray.Icon, item: pystray.MenuItem) -> None:
    if _listener is None:
        return
    _listener.stop()
    err = _listener.start()
    if err is None:
        log(f"[main] Hotkey re-registered: {_hotkey_combo}")
        _notify(f"Hotkey {_hotkey_combo} is registered.")
    else:
        text = hotkey.error_text(err)
        log(f"[main] ERROR: could not register hotkey '{_hotkey_combo}': {text}",
            err=True)
        _notify(f"Could not register {_hotkey_combo} — {text}")


# ----------------------------------------------------------------------- quit

def _rescue_recording_on_quit() -> None:
    """Quitting mid-recording used to throw the audio away. Write it to the
    recovery folder instead; transcription is skipped because the process
    worker is a daemon and dies with the interpreter."""
    global _capture, _start_time, _recording

    with _state_lock:
        cap_to_stop = _capture
        started_at = _start_time or datetime.now()
        _capture = None
        _recording = False
    if cap_to_stop is None:
        return
    if _overlay is not None:
        _overlay.hide()
    try:
        audio_array, temp_file, _warnings = cap_to_stop.stop()
        cfg = _get_config()
        name_stem = (
            f"{started_at.strftime('%Y-%m-%d')}_meeting-{started_at.strftime('%H%M')}"
        )
        retained = retain_session_audio(
            audio_array, temp_file, _recovery_audio_dir(cfg), name_stem
        )
        if retained:
            log(f"[main] Quit while recording — audio saved for recovery: {retained}",
                err=True)
            _notify(f"Recording saved for recovery:\n{retained}")
        else:
            log("[main] Quit while recording — no audio had been captured.", err=True)
    except Exception as exc:
        log(f"[main] ERROR: Could not save audio on quit: {exc}", err=True)
        traceback.print_exc(file=sys.stderr)


def _quit_handler(icon: pystray.Icon, item: pystray.MenuItem) -> None:
    with _state_lock:
        rec = _recording
        pending = _jobs_pending
    if rec:
        _rescue_recording_on_quit()
    if pending:
        log(
            f"[main] WARNING: Quitting with {pending} queued/in-flight "
            "transcription(s). These will be discarded.",
            err=True,
        )
    if _listener is not None:
        _listener.stop()
    _actions.put("quit")
    if _overlay is not None:
        _overlay.stop()
    icon.stop()


# ----------------------------------------------------------------------- main

def _prune_old_logs(days: int = 30) -> None:
    log_dir = os.path.join(_SCRIPT_DIR, "Logs")
    if not os.path.isdir(log_dir):
        return
    cutoff = time.time() - days * 86_400
    for name in os.listdir(log_dir):
        path = os.path.join(log_dir, name)
        try:
            if name.endswith(".log") and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _recovery_audio_dir(cfg: dict) -> str:
    """Resolve the folder retained meeting audio is kept in (config or default)."""
    configured = cfg.get("recovery", {}).get("audio_folder", "")
    if configured:
        return configured
    return os.path.join(_SCRIPT_DIR, "Recovery", "audio")


def _prune_old_audio(cfg: dict) -> None:
    """Delete retained meeting audio older than recovery.retain_audio_days."""
    days = int(cfg.get("recovery", {}).get("retain_audio_days", 0) or 0)
    if days <= 0:
        return
    audio_dir = _recovery_audio_dir(cfg)
    if not os.path.isdir(audio_dir):
        return
    cutoff = time.time() - days * 86_400
    for name in os.listdir(audio_dir):
        if not name.lower().endswith(".wav"):
            continue
        path = os.path.join(audio_dir, name)
        try:
            if os.path.getmtime(path) < cutoff:
                os.remove(path)
        except OSError:
            pass


def _quiet_noisy_libraries() -> None:
    """Silence known-harmless startup warnings from the ML stack so the log
    stays readable. Each suppression is deliberately narrow:
    - pyannote's torchcodec/FFmpeg wall of text: we never decode files with
      pyannote (audio is always passed in-memory), so the warning is moot.
    - torch's triton flop-counter notice: optional profiling extra.
    - lightning's checkpoint-upgrade notice for whisperx's bundled VAD model.
    """
    import logging
    import warnings

    warnings.filterwarnings(
        "ignore", category=UserWarning, module=r"pyannote\.audio\.core\.io"
    )
    logging.getLogger("torch.utils.flop_counter").setLevel(logging.ERROR)
    logging.getLogger("lightning.pytorch.utilities.migration.utils").setLevel(
        logging.ERROR
    )


def _warm_model() -> None:
    """Import the ML stack and load the configured engine's model.

    Runs on a background thread so startup does not wait on it. Everything
    heavy — including the `transcribe` import itself — happens here. A meeting
    started before this finishes simply blocks on the engine's model lock.
    """
    engine = "faster-whisper"
    started = time.monotonic()
    try:
        _quiet_noisy_libraries()
        engine = _get_config().get("transcription", {}).get("engine", "faster-whisper")
        if engine == "whisperx":
            import whisperx_engine

            whisperx_engine._get_model()
        else:
            import transcribe

            transcribe.preload_model()
        log(
            f"[main] Transcription model preloaded ({engine}) in "
            f"{time.monotonic() - started:.1f}s."
        )
    except Exception as exc:
        log(f"[main] WARNING: model preload failed ({engine}): {exc}", err=True)


def main() -> None:
    global _tray, _overlay, _listener, _hotkey_combo

    _setup_logging()
    _prune_old_logs()
    log("[main] Meeting Transcriber starting.")

    _overlay = ActivityOverlay()
    _overlay.start()

    cfg = _get_config()
    _hotkey_combo = cfg.get("hotkey", {}).get("hotkey", "ctrl+shift+r")

    threading.Thread(target=_action_worker, daemon=True, name="action-worker").start()
    threading.Thread(target=_process_worker, daemon=True, name="process-worker").start()

    # Register the hotkey before anything slow, so it is live as early as
    # possible. A failure here is not fatal — the tray menu does the same job.
    hotkey_error: int | None = None
    try:
        _listener = HotkeyListener(_hotkey_combo, _on_hotkey)
        hotkey_error = _listener.start()
    except ValueError as exc:
        hotkey_error = hotkey.ERROR_LISTENER_TIMEOUT
        log(f"[main] ERROR: bad hotkey in config.toml: {exc}", err=True)

    if hotkey_error is None:
        log(f"[main] Hotkey: {_hotkey_combo}  (first press: start | second press: stop)")
    else:
        log(
            f"[main] ERROR: could not register hotkey '{_hotkey_combo}': "
            f"{hotkey.error_text(hotkey_error)}. Use the tray menu instead.",
            err=True,
        )
    log("[main] Right-click the tray icon to start/stop recording or quit.")

    menu = pystray.Menu(
        pystray.MenuItem(_toggle_menu_text, _toggle_handler),
        pystray.MenuItem("Re-register hotkey", _reregister_handler),
        pystray.MenuItem("Quit", _quit_handler),
    )
    _tray = pystray.Icon(
        name="meeting-transcriber",
        icon=_ICONS[State.IDLE],
        title=_LABELS[State.IDLE],
        menu=menu,
    )

    def _on_tray_ready(icon: pystray.Icon) -> None:
        icon.visible = True
        if hotkey_error is not None:
            _notify(
                f"Hotkey {_hotkey_combo} could not be registered "
                f"({hotkey.error_text(hotkey_error)}). Another app may own it. "
                "Start and stop recording from this tray menu."
            )

    threading.Thread(target=_warm_model, daemon=True, name="model-warmup").start()
    if hotkey_error is None:
        log("[main] Ready: hotkey live, transcription model warming in background.")
    else:
        log("[main] Ready: tray menu live (no hotkey), model warming in background.")

    # pystray.Icon.run() blocks the main thread on Windows (required by Win32 message loop).
    _tray.run(setup=_on_tray_ready)

    if _listener is not None:
        _listener.stop()
    log("[main] Meeting Transcriber stopped.")


if __name__ == "__main__":
    main()
