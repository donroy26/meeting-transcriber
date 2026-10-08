"""
capture.py — Audio capture: mic + WASAPI loopback, mixed to 16 kHz mono.

Normal path: all audio stays in memory (numpy buffer).
Long-session guard: when accumulated audio exceeds CHUNK_THRESHOLD_SECONDS, the
buffer is flushed to a temp WAV file on disk to keep live memory bounded. The temp
file path is returned to the caller, which must delete it after successful use.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import traceback
import wave
from collections.abc import Callable
from math import gcd

import numpy as np

try:
    import pyaudiowpatch as pyaudio  # Windows: PyAudio fork with WASAPI loopback
except ImportError:
    import pyaudio  # macOS/Linux: plain PyAudio; see _is_loopback()

_IS_WINDOWS = sys.platform == "win32"
# macOS has no loopback endpoints. A virtual output device shows up as an
# ordinary input instead; BlackHole is the one install-mac.sh sets up.
VIRTUAL_LOOPBACK_NAME = "blackhole"

# Whisper expects 16 kHz mono float32.
SAMPLE_RATE = 16_000

# Frames per PyAudio read call.
READ_CHUNK = 1_024

# Long-session guard: flush in-memory buffer to temp WAV when accumulated audio
# exceeds this many seconds. At 16 kHz mono float32 the buffer is ~115 MB / 30 min;
# each flush keeps live memory under ~10 MB while the temp file accumulates the session.
CHUNK_THRESHOLD_SECONDS = 1_800  # 30 minutes

# Mid-session dead-mic check: after roughly this many reads (~3 s of audio), a
# mic peak still at exact digital zero means the device is delivering nothing.
_SILENCE_CHECK_READS = int(3 * SAMPLE_RATE / READ_CHUNK)
_DIGITAL_SILENCE_PEAK = 1e-6


class AudioCapture:
    """
    Captures microphone + WASAPI loopback simultaneously and mixes them to a single
    mono float32 stream at SAMPLE_RATE.

    Usage::

        cap = AudioCapture(output_device_name="Speakers (Realtek HD Audio)")
        cap.start()
        # ... user records the meeting ...
        audio_array, temp_file, warnings = cap.stop()
        # Exactly one of the first two return values is not None.
        # If temp_file is not None the caller owns its deletion.
    """

    def __init__(
        self,
        output_device_name: str | None = None,
        level_callback: Callable[[float], None] | None = None,
        on_event: Callable[[str, str], None] | None = None,
    ) -> None:
        # Empty/auto/all means "capture every loopback device we can open."
        # This is intentionally the default because meetings may move between
        # speakers, monitor audio, USB headsets, and Bluetooth headphones.
        self._output_device_name = output_device_name
        self._level_callback = level_callback
        # Called as on_event(kind, message) from the capture thread the moment
        # something goes wrong, so the owner can react while the meeting is
        # still happening instead of discovering it at stop(). Kinds:
        #   "error"   — the capture thread is dying; nothing more is recorded.
        #   "warning" — recording continues but the result may be unusable.
        self._on_event = on_event
        # Session warnings, returned by stop() so they can be surfaced in the
        # note. Appended only by the capture thread.
        self._warnings: list[str] = []
        self._pa: pyaudio.PyAudio | None = None
        self._buffer: list[np.ndarray] = []
        self._buffer_seconds: float = 0.0
        self._recording = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._temp_file: str | None = None
        self._wav_writer: wave.Wave_write | None = None
        # Set when stop() gives up waiting; the capture thread then owns
        # PyAudio termination instead of leaking it.
        self._orphaned = False

    # ------------------------------------------------------------------ public

    def start(self) -> None:
        """Start capture in a daemon background thread."""
        if self._recording:
            return
        self._pa = pyaudio.PyAudio()
        self._buffer = []
        self._buffer_seconds = 0.0
        self._recording = True
        self._temp_file = None
        self._warnings = []
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="audio-capture"
        )
        self._thread.start()

    def stop(self) -> tuple[np.ndarray | None, str | None, list[str]]:
        """
        Stop capture and return ``(audio_array, temp_file_path, warnings)``.

        Exactly one of the first two values is ``None``.  If *temp_file_path*
        is returned the caller is responsible for deleting it after use.
        *warnings* lists everything that went wrong during the session but did
        not stop it — an empty list means a clean recording.
        """
        self._recording = False
        if self._thread:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                print(
                    "[capture] WARNING: Capture thread did not stop within 10s; "
                    "it will release PortAudio itself when the read returns.",
                    file=sys.stderr,
                )
                self._orphaned = True
                audio, temp = self._assemble()
                return audio, temp, list(self._warnings)
            self._thread = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        audio, temp = self._assemble()
        return audio, temp, list(self._warnings)

    # ----------------------------------------------------------------- private

    def _warn(self, message: str, *, notify: bool = False) -> None:
        """Log a session warning and keep it for ``stop()`` to return.

        *notify* also pushes it to ``on_event`` so the owner can alert the user
        mid-recording — use it only for problems worth interrupting for.
        """
        print(f"[capture] WARNING: {message}", file=sys.stderr)
        self._warnings.append(message)
        if notify:
            self._fire("warning", message)

    def _fire(self, kind: str, message: str) -> None:
        """Hand a live capture event to the owner. Never raises into capture."""
        if self._on_event is None:
            return
        try:
            self._on_event(kind, message)
        except Exception:
            traceback.print_exc(file=sys.stderr)

    def _capture_loop(self) -> None:
        mic_stream = None
        loopbacks: list[dict] = []

        try:
            mic_idx = self._pa.get_default_input_device_info()["index"]
            mic_info = self._pa.get_device_info_by_index(mic_idx)
            print(f"[capture] Microphone input: {mic_info['name']}", flush=True)
            mic_stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=mic_idx,
                frames_per_buffer=READ_CHUNK,
            )

            for lb_info in self._find_loopback_devices():
                try:
                    lb_rate = int(lb_info.get("defaultSampleRate", SAMPLE_RATE))
                    lb_channels = max(1, int(lb_info.get("maxInputChannels", 1)))
                    lb_stream = self._pa.open(
                        format=pyaudio.paFloat32,
                        channels=lb_channels,
                        rate=lb_rate,
                        input=True,
                        input_device_index=int(lb_info["index"]),
                        frames_per_buffer=READ_CHUNK,
                    )
                    loopbacks.append(
                        {
                            "stream": lb_stream,
                            "name": lb_info["name"],
                            "rate": lb_rate,
                            "channels": lb_channels,
                            # Backlog and resampler history carried between reads
                            # so a device that falls behind catches up instead of
                            # drifting, and block seams stay continuous.
                            "state": new_loopback_state(),
                            # Loudest sample seen from this device this session —
                            # used to report which streams were actually live.
                            "peak": 0.0,
                        }
                    )
                    print(f"[capture] PC/caller audio loopback: {lb_info['name']}", flush=True)
                except Exception as exc:
                    print(
                        f"[capture] Loopback unavailable for '{lb_info['name']}' ({exc}); skipping.",
                        file=sys.stderr,
                    )
            if not loopbacks:
                print("[capture] No loopback devices opened; recording mic-only.", file=sys.stderr)

            mic_peak = 0.0
            reads = 0
            silence_checked = False
            while self._recording:
                mic_chunk = np.frombuffer(
                    mic_stream.read(READ_CHUNK, exception_on_overflow=False),
                    dtype=np.float32,
                )
                if len(mic_chunk):
                    mic_peak = max(mic_peak, float(np.max(np.abs(mic_chunk))))

                # Early dead-mic alarm. Exact digital zero for the first few
                # seconds means the device is delivering nothing at all (the
                # 2026-09-11 session recorded four meetings this way) — a quiet
                # room still reads well above zero. Worth interrupting for,
                # because the fix is to restart the app before the meeting
                # gets going. Recording continues either way: loopback audio
                # may still be worth keeping.
                reads += 1
                if not silence_checked and reads >= _SILENCE_CHECK_READS:
                    silence_checked = True
                    if mic_peak < _DIGITAL_SILENCE_PEAK:
                        self._warn(
                            "Microphone is delivering digital silence — your "
                            "voice is NOT being recorded. Check the input "
                            "device and mute switch, then restart the app.",
                            notify=True,
                        )

                lb_mix = np.zeros(len(mic_chunk), dtype=np.float32)
                for loopback in loopbacks:
                    try:
                        chunk = _read_loopback_chunk(
                            loopback["stream"],
                            int(loopback["channels"]),
                            int(loopback["rate"]),
                            len(mic_chunk),
                            loopback["state"],
                        )
                        lb_mix += chunk
                        if len(chunk):
                            loopback["peak"] = max(
                                loopback["peak"], float(np.max(np.abs(chunk)))
                            )
                    except Exception as exc:
                        print(
                            f"[capture] Loopback read failed for '{loopback['name']}' ({exc}); "
                            "continuing without that chunk.",
                            file=sys.stderr,
                        )

                mixed = np.clip(mic_chunk + lb_mix, -1.0, 1.0)
                self._report_level(mixed)

                chunk_secs = len(mixed) / SAMPLE_RATE
                with self._lock:
                    self._buffer.append(mixed)
                    self._buffer_seconds += chunk_secs
                    if self._buffer_seconds >= CHUNK_THRESHOLD_SECONDS:
                        self._flush_to_temp_locked()

            self._log_stream_summary(mic_peak, loopbacks)

        except Exception as exc:
            # Nothing is being recorded from here on. Tell the owner: without
            # this the tray sat on red "RECORDING" for the whole meeting while
            # the capture thread was already dead (2026-08-13, 2026-09-11).
            print(f"[capture] Capture thread error: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            self._fire("error", str(exc))
        finally:
            streams = [mic_stream] + [lb["stream"] for lb in loopbacks]
            for stream in streams:
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass
            if self._orphaned and self._pa is not None:
                try:
                    self._pa.terminate()
                except Exception:
                    pass
                self._pa = None

    # Below this peak a stream is treated as effectively silent (nothing played
    # through it this session). ~ -60 dBFS.
    _SILENCE_PEAK = 0.001


    def _log_stream_summary(self, mic_peak: float, loopbacks: list[dict]) -> None:
        """Report which streams carried audio and warn on a mic-only session.

        The app mixes the mic with every system-audio (loopback) device it can
        open, so whichever device is actually playing is captured. This makes
        the outcome visible: if the mic had speech but every loopback stayed
        silent, the caller's side was NOT captured — the usual cause is audio
        routed to a device with no loopback endpoint (some Bluetooth headsets in
        hands-free/mic mode, or an external monitor/dock).
        """
        live = [lb for lb in loopbacks if lb["peak"] >= self._SILENCE_PEAK]
        summary = ", ".join(f"{lb['name']}={lb['peak']:.3f}" for lb in loopbacks) or "none"
        print(
            f"[capture] Stream levels — mic peak={mic_peak:.3f}; "
            f"loopbacks: {summary}",
            flush=True,
        )
        mic_had_speech = mic_peak >= self._SILENCE_PEAK
        if not mic_had_speech:
            self._warn(
                f"your microphone recorded nothing this session (peak "
                f"{mic_peak:.3f}). Your side of the conversation is missing."
            )
        elif not live:
            self._warn(
                "caller/PC audio was NOT captured this session (all system-audio "
                "loopbacks were silent). Only your microphone was recorded. If you "
                "used Bluetooth headphones/mic or an external monitor's speakers, "
                "Windows may not expose a loopback for that device — route meeting "
                "audio through the laptop speakers to capture both sides."
            )

    def _find_loopback_devices(self) -> list[dict]:
        """Return loopback devices to capture for PC/caller audio."""
        capture_all = not self._output_device_name or self._output_device_name.lower() in {
            "all",
            "auto",
            "*",
        }
        matches = []
        for i in range(self._pa.get_device_count()):
            info = self._pa.get_device_info_by_index(i)
            if not _is_loopback(info, None if capture_all else self._output_device_name):
                continue
            if capture_all or self._output_device_name.lower() in info["name"].lower():
                matches.append(info)
        if self._output_device_name and not matches:
            print(
                f"[capture] Named loopback device '{self._output_device_name}' not found; "
                "recording mic-only.",
                file=sys.stderr,
            )
        return matches

    def _report_level(self, audio: np.ndarray) -> None:
        if self._level_callback is None or len(audio) == 0:
            return
        try:
            rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
            peak = float(np.max(np.abs(audio)))
            level = min(1.0, max(rms * 8.0, peak * 0.8))
            self._level_callback(level)
        except Exception:
            pass

    def _flush_to_temp_locked(self) -> None:
        """Flush in-memory buffer to the temp WAV file. Lock must be held by caller.

        The wave writer stays open for the whole session so each flush is an
        incremental frame append (the wave module fixes up the header on close)
        rather than a full-file read-and-rewrite.
        """
        if not self._buffer:
            return
        audio = np.concatenate(self._buffer)
        self._buffer = []
        self._buffer_seconds = 0.0
        pcm = (np.clip(audio, -1.0, 1.0) * 32_767).astype(np.int16)

        if self._wav_writer is None:
            fd, self._temp_file = tempfile.mkstemp(suffix=".wav", prefix="mtranscriber_")
            os.close(fd)
            self._wav_writer = wave.open(self._temp_file, "wb")
            self._wav_writer.setnchannels(1)
            self._wav_writer.setsampwidth(2)
            self._wav_writer.setframerate(SAMPLE_RATE)
        self._wav_writer.writeframes(pcm.tobytes())

    def _assemble(self) -> tuple[np.ndarray | None, str | None]:
        """Consolidate capture result after stop()."""
        with self._lock:
            if self._wav_writer is not None:
                # Long-session path: flush any remaining in-memory audio, close
                # the writer (finalizes the WAV header), return the file path.
                if self._buffer:
                    self._flush_to_temp_locked()
                try:
                    self._wav_writer.close()
                finally:
                    self._wav_writer = None
                return None, self._temp_file
            # Normal path: return concatenated in-memory array.
            if not self._buffer:
                return np.zeros(0, dtype=np.float32), None
            return np.concatenate(self._buffer), None


# -------------------------------------------------------------------- helpers

# Anti-alias filter design. The filter runs at the *input* rate, so the
# transition band is a comfortable fraction of the input Nyquist and a few
# hundred taps suffice. The passband stops at 85% of the output Nyquist
# (6.8 kHz for a 16 kHz target — above the useful band for speech) and the
# stopband begins at the output Nyquist itself, which is what keeps higher
# content from folding back down into the audible band.
_STOPBAND_DB = 70.0
_PASSBAND_FRACTION = 0.85
_MAX_TAPS = 2_047

_FILTER_CACHE: dict[tuple[int, int], np.ndarray] = {}


def _antialias_filter(from_rate: int, to_rate: int) -> np.ndarray:
    """
    Kaiser-windowed sinc low-pass for decimating *from_rate* to *to_rate*, sized
    by the standard Kaiser design formula. numpy only, and cached — the design
    cost is paid once per rate pair.
    """
    key = (from_rate, to_rate)
    h = _FILTER_CACHE.get(key)
    if h is None:
        in_nyq = from_rate / 2.0
        out_nyq = to_rate / 2.0
        passband = _PASSBAND_FRACTION * out_nyq
        transition = np.pi * (out_nyq - passband) / in_nyq
        beta = 0.1102 * (_STOPBAND_DB - 8.7)
        taps = int(np.ceil((_STOPBAND_DB - 8) / (2.285 * transition))) | 1
        taps = min(taps, _MAX_TAPS) | 1
        n = np.arange(taps, dtype=np.float64) - (taps - 1) / 2.0
        h = np.sinc(n * passband / in_nyq) * np.kaiser(taps, beta)
        h = (h / h.sum()).astype(np.float32)
        _FILTER_CACHE[key] = h
    return h


def _resample(data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """
    One-shot rate conversion.

    Band-limiting before the rate change is the whole point: plain interpolation
    folds everything above the output Nyquist back into the audible band at full
    amplitude. That is what made loopback (caller) audio sound garbled while the
    mic — resampled by the audio driver rather than by this code — stayed clean.
    """
    if from_rate == to_rate or len(data) == 0:
        return data.astype(np.float32)

    if to_rate < from_rate:  # downsampling: aliasing is the risk, so filter first
        h = _antialias_filter(from_rate, to_rate)
        data = np.convolve(data, h, mode="same")

    step = from_rate / to_rate
    n_out = max(1, int(len(data) / step))
    idx = np.arange(n_out, dtype=np.float64) * step
    idx = idx[idx <= len(data) - 1]
    return np.interp(idx, np.arange(len(data)), data).astype(np.float32)


# Cap on per-device backlog: beyond this the device is hopelessly behind and we
# drop the oldest audio rather than let latency (and memory) grow unboundedly.
_MAX_RESIDUAL_SECONDS = 10


def new_loopback_state() -> dict:
    """Per-device buffers carried between reads by ``_read_loopback_chunk``."""
    return {
        # Native-rate audio read but not yet band-limited.
        "raw": np.zeros(0, dtype=np.float32),
        # Filter history, so no block edge is ever filtered against zero padding.
        "fir": None,
        # Native-rate audio, band-limited, awaiting interpolation.
        "filt": np.zeros(0, dtype=np.float32),
        # Fractional read position within "filt" for the next output sample.
        "phase": 0.0,
        # SAMPLE_RATE audio not yet handed to the mixer.
        "out": np.zeros(0, dtype=np.float32),
    }


def _convert_pending(state: dict, rate: int) -> None:
    """Band-limit and rate-convert everything buffered, appending to ``state['out']``."""
    if rate == SAMPLE_RATE:
        state["out"] = np.concatenate([state["out"], state["raw"]])
        state["raw"] = np.zeros(0, dtype=np.float32)
        return

    h = _antialias_filter(rate, SAMPLE_RATE)
    if state["fir"] is None:
        state["fir"] = np.zeros(len(h) - 1, dtype=np.float32)

    block, state["raw"] = state["raw"], np.zeros(0, dtype=np.float32)
    if len(block):
        padded = np.concatenate([state["fir"], block])
        # 'valid' keeps only fully-filtered samples, so a block edge is never
        # filtered against zeros — that was the source of the per-read clicks.
        filtered = np.convolve(padded, h, mode="valid").astype(np.float32)
        state["fir"] = padded[-(len(h) - 1):]
        state["filt"] = np.concatenate([state["filt"], filtered])

    filt = state["filt"]
    if len(filt) < 2:
        return

    # Interpolation positions carry across calls, so there is no phase reset and
    # no accumulated length drift at read boundaries.
    step = rate / SAMPLE_RATE
    count = int(np.floor((len(filt) - 1 - state["phase"]) / step)) + 1
    if count <= 0:
        return
    idx = state["phase"] + step * np.arange(count, dtype=np.float64)
    out = np.interp(idx, np.arange(len(filt)), filt).astype(np.float32)
    state["out"] = np.concatenate([state["out"], out])

    next_pos = idx[-1] + step
    drop = min(int(np.floor(next_pos)), len(filt))
    state["filt"] = filt[drop:]
    state["phase"] = float(next_pos - drop)


def _read_loopback_chunk(
    stream,
    channels: int,
    rate: int,
    target_samples: int,
    state: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Read everything the loopback device has buffered, normalize to mono at
    SAMPLE_RATE, and return exactly *target_samples* of audio.

    Draining the full backlog every cycle keeps the device aligned with the mic:
    a device that briefly falls behind catches up via the backlog instead of
    accumulating unbounded latency in PortAudio's buffer.

    Rate conversion band-limits at the native rate and interpolates with a phase
    that carries across calls, so read boundaries introduce neither a filter
    discontinuity nor accumulated drift. Converting each read independently put
    an audible click at every seam.

    *state* (see ``new_loopback_state``) is mutated in place.
    """
    available = stream.get_read_available()
    if available > 0:
        raw = np.frombuffer(
            stream.read(available, exception_on_overflow=False),
            dtype=np.float32,
        )
        if channels > 1:
            usable = (len(raw) // channels) * channels
            raw = raw[:usable].reshape(-1, channels).mean(axis=1)
        state["raw"] = np.concatenate([state["raw"], raw.astype(np.float32)])

    _convert_pending(state, rate)

    # Bound the backlogs: past this the device is hopelessly behind, and old
    # audio is dropped rather than letting latency and memory grow unboundedly.
    # Discarding buffered audio invalidates the interpolation phase, so reset it.
    max_out = _MAX_RESIDUAL_SECONDS * SAMPLE_RATE
    if len(state["out"]) > max_out:
        state["out"] = state["out"][-max_out:]
    max_filt = _MAX_RESIDUAL_SECONDS * rate
    if len(state["filt"]) > max_filt:
        state["filt"] = state["filt"][-max_filt:]
        state["phase"] = 0.0

    out = state["out"]
    if len(out) >= target_samples:
        state["out"] = out[target_samples:]
        return out[:target_samples]
    state["out"] = np.zeros(0, dtype=np.float32)
    return np.pad(out, (0, target_samples - len(out))).astype(np.float32)


def _write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def save_recovery_wav(audio: np.ndarray) -> str:
    """Dump a float32 mono buffer to a temp WAV so a failed session is recoverable."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32_767).astype(np.int16)
    fd, path = tempfile.mkstemp(suffix=".wav", prefix="mtranscriber_recovery_")
    os.close(fd)
    _write_wav(path, pcm, SAMPLE_RATE)
    return path


def retain_session_audio(
    audio_array: np.ndarray | None,
    temp_file: str | None,
    dest_dir: str,
    name_stem: str,
) -> str | None:
    """
    Persist a session's audio into ``dest_dir`` (named ``<name_stem>.wav``) so a
    successful-but-garbled transcript can be re-run from the source.

    Moves the long-session temp WAV when one exists (no copy, no leftover temp);
    otherwise writes the in-memory buffer. Never overwrites an existing file.
    Returns the retained path, or ``None`` when there was no audio to keep.
    """
    if (temp_file is None or not os.path.exists(temp_file)) and (
        audio_array is None or len(audio_array) == 0
    ):
        return None

    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, f"{name_stem}.wav")
    counter = 2
    while os.path.exists(dest):
        dest = os.path.join(dest_dir, f"{name_stem}-{counter}.wav")
        counter += 1

    if temp_file is not None and os.path.exists(temp_file):
        shutil.move(temp_file, dest)
    else:
        pcm = (np.clip(audio_array, -1.0, 1.0) * 32_767).astype(np.int16)
        _write_wav(dest, pcm, SAMPLE_RATE)
    return dest


def _is_loopback(info: dict, wanted_name: str | None = None) -> bool:
    """Windows: a WASAPI loopback endpoint. Elsewhere: an input device whose
    name contains *wanted_name* (from config), or BlackHole when none is set."""
    if _IS_WINDOWS:
        return bool(info.get("isLoopbackDevice", False))
    if int(info.get("maxInputChannels", 0)) < 1:
        return False
    return (wanted_name or VIRTUAL_LOOPBACK_NAME).lower() in info["name"].lower()


def list_devices() -> None:
    """Print all audio devices — use to find output_device_name for config.toml."""
    pa = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        tag = " [LOOPBACK]" if _is_loopback(info) else ""
        print(
            f"  [{i:2d}] {info['name']}{tag}"
            f"  in:{info['maxInputChannels']}  out:{info['maxOutputChannels']}"
        )
    pa.terminate()


if __name__ == "__main__":
    if "--list-devices" in sys.argv:
        list_devices()
    else:
        print("Usage: python capture.py --list-devices")
