"""
capture.py — Audio capture: mic + WASAPI loopback, mixed to 16 kHz mono.

Normal path: all audio stays in memory (numpy buffer).
Long-session guard: when accumulated audio exceeds CHUNK_THRESHOLD_SECONDS, the
buffer is flushed to a temp WAV file on disk to keep live memory bounded. The temp
file path is returned to the caller, which must delete it after successful use.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import wave
from collections.abc import Callable

import numpy as np
import pyaudiowpatch as pyaudio

# Whisper expects 16 kHz mono float32.
SAMPLE_RATE = 16_000

# Frames per PyAudio read call.
READ_CHUNK = 1_024

# Long-session guard: flush in-memory buffer to temp WAV when accumulated audio
# exceeds this many seconds. At 16 kHz mono float32 the buffer is ~115 MB / 30 min;
# each flush keeps live memory under ~10 MB while the temp file accumulates the session.
CHUNK_THRESHOLD_SECONDS = 1_800  # 30 minutes


class AudioCapture:
    """
    Captures microphone + WASAPI loopback simultaneously and mixes them to a single
    mono float32 stream at SAMPLE_RATE.

    Usage::

        cap = AudioCapture(output_device_name="Speakers (Realtek HD Audio)")
        cap.start()
        # ... user records the meeting ...
        audio_array, temp_file = cap.stop()
        # Exactly one of the two return values is not None.
        # If temp_file is not None the caller owns its deletion.
    """

    def __init__(
        self,
        output_device_name: str | None = None,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        # Empty/auto/all means "capture every loopback device we can open."
        # This is intentionally the default because meetings may move between
        # speakers, monitor audio, USB headsets, and Bluetooth headphones.
        self._output_device_name = output_device_name
        self._level_callback = level_callback
        self._pa: pyaudio.PyAudio | None = None
        self._buffer: list[np.ndarray] = []
        self._buffer_seconds: float = 0.0
        self._recording = False
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._temp_file: str | None = None

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
        self._thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="audio-capture"
        )
        self._thread.start()

    def stop(self) -> tuple[np.ndarray | None, str | None]:
        """
        Stop capture and return ``(audio_array, temp_file_path)``.

        Exactly one of the two values is ``None``.  If *temp_file_path* is
        returned the caller is responsible for deleting it after use.
        """
        self._recording = False
        if self._thread:
            self._thread.join(timeout=10)
            if self._thread.is_alive():
                print(
                    "[capture] WARNING: Capture thread did not stop within 10s; "
                    "leaving PortAudio open until the read returns.",
                    file=sys.stderr,
                )
                return self._assemble()
            self._thread = None
        if self._pa:
            try:
                self._pa.terminate()
            except Exception:
                pass
            self._pa = None
        return self._assemble()

    # ----------------------------------------------------------------- private

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

            while self._recording:
                mic_chunk = np.frombuffer(
                    mic_stream.read(READ_CHUNK, exception_on_overflow=False),
                    dtype=np.float32,
                )

                lb_mix = np.zeros(len(mic_chunk), dtype=np.float32)
                for loopback in loopbacks:
                    try:
                        lb_mix += _read_loopback_chunk(
                            loopback["stream"],
                            int(loopback["channels"]),
                            int(loopback["rate"]),
                            len(mic_chunk),
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

        except Exception as exc:
            print(f"[capture] Capture thread error: {exc}", file=sys.stderr)
        finally:
            streams = [mic_stream] + [lb["stream"] for lb in loopbacks]
            for stream in streams:
                if stream is not None:
                    try:
                        stream.stop_stream()
                        stream.close()
                    except Exception:
                        pass

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
            if not info.get("isLoopbackDevice", False):
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
        """Flush in-memory buffer to the temp WAV file. Lock must be held by caller."""
        if not self._buffer:
            return
        audio = np.concatenate(self._buffer)
        self._buffer = []
        self._buffer_seconds = 0.0
        pcm = (np.clip(audio, -1.0, 1.0) * 32_767).astype(np.int16)

        if self._temp_file is None:
            fd, self._temp_file = tempfile.mkstemp(suffix=".wav", prefix="mtranscriber_")
            os.close(fd)
            _write_wav(self._temp_file, pcm, SAMPLE_RATE)
        else:
            _append_wav(self._temp_file, pcm, SAMPLE_RATE)

    def _assemble(self) -> tuple[np.ndarray | None, str | None]:
        """Consolidate capture result after stop()."""
        with self._lock:
            if self._temp_file is not None:
                # Long-session path: flush any remaining in-memory audio, return file path.
                if self._buffer:
                    self._flush_to_temp_locked()
                return None, self._temp_file
            # Normal path: return concatenated in-memory array.
            if not self._buffer:
                return np.zeros(0, dtype=np.float32), None
            return np.concatenate(self._buffer), None


# -------------------------------------------------------------------- helpers

def _resample(data: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    """Linear-interpolation resample — adequate for speech, avoids scipy dependency."""
    if from_rate == to_rate or len(data) == 0:
        return data
    n_out = max(1, round(len(data) * to_rate / from_rate))
    x_old = np.arange(len(data), dtype=np.float64)
    x_new = np.linspace(0, len(data) - 1, n_out)
    return np.interp(x_new, x_old, data).astype(np.float32)


def _read_loopback_chunk(
    stream,
    channels: int,
    rate: int,
    target_samples: int,
) -> np.ndarray:
    """Read one loopback chunk and normalize it to target_samples at SAMPLE_RATE."""
    target_frames = max(1, round(target_samples * rate / SAMPLE_RATE))
    available = stream.get_read_available()
    if available <= 0:
        return np.zeros(target_samples, dtype=np.float32)

    frames_to_read = min(target_frames, available)
    raw = np.frombuffer(
        stream.read(frames_to_read, exception_on_overflow=False),
        dtype=np.float32,
    )

    if channels > 1:
        usable = (len(raw) // channels) * channels
        raw = raw[:usable].reshape(-1, channels).mean(axis=1)

    if rate != SAMPLE_RATE:
        raw = _resample(raw, rate, SAMPLE_RATE)

    if len(raw) < target_samples:
        return np.pad(raw, (0, target_samples - len(raw))).astype(np.float32)
    return raw[:target_samples].astype(np.float32)


def _write_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm.tobytes())


def _append_wav(path: str, pcm: np.ndarray, rate: int) -> None:
    """Append 16-bit mono PCM frames to an existing WAV file."""
    with wave.open(path, "rb") as wf:
        existing = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    _write_wav(path, np.concatenate([existing, pcm]), rate)


def list_devices() -> None:
    """Print all audio devices — use to find output_device_name for config.toml."""
    pa = pyaudio.PyAudio()
    print("Available audio devices:")
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        tag = " [LOOPBACK]" if info.get("isLoopbackDevice") else ""
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
