"""
transcribe.py — Transcription via faster-whisper large-v3-turbo on CUDA float16.

Accepts either a float32 numpy array (normal path) or a WAV file path (long-session path).
Returns (transcript_str, duration_seconds).
No file writes. No network calls (model is pre-downloaded to HuggingFace cache).
"""

from __future__ import annotations

import os
import sys
import threading
import wave
from pathlib import Path
from typing import Union

import numpy as np

_DLL_DIR_HANDLES = []


def _add_nvidia_dll_dirs() -> None:
    """Let Windows find CUDA DLLs installed by NVIDIA Python wheels."""
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    site_packages = Path(sys.prefix) / "Lib" / "site-packages"
    for package in ("cublas", "cuda_nvrtc", "cuda_runtime", "cudnn"):
        dll_dir = site_packages / "nvidia" / package / "bin"
        if dll_dir.is_dir():
            os.environ["PATH"] = f"{dll_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(dll_dir)))


_add_nvidia_dll_dirs()

from faster_whisper import WhisperModel

MODEL_SIZE = "large-v3-turbo"
DEVICE = "cuda"
# int8_float16 halves VRAM vs float16 with negligible quality difference for
# meeting speech — headroom matters: a full GPU stalls Windows desktop
# compositing (system-wide freezes observed 2026-07-02).
COMPUTE_TYPE = "int8_float16"

# Beam size for decoding; 5 is the faster-whisper default.
BEAM_SIZE = 5

# VAD filter: skip non-speech segments to reduce hallucinations and improve speed.
VAD_FILTER = True
VAD_PARAMETERS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": float("inf"),
    "min_silence_duration_ms": 2_000,
    "speech_pad_ms": 400,
}

# Feeding prior segment tokens as context prevents boundary words from being clipped
# at the seam between decode windows — the overlapping-window effect the blueprint requires.
CONDITION_ON_PREVIOUS_TEXT = True


# Module-level model singleton — loaded once on first call, reused for all subsequent calls.
_model: WhisperModel | None = None
_model_lock = threading.Lock()


def _get_model() -> WhisperModel:
    global _model
    with _model_lock:
        if _model is None:
            try:
                _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
            except Exception as exc:
                # GPU unavailable (driver update, VRAM exhausted, other machine):
                # degrade to CPU int8 rather than losing the meeting.
                print(
                    f"[transcribe] WARNING: CUDA model load failed ({exc}); "
                    "falling back to CPU int8 — transcription will be slower.",
                    file=sys.stderr,
                )
                _model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
        return _model


def preload_model() -> None:
    """Load the model eagerly (called from a startup thread in main.py)."""
    _get_model()


def transcribe(
    audio: Union[np.ndarray, str],
    diarization_cfg: dict | None = None,
    engine: str = "faster-whisper",
    language: str | None = None,
) -> tuple[str, float]:
    """
    Transcribe audio to a text string, optionally with speaker labels.

    Parameters
    ----------
    audio
        Float32 numpy array at 16 kHz mono **or** an absolute path to a WAV file
        (returned by ``AudioCapture.stop()`` on long-session recordings).
    diarization_cfg
        The ``[diarization]`` config table. When ``enabled`` is true and
        pyannote.audio is installed with a valid HuggingFace token, the
        transcript is grouped into speaker turns. Any diarization failure
        falls back to the plain unlabeled transcript.
    engine
        ``"faster-whisper"`` (default) or ``"whisperx"``. WhisperX adds
        wav2vec2 forced alignment for tighter word timestamps and uses its own
        diarization assignment. If the WhisperX path fails for any reason the
        faster-whisper path runs instead — a meeting is never lost to it.
    language
        ISO language code to force (e.g. ``"en"``). ``None`` or empty string
        auto-detects from the first 30s — which can misfire on a quiet opening
        and lock the whole meeting to a wrong language. Pin it when the spoken
        language is known.

    Returns
    -------
    transcript : str
        Full transcript; speaker-labeled turns when diarization succeeds.
    duration : float
        Measured audio duration in seconds.
    """
    diarization_cfg = diarization_cfg or {}
    language = language or None  # normalize empty string to auto-detect

    if engine == "whisperx":
        try:
            import whisperx_engine

            return whisperx_engine.transcribe_whisperx(audio, diarization_cfg, language)
        except Exception as exc:
            print(
                f"[transcribe] WARNING: whisperx engine failed ({exc}); "
                "falling back to faster-whisper.",
                file=sys.stderr,
            )

    model = _get_model()
    want_diarization = bool(diarization_cfg.get("enabled", False))

    if isinstance(audio, str):
        duration = _wav_duration(audio)
        audio_input: Union[np.ndarray, str] = audio
    else:
        duration = float(len(audio)) / 16_000.0
        audio_input = audio
        if len(audio) == 0:
            return "", 0.0

    segments, info = model.transcribe(
        audio_input,
        beam_size=BEAM_SIZE,
        language=language,      # None auto-detects; a pinned code skips detection
        vad_filter=VAD_FILTER,
        vad_parameters=VAD_PARAMETERS,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        word_timestamps=want_diarization,  # per-word timing only needed for diarization
    )

    # Drain the generator (transcription happens lazily as it is consumed).
    segment_list = list(segments)
    transcript = " ".join(seg.text.strip() for seg in segment_list).strip()

    if not transcript:
        print(
            "[transcribe] WARNING: transcription produced an empty transcript — "
            "the VAD found no speech in this audio.",
            file=sys.stderr,
        )

    # Prefer the duration reported by the model (accounts for VAD trimming).
    if hasattr(info, "duration") and info.duration:
        duration = float(info.duration)

    if want_diarization and transcript:
        try:
            from diarize import label_speakers

            labeled = label_speakers(audio_input, segment_list, diarization_cfg)
            if labeled:
                transcript = labeled
        except Exception as exc:
            print(
                f"[transcribe] WARNING: diarization failed ({exc}); "
                "writing plain transcript instead.",
                file=sys.stderr,
            )

    return transcript, duration


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())
