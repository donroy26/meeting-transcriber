"""
transcribe.py — Transcription via faster-whisper large-v3-turbo on CUDA float16.

Accepts either a float32 numpy array (normal path) or a WAV file path (long-session path).
Returns (transcript_str, duration_seconds).
No file writes. No network calls (model is pre-downloaded to HuggingFace cache).
"""

from __future__ import annotations

import os
import sys
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
COMPUTE_TYPE = "float16"

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


def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
    return _model


def transcribe(audio: Union[np.ndarray, str]) -> tuple[str, float]:
    """
    Transcribe audio to a continuous text string.

    Parameters
    ----------
    audio
        Float32 numpy array at 16 kHz mono **or** an absolute path to a WAV file
        (returned by ``AudioCapture.stop()`` on long-session recordings).

    Returns
    -------
    transcript : str
        Full continuous transcript with no speaker labels.
    duration : float
        Measured audio duration in seconds.
    """
    model = _get_model()

    if isinstance(audio, str):
        duration = _wav_duration(audio)
        audio_input: Union[np.ndarray, str] = audio
    else:
        duration = float(len(audio)) / 16_000.0
        audio_input = audio

    segments, info = model.transcribe(
        audio_input,
        beam_size=BEAM_SIZE,
        language=None,          # auto-detect; covers mixed-language meetings
        vad_filter=VAD_FILTER,
        vad_parameters=VAD_PARAMETERS,
        condition_on_previous_text=CONDITION_ON_PREVIOUS_TEXT,
        word_timestamps=False,  # segment-level text only; no per-word timing needed
    )

    # Drain the generator; joins all segment text.
    transcript = " ".join(seg.text.strip() for seg in segments).strip()

    # Prefer the duration reported by the model (accounts for VAD trimming).
    if hasattr(info, "duration") and info.duration:
        duration = float(info.duration)

    return transcript, duration


def _wav_duration(path: str) -> float:
    with wave.open(path, "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())
