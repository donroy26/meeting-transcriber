"""
mlx_engine.py — Transcription on Apple Silicon via ``mlx-whisper`` (Metal GPU).

Optional third engine next to faster-whisper and whisperx. faster-whisper's
ctranslate2 backend has no Metal support, so on a Mac it runs on the CPU and a
one-hour meeting takes many minutes; mlx-whisper runs the same Whisper
large-v3-turbo weights on the GPU. Select it in config.toml:

    [transcription]
    engine = "mlx"

Install (Apple Silicon only):  pip install mlx-whisper
The model (~1.6 GB) downloads from the Hugging Face Hub on first use.

Same contract as the other engines: ``transcribe_mlx(audio, diarization_cfg,
language) -> (transcript, duration)``. Any failure raises; transcribe.py
catches it and falls back to faster-whisper so a meeting is never lost.

Known difference: mlx-whisper has no VAD filter, so long silences can produce
a repeated/hallucinated phrase. Diarization reuses diarize.py unchanged.
"""

from __future__ import annotations

import sys
import threading
from types import SimpleNamespace
from typing import Union

import numpy as np

# Same weights as faster-whisper's "large-v3-turbo", converted for MLX.
MODEL_REPO = "mlx-community/whisper-large-v3-turbo"

_load_lock = threading.Lock()
_warm = False


def _get_model() -> None:
    """mlx-whisper loads lazily inside transcribe(); warm it with 1 s of silence
    so the first real meeting does not pay the download/compile cost."""
    global _warm
    with _load_lock:
        if not _warm:
            import mlx_whisper

            print(f"[mlx] Loading {MODEL_REPO} on the Metal GPU.", flush=True)
            mlx_whisper.transcribe(np.zeros(16_000, dtype=np.float32), path_or_hf_repo=MODEL_REPO)
            _warm = True


def transcribe_mlx(
    audio: Union[np.ndarray, str],
    diarization_cfg: dict | None = None,
    language: str | None = None,
) -> tuple[str, float]:
    import mlx_whisper

    diarization_cfg = diarization_cfg or {}
    want_diarization = bool(diarization_cfg.get("enabled", False))

    if isinstance(audio, str):
        from diarize import _load_wav_mono_16k  # stdlib wave → float32, no ffmpeg

        samples = _load_wav_mono_16k(audio)
    else:
        samples = np.ascontiguousarray(audio, dtype=np.float32)
    if len(samples) == 0:
        return "", 0.0
    duration = float(len(samples)) / 16_000.0

    _get_model()
    result = mlx_whisper.transcribe(
        samples,
        path_or_hf_repo=MODEL_REPO,
        language=language or None,
        word_timestamps=want_diarization,
        condition_on_previous_text=True,
    )
    segments = result.get("segments", [])
    transcript = " ".join(seg.get("text", "").strip() for seg in segments).strip()

    if not transcript:
        print("[mlx] WARNING: transcription produced an empty transcript.", file=sys.stderr)

    if want_diarization and transcript:
        try:
            from diarize import label_speakers

            # diarize.py reads faster-whisper style objects (seg.words[].start/.end/.word).
            shaped = [
                SimpleNamespace(
                    words=[SimpleNamespace(**w) for w in seg.get("words", []) if "start" in w]
                )
                for seg in segments
            ]
            labeled = label_speakers(audio, shaped, diarization_cfg)
            if labeled:
                transcript = labeled
        except Exception as exc:
            print(
                f"[mlx] WARNING: diarization failed ({exc}); writing plain transcript instead.",
                file=sys.stderr,
            )

    return transcript, duration
