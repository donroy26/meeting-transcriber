"""
whisperx_engine.py — Optional WhisperX transcription engine.

Selected via config.toml:

    [transcription]
    engine = "whisperx"      # default is "faster-whisper"

WhisperX runs the same faster-whisper model underneath, then adds wav2vec2
forced alignment (tighter word timestamps) and pyannote diarization with its
own word-to-speaker assignment. Everything runs locally on the GPU; the only
network access is the one-time model downloads from HuggingFace.

Install note: whisperx pins ctranslate2==4.4.0, which has no Python 3.14
wheels, so it must be installed WITHOUT dependencies against our newer stack:

    pip install whisperx --no-deps
    pip install pandas transformers nltk

Any failure in this module is caught by transcribe.py, which falls back to the
plain faster-whisper engine.
"""

from __future__ import annotations

import sys
import threading
import wave
from typing import Union

import numpy as np

MODEL_SIZE = "large-v3-turbo"

# Low-VRAM posture: this pipeline must coexist with Windows desktop
# compositing on an 8 GB GPU. Filling VRAM makes the driver page GPU memory
# to system RAM, which freezes the whole desktop (observed 2026-07-02 on a
# long meeting). So: small batch, int8 weights, and each stage releases its
# models before the next loads. Reload costs ~seconds per meeting.
BATCH_SIZE = 4
GPU_COMPUTE_TYPE = "int8_float16"

_lock = threading.Lock()
_model = None
_device: str | None = None


def _get_device() -> str:
    global _device
    if _device is None:
        import torch

        _device = "cuda" if torch.cuda.is_available() else "cpu"
    return _device


_pyannote_patched = False


def _patch_pyannote_compat() -> None:
    """
    whisperx 3.2.0 targets pyannote 3.x and passes ``use_auth_token`` into
    APIs that pyannote 4 removed it from. Strip the argument before it lands.
    Idempotent, applied once before whisperx loads its VAD model.
    """
    global _pyannote_patched
    if _pyannote_patched:
        return
    from pyannote.audio import Inference, Model

    orig_inference_init = Inference.__init__

    def inference_init(self, *args, **kwargs):
        kwargs.pop("use_auth_token", None)
        orig_inference_init(self, *args, **kwargs)

    Inference.__init__ = inference_init

    orig_from_pretrained = Model.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        kwargs.pop("use_auth_token", None)
        return orig_from_pretrained(cls, *args, **kwargs)

    Model.from_pretrained = classmethod(from_pretrained)
    _pyannote_patched = True


def _get_model():
    global _model
    with _lock:
        if _model is None:
            _patch_pyannote_compat()
            import whisperx

            device = _get_device()
            compute_type = GPU_COMPUTE_TYPE if device == "cuda" else "int8"
            print(
                f"[whisperx] Loading model {MODEL_SIZE} on {device} ({compute_type}).",
                flush=True,
            )
            # asr_options: faster-whisper >=1.1 requires two extra
            # TranscriptionOptions fields whisperx 3.2.0 doesn't set.
            _model = whisperx.load_model(
                MODEL_SIZE,
                device,
                compute_type=compute_type,
                asr_options={"multilingual": False, "hotwords": None},
            )
        return _model


def _free_gpu() -> None:
    """Return freed VRAM to the pool (callers must drop their references first)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _release_model() -> None:
    global _model
    with _lock:
        _model = None
    _free_gpu()


def _run_diarization(samples: np.ndarray, cfg: dict):
    """
    Run pyannote diarization (via diarize.py, which handles the pyannote 4 API
    and GPU visit-then-release) and return the DataFrame shape
    whisperx.assign_word_speakers expects. whisperx's own DiarizationPipeline
    targets the removed pyannote 3.x API.
    """
    import pandas as pd
    import torch

    import diarize

    waveform = torch.from_numpy(np.ascontiguousarray(samples)).unsqueeze(0)
    annotation = diarize.run_pipeline(
        {"waveform": waveform, "sample_rate": 16_000}, cfg
    )

    rows = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    return pd.DataFrame(rows)


def transcribe_whisperx(
    audio: Union[np.ndarray, str],
    diarization_cfg: dict,
    language: str | None = None,
) -> tuple[str, float]:
    """Transcribe (and optionally diarize) with WhisperX. Returns (text, duration).

    ``language`` pins the transcription language (e.g. ``"en"``); ``None`` lets
    WhisperX auto-detect from the first window, which can misfire on a quiet
    opening and lock the whole meeting to a wrong language.
    """
    import whisperx

    if isinstance(audio, str):
        samples = _load_wav_mono_16k(audio)
    else:
        samples = np.ascontiguousarray(audio, dtype=np.float32)
    if len(samples) == 0:
        return "", 0.0
    duration = len(samples) / 16_000.0

    # Stage 1: transcribe, then release Whisper's VRAM before alignment loads.
    # A pinned language is passed straight through so detection is skipped.
    # The release is in a finally because this call does raise in practice —
    # silent audio yields no VAD segments and transformers then indexes an
    # empty list — and leaking the model here left it resident in VRAM for the
    # rest of the session while the faster-whisper fallback loaded a second one.
    model = _get_model()
    try:
        result = model.transcribe(samples, batch_size=BATCH_SIZE, language=language)
    except IndexError:
        # whisperx's VAD found no speech, so it hands transformers an empty
        # batch and pipelines/base.py does inputs[0] on an empty list. That is
        # "nothing was said", not an engine fault — reporting it as a failure
        # made transcribe.py fall back and load a second model for nothing.
        print(
            f"[whisperx] WARNING: no speech detected in {duration:.1f}s of audio; "
            "returning an empty transcript.",
            file=sys.stderr,
        )
        return "", duration
    finally:
        model = None
        _release_model()

    language = result.get("language", language or "en")

    # Same outcome, reached cleanly: nothing to align and nothing to diarize.
    if not result.get("segments"):
        print(
            "[whisperx] WARNING: no speech segments detected in "
            f"{duration:.1f}s of audio; returning an empty transcript.",
            file=sys.stderr,
        )
        return "", duration

    # Stage 2: forced alignment for accurate word timestamps. Not fatal if it
    # fails (e.g. no align model for the detected language) — segments still
    # have text. Model is loaded per meeting and freed immediately after.
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=_get_device()
        )
        try:
            result = whisperx.align(
                result["segments"], align_model, metadata, samples, _get_device()
            )
        finally:
            del align_model, metadata
            _free_gpu()
    except Exception as exc:
        print(
            f"[whisperx] WARNING: alignment failed ({exc}); "
            "continuing with unaligned segments.",
            file=sys.stderr,
        )

    plain_text = " ".join(
        seg.get("text", "").strip() for seg in result.get("segments", [])
    ).strip()

    if not bool(diarization_cfg.get("enabled", False)) or not plain_text:
        return plain_text, duration

    try:
        print("[whisperx] Running speaker diarization.", flush=True)
        diarize_segments = _run_diarization(samples, diarization_cfg)
        if diarize_segments.empty:
            return plain_text, duration
        result = whisperx.assign_word_speakers(diarize_segments, result)
        labeled = _format_speaker_turns(result.get("segments", []))
        return (labeled or plain_text), duration
    except Exception as exc:
        print(
            f"[whisperx] WARNING: diarization failed ({exc}); "
            "writing plain transcript instead.",
            file=sys.stderr,
        )
        return plain_text, duration


def _format_speaker_turns(segments: list[dict]) -> str:
    """Group consecutive same-speaker segments into '**Speaker N:** …' blocks."""
    display_names: dict[str, str] = {}

    def display(speaker: str) -> str:
        if speaker not in display_names:
            display_names[speaker] = f"Speaker {len(display_names) + 1}"
        return display_names[speaker]

    blocks: list[tuple[str, list[str]]] = []
    last_known: str | None = None
    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue
        speaker = seg.get("speaker") or last_known
        if speaker is None:
            speaker = "SPEAKER_UNKNOWN"
        last_known = speaker
        name = display(speaker)
        if blocks and blocks[-1][0] == name:
            blocks[-1][1].append(text)
        else:
            blocks.append((name, [text]))

    return "\n\n".join(f"**{name}:** {' '.join(parts)}" for name, parts in blocks)


def _load_wav_mono_16k(path: str) -> np.ndarray:
    """Load a 16-bit mono 16 kHz WAV (as written by capture.py) to float32."""
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != 16_000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"Unexpected WAV format in {path}: "
                f"{wf.getframerate()} Hz, {wf.getnchannels()} ch, {wf.getsampwidth() * 8}-bit"
            )
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return pcm.astype(np.float32) / 32_768.0
