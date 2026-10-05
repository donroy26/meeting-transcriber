"""
diarize.py — Optional speaker diarization via pyannote.audio.

Takes the word-timestamped segments produced by faster-whisper plus the raw
audio, runs the pyannote speaker-diarization pipeline, and merges the two by
assigning each word to the speaker segment covering its midpoint. The output
is a transcript grouped into speaker turns:

    **Speaker 1:** Good morning everyone, let's get started...

    **Speaker 2:** Thanks — quick update on the panel schedule...

Requirements (all optional — transcription works without them):
  - pip install torch (CUDA build) and pyannote.audio
  - A HuggingFace token in config.toml [diarization] hf_token, with the
    pipeline's model terms accepted on huggingface.co
"""

from __future__ import annotations

import sys
from typing import Union

import numpy as np

DEFAULT_PIPELINE = "pyannote/speaker-diarization-community-1"

# Pipeline singleton — expensive to load, reused across meetings.
_pipeline = None
_pipeline_name: str | None = None


def _get_pipeline(model_name: str, hf_token: str):
    global _pipeline, _pipeline_name
    if _pipeline is not None and _pipeline_name == model_name:
        return _pipeline

    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(model_name, token=hf_token or None)
    except TypeError:
        # pyannote.audio 3.x uses use_auth_token instead of token
        pipeline = Pipeline.from_pretrained(model_name, use_auth_token=hf_token or None)

    if pipeline is None:
        raise RuntimeError(
            f"Could not load pipeline '{model_name}'. Most likely the HuggingFace "
            "token is missing/invalid or the model's terms have not been accepted "
            f"at https://huggingface.co/{model_name}"
        )

    _pipeline = pipeline
    _pipeline_name = model_name
    return _pipeline


def run_pipeline(pipeline_input: dict, cfg: dict):
    """
    Run the diarization pipeline and return the Annotation.

    The pipeline lives on the CPU between runs and visits the GPU only for the
    duration of a run — keeping VRAM free for Whisper and, critically, for
    Windows desktop compositing (a full GPU freezes the whole desktop).
    """
    import gc

    import torch

    model_name = cfg.get("model") or DEFAULT_PIPELINE
    pipeline = _get_pipeline(model_name, cfg.get("hf_token", ""))

    kwargs = {}
    num_speakers = int(cfg.get("num_speakers", 0) or 0)
    if num_speakers > 0:
        kwargs["num_speakers"] = num_speakers

    use_cuda = torch.cuda.is_available()
    try:
        if use_cuda:
            pipeline.to(torch.device("cuda"))
        annotation = pipeline(pipeline_input, **kwargs)
    finally:
        if use_cuda:
            try:
                pipeline.to(torch.device("cpu"))
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()

    # pyannote 4 pipelines wrap the Annotation in an output object;
    # pyannote 3 returns the Annotation directly.
    if not hasattr(annotation, "itertracks"):
        annotation = annotation.speaker_diarization
    return annotation


def label_speakers(
    audio: Union[np.ndarray, str],
    segments: list,
    cfg: dict,
) -> str | None:
    """
    Run diarization and return a speaker-labeled transcript, or ``None`` when
    there is nothing to label. Raises on pipeline/dependency failures — the
    caller (transcribe.py) catches and falls back to the plain transcript.

    Parameters
    ----------
    audio
        Same input given to faster-whisper: 16 kHz mono float32 array or WAV path.
    segments
        Drained faster-whisper segments with ``words`` populated
        (``word_timestamps=True``).
    cfg
        The ``[diarization]`` config table.
    """
    words = [w for seg in segments for w in (seg.words or [])]
    if not words:
        return None

    import torch

    # Always hand pyannote an in-memory waveform: decoding a file path would
    # pull in torchcodec/FFmpeg, a fragile dependency on Windows. Our audio is
    # always 16 kHz mono PCM produced by capture.py, so loading it is trivial.
    if isinstance(audio, str):
        samples = _load_wav_mono_16k(audio)
    else:
        samples = np.ascontiguousarray(audio)
    waveform = torch.from_numpy(samples).unsqueeze(0)
    pipeline_input = {"waveform": waveform, "sample_rate": 16_000}

    print(
        f"[diarize] Running speaker diarization ({cfg.get('model') or DEFAULT_PIPELINE}).",
        flush=True,
    )
    annotation = run_pipeline(pipeline_input, cfg)

    turns = [
        (turn.start, turn.end, speaker)
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    if not turns:
        return None

    return _merge_words_with_turns(words, turns)


def _load_wav_mono_16k(path: str) -> np.ndarray:
    """Load a 16-bit mono 16 kHz WAV (as written by capture.py) to float32."""
    import wave

    with wave.open(path, "rb") as wf:
        if wf.getframerate() != 16_000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError(
                f"Unexpected WAV format in {path}: "
                f"{wf.getframerate()} Hz, {wf.getnchannels()} ch, {wf.getsampwidth() * 8}-bit"
            )
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    return (pcm.astype(np.float32) / 32_768.0)


def _speaker_for(midpoint: float, turns: list[tuple[float, float, str]]) -> str:
    """Speaker whose turn covers the midpoint; else the nearest turn."""
    best_speaker = turns[0][2]
    best_distance = float("inf")
    for start, end, speaker in turns:
        if start <= midpoint <= end:
            return speaker
        distance = min(abs(midpoint - start), abs(midpoint - end))
        if distance < best_distance:
            best_distance = distance
            best_speaker = speaker
    return best_speaker


def _merge_words_with_turns(
    words: list,
    turns: list[tuple[float, float, str]],
) -> str:
    """Group consecutive same-speaker words into labeled paragraphs."""
    # Stable, human-friendly names in order of first appearance.
    display_names: dict[str, str] = {}

    def display(speaker: str) -> str:
        if speaker not in display_names:
            display_names[speaker] = f"Speaker {len(display_names) + 1}"
        return display_names[speaker]

    blocks: list[tuple[str, list[str]]] = []
    for word in words:
        midpoint = (word.start + word.end) / 2.0
        name = display(_speaker_for(midpoint, turns))
        text = word.word.strip()
        if not text:
            continue
        if blocks and blocks[-1][0] == name:
            blocks[-1][1].append(text)
        else:
            blocks.append((name, [text]))

    return "\n\n".join(f"**{name}:** {' '.join(tokens)}" for name, tokens in blocks)
