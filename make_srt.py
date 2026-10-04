"""
make_srt.py — Standalone subtitle generator for editing (DaVinci Resolve, Premiere, YouTube).

Uses the same faster-whisper large-v3-turbo / CUDA setup as transcribe.py, but asks for
word-level timestamps and re-cuts them into caption-shaped cues instead of prose. No
diarization, no vault writing, no network.

    python make_srt.py "C:\\path\\to\\audio.wav"
    python make_srt.py video.mp4 --offset 01:00:00,000 --out subs.srt

Writes <input>.srt beside the input unless --out is given.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_DLL_DIR_HANDLES = []


def _add_nvidia_dll_dirs() -> None:
    """Let Windows find CUDA DLLs installed by NVIDIA Python wheels (see transcribe.py)."""
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
from faster_whisper.audio import decode_audio

MODEL_SIZE = "large-v3-turbo"
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5

VAD_PARAMETERS = {
    "threshold": 0.5,
    "min_speech_duration_ms": 250,
    "max_speech_duration_s": float("inf"),
    "min_silence_duration_ms": 2_000,
    "speech_pad_ms": 400,
}

# Broadcast-ish caption shape. Two lines of ~42 characters is the readable ceiling;
# 6s is the longest a cue should sit on screen, 1.2s the shortest it can be read in.
MAX_LINE_CHARS = 42
MAX_LINES = 2
MAX_CUE_CHARS = MAX_LINE_CHARS * MAX_LINES
MAX_CUE_SECONDS = 6.0
MIN_CUE_SECONDS = 1.2
# A pause this long inside a cue reads as a break — start a new one instead.
PAUSE_BREAK_SECONDS = 0.7
# Don't split on a sentence end until the cue has enough on it to stand alone.
MIN_CHARS_BEFORE_SENTENCE_BREAK = 18
# Gap left between adjacent cues when a short one is padded out to MIN_CUE_SECONDS.
CUE_GAP_SECONDS = 0.08

SENTENCE_ENDINGS = (".", "!", "?", "…")
CLAUSE_ENDINGS = (",", ";", ":", "—")

# Whisper loops on trailing silence/room tone and emits a token over and over
# ("2021 2021 2021 ..."). Drop a cue whose words are this dominated by one token.
REPEAT_MIN_WORDS = 4
REPEAT_MAX_UNIQUE_RATIO = 0.4


def parse_timecode(value: str) -> float:
    """Parse an offset given as seconds ("12.5") or SRT/timecode form ("01:00:00,000")."""
    value = value.strip()
    if ":" not in value:
        return float(value)
    parts = value.replace(",", ".").split(":")
    if len(parts) > 3:
        raise ValueError(f"unrecognized timecode: {value!r}")
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60.0 + float(part)
    return seconds


def format_timestamp(seconds: float) -> str:
    seconds = max(0.0, seconds)
    millis = int(round(seconds * 1000.0))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1_000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def collect_words(segments) -> list[dict]:
    """Flatten segments into a single word stream, falling back to whole segments
    when the model returned no per-word timings for one."""
    words: list[dict] = []
    for seg in segments:
        seg_words = getattr(seg, "words", None)
        if seg_words:
            for w in seg_words:
                text = w.word.strip()
                if text:
                    words.append({"text": text, "start": float(w.start), "end": float(w.end)})
            continue
        text = seg.text.strip()
        if text:
            words.append({"text": text, "start": float(seg.start), "end": float(seg.end)})
    return words


def build_cues(words: list[dict]) -> list[dict]:
    """Group the word stream into caption cues on length, duration, pause and punctuation."""
    cues: list[dict] = []
    current: list[dict] = []

    def flush() -> None:
        if not current:
            return
        cues.append(
            {
                "start": current[0]["start"],
                "end": current[-1]["end"],
                "text": " ".join(w["text"] for w in current),
            }
        )
        current.clear()

    for word in words:
        if current:
            pending = len(" ".join(w["text"] for w in current)) + 1 + len(word["text"])
            gap = word["start"] - current[-1]["end"]
            span = word["end"] - current[0]["start"]
            if pending > MAX_CUE_CHARS or gap > PAUSE_BREAK_SECONDS or span > MAX_CUE_SECONDS:
                flush()
        current.append(word)

        chars = len(" ".join(w["text"] for w in current))
        if word["text"].endswith(SENTENCE_ENDINGS) and chars >= MIN_CHARS_BEFORE_SENTENCE_BREAK:
            flush()

    flush()
    return cues


def is_repetition_loop(text: str) -> bool:
    tokens = [t.strip(".,!?;:").lower() for t in text.split()]
    tokens = [t for t in tokens if t]
    if len(tokens) < REPEAT_MIN_WORDS:
        return False
    return len(set(tokens)) / len(tokens) <= REPEAT_MAX_UNIQUE_RATIO


def drop_hallucinated_cues(cues: list[dict]) -> list[dict]:
    kept = [c for c in cues if not is_repetition_loop(c["text"])]
    dropped = len(cues) - len(kept)
    if dropped:
        print(f"[make_srt] dropped {dropped} repetition-loop cue(s)", file=sys.stderr)
    return kept


def enforce_min_duration(cues: list[dict]) -> None:
    """Stretch cues that flash by, without letting one run into the next."""
    for i, cue in enumerate(cues):
        if cue["end"] - cue["start"] >= MIN_CUE_SECONDS:
            continue
        wanted = cue["start"] + MIN_CUE_SECONDS
        if i + 1 < len(cues):
            wanted = min(wanted, cues[i + 1]["start"] - CUE_GAP_SECONDS)
        cue["end"] = max(cue["end"], wanted)


def wrap_lines(text: str) -> str:
    """Wrap to at most MAX_LINES lines, preferring a break at a clause boundary near
    the middle so the two lines read as balanced halves."""
    if len(text) <= MAX_LINE_CHARS:
        return text

    tokens = text.split()
    if len(tokens) < 2:
        return text  # one unbreakable token — nothing to wrap

    best_index = None
    best_score = None
    for i in range(1, len(tokens)):
        head = " ".join(tokens[:i])
        tail = " ".join(tokens[i:])
        if len(head) > MAX_LINE_CHARS:
            break
        if len(tail) > MAX_LINE_CHARS:
            continue
        score = abs(len(head) - len(tail))
        if tokens[i - 1].endswith(CLAUSE_ENDINGS):
            score -= 12  # a clause break beats a marginally more even split
        if best_score is None or score < best_score:
            best_score, best_index = score, i

    if best_index is None:
        # No split leaves both halves under the cap — take the one whose longer line
        # is shortest, so the overflow is as small as it can be.
        best_index = min(
            range(1, len(tokens)),
            key=lambda i: max(len(" ".join(tokens[:i])), len(" ".join(tokens[i:]))),
        )

    return " ".join(tokens[:best_index]) + "\n" + " ".join(tokens[best_index:])


def to_srt(cues: list[dict], offset: float = 0.0) -> str:
    blocks = []
    for i, cue in enumerate(cues, start=1):
        start = format_timestamp(cue["start"] + offset)
        end = format_timestamp(cue["end"] + offset)
        blocks.append(f"{i}\n{start} --> {end}\n{wrap_lines(cue['text'])}\n")
    return "\n".join(blocks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate an SRT subtitle file from audio/video.")
    parser.add_argument("input", help="Audio or video file to transcribe.")
    parser.add_argument("--out", help="Output .srt path (default: input path with .srt).")
    parser.add_argument("--language", default="en", help="Language code, or '' to auto-detect.")
    parser.add_argument(
        "--offset",
        default="0",
        help="Shift every cue later by this much — seconds or HH:MM:SS,mmm. "
        "Use when the audio starts partway into the timeline.",
    )
    parser.add_argument(
        "--end",
        help="Drop cues starting at or after this point (seconds or HH:MM:SS,mmm). "
        "Applause and room tone past the last word make Whisper hallucinate — trim there.",
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--model", default=MODEL_SIZE)
    parser.add_argument("--no-vad", action="store_true", help="Disable the VAD filter.")
    args = parser.parse_args()

    src = Path(args.input)
    if not src.is_file():
        print(f"[make_srt] ERROR: no such file: {src}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else src.with_suffix(".srt")
    offset = parse_timecode(args.offset)

    print(f"[make_srt] decoding {src.name} ...", flush=True)
    audio = decode_audio(str(src), sampling_rate=16_000)
    print(f"[make_srt] {len(audio) / 16_000.0:.1f}s of audio", flush=True)

    compute_type = COMPUTE_TYPE if args.device == "cuda" else "int8"
    try:
        model = WhisperModel(args.model, device=args.device, compute_type=compute_type)
    except Exception as exc:
        print(f"[make_srt] WARNING: {args.device} load failed ({exc}); using CPU.", file=sys.stderr)
        model = WhisperModel(args.model, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        audio,
        beam_size=BEAM_SIZE,
        language=args.language or None,
        vad_filter=not args.no_vad,
        vad_parameters=VAD_PARAMETERS,
        condition_on_previous_text=True,
        word_timestamps=True,
    )

    words = collect_words(segments)
    if not words:
        print("[make_srt] ERROR: no speech found.", file=sys.stderr)
        return 1

    cues = drop_hallucinated_cues(build_cues(words))
    if args.end:
        limit = parse_timecode(args.end)
        before = len(cues)
        cues = [c for c in cues if c["start"] < limit]
        for c in cues:
            c["end"] = min(c["end"], limit)
        print(f"[make_srt] trimmed {before - len(cues)} cue(s) at/after {args.end}", file=sys.stderr)
    if not cues:
        print("[make_srt] ERROR: nothing left after filtering.", file=sys.stderr)
        return 1
    enforce_min_duration(cues)
    out_path.write_text(to_srt(cues, offset), encoding="utf-8")

    print(
        f"[make_srt] {len(cues)} cues from {len(words)} words "
        f"({getattr(info, 'language', '?')}) -> {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
