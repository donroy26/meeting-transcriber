"""
note_writer.py — Compose the frozen transcript-note shape and write it to the
configured vault Meetings/ folder (or local fallback if unreachable).

The note includes optional attendees and operator notes captured during recording.
Summary / Decisions / Action Items remain empty with their comment markers intact
for the downstream summarization system to fill.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

# Frozen note shape — do not change section order or marker content.
_NOTE_TEMPLATE = """\
---
date: {date}
type: meeting
duration: {duration_m}m
summary: pending
---

# Meeting — {date} {time}

## Attendees
{attendees}

## Notes
{operator_notes}

## Summary
<!-- filled by the later summarization system -->

## Decisions
<!-- filled by the later summarization system -->

## Action Items
<!-- filled by the later summarization system -->

## Transcript
{transcript}
"""


def write_note(
    transcript: str,
    start_dt: datetime,
    duration_seconds: float,
    vault_meetings_path: str,
    fallback_folder: str,
    attendees: str = "",
    operator_notes: str = "",
) -> str:
    """
    Write a transcript note in the frozen shape.

    Returns the absolute path of the file written.  Emits a console warning
    (stderr) naming the fallback location when the vault path is unavailable.
    """
    date_str = start_dt.strftime("%Y-%m-%d")
    time_str = start_dt.strftime("%H:%M")
    duration_m = max(1, round(duration_seconds / 60))
    filename = f"{date_str}_meeting-{start_dt.strftime('%H%M')}.md"

    content = _NOTE_TEMPLATE.format(
        date=date_str,
        time=time_str,
        duration_m=duration_m,
        attendees=_format_attendees(attendees),
        operator_notes=_format_operator_notes(operator_notes),
        transcript=transcript,
    )

    target_dir, used_fallback = _resolve_target(vault_meetings_path, fallback_folder)
    os.makedirs(target_dir, exist_ok=True)

    out_path = _unique_path(target_dir, filename)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(content)

    if used_fallback:
        print(
            f"[note_writer] WARNING: vault path unavailable; note written to fallback: {out_path}",
            file=sys.stderr,
        )
    else:
        print(f"[note_writer] Note written: {out_path}")

    return out_path


def write_failure_note(
    start_dt: datetime,
    error_description: str,
    vault_meetings_path: str,
    fallback_folder: str,
    attendees: str = "",
    operator_notes: str = "",
    audio_path: str | None = None,
) -> str:
    """
    Write a placeholder note when transcription fails so the session is recoverable.

    The transcript section records the failure reason and, when available, the
    path of the retained audio file for manual re-transcription.
    Returns the path of the placeholder note written.
    """
    if audio_path:
        recovery_line = f"Raw audio retained for manual re-transcription:\n{audio_path}"
    else:
        recovery_line = "No audio file could be retained for this session."
    placeholder_transcript = (
        f"[TRANSCRIPTION FAILED]\n"
        f"Error: {error_description}\n\n"
        f"{recovery_line}"
    )
    return write_note(
        transcript=placeholder_transcript,
        start_dt=start_dt,
        duration_seconds=0,
        vault_meetings_path=vault_meetings_path,
        fallback_folder=fallback_folder,
        attendees=attendees,
        operator_notes=operator_notes,
    )


def _unique_path(directory: str, filename: str) -> str:
    """Never overwrite an existing note: append -2, -3, … on collision."""
    base, ext = os.path.splitext(filename)
    candidate = os.path.join(directory, filename)
    counter = 2
    while os.path.exists(candidate):
        candidate = os.path.join(directory, f"{base}-{counter}{ext}")
        counter += 1
    return candidate


def _format_attendees(attendees: str) -> str:
    names = [
        item.strip()
        for chunk in attendees.splitlines()
        for item in chunk.split(",")
        if item.strip()
    ]
    if not names:
        return "<!-- optional: add attendees -->"
    return "\n".join(f"- {name}" for name in names)


def _format_operator_notes(operator_notes: str) -> str:
    if not operator_notes.strip():
        return "<!-- optional: notes captured during recording -->"
    return operator_notes.strip()


def _resolve_target(vault_meetings_path: str, fallback_folder: str) -> tuple[str, bool]:
    """Return ``(directory_path, used_fallback)``."""
    if vault_meetings_path and os.path.isdir(vault_meetings_path):
        return vault_meetings_path, False

    # Vault path is absent or unreachable — use the configured fallback.
    if fallback_folder:
        return fallback_folder, True

    # Default fallback: a Meetings/ folder beside the scripts.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, "Meetings"), True
