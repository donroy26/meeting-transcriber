# Meeting Transcriber — Setup Guide

Portable Windows script bundle. Records mic + system audio on a global hotkey toggle,
transcribes locally on the NVIDIA GPU, and writes a summarization-ready markdown note
to your Obsidian Meetings/ folder.

**Requirements:** Windows 10/11 + NVIDIA GPU with current NVIDIA driver + Python
3.13/3.14. No cloud API. No internet during recording.

---

## 1. Python environment

Create and activate a virtual environment inside the System/ folder (or anywhere you prefer):

```bat
python -m venv .venv
.venv\Scripts\activate
```

---

## 2. Install dependencies

The transfer bundle includes `Install Meeting Transcriber.bat` and
`Install Meeting Transcriber.ps1`. Prefer those on a fresh machine:

```bat
Install Meeting Transcriber.bat
```

Manual install:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

`requirements.txt` includes the CUDA 12 runtime, cuBLAS, cuDNN, CTranslate2, and
faster-whisper versions verified on this machine. Launch with
`Start Meeting Transcriber.bat` so the NVIDIA DLL folders are placed on PATH.

**Verify CUDA is visible:**

```python
import ctranslate2
print(ctranslate2.get_cuda_device_count())   # should print >= 1
```

---

## 3. Model download (first run)

On the first transcription call, faster-whisper auto-downloads `large-v3-turbo`
(~800 MB) from Hugging Face Hub into `~/.cache/huggingface/hub/`. Ensure you have an
internet connection for this one-time download. Subsequent runs are fully offline.

Model spec: `large-v3-turbo`, `device=cuda`, `compute_type=float16`.

---

## 4. Audio capture setup

`pyaudiowpatch` provides WASAPI loopback access — it captures the system's output
(far-side call audio) as an additional input stream.

```bat
pip install pyaudiowpatch==0.2.12.8
```

This is already in `requirements.txt`. No additional driver install is needed.

To find the name of your system audio output device for `config.toml`:

```bat
python capture.py --list-devices
```

Leave `output_device_name` empty to capture all `[LOOPBACK]` devices found at
recording start. This is recommended when you sometimes use headphones and
sometimes use speakers.

Caller / PC audio is captured from loopback devices. Teams, Zoom, Meet, your
browser, or any call app can use any Windows output device that exposes a
loopback channel. Set `output_device_name` only if you need to force one specific
speaker/headset.

---

## 5. Hotkey setup

No external tool required. The `keyboard` Python library registers a low-level Windows
keyboard hook. It fires from any focused application.

**Note on elevated windows:** If you need the hotkey to trigger while an elevated
(Administrator) window has focus (e.g., Task Manager), run `main.py` as Administrator
or accept mic-only capture in those contexts.

The default hotkey is `ctrl+shift+r`. Change it in `config.toml → [hotkey] hotkey`.
Supported formats: `"ctrl+shift+r"`, `"f9"`, `"alt+r"`, etc. (keyboard library syntax).

---

## 6. Config setup

```bat
copy config.example.toml config.toml
```

Edit `config.toml` with your machine values:

```toml
[paths]
vault_meetings_path = "D:/Obsidian/MyVault/Meetings"
fallback_folder     = ""   # leave empty to use Meetings/ beside these scripts

[audio]
output_device_name = ""    # empty = capture all loopback devices

[hotkey]
hotkey = "ctrl+shift+r"
```

`config.toml` is machine-specific. Do **not** commit it. Only `config.example.toml`
is committed.

---

## 7. Run

```bat
.venv\Scripts\activate
python main.py
```

The process runs persistently (hotkey always live). A tray icon appears in the system tray:

| Tray color | State        |
|------------|--------------|
| Gray       | Idle         |
| Red        | Recording    |
| Orange     | Processing   |

While recording, a small floating meter appears near the upper-right of the
screen. It shows rough mixed-audio activity from the microphone plus PC/caller
loopback audio. Moving bars mean the app is seeing audio. It is an activity
indicator, not a calibrated volume meter.

A notes window also opens while recording. Use **Attendees** for participant
names and **Notes to add** for anything you want preserved with the transcript.
Those fields are written into `## Attendees` and `## Notes` in the output note
when you stop the recording.

Use the **To-do** button in the notes window to mark the current note line as:
`- [ ] TODO: ...`. This gives the later summarizer an explicit action-item
signal without filling the formal `## Action Items` section yet.

**Workflow:**
1. Press `ctrl+shift+r` → recording starts (tray turns red).
2. Press `ctrl+shift+r` again → recording stops, transcription begins (tray turns orange).
3. When done, the markdown note appears in your Obsidian `Meetings/` folder (tray returns gray).

Right-click the tray icon → **Quit** to exit. Do not quit while the tray shows orange —
wait for it to return to gray first.

---

## 8. Output note shape

Each session produces one file: `YYYY-MM-DD_meeting-HHMM.md`

```markdown
---
date: YYYY-MM-DD
type: meeting
duration: <mm>m
summary: pending
---

# Meeting — YYYY-MM-DD HH:MM

## Attendees
- <attendee names entered during recording>

## Notes
<notes entered during recording>
<to-do lines appear as - [ ] TODO: ...>

## Summary
<!-- filled by the later summarization system -->

## Decisions
<!-- filled by the later summarization system -->

## Action Items
<!-- filled by the later summarization system -->

## Transcript
<full continuous transcript>
```

Summary / Decisions / Action Items are intentionally empty — they are filled by a
separate downstream summarization system.

---

## 9. Failure recovery

If transcription fails, the process:

1. Retains the raw audio as a temp WAV file (path printed to console).
2. Writes a placeholder note to the vault (or fallback) noting the failure.

To recover: transcribe the retained WAV manually, or re-run the transcriber on it:

```python
from transcribe import transcribe
transcript, duration = transcribe("C:/path/to/mtranscriber_XXXXX.wav")
```

Then write the note manually or delete the placeholder and re-run.

---

## 10. Long-session behavior

For sessions exceeding **30 minutes**, the in-memory audio buffer is flushed to a temp
WAV file on disk in 30-minute chunks to keep memory bounded (~115 MB per chunk at
16 kHz mono). On successful transcription the temp file is deleted automatically.
The console logs the deletion explicitly; if you see a warning that deletion failed,
check for and manually remove `%TEMP%\mtranscriber_*.wav`.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: config.toml not found` | Copy `config.example.toml` → `config.toml` and fill in paths. |
| Hotkey doesn't fire from a specific app | Run `main.py` as Administrator (elevated apps ignore low-level hooks from non-admin processes). |
| Loopback device not found | Run `python capture.py --list-devices`, find your `[LOOPBACK]` entry, update `config.toml`. |
| `ctranslate2` CUDA error on startup | Verify CUDA DLLs are in PATH; run `ctranslate2.get_cuda_device_count()` to diagnose. |
| First run is slow | faster-whisper downloads `large-v3-turbo` (~800 MB) on first call. Subsequent runs use the cache. |
