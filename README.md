# Meeting Transcriber — Setup Guide

Portable Windows script bundle. Records mic + system audio on a global hotkey toggle,
transcribes locally on the NVIDIA GPU, and writes a summarization-ready markdown note
to your Obsidian Meetings/ folder.

**Requirements:** Windows 10/11 + NVIDIA GPU with current NVIDIA driver + Python 3.13/3.14. No cloud API. No internet during recording.

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
faster-whisper versions that were verified working together. Launch with
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

No external tool and no package required. `hotkey.py` calls the Win32
`RegisterHotKey` API through stdlib `ctypes`. Windows itself matches the combo
and posts the event to the app, so the hotkey survives sleep, Win+L, UAC
prompts, and Remote Desktop switches.

(This replaced the `keyboard` library, which installed a low-level keyboard
hook. A hook tracks which keys it believes are down, and any key-release it
misses — which happens every time Windows switches to the secure desktop for
Win+L, Ctrl+Alt+Del, UAC, or the lock screen — sticks in that set permanently
and the combo silently never matches again. That was the cause of recordings
that could not be stopped.)

The default hotkey is `ctrl+shift+r`. Change it in `config.toml → [hotkey] hotkey`:

- Modifiers: `ctrl` (or `control`), `shift`, `alt`, `win` (or `windows`)
- Main key: one letter or digit, `f1`–`f24`, or `space`, `pause`, `insert`,
  `delete`, `home`, `end`, `pageup`, `pagedown`, `esc`
- Examples: `"ctrl+shift+r"`, `"f9"`, `"alt+shift+r"`, `"ctrl+alt+space"`

**The combo is exclusive.** Windows grants it to one process at a time. If
another app already owns it (or a second copy of this app is running), startup
logs the conflict as `error 1409`, shows a tray notification, and keeps
running — use the tray menu to start and stop recording, then fix the conflict
and choose **Re-register hotkey** from the same menu.

**Note on elevated windows:** whether the hotkey fires while an elevated
(Administrator) window such as Task Manager has focus has *not* been verified
— it cannot be tested with synthetic keystrokes, because
Windows blocks a non-elevated process from injecting input at all while an
elevated window is foreground. If you find it does not respond there, run
`main.py` as Administrator.

---

## 6. Config setup

```bat
copy config.example.toml config.toml
```

Edit `config.toml` with your machine values:

```toml
[paths]
vault_meetings_path = "C:/Users/yourname/Obsidian/MyVault/Meetings"
fallback_folder     = ""   # leave empty to use Meetings/ beside these scripts

[audio]
output_device_name = ""    # empty = capture all loopback devices

[hotkey]
hotkey = "ctrl+shift+r"
```

`config.toml` is machine-specific. Do **not** commit it. Only `config.example.toml`
is committed.

### Start at login

The app is only useful if it is already running when a meeting starts, so put a
shortcut to the launcher in the Startup folder:

1. Right-click `Start Meeting Transcriber.bat` → **Create shortcut**.
2. Move the shortcut into
   `%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup`
   (paste that into the Explorer address bar to open it).
3. Right-click the shortcut → **Properties** → set **Run:** to **Minimized**,
   so the console window does not land in the foreground at every login.

The hotkey and tray icon are live within a few seconds of launch; the
transcription model finishes loading in the background afterwards.

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
2. Press `ctrl+shift+r` again → recording stops and is queued for transcription
   (tray turns orange).
3. When done, the markdown note appears in your Obsidian `Meetings/` folder (tray returns gray).

Recording and transcription are independent: you can start a new recording
immediately, even while earlier meetings are still transcribing (tray stays red
with a "+N transcribing" note in its tooltip). Transcriptions queue up and run
one at a time in the background.

**Tray menu** (right-click the icon):

| Item | What it does |
|------|--------------|
| **Start recording** / **Stop recording** | Same as the hotkey. Use it if the hotkey is unavailable. |
| **Re-register hotkey** | Re-claims the combo from Windows — use after closing whatever app was holding it. Reports the result as a notification. |
| **Quit** | Exits. |

Quitting **while recording** no longer throws the audio away: capture is
stopped and the WAV is written to `Recovery/audio` (the log names the file). It
is not transcribed — re-run it from the saved audio.

Still avoid quitting while the tray is orange or the tooltip mentions
transcribing: queued meetings are discarded. Wait for gray first.

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

## 9. Transcription engine choice (optional)

Two engines are available; switch in `config.toml` and compare:

```toml
[transcription]
engine = "faster-whisper"   # or "whisperx"
```

Both run the same Whisper `large-v3-turbo` model locally on the GPU.
**whisperx** adds wav2vec2 forced alignment (tighter word timestamps for
diarization) and batched inference; its transcripts come out lowercase without
punctuation — that is normal whisperx behavior. If the whisperx path fails for
any reason, that meeting automatically falls back to faster-whisper.

whisperx requires the extra packages below, installed exactly this way
(whisperx pins old dependency versions that don't exist for Python 3.14, so it
must be installed without them — `whisperx_engine.py` bridges the API gaps):

```bat
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-diarization.txt
pip install whisperx --no-deps
```

---

## 10. Speaker diarization (optional)

The transcriber can label transcript turns as **Speaker 1 / Speaker 2 / …** using
pyannote speaker diarization. With `engine = "faster-whisper"` it merges pyannote
turns with faster-whisper word timestamps; with `engine = "whisperx"` it uses
whisperx's word-speaker assignment on aligned words. Both run fully locally.

**Install the extra packages** (~3 GB, one time — same as the whisperx install
above, minus the whisperx line if you only want diarization):

```bat
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-diarization.txt
```

**Get a HuggingFace token** (free):

1. Create a token at <https://huggingface.co/settings/tokens> (read access is enough).
2. Accept the model terms at
   <https://huggingface.co/pyannote/speaker-diarization-community-1>.

The token can go in `config.toml → hf_token`, or be left empty if the
`HF_TOKEN` environment variable is set.

**Enable it** in `config.toml`:

```toml
[diarization]
enabled = true
hf_token = ""      # empty = use the HF_TOKEN environment variable
num_speakers = 0   # 0 = auto-detect; set an exact count if known
```

The diarization models download on first use, then run fully offline on the
GPU. If diarization fails for any reason — missing packages, bad token,
unaccepted terms — the plain unlabeled transcript is written instead and the
warning is logged; a meeting is never lost to a diarization problem.

---

## 11. Failure recovery

If transcription fails, the process:

1. Retains the raw audio as a WAV file — the long-session temp file if one
   exists, otherwise the in-memory buffer is dumped to
   `%TEMP%\mtranscriber_recovery_*.wav`.
2. Writes a placeholder note to the vault (or fallback) that includes the
   retained WAV's path.

To recover: transcribe the retained WAV manually, or re-run the transcriber on it:

```python
from transcribe import transcribe
transcript, duration = transcribe("C:/path/to/mtranscriber_XXXXX.wav")
```

Then write the note manually or delete the placeholder and re-run.

---

## 12. Long-session behavior

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
| Hotkey doesn't fire at all, log shows `error 1409` | Another app (or a second copy of this one) owns the combo. Close it, or pick a different combo in `config.toml`, then tray menu → **Re-register hotkey**. Recording still works from the tray menu meanwhile. |
| Hotkey doesn't fire while an elevated window has focus | Run `main.py` as Administrator. (Unverified — see section 5.) |
| Tray is red but nothing is being recorded | Fixed: a capture failure now stops the recording, names the error in a notification, and writes a failure note instead of an empty one. If you see it again, check the log for `[capture] Capture thread error`. |
| Loopback device not found | Run `python capture.py --list-devices`, find your `[LOOPBACK]` entry, update `config.toml`. |
| `ctranslate2` CUDA error on startup | Verify CUDA DLLs are in PATH; run `ctranslate2.get_cuda_device_count()` to diagnose. |
| First run is slow | faster-whisper downloads `large-v3-turbo` (~800 MB) on first call. Subsequent runs use the cache. |
