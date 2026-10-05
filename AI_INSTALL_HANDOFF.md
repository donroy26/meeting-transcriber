# Meeting Transcriber: Install Handoff for AI

You are installing a local Windows meeting transcriber on another PC.

## What This Bundle Does

This is a portable Windows script bundle. It records microphone plus system audio with a global hotkey, transcribes locally on an NVIDIA GPU using faster-whisper, and writes a markdown note into an Obsidian `Meetings` folder.

It does not use cloud transcription, Anthropic, OpenAI, or any LLM API.

## Target Requirements

- Windows 10/11.
- NVIDIA GPU with a working driver.
- Python 3.13 or 3.14 available as `py` or `python`.
- Internet during install for Python packages and the first model download.
- Obsidian vault path for the target PC's `Meetings` folder.

The verified dependency set is in `requirements.txt`. It includes NVIDIA CUDA runtime wheels, so do not install random CUDA packages unless the checks fail.

## Files To Know

- `Install Meeting Transcriber.bat` - double-click installer wrapper.
- `Install Meeting Transcriber.ps1` - real installer.
- `Start Meeting Transcriber.bat` - user launcher after install.
- `config.toml` - machine-specific config, created by installer.
- `config.example.toml` - template.
- `Logs/` - created at runtime; inspect newest log if the app crashes.

## Recommended Install

Open PowerShell in this folder and run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\Install Meeting Transcriber.ps1"
```

Or double-click:

```text
Install Meeting Transcriber.bat
```

The installer will:

1. Create `.venv`.
2. Upgrade pip/setuptools/wheel.
3. Install `requirements.txt`.
4. Create `config.toml` if missing.
5. Check CUDA visibility with `ctranslate2.get_cuda_device_count()`.
6. Download/load `large-v3-turbo`.
7. Print available audio devices.

## Optional One-Command Config

If you already know the target vault and loopback device, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\Install Meeting Transcriber.ps1" `
  -VaultMeetingsPath "D:\Obsidian\MyVault\Meetings" `
  -OutputDeviceName "DEVICE NAME FROM [LOOPBACK] LIST" `
  -Hotkey "ctrl+shift+r"
```

Use the real target path and loopback device. If unsure about the loopback name, omit `-OutputDeviceName`, run the installer, read the printed device list, then edit `config.toml`.

## Configure

Open `config.toml`:

```toml
[paths]
vault_meetings_path = "D:/Obsidian/MyVault/Meetings"
fallback_folder = ""

[audio]
output_device_name = ""

[hotkey]
hotkey = "ctrl+shift+r"
```

Set `vault_meetings_path` to the target PC's real Obsidian `Meetings` folder. Use forward slashes or escaped backslashes.

Usually leave `output_device_name` blank. Blank means the app captures every
`[LOOPBACK]` device found at recording start, which is best when the user sometimes
uses headphones and sometimes uses speakers.

To inspect loopback devices:

```powershell
.\.venv\Scripts\python.exe capture.py --list-devices
```

Set `output_device_name` only if you must force one specific output device.

Loopback is how the app captures callers / PC audio. With blank
`output_device_name`, Teams, Zoom, Meet, or the browser can use any Windows
output device that exposes a loopback channel.

## Run

Double-click:

```text
Start Meeting Transcriber.bat
```

The launcher sets the NVIDIA DLL folders on `PATH` before running Python. Use this launcher instead of double-clicking `main.py`.

Workflow:

1. Press `ctrl+shift+r` once to start recording.
2. Press `ctrl+shift+r` again to stop and transcribe.
3. Wait until processing finishes.
4. Confirm a note appears in the configured `Meetings` folder.

Tray colors:

- Gray: idle.
- Red: recording.
- Orange: processing.

While recording, a small floating meter should appear near the upper-right of
the screen. Its moving bars are a rough mixed-audio activity indicator for
microphone plus PC/caller loopback audio. It is not a calibrated volume meter.

A notes window should also open while recording. The user can enter attendees and
freeform notes there. On stop, those fields are written into `## Attendees` and
`## Notes` in the markdown output.

The notes window includes a **To-do** button. It marks the current notes line as
`- [ ] TODO: ...` so the later summarizer can treat that line as an action-item
candidate without this recorder filling `## Action Items` itself.

Do not quit while orange.

## Acceptance Test

Run a short 5-10 second test:

1. Start the app.
2. Play a YouTube/video/audio clip or system sound.
3. Speak a sentence into the mic.
4. Confirm the floating meter appears and responds with moving bars.
5. Enter at least one attendee and one note in the notes window. Click **To-do**
   on one notes line.
6. Stop the recording.
7. Confirm the note appears in the vault.
8. Confirm the note includes `## Attendees` and `## Notes` with the entered text,
   including a `- [ ] TODO:` line.
9. Confirm the transcript contains both the user's microphone sentence and at least
   some PC/caller audio from the clip or call output.

If it fails or disappears, inspect the newest log:

```text
Logs\meeting-transcriber-YYYYMMDD-HHMMSS.log
```

## Known Fixes Already Included

- Python 3.14-compatible package pins.
- NVIDIA CUDA runtime/cublas/cudnn wheels in `requirements.txt`.
- Runtime NVIDIA DLL path setup in both `Start Meeting Transcriber.bat` and `transcribe.py`.
- Crash logging with `faulthandler` to `Logs/`.
- Fix for hotkey handler `_state` scope bug.
- Fix for `pyaudiowpatch` native crash when stopping while loopback read is blocked.

## Common Problems

### `cuda_devices 0`

Update or install NVIDIA drivers, reboot, then rerun the installer.

### `cublas64_12.dll is not found`

Make sure the user launches via `Start Meeting Transcriber.bat`, not `main.py`. Then rerun:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Hotkey does not work in some apps

Run `Start Meeting Transcriber.bat` as Administrator if the focused app is elevated.

### No far-side/system audio

Run:

```powershell
.\.venv\Scripts\python.exe capture.py --list-devices
```

Set `output_device_name` to the correct `[LOOPBACK]` device in `config.toml`.

### App crashes

Read the newest file in `Logs/`. The current build logs Python tracebacks and native Windows access violations.
