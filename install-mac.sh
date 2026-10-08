#!/bin/bash
# Meeting Transcriber — macOS installer. Counterpart of "Install Meeting Transcriber.ps1".
#
#   ./install-mac.sh [--vault "/path/to/Vault/Meetings"] [--device "BlackHole 2ch"]
#                    [--hotkey "ctrl+shift+r"] [--skip-model-download]
#
# Installs Homebrew prerequisites (portaudio, BlackHole), creates .venv, installs
# Python packages, writes config.toml, and downloads the model. Re-runnable.
set -euo pipefail
cd "$(dirname "$0")"

VAULT=""; DEVICE=""; HOTKEY=""; SKIP_MODEL=0
while [ $# -gt 0 ]; do
  case "$1" in
    --vault)  VAULT="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --hotkey) HOTKEY="$2"; shift 2 ;;
    --skip-model-download) SKIP_MODEL=1; shift ;;
    *) echo "Unknown option: $1"; exit 2 ;;
  esac
done

echo "== Meeting Transcriber install (macOS) =="
echo "Folder: $PWD"

# --- Homebrew prerequisites ---------------------------------------------------
if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required (for PortAudio and BlackHole). Install it from https://brew.sh then rerun."
  exit 1
fi
echo "Installing PortAudio and BlackHole via Homebrew..."
brew list portaudio >/dev/null 2>&1 || brew install portaudio
brew list --cask blackhole-2ch >/dev/null 2>&1 || brew install --cask blackhole-2ch

# --- Python ---------------------------------------------------------------------
PY=""
for candidate in python3.13 python3.14 python3; do
  if command -v "$candidate" >/dev/null 2>&1 &&
     "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 13) else 1)' 2>/dev/null; then
    PY="$candidate"; break
  fi
done
if [ -z "$PY" ]; then
  echo "Python 3.13+ not found. Install it with:  brew install python@3.13 python-tk@3.13"
  echo "(or use the installer from https://www.python.org/downloads/macos/, which bundles Tk)"
  exit 1
fi
echo "Using $($PY --version) at $(command -v "$PY")"
if ! "$PY" -c 'import tkinter' 2>/dev/null; then
  echo "tkinter is missing for this Python. Fix with:  brew install python-tk@$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  exit 1
fi

[ -x .venv/bin/python ] || { echo "Creating virtual environment..."; "$PY" -m venv .venv; }
VENV_PY="$PWD/.venv/bin/python"

echo "Upgrading pip/setuptools/wheel..."
"$VENV_PY" -m pip install --upgrade pip setuptools wheel
echo "Installing dependencies..."
"$VENV_PY" -m pip install -r requirements-mac.txt

ENGINE="faster-whisper"
if [ "$(uname -m)" = "arm64" ]; then
  echo "Apple Silicon detected: installing mlx-whisper (GPU transcription)..."
  if "$VENV_PY" -m pip install mlx-whisper; then
    ENGINE="mlx"
  else
    echo "WARNING: mlx-whisper install failed; transcription will run on the CPU (slower)."
  fi
fi

# --- config.toml ------------------------------------------------------------------
if [ ! -f config.toml ]; then
  cp config.example.toml config.toml
  sed -i '' "s|^engine = .*|engine = \"$ENGINE\"|" config.toml
fi
# Only overwrite what was passed on the command line; config.toml keeps its comments.
[ -n "$VAULT" ]  && sed -i '' "s|^vault_meetings_path = .*|vault_meetings_path = \"$VAULT\"|" config.toml
[ -n "$DEVICE" ] && sed -i '' "s|^output_device_name = .*|output_device_name = \"$DEVICE\"|" config.toml
[ -n "$HOTKEY" ] && sed -i '' "s|^hotkey = .*|hotkey = \"$HOTKEY\"|" config.toml

chmod +x "Start Meeting Transcriber.command"

# --- checks -------------------------------------------------------------------------
echo
echo "Available audio devices:"
DEVICES="$("$VENV_PY" capture.py --list-devices)"
echo "$DEVICES"
if ! echo "$DEVICES" | grep -q '\[LOOPBACK\]'; then
  echo "WARNING: BlackHole is not visible yet. Reboot (or log out and back in), then rerun this script."
fi

if [ "$SKIP_MODEL" -eq 0 ]; then
  echo
  echo "Downloading/loading the large-v3-turbo model ($ENGINE). This can take several minutes..."
  if [ "$ENGINE" = "mlx" ]; then
    "$VENV_PY" -c "import mlx_engine; mlx_engine._get_model(); print('model_loaded')"
  else
    "$VENV_PY" -c "from faster_whisper import WhisperModel; WhisperModel('large-v3-turbo', device='cpu', compute_type='int8'); print('model_loaded')"
  fi
fi

cat <<EOF

Install complete. Engine: $ENGINE

Next:
1. Edit config.toml: set vault_meetings_path to your Obsidian Meetings folder.
2. Route call audio through BlackHole (one time, see README "macOS install"):
   Audio MIDI Setup -> "+" -> Create Multi-Output Device -> tick your speakers
   AND "BlackHole 2ch" -> then System Settings -> Sound -> Output -> that device.
   Keep your normal microphone as the Input device.
3. Double-click "Start Meeting Transcriber.command".
4. On first run grant Microphone and Accessibility to Terminal when macOS asks
   (Accessibility is what makes the hotkey work). Then use the menu bar icon ->
   "Re-register hotkey", or restart the app.
EOF
