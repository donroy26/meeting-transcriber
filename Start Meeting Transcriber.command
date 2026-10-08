#!/bin/bash
# macOS launcher — double-click in Finder. Counterpart of "Start Meeting Transcriber.bat".
cd "$(dirname "$0")" || exit 1
PY=".venv/bin/python"

if [ ! -x "$PY" ]; then
  echo "Could not find the project Python environment: $PWD/$PY"
  echo "Run ./install-mac.sh first."
  read -n 1 -r -s -p "Press any key to close."
  exit 1
fi

"$PY" main.py
status=$?
if [ "$status" -ne 0 ]; then
  echo
  echo "Meeting Transcriber exited with an error (code $status). See the newest file in Logs/."
  read -n 1 -r -s -p "Press any key to close."
fi
