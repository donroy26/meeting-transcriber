@echo off
setlocal

cd /d "%~dp0"
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0Install Meeting Transcriber.ps1"

if errorlevel 1 (
  echo.
  echo Install failed. Read the error above.
  pause
  exit /b 1
)

echo.
echo Install finished.
pause
