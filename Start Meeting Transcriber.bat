@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "NVIDIA_DIR=%SCRIPT_DIR%.venv\Lib\site-packages\nvidia"

if not exist "%PYTHON%" (
  echo Could not find the project Python environment:
  echo   %PYTHON%
  echo.
  echo Recreate it with:
  echo   python -m venv .venv
  echo   .venv\Scripts\python.exe -m pip install -r requirements.txt
  echo.
  pause
  exit /b 1
)

cd /d "%SCRIPT_DIR%"
set "PATH=%NVIDIA_DIR%\cublas\bin;%NVIDIA_DIR%\cuda_nvrtc\bin;%NVIDIA_DIR%\cuda_runtime\bin;%NVIDIA_DIR%\cudnn\bin;%PATH%"
"%PYTHON%" "%SCRIPT_DIR%main.py"

if errorlevel 1 (
  echo.
  echo Meeting Transcriber exited with an error.
  pause
)
