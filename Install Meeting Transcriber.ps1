param(
  [string]$VaultMeetingsPath = "",
  [string]$OutputDeviceName = "",
  [string]$Hotkey = "ctrl+shift+r",
  [switch]$SkipModelDownload
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

function Find-Python {
  $candidates = @(
    @{ Command = "py"; Args = @("-3.14") },
    @{ Command = "py"; Args = @("-3.13") },
    @{ Command = "py"; Args = @("-3") },
    @{ Command = "python"; Args = @() }
  )

  foreach ($candidate in $candidates) {
    $cmd = Get-Command $candidate.Command -ErrorAction SilentlyContinue
    if (-not $cmd) { continue }

    & $candidate.Command @($candidate.Args + @("-c", "import sys; print(sys.version_info[:2])")) *> $null
    if ($LASTEXITCODE -eq 0) {
      return $candidate
    }
  }

  throw "Python 3.13+ or 3.14+ was not found. Install Python from https://www.python.org/downloads/windows/ and rerun this script."
}

Write-Host "== Meeting Transcriber install =="
Write-Host "Folder: $Root"

$python = Find-Python
Write-Host "Using Python launcher: $($python.Command) $($python.Args -join ' ')"

if (-not (Test-Path ".venv\Scripts\python.exe")) {
  Write-Host "Creating virtual environment..."
  & $python.Command @($python.Args + @("-m", "venv", ".venv"))
}

$VenvPython = Join-Path $Root ".venv\Scripts\python.exe"

Write-Host "Upgrading pip/setuptools/wheel..."
& $VenvPython -m pip install --upgrade pip setuptools wheel

Write-Host "Installing dependencies..."
& $VenvPython -m pip install -r requirements.txt

if (-not (Test-Path "config.toml")) {
  Copy-Item "config.example.toml" "config.toml"
}

if ($VaultMeetingsPath -or $OutputDeviceName -or $Hotkey) {
  $vault = $VaultMeetingsPath.Replace("\", "/")
  $config = @"
[paths]
vault_meetings_path = "$vault"
fallback_folder = ""

[audio]
output_device_name = "$OutputDeviceName"

[hotkey]
hotkey = "$Hotkey"
"@
  Set-Content -Path "config.toml" -Value $config -Encoding UTF8
}

Write-Host "Checking CUDA visibility..."
& $VenvPython -c "import ctranslate2; print('cuda_devices', ctranslate2.get_cuda_device_count())"
if ($LASTEXITCODE -ne 0) {
  throw "CUDA check failed. Verify NVIDIA drivers are installed and reboot if drivers were just updated."
}

if (-not $SkipModelDownload) {
  Write-Host "Downloading/loading large-v3-turbo model. This can take several minutes..."
  & $VenvPython -c "from faster_whisper import WhisperModel; print('loading large-v3-turbo...'); WhisperModel('large-v3-turbo', device='cuda', compute_type='float16'); print('model_loaded')"
}

Write-Host ""
Write-Host "Available audio devices:"
& $VenvPython capture.py --list-devices

Write-Host ""
Write-Host "Install complete."
Write-Host "Next:"
Write-Host "1. Edit config.toml and set vault_meetings_path to the target Obsidian Meetings folder."
Write-Host "2. Leave output_device_name blank to capture all [LOOPBACK] devices, or set one specific device if needed."
Write-Host "3. Double-click 'Start Meeting Transcriber.bat'."
