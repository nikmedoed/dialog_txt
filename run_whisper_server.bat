@echo off
setlocal
cd /d "%~dp0"
if not defined DIALOG_TXT_SERVER_TOKEN (
  echo Set DIALOG_TXT_SERVER_TOKEN before starting the LAN server.
  echo Example: set DIALOG_TXT_SERVER_TOKEN=a-long-random-secret
  exit /b 2
)
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m dialog_txt.whisper_server --host 0.0.0.0 --port 8765
) else (
  python -m dialog_txt.whisper_server --host 0.0.0.0 --port 8765
)
