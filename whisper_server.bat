@echo off
setlocal
pushd "%~dp0"
title Dialog TXT - Network Whisper Server

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

echo.
echo Starting server on all local network interfaces, port 8765...
echo Stop it with Ctrl+C or by closing this window.
echo.
"%PY_EXE%" "%~dp0whisper_server.py"
set "EC=%ERRORLEVEL%"

echo.
if not "%EC%"=="0" echo Server exited with code %EC%.
pause
popd
exit /b %EC%
