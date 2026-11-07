@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0server.ps1" %*
set "EC=%ERRORLEVEL%"
if "%~1"=="" if not "%EC%"=="0" pause
exit /b %EC%
