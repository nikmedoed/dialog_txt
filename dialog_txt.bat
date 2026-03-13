@echo off
setlocal
pushd "%~dp0"

if not "%DIALOG_TXT_HOME%"=="" (
    if not exist "%DIALOG_TXT_HOME%" mkdir "%DIALOG_TXT_HOME%" >nul 2>&1
)

set "PY_EXE=%~dp0.venv\Scripts\python.exe"
if not exist "%PY_EXE%" set "PY_EXE=python"

"%PY_EXE%" "%~dp0main.py" %*
set "EC=%ERRORLEVEL%"

popd
if %EC% neq 0 (
    echo.
    echo dialog-txt exited with code %EC%.
    pause
)
exit /b %EC%
