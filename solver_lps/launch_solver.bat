@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "BOOTSTRAP=%SCRIPT_DIR%bootstrap_solver.py"

if exist "%SCRIPT_DIR%.solver_env\Scripts\python.exe" (
    "%SCRIPT_DIR%.solver_env\Scripts\python.exe" "%BOOTSTRAP%" %*
    exit /b %errorlevel%
)

if exist "%USERPROFILE%\.venv\Scripts\python.exe" (
    "%USERPROFILE%\.venv\Scripts\python.exe" "%BOOTSTRAP%" %*
    exit /b %errorlevel%
)

if exist "%SCRIPT_DIR%..\NeptuVisionS2\.venv\Scripts\python.exe" (
    "%SCRIPT_DIR%..\NeptuVisionS2\.venv\Scripts\python.exe" "%BOOTSTRAP%" %*
    exit /b %errorlevel%
)

if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" "%BOOTSTRAP%" %*
    exit /b %errorlevel%
)

if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" "%BOOTSTRAP%" %*
    exit /b %errorlevel%
)

if exist "C:\Program Files\KiCad\10.0\bin\python.exe" (
    "C:\Program Files\KiCad\10.0\bin\python.exe" "%BOOTSTRAP%" %*
    exit /b %errorlevel%
)

if exist "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" (
    "%USERPROFILE%\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" "%BOOTSTRAP%" %*
    exit /b %errorlevel%
)

echo Aucun interpreteur Python compatible n'a ete trouve.
echo Installe Python 3.11+ ou reviens avec KiCad Python disponible.
exit /b 1
