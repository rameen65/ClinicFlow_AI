@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run run_windows.bat once first so it can create the environment and install dependencies.
  exit /b 1
)
".venv\Scripts\python.exe" -m unittest discover -v
exit /b %errorlevel%
