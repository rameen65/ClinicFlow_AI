@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  where py >nul 2>nul
  if errorlevel 1 (
    echo Python 3.13 was not found. Install Python 3.13 from python.org and enable the py launcher.
    pause
    exit /b 1
  )
  py -3.13 -m venv .venv
  if errorlevel 1 py -3 -m venv .venv
  if errorlevel 1 (
    echo Could not create the virtual environment.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -c "import streamlit, pandas, sklearn" >nul 2>nul
if errorlevel 1 (
  echo Installing the required Python packages. This is needed only once.
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo Package installation failed. Check your internet connection and try again.
    pause
    exit /b 1
  )
)
".venv\Scripts\python.exe" -m streamlit run app.py
endlocal
