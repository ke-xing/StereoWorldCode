@echo off
REM StereoWorld Studio launcher - finds the "stereoworld" conda env and starts the GUI with it
setlocal
cd /d "%~dp0"
set "PY="
for %%R in ("%CONDA_PREFIX%\..\.." "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
  if not defined PY if exist "%%~R\envs\stereoworld\python.exe" set "PY=%%~R\envs\stereoworld\python.exe"
)
if not defined PY (
  echo Could not find the "stereoworld" conda environment.
  echo Create it first:  conda create -n stereoworld python=3.11 -y
  pause
  exit /b 1
)
echo Using %PY%
"%PY%" -c "import PySide6" 2>nul || "%PY%" -m pip install PySide6
"%PY%" gui.py %*
if errorlevel 1 pause
