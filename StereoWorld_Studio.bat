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
"%PY%" -c "from PySide6 import QtWidgets" 2>nul
if errorlevel 1 (
  echo.
  echo PySide6 is installed but Qt cannot load ^(usually an outdated MSVC runtime in the conda env^).
  echo Fix in the Anaconda Prompt:
  echo   conda activate stereoworld
  echo   conda install -c conda-forge "vc14_runtime>=14.42" -y
  echo If that is not enough:
  echo   pip uninstall -y PySide6 PySide6-Essentials PySide6-Addons shiboken6
  echo   conda install -c conda-forge pyside6 -y
  pause
  exit /b 1
)
"%PY%" gui.py %*
if errorlevel 1 pause
