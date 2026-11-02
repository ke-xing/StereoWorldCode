@echo off
REM StereoWorld Studio launcher
REM  - jobs run in the "stereoworld" conda env (found automatically by the GUI)
REM  - the window itself runs in any Python whose PySide6/Qt loads; conda envs
REM    often break Qt DLLs, so a plain system Python is tried as well.
setlocal
cd /d "%~dp0"

set "ENVPY="
for %%R in ("%CONDA_PREFIX%\..\.." "%USERPROFILE%\miniconda3" "%USERPROFILE%\anaconda3" "%USERPROFILE%\miniforge3" "%LOCALAPPDATA%\miniconda3" "%LOCALAPPDATA%\anaconda3" "%ProgramData%\miniconda3" "%ProgramData%\anaconda3" "C:\miniconda3" "C:\anaconda3") do (
  if not defined ENVPY if exist "%%~R\envs\stereoworld\python.exe" set "ENVPY=%%~R\envs\stereoworld\python.exe"
)
if not defined ENVPY (
  echo Could not find the "stereoworld" conda environment.
  echo Create it first:  conda create -n stereoworld python=3.11 -y
  pause
  exit /b 1
)
echo Jobs will run with: %ENVPY%

REM 1) env python, if its Qt works
"%ENVPY%" -c "from PySide6 import QtWidgets" 1>nul 2>nul && (set "GUIPY=%ENVPY%" & goto run)

REM 2) a normal Python install (py launcher, then python on PATH)
for %%P in (py python) do (
  where %%P 1>nul 2>nul && (
    %%P -c "from PySide6 import QtWidgets" 1>nul 2>nul && (set "GUIPY=%%P" & goto run)
  )
)
REM 3) install PySide6 into a normal Python install and retry
for %%P in (py python) do (
  where %%P 1>nul 2>nul && (
    %%P -c "import sys; sys.exit('conda' in sys.version.lower() or 'envs' in sys.executable.lower())" 1>nul 2>nul && (
      echo Installing PySide6 for the GUI window into: & %%P -c "import sys; print(sys.executable)"
      %%P -m pip install --user PySide6
      %%P -c "from PySide6 import QtWidgets" 1>nul 2>nul && (set "GUIPY=%%P" & goto run)
    )
  )
)
echo.
echo Could not get Qt/PySide6 working in any Python.
echo Option A: install python.org Python 3.12, then:  py -m pip install PySide6
echo Option B: conda activate stereoworld
echo           pip uninstall -y PySide6 PySide6-Essentials PySide6-Addons shiboken6
echo           conda install -c conda-forge pyside6 -y
pause
exit /b 1

:run
echo GUI runs with: %GUIPY%
"%GUIPY%" gui.py %*
if errorlevel 1 pause
