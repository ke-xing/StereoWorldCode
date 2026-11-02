@echo off
REM StereoWorld Studio launcher - activates the conda env and starts the GUI
cd /d "%~dp0"
call conda activate stereoworld 2>nul
python -c "import PySide6" 2>nul || pip install PySide6
python gui.py %*
if errorlevel 1 pause
