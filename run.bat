@echo off
echo ======================================================
echo           AgriVision v3.0 - Auto Launcher
echo ======================================================

:: Step 1: Virtual Environment
echo [1/4] Checking Virtual Environment...
if not exist venv (
    echo [INFO] Creating new virtual environment...
    python -m venv venv
)

:: Step 2: Activation
echo [2/4] Activating Environment...
call venv\Scripts\activate

:: Step 3: Dependencies
echo [3/4] Installing/Updating Dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt

:: Step 4: Run Application
echo [4/4] Starting AgriVision Server...
echo.
echo 🚀 App will be available at: http://127.0.0.1:5000
echo.
python app.py

pause
