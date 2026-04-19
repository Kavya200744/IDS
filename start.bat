@echo off
echo ===================================
echo IDS System - Quick Start
echo ===================================
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate
echo.

REM Install requirements
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Check if models exist
if not exist "models\ids_model_ensemble.pkl" (
    echo Models not found. Training models...
    echo This will take 3-5 minutes on i3 processor...
    echo.
    python train.py
    echo.
)

REM Run the application
echo ===================================
echo Starting IDS Web Application...
echo ===================================
echo.
echo Open your browser and go to:
echo http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python app.py

pause
