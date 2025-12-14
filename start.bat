@echo off
echo ================================================================================
echo Amazon Book Review Analyzer - Quick Start
echo ================================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.11 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo [1/5] Checking Python version...
python --version

echo.
echo [2/5] Installing dependencies...
pip install -r requirements.txt

echo.
echo [3/5] Downloading NLTK data...
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"

echo.
echo [4/5] Training model (this may take 3-5 minutes)...
if not exist sentiment_model.pkl (
    python train_and_save_model.py
) else (
    echo Model files already exist, skipping training...
)

echo.
echo [5/5] Starting Flask server...
echo.
echo ================================================================================
echo Server starting at: http://localhost:5000
echo.
echo - Press Ctrl+C to stop the server
echo - Open http://localhost:5000 in your browser
echo ================================================================================
echo.

python app.py

pause
