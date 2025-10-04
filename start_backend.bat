@echo off
REM Backend startup script for Windows

echo 🚀 Starting Voice Activity Recommendation Backend...

REM Navigate to project directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist venv (
    echo 📁 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Install dependencies if needed
if not exist venv\lib (
    echo 📦 Installing dependencies...
    pip install -r requirements.txt
)

REM Start the backend
echo 🔧 Starting FastAPI backend...
cd app
python main.py
pause
