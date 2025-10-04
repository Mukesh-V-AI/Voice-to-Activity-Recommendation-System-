@echo off
REM Frontend startup script for Windows

echo 🎨 Starting Voice Activity Recommendation Frontend...

REM Navigate to project directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist venv (
    echo 📁 Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Start the frontend
echo 💻 Starting Streamlit frontend...
streamlit run frontend/streamlit_app.py
pause
