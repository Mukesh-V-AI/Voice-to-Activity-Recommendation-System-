#!/bin/bash
# Backend startup script

echo "🚀 Starting Voice Activity Recommendation Backend..."

# Navigate to project directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📁 Activating virtual environment..."
    source venv/bin/activate
fi

# Install dependencies if needed
if [ ! -d "venv/lib" ]; then
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
fi

# Start the backend
echo "🔧 Starting FastAPI backend..."
cd app
python main.py
