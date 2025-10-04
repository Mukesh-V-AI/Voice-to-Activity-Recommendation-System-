#!/bin/bash
# Frontend startup script

echo "🎨 Starting Voice Activity Recommendation Frontend..."

# Navigate to project directory  
cd "$(dirname "$0")"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "📁 Activating virtual environment..."
    source venv/bin/activate
fi

# Start the frontend
echo "💻 Starting Streamlit frontend..."
streamlit run frontend/streamlit_app.py
