#!/bin/bash

echo "🐦 Starting Sentiment Analysis App..."
echo "====================================="
echo "Python version: $(python --version)"
echo "Opening browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo "====================================="

# Check if streamlit is available
if ! command -v streamlit &> /dev/null; then
    echo "❌ Streamlit not found. Installing..."
    python -m pip install streamlit
fi

# Run streamlit app
python -m streamlit run main.py
