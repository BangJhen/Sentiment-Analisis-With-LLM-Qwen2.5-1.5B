#!/bin/bash

# Script untuk menjalankan aplikasi Streamlit Analisis Sentimen

echo "🐦 Starting Sentiment Analysis App..."
echo "=================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install -r requirements.txt

# Run streamlit app
echo "🚀 Starting Streamlit app..."
echo "App will be available at: http://localhost:8501"
echo "Press Ctrl+C to stop the application"
echo "=================================="

streamlit run main.py
