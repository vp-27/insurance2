#!/bin/bash

# Live Insurance Risk & Quote Co-Pilot Startup Script

echo "🏢 Starting Live Insurance Risk & Quote Co-Pilot..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    exit 1
fi

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Create data directory if it doesn't exist
mkdir -p live_data_feed

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Using default configuration..."
    echo "   You can add API keys to .env for enhanced functionality."
fi

echo ""
echo "🚀 Starting the application..."
echo ""
echo "The application will be available at: http://localhost:8000"
echo ""
echo "To stop the application, press Ctrl+C"
echo ""

# Start the application
python app.py
