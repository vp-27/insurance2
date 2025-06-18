#!/bin/bash

# 🏢 Live Insurance Risk & Quote Co-Pilot - Complete Setup & Demo
# This script sets up and demonstrates the entire system

echo "🚀 Starting Live Insurance Risk & Quote Co-Pilot Setup"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null; then
    echo "❌ pip is required but not installed"
    exit 1
fi

echo "✅ Python and pip are available"

# Install dependencies if needed
echo "📦 Checking dependencies..."
pip install -q -r requirements.txt

# Create live data directory
mkdir -p live_data_feed

echo "🔧 System setup complete!"
echo ""

# Start the server in background
echo "🌐 Starting FastAPI server..."
python -m uvicorn app:app --host 0.0.0.0 --port 8000 --log-level error &
SERVER_PID=$!

# Wait for server to start
echo "⏳ Waiting for server to initialize..."
sleep 5

# Check if server is running
if curl -s http://localhost:8000/health > /dev/null; then
    echo "✅ Server is running successfully!"
else
    echo "❌ Server failed to start"
    kill $SERVER_PID 2>/dev/null
    exit 1
fi

echo ""
echo "🎉 Live Insurance Risk & Quote Co-Pilot is now running!"
echo "=================================================="
echo ""
echo "🌐 Web Interface: http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "❤️ Health Check: http://localhost:8000/health"
echo ""
echo "🎮 Demo Options:"
echo "1. 🖥️  Open web interface for manual testing"
echo "2. 🤖 Run automated demo script"
echo "3. 📱 Use API endpoints directly"
echo ""

# Ask user for demo preference
read -p "Choose demo option (1, 2, 3, or 'skip'): " choice

case $choice in
    1)
        echo "🖥️ Opening web interface..."
        if command -v open &> /dev/null; then
            open http://localhost:8000
        elif command -v xdg-open &> /dev/null; then
            xdg-open http://localhost:8000
        else
            echo "📱 Please manually open: http://localhost:8000"
        fi
        ;;
    2)
        echo "🤖 Running automated demo..."
        echo ""
        python demo_enhanced.py
        ;;
    3)
        echo "📱 API Demo - Testing endpoints..."
        echo ""
        echo "🔍 Health Check:"
        curl -s http://localhost:8000/health | python -m json.tool
        echo ""
        echo ""
        echo "📊 System Stats:"
        curl -s http://localhost:8000/stats | python -m json.tool
        echo ""
        echo ""
        echo "🏢 Sample Risk Assessment:"
        curl -s -X POST "http://localhost:8000/get_assessment" \
             -H "Content-Type: application/json" \
             -d '{"address": "25 Columbus Dr, Jersey City, NJ", "query": "What are the current risks?"}' | \
             python -m json.tool
        ;;
    *)
        echo "⏭️ Skipping demo - server is running for manual testing"
        ;;
esac

echo ""
echo "📋 System Information:"
echo "=================="
echo "• Real-time data processing with Pathway streaming engine"
echo "• Dynamic vector embeddings with sentence-transformers"
echo "• AI-powered risk assessment with LLM integration"
echo "• Live data feeds from weather, news, crime, earthquake APIs"
echo "• Modern React frontend with real-time updates"
echo "• RESTful API with comprehensive error handling"
echo ""

echo "🛠️ Quick Testing Guide:"
echo "===================="
echo "1. Navigate to http://localhost:8000"
echo "2. Enter a property address (e.g., '25 Columbus Dr, Jersey City, NJ')"
echo "3. Click 'Get Live Assessment' to see baseline risk"
echo "4. Use demo controls to inject test alerts:"
echo "   • 🔥 Fire Alert - See risk jump to 9/10"
echo "   • 🌊 Flood Alert - Watch quotes increase"
echo "   • 🚨 Crime Alert - Observe security risk analysis"
echo "   • 🏗️ Earthquake Alert - View seismic risk assessment"
echo "5. Enable auto-refresh for live monitoring"
echo ""

echo "🎯 Expected Demo Results:"
echo "======================="
echo "• Baseline risk: 2/10, Quote: $600/month"
echo "• Fire emergency: 9/10, Quote: $950/month (+58% increase)"
echo "• Flood warning: 7/10, Quote: $850/month (+42% increase)"
echo "• Crime incident: 6/10, Quote: $800/month (+33% increase)"
echo "• Earthquake: 8/10, Quote: $900/month (+50% increase)"
echo ""

# Keep the server running
echo "🔄 Server is running... Press Ctrl+C to stop"
echo ""

# Wait for user interruption
trap "echo ''; echo '🛑 Shutting down server...'; kill $SERVER_PID 2>/dev/null; echo '✅ Server stopped. Thank you for trying the Live Insurance Risk & Quote Co-Pilot!'; exit 0" INT

# Keep script running
while true; do
    sleep 1
done
