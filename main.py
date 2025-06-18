#!/usr/bin/env python3
"""
Main application launcher with integrated data processing
This script starts all components of the Live Insurance Risk & Quote Co-Pilot
"""

import asyncio
import threading
import time
import signal
import sys
import os
from multiprocessing import Process
from dotenv import load_dotenv

# Import our modules
from app import app
from data_fetcher import DataFetcher
from pipeline import LiveRAGPipeline

# Load environment variables
load_dotenv()

class ApplicationManager:
    def __init__(self):
        self.data_fetcher = None
        self.rag_pipeline = None
        self.running = False
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        print("\n🛑 Shutdown signal received...")
        self.running = False
        sys.exit(0)
    
    def start_data_fetcher(self):
        """Start data fetching in background thread"""
        print("🔄 Starting data fetcher...")
        
        self.data_fetcher = DataFetcher()
        
        # Run initial data fetch
        self.data_fetcher.fetch_weather_alerts()
        self.data_fetcher.fetch_news_alerts()
        self.data_fetcher.fetch_crime_data()
        
        # Start scheduled fetching
        self.data_fetcher.start_scheduled_fetching()
        
        print("✅ Data fetcher started")
    
    def start_rag_pipeline(self):
        """Start RAG pipeline in background thread"""
        print("🧠 Starting RAG pipeline...")
        
        self.rag_pipeline = LiveRAGPipeline()
        self.rag_pipeline.load_existing_data()
        
        # Start file monitoring
        monitor_thread = threading.Thread(
            target=self.rag_pipeline.monitor_new_files, 
            daemon=True
        )
        monitor_thread.start()
        
        print("✅ RAG pipeline started")
    
    def start_web_server(self):
        """Start the FastAPI web server"""
        print("🌐 Starting web server...")
        
        import uvicorn
        
        # Configure server
        config = uvicorn.Config(
            app=app,
            host="0.0.0.0",
            port=8000,
            log_level="info",
            access_log=False
        )
        
        server = uvicorn.Server(config)
        
        print("✅ Web server started at http://localhost:8000")
        
        # Run server
        asyncio.run(server.serve())
    
    def check_dependencies(self):
        """Check if all dependencies are available"""
        try:
            import pathway
            import sentence_transformers
            import fastapi
            import uvicorn
            print("✅ All dependencies available")
            return True
        except ImportError as e:
            print(f"❌ Missing dependency: {e}")
            print("Run: pip install -r requirements.txt")
            return False
    
    def print_startup_info(self):
        """Print startup information"""
        print("🏢 Live Insurance Risk & Quote Co-Pilot")
        print("=" * 50)
        print("Real-time insurance underwriting with AI")
        print("=" * 50)
        print()
        
        # System info
        print("📊 System Information:")
        print(f"   Python Version: {sys.version.split()[0]}")
        print(f"   Data Directory: {os.path.abspath('./live_data_feed')}")
        print(f"   Config File: {os.path.abspath('.env')}")
        
        # Check simulation mode
        if (os.getenv("OPENROUTER_API_KEY", "").startswith("your_") or 
            os.getenv("NEWS_API_KEY", "").startswith("your_")):
            print("   Mode: Simulation (no API keys configured)")
        else:
            print("   Mode: Live (API keys configured)")
        
        print()
    
    def wait_for_components(self):
        """Wait for all components to initialize"""
        print("⏳ Initializing components...")
        time.sleep(2)  # Give components time to start
        
        # Check if data directory exists and has files
        data_dir = "./live_data_feed"
        if os.path.exists(data_dir):
            file_count = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
            print(f"📁 Found {file_count} data files")
        else:
            print("📁 Creating data directory...")
            os.makedirs(data_dir, exist_ok=True)
    
    def run(self):
        """Run the complete application"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        # Print startup info
        self.print_startup_info()
        
        # Check dependencies
        if not self.check_dependencies():
            return False
        
        try:
            # Start components in order
            self.start_data_fetcher()
            self.start_rag_pipeline()
            self.wait_for_components()
            
            print("\n🚀 All components started successfully!")
            print("🌐 Web interface: http://localhost:8000")
            print("📊 API endpoints: http://localhost:8000/docs")
            print("\n💡 Press Ctrl+C to stop")
            print("-" * 50)
            
            # Start web server (this will block)
            self.start_web_server()
            
        except KeyboardInterrupt:
            print("\n🛑 Shutting down...")
            self.running = False
        except Exception as e:
            print(f"❌ Error starting application: {e}")
            return False
        
        return True

def main():
    """Main entry point"""
    
    # Check if running with arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "--data-only":
            # Run only data fetcher for testing
            print("🔄 Running data fetcher only...")
            fetcher = DataFetcher()
            fetcher.start_scheduled_fetching()
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("🛑 Stopping data fetcher")
                return
        
        elif sys.argv[1] == "--pipeline-only":
            # Run only RAG pipeline for testing
            print("🧠 Running RAG pipeline only...")
            pipeline = LiveRAGPipeline()
            pipeline.load_existing_data()
            
            # Start monitoring
            monitor_thread = threading.Thread(
                target=pipeline.monitor_new_files, 
                daemon=True
            )
            monitor_thread.start()
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("🛑 Stopping RAG pipeline")
                return
        
        elif sys.argv[1] == "--help":
            print("Live Insurance Risk & Quote Co-Pilot")
            print("\nUsage:")
            print("  python main.py              # Start complete application")
            print("  python main.py --data-only  # Start data fetcher only")
            print("  python main.py --pipeline-only # Start RAG pipeline only")
            print("  python main.py --help       # Show this help")
            return
    
    # Run complete application
    app_manager = ApplicationManager()
    success = app_manager.run()
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
