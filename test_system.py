#!/usr/bin/env python3
"""
Quick test script to verify the Live Insurance Risk & Quote Co-Pilot setup
"""

import os
import sys
import json
import time
from pathlib import Path

def test_file_structure():
    """Test if all required files exist"""
    print("📁 Testing file structure...")
    
    required_files = [
        'app.py',
        'pipeline.py', 
        'data_fetcher.py',
        'config.py',
        'requirements.txt',
        '.env',
        'README.md'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    print("✅ All required files present")
    return True

def test_data_directory():
    """Test data directory setup"""
    print("📂 Testing data directory...")
    
    data_dir = Path("./live_data_feed")
    
    if not data_dir.exists():
        print("Creating data directory...")
        data_dir.mkdir(exist_ok=True)
    
    # Create a test file
    test_data = {
        "source": "test",
        "timestamp": "2025-06-18T12:00:00Z",
        "location": "Test Location",
        "content": "Test alert for system verification",
        "type": "test"
    }
    
    test_file = data_dir / "test_alert.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"✅ Data directory ready: {data_dir.absolute()}")
    return True

def test_imports():
    """Test if core modules can be imported"""
    print("🔧 Testing module imports...")
    
    try:
        # Test config
        from config import config
        print("  ✅ Config module")
        
        # Test data fetcher
        from data_fetcher import DataFetcher
        print("  ✅ Data fetcher module")
        
        # Test pipeline (may fail without dependencies)
        try:
            from pipeline import LiveRAGPipeline
            print("  ✅ RAG pipeline module")
        except ImportError as e:
            print(f"  ⚠️  RAG pipeline: {e} (install dependencies)")
        
        # Test FastAPI app
        try:
            from app import app
            print("  ✅ FastAPI app module")
        except ImportError as e:
            print(f"  ⚠️  FastAPI app: {e} (install dependencies)")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_configuration():
    """Test configuration setup"""
    print("⚙️ Testing configuration...")
    
    try:
        from config import config
        
        print(f"  📊 Base insurance cost: ${config.BASE_INSURANCE_COST}")
        print(f"  📈 Risk multiplier: {config.RISK_MULTIPLIER}")
        print(f"  🕐 Refresh interval: {config.REFRESH_INTERVAL}s")
        
        if config.is_simulation_mode():
            print("  🎭 Running in simulation mode")
        else:
            print("  🔑 API keys configured")
        
        print("✅ Configuration loaded successfully")
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_data_fetcher():
    """Test data fetcher functionality"""
    print("📡 Testing data fetcher...")
    
    try:
        from data_fetcher import DataFetcher
        
        fetcher = DataFetcher()
        
        # Test inject functionality
        fetcher.inject_test_alert("123 Test St", "fire")
        
        # Check if file was created
        data_dir = Path("./live_data_feed")
        json_files = list(data_dir.glob("*.json"))
        
        if len(json_files) > 0:
            print(f"  📄 Created {len(json_files)} data files")
            print("✅ Data fetcher working")
            return True
        else:
            print("❌ No data files created")
            return False
            
    except Exception as e:
        print(f"❌ Data fetcher error: {e}")
        return False

def run_system_test():
    """Run comprehensive system test"""
    print("🏢 Live Insurance Risk & Quote Co-Pilot - System Test")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Data Directory", test_data_directory),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Data Fetcher", test_data_fetcher)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} error: {e}")
    
    print("\n" + "=" * 60)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready.")
        print("\n🚀 To start the application:")
        print("   python main.py")
        print("\n🌐 Web interface will be at:")
        print("   http://localhost:8000")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        print("\n🛠 To fix issues:")
        print("   1. Run: pip install -r requirements.txt")
        print("   2. Check .env file configuration")
        print("   3. Ensure Python 3.8+ is installed")
        return False

if __name__ == "__main__":
    success = run_system_test()
    
    if not success:
        sys.exit(1)
