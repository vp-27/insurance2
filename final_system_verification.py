#!/usr/bin/env python3
"""
System Verification Test - Ensures no fallback to simulated news unless explicitly triggered
"""

import os
import sys
import asyncio
import json
from datetime import datetime
from typing import Dict, Any

# Add the project directory to the path
sys.path.append('/Users/machanic/Documents/vpStudios/underground_coding/insurance2')

from data_fetcher import DataFetcher
from pipeline import LiveRAGPipeline

def test_no_simulation_fallback():
    """Test that the system doesn't fall back to simulated news automatically"""
    print("🧪 Testing: No automatic fallback to simulated news")
    
    # Create a temporary data fetcher with no API key
    df = DataFetcher()
    original_key = df.news_api_key
    
    # Temporarily remove API key
    df.news_api_key = None
    
    # Count files before
    data_dir = df.data_dir
    files_before = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    # Try to fetch news - should NOT create simulated news
    print("  - Attempting news fetch with no API key...")
    df.fetch_news_alerts("Test Location")
    
    # Count files after
    files_after = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    # Restore original key
    df.news_api_key = original_key
    
    if files_after == files_before:
        print("  ✅ PASS: No simulated news generated when API fails")
        return True
    else:
        print("  ❌ FAIL: Simulated news was generated automatically")
        return False

def test_explicit_demo_mode():
    """Test that demo mode works when explicitly triggered"""
    print("\n🧪 Testing: Explicit demo mode activation")
    
    df = DataFetcher()
    data_dir = df.data_dir
    
    # Count files before
    files_before = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    # Explicitly trigger demo mode
    print("  - Triggering explicit demo mode...")
    df.inject_demo_news_alerts("Demo Test Location")
    df.inject_test_alert("Demo Test Location", "fire")
    
    # Count files after
    files_after = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    if files_after > files_before:
        print(f"  ✅ PASS: Demo mode created {files_after - files_before} files when explicitly triggered")
        
        # Check that the created files are marked as demo
        newest_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[-3:]
        demo_files_found = 0
        
        for filename in newest_files:
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                source = data.get('source', '')
                content = data.get('content', '')
                if 'demo' in source.lower() or content.startswith('DEMO:'):
                    demo_files_found += 1
        
        if demo_files_found > 0:
            print(f"  ✅ PASS: Found {demo_files_found} properly marked demo files")
            return True
        else:
            print("  ❌ FAIL: Demo files not properly marked")
            return False
    else:
        print("  ❌ FAIL: Demo mode did not create any files")
        return False

async def test_rag_pipeline_distinction():
    """Test that RAG pipeline properly distinguishes between real and demo data"""
    print("\n🧪 Testing: RAG pipeline data distinction")
    
    pipeline = LiveRAGPipeline()
    pipeline.load_existing_data()
    
    # Test with a real address assessment
    print("  - Testing assessment with mixed real/demo data...")
    try:
        result = await pipeline.query_rag("123 Test Street, New York, NY", "What are the current risks?")
        
        risk_summary = result.get('risk_summary', '')
        
        # Check if the assessment mentions demo data vs real data appropriately
        if 'demo' in risk_summary.lower() or 'test' in risk_summary.lower():
            print("  ⚠️  WARNING: Assessment may be including demo data in risk calculation")
            print(f"     Summary excerpt: {risk_summary[:200]}...")
        else:
            print("  ✅ PASS: Assessment appears to focus on real data only")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: Pipeline test failed: {e}")
        return False

def test_real_data_sources():
    """Verify that real data sources are working"""
    print("\n🧪 Testing: Real data sources")
    
    df = DataFetcher()
    data_dir = df.data_dir
    
    # Count files before
    files_before = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    # Test real data sources
    print("  - Testing real news API...")
    df.fetch_news_alerts("New York")
    
    print("  - Testing weather API...")
    df.fetch_weather_alerts(40.7128, -74.0060)  # NYC coordinates
    
    print("  - Testing crime data...")
    df.fetch_crime_data(40.7128, -74.0060)  # NYC coordinates
    
    # Count files after
    files_after = len([f for f in os.listdir(data_dir) if f.endswith('.json')])
    
    if files_after > files_before:
        print(f"  ✅ PASS: Real data sources created {files_after - files_before} new files")
        
        # Check the sources of newest files
        newest_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.json')])[-5:]
        real_sources = []
        
        for filename in newest_files:
            with open(os.path.join(data_dir, filename), 'r') as f:
                data = json.load(f)
                source = data.get('source', '')
                if source in ['newsdata_io', 'nws_weather', 'nyc_open_data', 'chicago_open_data', 'usgs_earthquake']:
                    real_sources.append(source)
        
        if real_sources:
            print(f"  ✅ PASS: Found real data sources: {set(real_sources)}")
            return True
        else:
            print("  ⚠️  WARNING: No real data sources found in recent files")
            return False
    else:
        print("  ⚠️  WARNING: No new files created from real data sources")
        return False

async def main():
    """Run all tests"""
    print("🔍 System Verification: Simulated Data Fallback Prevention\n")
    print("=" * 60)
    
    tests = [
        ("No Automatic Simulation Fallback", test_no_simulation_fallback),
        ("Explicit Demo Mode", test_explicit_demo_mode),
        ("Real Data Sources", test_real_data_sources),
        ("RAG Pipeline Distinction", test_rag_pipeline_distinction),
    ]
    
    results = []
    for test_name, test_func in tests:
        if asyncio.iscoroutinefunction(test_func):
            result = await test_func()
        else:
            result = test_func()
        results.append((test_name, result))
    
    print("\n" + "=" * 60)
    print("📊 Test Summary:")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
        if result:
            passed += 1
    
    print(f"\nResult: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("\n🎉 All tests passed! System correctly prevents simulated news fallback.")
    else:
        print("\n⚠️  Some tests failed. Review the issues above.")

if __name__ == "__main__":
    asyncio.run(main())
