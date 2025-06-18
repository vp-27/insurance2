#!/usr/bin/env python3
"""
Environment diagnostic script to check API key loading
"""

import os
from dotenv import load_dotenv

def diagnose_environment():
    """Diagnose environment variable loading"""
    print("🔍 Environment Variable Diagnostic")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = ".env"
    if os.path.exists(env_file):
        print(f"✅ .env file exists: {env_file}")
        
        # Read .env file content
        with open(env_file, 'r') as f:
            content = f.read()
            print(f"📄 .env file content ({len(content)} chars):")
            lines = content.strip().split('\n')
            for line in lines[:10]:  # Show first 10 lines
                if 'KEY' in line and '=' in line:
                    key, value = line.split('=', 1)
                    if len(value) > 10:
                        print(f"   {key}={value[:10]}...{value[-10:]}")
                    else:
                        print(f"   {key}={value}")
                else:
                    print(f"   {line}")
    else:
        print(f"❌ .env file not found: {env_file}")
    
    print("\n🔧 Loading environment variables...")
    
    # Load environment variables
    load_dotenv(override=True)
    
    # Check specific variables
    variables = [
        'OPENROUTER_API_KEY',
        'OPENROUTER_BASE_URL', 
        'MODEL_NAME',
        'NEWS_API_KEY'
    ]
    
    print("\n📊 Environment Variables Status:")
    for var in variables:
        value = os.getenv(var)
        if value:
            if 'KEY' in var and len(value) > 10:
                print(f"   ✅ {var}: {value[:10]}...{value[-10:]}")
            else:
                print(f"   ✅ {var}: {value}")
        else:
            print(f"   ❌ {var}: Not set")
    
    # Test import of pipeline module
    print("\n🧪 Testing Pipeline Import...")
    try:
        from pipeline import LiveRAGPipeline
        pipeline = LiveRAGPipeline()
        
        print(f"   ✅ Pipeline initialized")
        print(f"   API Key: {pipeline.openrouter_api_key[:10] if pipeline.openrouter_api_key else 'None'}...")
        print(f"   Base URL: {pipeline.openrouter_base_url}")
        print(f"   Model: {pipeline.model_name}")
        print(f"   Client available: {pipeline.openai_client is not None}")
        
    except Exception as e:
        print(f"   ❌ Pipeline import error: {e}")

if __name__ == "__main__":
    diagnose_environment()
