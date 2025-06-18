#!/usr/bin/env python3
"""
Test OpenRouter with direct HTTP request
"""

import os
import requests
from dotenv import load_dotenv

def test_openrouter_http():
    """Test OpenRouter API with direct HTTP request"""
    # Force reload of environment
    load_dotenv(override=True)
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("MODEL_NAME", "google/gemma-3n-e4b-it:free")
    
    print(f"API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    if not api_key:
        print("❌ No API key found")
        return
    
    # Test with direct HTTP request
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost:8000",
        "X-Title": "Insurance Risk Analysis"
    }
    
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "What is 2+2? Answer in one sentence."}
        ],
        "max_tokens": 50
    }
    
    try:
        print("\n🧪 Testing HTTP request...")
        response = requests.post(f"{base_url}/chat/completions", headers=headers, json=data)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Success: {result['choices'][0]['message']['content']}")
        else:
            print(f"❌ Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Request failed: {e}")

if __name__ == "__main__":
    test_openrouter_http()
