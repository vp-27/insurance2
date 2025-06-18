#!/usr/bin/env python3
"""
Debug script to test OpenRouter API directly
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

def test_openrouter_direct():
    """Test OpenRouter API directly"""
    load_dotenv()
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("MODEL_NAME", "google/gemma-3-4b-it:free")
    
    print(f"API Key: {api_key[:10]}...{api_key[-10:] if api_key else 'None'}")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    if not api_key or api_key == "your_openrouter_api_key_here":
        print("❌ No valid API key found")
        return
    
    try:
        client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        
        print("\n🧪 Testing API call...")
        
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=50
        )
        
        print("✅ Success!")
        print(f"Response: {completion.choices[0].message.content}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    test_openrouter_direct()
