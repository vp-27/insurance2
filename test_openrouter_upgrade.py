#!/usr/bin/env python3
"""
Test script to verify the upgraded OpenRouter integration
"""

import asyncio
import os
from dotenv import load_dotenv
from pipeline import LiveRAGPipeline

async def test_openrouter_integration():
    """Test the upgraded OpenRouter integration"""
    
    # Load environment variables
    load_dotenv()
    
    print("🔧 Testing OpenRouter Upgrade")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        print("1. Initializing pipeline...")
        pipeline = LiveRAGPipeline()
        
        print(f"   ✓ Model: {pipeline.model_name}")
        print(f"   ✓ Base URL: {pipeline.openrouter_base_url}")
        print(f"   ✓ Client initialized: {pipeline.openai_client is not None}")
        
        # Test LLM call
        print("\n2. Testing LLM call...")
        test_prompt = "What are the main risk factors for home insurance?"
        
        response = await pipeline.call_llm(test_prompt)
        
        print(f"   ✓ Response received ({len(response)} characters)")
        print(f"   Preview: {response[:100]}...")
        
        print("\n✅ OpenRouter upgrade successful!")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_openrouter_integration())
    if not success:
        exit(1)
