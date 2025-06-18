#!/usr/bin/env python3
"""
Focused test to reproduce the 401 OpenRouter API error
"""

import os
import sys
import asyncio
from dotenv import load_dotenv

# Force reload environment variables
load_dotenv(override=True)

# Add current directory to path
sys.path.append('.')

from pipeline import LiveRAGPipeline

async def test_specific_scenario():
    """Test the specific scenario that's causing the 401 error"""
    
    print("🔍 Debugging OpenRouter API 401 Error")
    print("=" * 50)
    
    # Initialize pipeline
    print("1. Initializing pipeline...")
    pipeline = LiveRAGPipeline()
    
    print(f"   API Key: {pipeline.openrouter_api_key[:10]}...{pipeline.openrouter_api_key[-10:] if pipeline.openrouter_api_key else 'None'}")
    print(f"   Base URL: {pipeline.openrouter_base_url}")
    print(f"   Model: {pipeline.model_name}")
    print(f"   Client initialized: {pipeline.openai_client is not None}")
    
    # Test direct LLM call
    print("\n2. Testing direct LLM call...")
    test_prompt = "What is 2 + 2?"
    
    try:
        response = await pipeline.call_llm(test_prompt)
        print(f"   ✅ Success: {response[:100]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False
    
    # Test insurance analysis for the specific address
    print("\n3. Testing insurance analysis...")
    address = "25 Columbus Dr, Jersey City, NJ"
    
    try:
        result = await pipeline.query_rag(address, "What are the current risks at this address?")
        print(f"   ✅ Analysis complete")
        print(f"   Risk Score: {result.get('risk_score', 'N/A')}")
        print(f"   Premium: ${result.get('insurance_quote', 'N/A')}")
        print(f"   AI Response length: {len(result.get('risk_summary', ''))}")
        
        # Check for consistency
        risk_score = result.get('risk_score', 0)
        premium = result.get('insurance_quote', 0)
        ai_response = result.get('risk_summary', '')
        
        print("\n4. Checking consistency...")
        if risk_score and premium and ai_response:
            # Basic consistency check
            if '401' in ai_response or 'No auth credentials found' in ai_response:
                print("   ❌ Found 401 error in AI response!")
                print(f"   Full response: {ai_response}")
                return False
            else:
                print("   ✅ No 401 errors found")
                
                # Check if risk score and premium are logically consistent
                expected_premium = 500 * (1 + 0.1 * risk_score)
                if abs(premium - expected_premium) < 10:  # Allow small floating point differences
                    print(f"   ✅ Risk score ({risk_score}) and premium (${premium:.2f}) are consistent")
                else:
                    print(f"   ⚠️ Risk score ({risk_score}) and premium (${premium:.2f}) might be inconsistent")
                    print(f"   Expected premium: ${expected_premium:.2f}")
        
    except Exception as e:
        print(f"   ❌ Analysis error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_specific_scenario())
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Tests failed!")
