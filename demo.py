#!/usr/bin/env python3
"""
Demo script for Live Insurance Risk & Quote Co-Pilot
This script demonstrates the key features of the system
"""

import asyncio
import time
import requests
import json
from data_fetcher import DataFetcher
from pipeline import LiveRAGPipeline

class LiveDemo:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.demo_address = "25 Columbus Dr, Jersey City, NJ"
        
    def print_banner(self, text):
        print("\n" + "="*60)
        print(f"🎯 {text}")
        print("="*60)
        
    def print_step(self, step, description):
        print(f"\n{step}. {description}")
        print("-" * 40)
        
    async def demo_basic_assessment(self):
        """Demonstrate basic risk assessment"""
        self.print_step(1, "Basic Risk Assessment")
        
        payload = {
            "address": self.demo_address,
            "query": "What are the current risks at this address?"
        }
        
        try:
            response = requests.post(f"{self.base_url}/get_assessment", json=payload)
            if response.status_code == 200:
                data = response.json()["data"]
                print(f"📍 Address: {self.demo_address}")
                print(f"⚡ Risk Score: {data['risk_score']}/10")
                print(f"💰 Insurance Quote: ${data['insurance_quote']}/month")
                print(f"📄 Based on: {data['relevant_documents']} data points")
                return data
            else:
                print(f"❌ Error: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None
    
    async def demo_alert_injection(self, alert_type="fire"):
        """Demonstrate alert injection and live updates"""
        self.print_step(2, f"Injecting {alert_type.title()} Alert")
        
        payload = {
            "address": self.demo_address,
            "alert_type": alert_type
        }
        
        try:
            response = requests.post(f"{self.base_url}/inject_test_alert", json=payload)
            if response.status_code == 200:
                print(f"🔥 {alert_type.title()} alert injected!")
                print("⏳ Waiting for system to process...")
                
                # Wait for processing
                await asyncio.sleep(3)
                
                # Get updated assessment
                return await self.demo_basic_assessment()
            else:
                print(f"❌ Error injecting alert: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ Connection error: {e}")
            return None
    
    def compare_assessments(self, before, after, alert_type):
        """Compare before and after assessments"""
        self.print_step(3, "Impact Analysis")
        
        if not before or not after:
            print("❌ Cannot compare - missing data")
            return
        
        print(f"📊 Impact of {alert_type} alert:")
        print(f"   Risk Score: {before['risk_score']} → {after['risk_score']} ({after['risk_score'] - before['risk_score']:+d})")
        print(f"   Insurance Quote: ${before['insurance_quote']} → ${after['insurance_quote']} (${after['insurance_quote'] - before['insurance_quote']:+.0f})")
        
        if after['risk_score'] > before['risk_score']:
            print("📈 Risk increased due to emergency situation")
        else:
            print("📉 Risk level unchanged")
    
    async def demo_system_stats(self):
        """Show system statistics"""
        self.print_step(4, "System Statistics")
        
        try:
            response = requests.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                stats = response.json()
                print(f"📚 Documents Indexed: {stats['documents_indexed']}")
                print(f"🔍 Vector Store Size: {stats['vector_store_size']}")
                print(f"🤖 Embedding Model: {stats['model_info']['embedding_model']}")
                print(f"🧠 LLM Model: {stats['model_info']['llm_model']}")
        except Exception as e:
            print(f"❌ Error getting stats: {e}")
    
    def wait_for_server(self, max_attempts=30):
        """Wait for server to be ready"""
        print("⏳ Waiting for server to start...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=2)
                if response.status_code == 200:
                    print("✅ Server is ready!")
                    return True
            except:
                pass
            
            time.sleep(1)
            if attempt % 5 == 0:
                print(f"   Still waiting... ({attempt}/{max_attempts})")
        
        print("❌ Server failed to start within timeout")
        return False
    
    async def run_full_demo(self):
        """Run the complete demo"""
        self.print_banner("Live Insurance Risk & Quote Co-Pilot Demo")
        
        print("This demo will show how the system responds to live risk events.")
        print(f"Using demo address: {self.demo_address}")
        
        # Wait for server
        if not self.wait_for_server():
            print("\n❌ Cannot run demo - server not available")
            print("Please start the server first: python app.py")
            return
        
        # Get baseline assessment
        print("\n🔄 Getting baseline risk assessment...")
        baseline = await self.demo_basic_assessment()
        
        if not baseline:
            print("❌ Cannot get baseline assessment")
            return
        
        # Inject fire alert and compare
        print("\n🔄 Simulating emergency situation...")
        updated = await self.demo_alert_injection("fire")
        
        # Compare results
        self.compare_assessments(baseline, updated, "fire")
        
        # Show system stats
        await self.demo_system_stats()
        
        self.print_banner("Demo Complete!")
        print("🎉 The system successfully demonstrated:")
        print("   ✅ Real-time risk assessment")
        print("   ✅ Live data processing")
        print("   ✅ Dynamic pricing updates")
        print("   ✅ Emergency response simulation")
        print("\n🌐 Visit http://localhost:8000 for the web interface")

if __name__ == "__main__":
    demo = LiveDemo()
    asyncio.run(demo.run_full_demo())
