#!/usr/bin/env python3
"""
Enhanced Demo Script for Live Insurance Risk & Quote Co-Pilot
"""

import requests
import time
import json
from datetime import datetime

class InsuranceDemo:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def check_health(self):
        """Check if the system is healthy"""
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                data = response.json()
                print("✅ System Health Check:")
                print(f"  Status: {data['status']}")
                print(f"  RAG Pipeline: {'✅' if data['rag_pipeline'] else '❌'}")
                print(f"  Data Fetcher: {'✅' if data['data_fetcher'] else '❌'}")
                print(f"  Documents Indexed: {data['documents_indexed']}")
                return True
            return False
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def get_system_stats(self):
        """Get detailed system statistics"""
        try:
            response = self.session.get(f"{self.base_url}/stats")
            if response.status_code == 200:
                data = response.json()
                print("\n📊 System Statistics:")
                print(f"  Documents Indexed: {data['documents_indexed']}")
                print(f"  Vector Store Size: {data['vector_store_size']}")
                print(f"  Data Directory: {data['data_directory']}")
                print(f"  Embedding Model: {data['model_info']['embedding_model']}")
                print(f"  LLM Model: {data['model_info']['llm_model']}")
                return data
            return None
        except Exception as e:
            print(f"❌ Stats fetch failed: {e}")
            return None
    
    def get_risk_assessment(self, address, query="What are the current risks at this address?"):
        """Get risk assessment for an address"""
        try:
            payload = {
                "address": address,
                "query": query
            }
            response = self.session.post(f"{self.base_url}/get_assessment", json=payload)
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    data = result['data']
                    print(f"\n🏢 Risk Assessment for: {address}")
                    print(f"  Risk Score: {data['risk_score']}/10")
                    print(f"  Insurance Quote: ${data['insurance_quote']}/month")
                    print(f"  Documents Analyzed: {data['relevant_documents']}")
                    print(f"  Timestamp: {data['timestamp']}")
                    print("\n📋 Risk Summary:")
                    print(f"  {data['risk_summary'][:200]}...")
                    return data
                else:
                    print(f"❌ Assessment failed: {result}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
            return None
        except Exception as e:
            print(f"❌ Assessment request failed: {e}")
            return None
    
    def inject_test_alert(self, address, alert_type):
        """Inject a test alert"""
        try:
            payload = {
                "address": address,
                "alert_type": alert_type
            }
            response = self.session.post(f"{self.base_url}/inject_test_alert", json=payload)
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    print(f"✅ {alert_type.upper()} alert injected for {address}")
                    return True
                else:
                    print(f"❌ Alert injection failed: {result}")
            else:
                print(f"❌ HTTP Error: {response.status_code}")
            return False
        except Exception as e:
            print(f"❌ Alert injection failed: {e}")
            return False
    
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration"""
        print("🚀 Starting Live Insurance Risk & Quote Co-Pilot Demo")
        print("=" * 60)
        
        # Health check
        if not self.check_health():
            print("❌ System not healthy, exiting demo")
            return
        
        # Get initial stats
        self.get_system_stats()
        
        # Test addresses
        test_addresses = [
            "25 Columbus Dr, Jersey City, NJ",
            "1600 Pennsylvania Avenue, Washington, DC",
            "350 Fifth Avenue, New York, NY"
        ]
        
        # Demo scenarios
        scenarios = [
            ("fire", "🔥 Fire Emergency Scenario"),
            ("flood", "🌊 Flood Warning Scenario"),
            ("crime", "🚨 Security Incident Scenario"),
            ("earthquake", "🏗️ Seismic Activity Scenario")
        ]
        
        for address in test_addresses:
            print(f"\n{'='*60}")
            print(f"🏢 Testing Address: {address}")
            print('='*60)
            
            # Get baseline assessment
            print("\n1️⃣ Baseline Risk Assessment:")
            baseline = self.get_risk_assessment(address)
            
            if not baseline:
                continue
            
            # Test each scenario
            for alert_type, scenario_name in scenarios:
                print(f"\n2️⃣ {scenario_name}")
                print("-" * 40)
                
                # Inject alert
                if self.inject_test_alert(address, alert_type):
                    print("⏳ Waiting 3 seconds for data processing...")
                    time.sleep(3)
                    
                    # Get updated assessment
                    updated = self.get_risk_assessment(address, f"What is the impact of the recent {alert_type} incident?")
                    
                    if updated and baseline:
                        # Compare results
                        print(f"\n📈 Impact Analysis:")
                        risk_change = updated['risk_score'] - baseline['risk_score']
                        quote_change = updated['insurance_quote'] - baseline['insurance_quote']
                        
                        print(f"  Risk Score Change: {risk_change:+.1f} points")
                        print(f"  Quote Change: ${quote_change:+.2f}/month")
                        print(f"  Percentage Increase: {(quote_change/baseline['insurance_quote']*100):+.1f}%")
                    
                    print("\n⏳ Waiting 2 seconds before next scenario...")
                    time.sleep(2)
                
                break  # Only test one scenario per address for demo brevity
        
        print(f"\n{'='*60}")
        print("✅ Demo completed successfully!")
        print("\n🎯 Key Features Demonstrated:")
        print("  ✅ Real-time data processing with Pathway")
        print("  ✅ Dynamic risk scoring (1-10 scale)")
        print("  ✅ Live insurance quote calculations")
        print("  ✅ Multiple risk factor analysis")
        print("  ✅ AI-powered risk assessment")
        print("  ✅ Vector-based document retrieval")
        print("  ✅ Streaming data ingestion")
        print("  ✅ Modern React-based UI")
        
        # Final stats
        print("\n📊 Final System Statistics:")
        self.get_system_stats()

def main():
    demo = InsuranceDemo()
    
    print("🌟 Welcome to the Live Insurance Risk & Quote Co-Pilot Demo!")
    print("\nThis system demonstrates:")
    print("• Real-time risk assessment using live data streams")
    print("• Dynamic insurance pricing based on current conditions")
    print("• AI-powered analysis with Pathway streaming engine")
    print("• Modern web interface with live updates")
    
    choice = input("\nRun comprehensive demo? (y/n): ").lower().strip()
    
    if choice == 'y':
        demo.run_comprehensive_demo()
    else:
        print("\n📱 Manual testing available at: http://localhost:8000")
        print("Use the web interface to:")
        print("1. Enter a property address")
        print("2. Get live risk assessment")
        print("3. Inject test alerts")
        print("4. See real-time quote updates")

if __name__ == "__main__":
    main()
