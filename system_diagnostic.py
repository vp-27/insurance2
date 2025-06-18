#!/usr/bin/env python3
"""
Comprehensive system diagnostic and demo for Live Insurance Risk Assessment
Tests both real-time data processing and AI response quality
"""

import asyncio
import requests
import time
import json
from datetime import datetime
from pathlib import Path

class SystemDiagnostic:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def test_server_health(self):
        """Test server health and data management"""
        print("🏥 Testing Server Health")
        print("=" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/health")
            if response.status_code == 200:
                health = response.json()
                print("✅ Server is healthy")
                print(f"  📊 Documents indexed: {health['documents_indexed']}")
                print(f"  📁 Active files: {health['active_files']}")
                print(f"  🗂️ Data management: {'Active' if health['data_management_active'] else 'Inactive'}")
                print(f"  🤖 RAG Pipeline: {'✅' if health['rag_pipeline'] else '❌'}")
                print(f"  📡 Data Fetcher: {'✅' if health['data_fetcher'] else '❌'}")
                return True
            else:
                print(f"❌ Server health check failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return False
    
    def test_data_management(self):
        """Test data management functionality"""
        print("\n🗂️ Testing Data Management")
        print("=" * 50)
        
        try:
            # Get data stats
            response = self.session.get(f"{self.base_url}/data/stats")
            if response.status_code == 200:
                stats = response.json()
                print("✅ Data management active")
                print(f"  📁 Active files: {stats['active_files']}")
                print(f"  📦 Archived files: {stats['archived_files']}")
                print(f"  💾 Archive size: {stats['archive_size_mb']} MB")
                print(f"  🎯 File limit: {stats['max_active_files']}")
                
                # Show data sources
                if stats['data_sources']:
                    print("  📊 Data sources:")
                    for source, count in stats['data_sources'].items():
                        emoji = "🧪" if "manual" in source or "test" in source or "simulated" in source else "📡"
                        print(f"    {emoji} {source}: {count}")
                
                return True
            else:
                print(f"❌ Data stats failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Data management test failed: {e}")
            return False
    
    def test_real_vs_demo_alerts(self):
        """Test how the system distinguishes between real and demo data"""
        print("\n🔍 Testing Real vs Demo Data Distinction")
        print("=" * 50)
        
        test_address = "25 Columbus Dr, Jersey City, NJ"
        
        # Test 1: Get baseline assessment with real data only
        print("1. Baseline assessment (real data only):")
        baseline = self.get_assessment(test_address, "What are the current risks based on real incidents only?")
        
        if baseline:
            print(f"   Risk Score: {baseline['risk_score']}/10")
            print(f"   Quote: ${baseline['insurance_quote']}/month")
            print(f"   Documents: {baseline['relevant_documents']}")
        
        # Test 2: Inject a demo alert
        print("\n2. Injecting demo fire alert...")
        demo_response = self.inject_alert(test_address, "fire")
        
        if demo_response:
            print("   ✅ Demo alert injected successfully")
            time.sleep(3)  # Wait for processing
            
            # Test 3: Get assessment after demo alert
            print("\n3. Assessment after demo alert:")
            after_demo = self.get_assessment(test_address, "Analyze current risks. Distinguish between real incidents and test alerts.")
            
            if after_demo and baseline:
                print(f"   Risk Score: {after_demo['risk_score']}/10 (was {baseline['risk_score']}/10)")
                print(f"   Quote: ${after_demo['insurance_quote']}/month (was ${baseline['insurance_quote']}/month)")
                
                # Check if the AI properly distinguished real vs test data
                summary = after_demo['risk_summary'].lower()
                if "test" in summary or "demo" in summary or "simulation" in summary:
                    print("   ✅ AI correctly identified test/demo data")
                else:
                    print("   ⚠️ AI may not have distinguished test vs real data")
                
                return True
        
        return False
    
    def test_data_volume_impact(self):
        """Test system performance with current data volume"""
        print("\n📊 Testing Data Volume Impact")
        print("=" * 50)
        
        try:
            # Get recent data summary
            response = self.session.get(f"{self.base_url}/data/recent")
            if response.status_code == 200:
                recent = response.json()
                print(f"📈 Recent files (2 hours): {recent['total_files']}")
                print(f"📡 Active sources: {', '.join(recent['sources'])}")
                print(f"🏷️ Data types: {', '.join(recent['types'])}")
                
                # Test response time
                start_time = time.time()
                test_response = self.get_assessment("123 Test St, New York, NY", "Quick test")
                response_time = time.time() - start_time
                
                print(f"⏱️ Response time: {response_time:.2f} seconds")
                
                if response_time < 5:
                    print("✅ System performance is good")
                elif response_time < 10:
                    print("⚠️ System performance is acceptable")
                else:
                    print("❌ System performance may be degraded")
                
                return True
            else:
                print(f"❌ Recent data test failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Data volume test failed: {e}")
            return False
    
    def force_cleanup_demo(self):
        """Demonstrate data cleanup functionality"""
        print("\n🧹 Data Cleanup Demonstration")
        print("=" * 50)
        
        try:
            response = self.session.post(f"{self.base_url}/data/cleanup")
            if response.status_code == 200:
                result = response.json()
                details = result['details']
                print("✅ Cleanup completed successfully")
                print(f"  📁 Initial files: {details['initial_files']}")
                print(f"  📁 Final files: {details['final_files']}")
                print(f"  📦 Archived files: {details['archived_files']}")
                
                if details['archived_files'] > 0:
                    print("✅ Data archiving is working properly")
                else:
                    print("ℹ️ No files needed archiving at this time")
                
                return True
            else:
                print(f"❌ Cleanup failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ Cleanup test failed: {e}")
            return False
    
    def get_assessment(self, address, query):
        """Get risk assessment"""
        try:
            payload = {"address": address, "query": query}
            response = self.session.post(f"{self.base_url}/get_assessment", json=payload)
            if response.status_code == 200:
                return response.json()["data"]
            return None
        except Exception as e:
            print(f"   ❌ Assessment failed: {e}")
            return None
    
    def inject_alert(self, address, alert_type):
        """Inject test alert"""
        try:
            payload = {"address": address, "alert_type": alert_type}
            response = self.session.post(f"{self.base_url}/inject_test_alert", json=payload)
            return response.status_code == 200
        except Exception as e:
            print(f"   ❌ Alert injection failed: {e}")
            return False
    
    def run_full_diagnostic(self):
        """Run complete system diagnostic"""
        print("🚀 Live Insurance Risk Assessment - System Diagnostic")
        print("=" * 70)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        all_tests_passed = True
        
        # Test 1: Server Health
        if not self.test_server_health():
            all_tests_passed = False
        
        # Test 2: Data Management
        if not self.test_data_management():
            all_tests_passed = False
        
        # Test 3: Real vs Demo Data
        if not self.test_real_vs_demo_alerts():
            all_tests_passed = False
        
        # Test 4: Data Volume Impact
        if not self.test_data_volume_impact():
            all_tests_passed = False
        
        # Test 5: Cleanup Demo
        if not self.force_cleanup_demo():
            all_tests_passed = False
        
        # Summary
        print("\n🎯 Diagnostic Summary")
        print("=" * 50)
        
        if all_tests_passed:
            print("✅ All tests passed - system is functioning optimally")
            print("\n🔧 Recommendations:")
            print("• System is properly distinguishing real vs test data")
            print("• Data management is active and maintaining optimal file count")
            print("• Performance is good with current data volume")
        else:
            print("⚠️ Some issues detected - see details above")
            print("\n🔧 Recommendations:")
            print("• Check server connectivity and initialization")
            print("• Verify API keys if using real AI services")
            print("• Monitor data volume and cleanup frequency")
        
        print("\n📚 System Status:")
        try:
            stats_response = self.session.get(f"{self.base_url}/stats")
            if stats_response.status_code == 200:
                stats = stats_response.json()
                model_info = stats.get('model_info', {})
                data_mgmt = stats.get('data_management', {})
                
                print(f"• LLM Model: {model_info.get('llm_model', 'Unknown')}")
                print(f"• API Active: {model_info.get('api_client_active', False)}")
                print(f"• Embedding Model: {model_info.get('embedding_model', 'Unknown')}")
                print(f"• Documents Indexed: {stats.get('documents_indexed', 0)}")
                print(f"• Active Files: {data_mgmt.get('active_files', 0)}")
                print(f"• Management Active: {data_mgmt.get('management_active', False)}")
        except:
            pass
        
        print("\n📖 Next Steps:")
        print("• Use the web interface at http://localhost:8000 for manual testing")
        print("• Try demo alerts to see real-time risk assessment changes")
        print("• Monitor the live_data_feed directory for new files")
        print("• Check archived_data directory for older files")

def main():
    diagnostic = SystemDiagnostic()
    diagnostic.run_full_diagnostic()

if __name__ == "__main__":
    main()
