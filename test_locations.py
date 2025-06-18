#!/usr/bin/env python3
"""
Test the enhanced location-aware risk assessment
"""

import requests
import json

def test_location_differences():
    """Test different addresses to see location-specific variations"""
    
    # Test different addresses with enhanced location analysis
    addresses = [
        "77 Poplar Rd, Glenburn, ME 04401",  # Rural Maine
        "2401 Forest Ave, Austin, TX 78704",  # Urban Texas
        "25 Columbus Dr, Jersey City, NJ",    # Urban area near NYC
        "1600 Pennsylvania Avenue, Washington, DC",  # DC
        "Tornado Alley, Moore, OK 73160",     # Oklahoma tornado area
        "123 Sunset Blvd, Los Angeles, CA"   # California
    ]

    for address in addresses:
        print(f"\n{'='*60}")
        print(f"🏢 Testing: {address}")
        print('='*60)
        
        try:
            response = requests.post("http://localhost:8000/get_assessment", 
                                   json={"address": address, "query": "What are the current risks?"})
            
            if response.status_code == 200:
                data = response.json()["data"]
                
                print(f"📊 Risk Score: {data['risk_score']}/10")
                print(f"💰 Quote: ${data['insurance_quote']}/month")
                
                if "location_factors" in data:
                    loc = data["location_factors"]
                    print(f"📍 Location Type: {loc['location_description']}")
                    print(f"🎯 Base Risk Level: {loc['base_risk_score']:.1f}/10")
                    
                    if loc.get('latitude') and loc.get('longitude'):
                        print(f"🌍 Coordinates: {loc['latitude']:.2f}, {loc['longitude']:.2f}")
                    
                    if loc['primary_risks']:
                        print(f"⚠️  Regional Risks: {', '.join(loc['primary_risks'])}")
                    
                    if loc['risk_multipliers']:
                        multipliers = []
                        for risk_type, multiplier in loc['risk_multipliers'].items():
                            multipliers.append(f"{risk_type} ({multiplier:.1f}x)")
                        print(f"📈 Risk Multipliers: {', '.join(multipliers)}")
                
                print(f"\n📝 Risk Analysis Preview:")
                print(f"   {data['risk_summary'][:150]}...")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"❌ Exception: {e}")

if __name__ == "__main__":
    print("🧪 Testing Enhanced Location-Aware Risk Assessment")
    print("=" * 60)
    test_location_differences()
    print(f"\n{'='*60}")
    print("✅ Location testing complete!")
    print("\n📋 Summary:")
    print("• Different addresses should now show different base risk scores")
    print("• Rural areas should have lower base risks")
    print("• Urban areas should have higher base risks")
    print("• Regional risks like earthquakes, hurricanes, tornadoes should appear")
    print("• Insurance quotes should vary based on location factors")
