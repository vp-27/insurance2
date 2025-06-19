"""
Location-specific risk analysis for insurance underwriting
"""

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
from typing import Dict, Any
import re


class LocationAnalyzer:
    """Analyze location-specific risk factors"""
    
    def __init__(self):
        self.geolocator = Nominatim(user_agent="insurance_risk_analyzer")
        
    def get_coordinates(self, address: str) -> tuple:
        """Get latitude and longitude for an address with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Increase timeout and add retry logic
                location = self.geolocator.geocode(address, timeout=15)
                if location:
                    print(f"✅ Geocoding successful for {address}: {location.latitude}, {location.longitude}")
                    return location.latitude, location.longitude
                else:
                    print(f"⚠️ No geocoding result for {address} (attempt {attempt + 1})")
            except GeocoderTimedOut:
                print(f"⚠️ Geocoding timeout for {address} (attempt {attempt + 1})")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)  # Wait before retry
                continue
            except Exception as e:
                print(f"❌ Geocoding error for {address}: {e}")
                break
        
        print(f"❌ Failed to geocode {address} after {max_retries} attempts, using fallback")
        return None, None
    
    def analyze_location_risk_factors(self, address: str) -> Dict[str, Any]:
        """Analyze location-specific risk factors"""
        lat, lon = self.get_coordinates(address)
        
        if lat is None or lon is None:
            return self.get_default_risk_factors(address)
        
        risk_factors = {
            'latitude': lat,
            'longitude': lon,
            'base_risk_score': 2,
            'risk_multipliers': {},
            'location_description': '',
            'primary_risks': []
        }
        
        # Coastal risk analysis
        if self.is_coastal_area(lat, lon):
            risk_factors['risk_multipliers']['coastal'] = 1.3
            risk_factors['primary_risks'].append('hurricane/storm surge')
            risk_factors['base_risk_score'] += 0.5
        
        # Seismic risk zones
        seismic_risk = self.get_seismic_risk(lat, lon)
        if seismic_risk > 1.2:
            risk_factors['risk_multipliers']['seismic'] = seismic_risk
            risk_factors['primary_risks'].append('earthquake activity')
            risk_factors['base_risk_score'] += (seismic_risk - 1) * 2
        
        # Climate zone risks
        climate_risk = self.get_climate_risks(lat, lon)
        risk_factors['risk_multipliers'].update(climate_risk['multipliers'])
        risk_factors['primary_risks'].extend(climate_risk['risks'])
        risk_factors['base_risk_score'] += climate_risk['base_increase']
        
        # Urban vs Rural
        urban_factor = self.get_urban_rural_factor(address, lat, lon)
        risk_factors['risk_multipliers']['urban_density'] = urban_factor['multiplier']
        risk_factors['primary_risks'].extend(urban_factor['risks'])
        risk_factors['base_risk_score'] += urban_factor['base_increase']
        risk_factors['location_description'] = urban_factor['description']
        
        # State-specific risks
        state_risks = self.get_state_specific_risks(address, lat, lon)
        risk_factors['risk_multipliers'].update(state_risks['multipliers'])
        risk_factors['primary_risks'].extend(state_risks['risks'])
        risk_factors['base_risk_score'] += state_risks['base_increase']
        
        # Cap the base risk score
        risk_factors['base_risk_score'] = min(max(risk_factors['base_risk_score'], 1), 6)
        
        return risk_factors
    
    def is_coastal_area(self, lat: float, lon: float) -> bool:
        """Determine if location is in coastal area"""
        # US coastal boundaries (approximate)
        east_coast = lon > -85 and lat > 25 and lat < 45
        west_coast = lon < -115 and lat > 32 and lat < 49
        gulf_coast = lat > 25 and lat < 32 and lon > -98 and lon < -80
        return east_coast or west_coast or gulf_coast
    
    def get_seismic_risk(self, lat: float, lon: float) -> float:
        """Get seismic risk factor based on location"""
        # California (high seismic activity)
        if -125 < lon < -114 and 32 < lat < 42:
            return 2.0
        # Pacific Northwest
        elif -125 < lon < -116 and 42 < lat < 49:
            return 1.8
        # New Madrid Seismic Zone (Missouri, Arkansas)
        elif -92 < lon < -88 and 35 < lat < 38:
            return 1.6
        # Alaska
        elif lon < -130:
            return 2.5
        # Eastern US (generally low)
        elif lon > -85:
            return 1.1
        else:
            return 1.2
    
    def get_climate_risks(self, lat: float, lon: float) -> Dict[str, Any]:
        """Analyze climate-related risks"""
        risks = []
        multipliers = {}
        base_increase = 0
        
        # Tornado Alley
        if -105 < lon < -93 and 31 < lat < 41:
            multipliers['tornado'] = 1.5
            risks.append('tornado activity')
            base_increase += 0.8
        
        # Hurricane zones (Atlantic/Gulf Coast)
        if (lon > -85 and lat > 25 and lat < 35) or (lat > 25 and lat < 32 and lon > -98 and lon < -80):
            multipliers['hurricane'] = 1.6
            risks.append('hurricane risk')
            base_increase += 1.0
        
        # Wildfire zones (Western US)
        if lon < -100 and lat > 32:
            multipliers['wildfire'] = 1.4
            risks.append('wildfire risk')
            base_increase += 0.5
        
        # Extreme cold (Northern states)
        if lat > 45:
            multipliers['extreme_weather'] = 1.2
            risks.append('extreme cold/snow')
            base_increase += 0.2
        
        # Desert/extreme heat
        if 32 < lat < 37 and -117 < lon < -109:
            multipliers['extreme_heat'] = 1.3
            risks.append('extreme heat')
            base_increase += 0.3
        
        return {
            'multipliers': multipliers,
            'risks': risks,
            'base_increase': base_increase
        }
    
    def get_urban_rural_factor(self, address: str, lat: float, lon: float) -> Dict[str, Any]:
        """Determine urban vs rural risk factors"""
        address_lower = address.lower()
        
        # Major metropolitan areas
        metro_areas = {
            'new york': {'multiplier': 1.3, 'base_increase': 0.8, 'desc': 'major metropolitan area'},
            'los angeles': {'multiplier': 1.25, 'base_increase': 0.6, 'desc': 'major metropolitan area'},
            'chicago': {'multiplier': 1.2, 'base_increase': 0.5, 'desc': 'major metropolitan area'},
            'houston': {'multiplier': 1.15, 'base_increase': 0.4, 'desc': 'major metropolitan area'},
            'phoenix': {'multiplier': 1.1, 'base_increase': 0.3, 'desc': 'major metropolitan area'},
            'philadelphia': {'multiplier': 1.15, 'base_increase': 0.4, 'desc': 'major metropolitan area'},
            'san antonio': {'multiplier': 1.05, 'base_increase': 0.2, 'desc': 'large urban area'},
            'san diego': {'multiplier': 1.1, 'base_increase': 0.3, 'desc': 'large urban area'},
            'dallas': {'multiplier': 1.15, 'base_increase': 0.4, 'desc': 'major metropolitan area'},
            'austin': {'multiplier': 1.1, 'base_increase': 0.3, 'desc': 'growing urban area'},
            'washington': {'multiplier': 1.2, 'base_increase': 0.5, 'desc': 'major metropolitan area'},
            'boston': {'multiplier': 1.15, 'base_increase': 0.4, 'desc': 'major metropolitan area'},
            'seattle': {'multiplier': 1.15, 'base_increase': 0.4, 'desc': 'major metropolitan area'},
            'miami': {'multiplier': 1.25, 'base_increase': 0.6, 'desc': 'major metropolitan area'},
            'atlanta': {'multiplier': 1.15, 'base_increase': 0.4, 'desc': 'major metropolitan area'},
            'jersey city': {'multiplier': 1.2, 'base_increase': 0.5, 'desc': 'urban area near NYC'},
        }
        
        for city, factors in metro_areas.items():
            if city in address_lower:
                return {
                    'multiplier': factors['multiplier'],
                    'base_increase': factors['base_increase'],
                    'description': factors['desc'],
                    'risks': ['urban crime potential', 'traffic density', 'infrastructure strain']
                }
        
        # Rural area detection
        rural_indicators = ['rd', 'route', 'county road', 'rural', 'farm', 'ranch']
        if any(indicator in address_lower for indicator in rural_indicators):
            return {
                'multiplier': 0.85,
                'base_increase': -0.5,
                'description': 'rural area',
                'risks': ['emergency response delays', 'infrastructure limitations']
            }
        
        # Default suburban
        return {
            'multiplier': 1.0,
            'base_increase': 0,
            'description': 'suburban area',
            'risks': ['standard suburban risks']
        }
    
    def get_state_specific_risks(self, address: str, lat: float, lon: float) -> Dict[str, Any]:
        """Get state-specific risk factors"""
        state_risks = {
            'california': {
                'multipliers': {'wildfire': 1.6, 'earthquake': 1.8},
                'risks': ['wildfire', 'earthquake', 'mudslides'],
                'base_increase': 1.2
            },
            'florida': {
                'multipliers': {'hurricane': 1.8, 'flood': 1.5},
                'risks': ['hurricane', 'flooding', 'sinkholes'],
                'base_increase': 1.5
            },
            'texas': {
                'multipliers': {'tornado': 1.4, 'extreme_heat': 1.2},
                'risks': ['tornado', 'extreme weather', 'hurricane (coastal)'],
                'base_increase': 0.6
            },
            'oklahoma': {
                'multipliers': {'tornado': 1.7, 'severe_storm': 1.4},
                'risks': ['tornado', 'severe thunderstorms', 'hail'],
                'base_increase': 1.0
            },
            'kansas': {
                'multipliers': {'tornado': 1.5, 'severe_storm': 1.3},
                'risks': ['tornado', 'severe weather', 'hail'],
                'base_increase': 0.8
            },
            'maine': {
                'multipliers': {'winter_storm': 1.2, 'coastal': 1.1},
                'risks': ['severe winter weather', 'coastal storms'],
                'base_increase': -0.3  # Generally lower risk
            },
            'alaska': {
                'multipliers': {'earthquake': 2.0, 'extreme_cold': 1.5},
                'risks': ['earthquake', 'extreme cold', 'isolation'],
                'base_increase': 1.8
            },
        }
        
        address_lower = address.lower()
        for state, risks in state_risks.items():
            if state in address_lower or f', {state[:2].upper()}' in address or self.detect_state_from_coordinates(lat, lon, state):
                return risks
        
        # Default for other states
        return {
            'multipliers': {},
            'risks': [],
            'base_increase': 0
        }
    
    def detect_state_from_coordinates(self, lat: float, lon: float, state: str) -> bool:
        """Detect state from coordinates (basic implementation)"""
        state_bounds = {
            'california': {'lat': (32.5, 42), 'lon': (-124.5, -114)},
            'florida': {'lat': (24.5, 31), 'lon': (-87.5, -79.8)},
            'texas': {'lat': (25.8, 36.5), 'lon': (-106.6, -93.5)},
            'maine': {'lat': (43, 47.5), 'lon': (-71.1, -66.9)},
            'alaska': {'lat': (54, 72), 'lon': (-179, -129)},
        }
        
        if state in state_bounds:
            bounds = state_bounds[state]
            return (bounds['lat'][0] <= lat <= bounds['lat'][1] and 
                   bounds['lon'][0] <= lon <= bounds['lon'][1])
        return False
    
    def get_default_risk_factors(self, address: str) -> Dict[str, Any]:
        """Fallback risk factors when geocoding fails"""
        # Try to extract some info from address text
        address_lower = address.lower()
        base_score = 2
        description = 'unknown location'
        
        # Simple text-based analysis
        if any(city in address_lower for city in ['new york', 'los angeles', 'chicago']):
            base_score = 3
            description = 'major metropolitan area (estimated)'
        elif any(indicator in address_lower for indicator in ['rd', 'route', 'rural']):
            base_score = 1.5
            description = 'rural area (estimated)'
        
        return {
            'latitude': None,
            'longitude': None,
            'base_risk_score': base_score,
            'risk_multipliers': {},
            'location_description': description,
            'primary_risks': ['standard property risks']
        }
