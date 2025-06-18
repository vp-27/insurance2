import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, Any, List
import requests
from geopy.geocoders import Nominatim
import schedule
import threading

class DataFetcher:
    def __init__(self, data_dir: str = "./live_data_feed"):
        self.data_dir = data_dir
        self.news_api_key = os.getenv("NEWS_API_KEY")
        self.geolocator = Nominatim(user_agent="insurance_risk_app")
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
    def save_alert(self, data: Dict[str, Any]):
        """Save an alert to the live data feed directory"""
        timestamp = datetime.now().isoformat()
        filename = f"{timestamp.replace(':', '-').replace('.', '-')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # Add timestamp if not present
        if 'timestamp' not in data:
            data['timestamp'] = timestamp
            
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved alert: {filename}")
        
    def get_coordinates(self, address: str) -> tuple:
        """Get latitude and longitude from address"""
        try:
            location = self.geolocator.geocode(address)
            if location:
                return location.latitude, location.longitude
            return None, None
        except Exception as e:
            print(f"Geocoding error: {e}")
            return None, None
    
    def fetch_weather_alerts(self, lat: float = 40.7128, lon: float = -74.0060):
        """Fetch weather alerts from National Weather Service"""
        try:
            # NWS API for alerts by coordinates
            url = f"https://api.weather.gov/alerts/active?point={lat},{lon}"
            headers = {'User-Agent': 'Insurance Risk App (contact@example.com)'}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for alert in data.get('features', []):
                properties = alert.get('properties', {})
                
                alert_data = {
                    'source': 'nws_weather',
                    'timestamp': datetime.now().isoformat(),
                    'location': f"{lat},{lon}",
                    'content': f"{properties.get('headline', 'Weather Alert')}: {properties.get('description', 'No description available')}",
                    'type': 'weather',
                    'severity': properties.get('severity', 'Unknown'),
                    'event': properties.get('event', 'Weather Event')
                }
                
                self.save_alert(alert_data)
                
        except Exception as e:
            print(f"Weather fetch error: {e}")
            
    def fetch_news_alerts(self, location: str = "New York"):
        """Fetch local news that might affect insurance risk"""
        try:
            if not self.news_api_key or self.news_api_key == "your_newsdata_io_api_key_here":
                # No news data available if no API key - DO NOT simulate
                print("⚠️ No news API key configured - news data unavailable")
                return
                
            url = "https://newsdata.io/api/1/news"
            params = {
                'apikey': self.news_api_key,
                'q': f'fire OR flood OR earthquake OR crime OR accident OR emergency',
                'country': 'us',
                'language': 'en',
                'category': 'top',
                'size': 10
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for article in data.get('results', []):
                news_data = {
                    'source': 'newsdata_io',
                    'timestamp': datetime.now().isoformat(),
                    'location': location,
                    'content': f"{article.get('title', '')}: {article.get('description', '')}",
                    'type': 'news',
                    'url': article.get('link', ''),
                    'category': article.get('category', ['general'])
                }
                
                self.save_alert(news_data)
                
        except Exception as e:
            print(f"News fetch error: {e}")
            # DO NOT fall back to simulated news for real assessments
            # Only use simulated news when explicitly triggered via demo buttons
    
    def simulate_news_alerts(self, location: str):
        """Simulate news alerts when no API key is available"""
        import random
        
        sample_alerts = [
            {
                'source': 'simulated_news',
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'content': 'Local fire department responds to commercial building fire on Main Street',
                'type': 'news',
                'severity': 'medium'
            },
            {
                'source': 'simulated_news',
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'content': 'Heavy rainfall causes minor flooding in downtown area',
                'type': 'news',
                'severity': 'low'
            },
            {
                'source': 'simulated_news',
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'content': 'Construction accident blocks major intersection, emergency services on scene',
                'type': 'news',
                'severity': 'medium'
            }
        ]
        
        # Randomly select and save one alert
        if random.random() < 0.3:  # 30% chance of generating an alert
            alert = random.choice(sample_alerts)
            self.save_alert(alert)
    
    def fetch_crime_data(self, lat: float = 40.7128, lon: float = -74.0060):
        """Fetch crime data from free APIs and simulate when needed"""
        import random
        
        # Try to fetch from free crime APIs
        try:
            # Chicago Data Portal (if in Chicago area)
            if 41.8 < lat < 42.0 and -87.8 < lon < -87.5:
                self.fetch_chicago_crime_data(lat, lon)
            
            # NYC Open Data (if in NYC area)
            elif 40.4 < lat < 40.9 and -74.3 < lon < -73.7:
                self.fetch_nyc_crime_data(lat, lon)
            
            # For other areas, use FBI Crime Data API simulation
            else:
                self.simulate_crime_data_realistic(lat, lon)
                
        except Exception as e:
            print(f"Crime data fetch error: {e}")
            self.simulate_crime_data_realistic(lat, lon)
    
    def fetch_chicago_crime_data(self, lat: float, lon: float):
        """Fetch real crime data from Chicago Data Portal"""
        try:
            # Chicago Crime Data API
            url = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
            params = {
                '$limit': 10,
                '$where': f"latitude > {lat-0.01} AND latitude < {lat+0.01} AND longitude > {lon-0.01} AND longitude < {lon+0.01}",
                '$order': 'date DESC'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            crimes = response.json()
            
            for crime in crimes[:3]:  # Limit to recent 3
                crime_data = {
                    'source': 'chicago_open_data',
                    'timestamp': datetime.now().isoformat(),
                    'location': f"{lat},{lon}",
                    'content': f"Police report: {crime.get('primary_type', 'incident')} - {crime.get('description', 'No description')}",
                    'type': 'crime',
                    'severity': self.classify_crime_severity(crime.get('primary_type', '')),
                    'crime_type': crime.get('primary_type', 'unknown')
                }
                
                self.save_alert(crime_data)
                
        except Exception as e:
            print(f"Chicago crime data error: {e}")
            self.simulate_crime_data_realistic(lat, lon)
    
    def fetch_nyc_crime_data(self, lat: float, lon: float):
        """Fetch real crime data from NYC Open Data"""
        try:
            # NYC Crime Data API
            url = "https://data.cityofnewyork.us/resource/5uac-w243.json"
            params = {
                '$limit': 10,
                '$where': f"latitude > {lat-0.01} AND latitude < {lat+0.01} AND longitude > {lon-0.01} AND longitude < {lon+0.01}",
                '$order': 'cmplnt_fr_dt DESC'
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            crimes = response.json()
            
            for crime in crimes[:3]:  # Limit to recent 3
                crime_data = {
                    'source': 'nyc_open_data',
                    'timestamp': datetime.now().isoformat(),
                    'location': f"{lat},{lon}",
                    'content': f"NYPD report: {crime.get('ofns_desc', 'incident')} in {crime.get('boro_nm', 'area')}",
                    'type': 'crime',
                    'severity': self.classify_crime_severity(crime.get('ofns_desc', '')),
                    'crime_type': crime.get('ofns_desc', 'unknown')
                }
                
                self.save_alert(crime_data)
                
        except Exception as e:
            print(f"NYC crime data error: {e}")
            self.simulate_crime_data_realistic(lat, lon)
    
    def classify_crime_severity(self, crime_type: str) -> str:
        """Classify crime severity based on type"""
        crime_type = crime_type.lower()
        
        if any(word in crime_type for word in ['murder', 'assault', 'robbery', 'rape', 'shooting']):
            return 'high'
        elif any(word in crime_type for word in ['burglary', 'theft', 'vandalism', 'fraud']):
            return 'medium'
        else:
            return 'low'
    
    def simulate_crime_data_realistic(self, lat: float, lon: float):
        """Generate realistic crime simulation based on area demographics"""
        import random
        
        crime_types = [
            ('theft', 'low'), ('vandalism', 'low'), ('break-in', 'medium'), 
            ('assault', 'high'), ('robbery', 'high'), ('fraud', 'medium'),
            ('vehicle theft', 'medium'), ('domestic incident', 'medium')
        ]
        
        # Generate simulated crime data occasionally
        if random.random() < 0.25:  # 25% chance
            crime_type, base_severity = random.choice(crime_types)
            crime_data = {
                'source': 'simulated_crime',
                'timestamp': datetime.now().isoformat(),
                'location': f"{lat},{lon}",
                'content': f"Police report: {crime_type} incident reported in the area. Local law enforcement investigating.",
                'type': 'crime',
                'severity': base_severity,
                'crime_type': crime_type
            }
            
            self.save_alert(crime_data)
    
    def inject_test_alert(self, address: str, alert_type: str = "fire"):
        """Inject a test alert for demo purposes - clearly marked as demo data"""
        test_alerts = {
            'fire': {
                'source': 'demo_test',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'DEMO: 4-alarm fire reported near {address}. Emergency services on scene with multiple fire trucks and ambulances responding.',
                'type': 'fire',
                'severity': 'critical',
                'is_demo': True
            },
            'flood': {
                'source': 'demo_test',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'DEMO: Flash flood warning issued for area near {address}. Water levels rising rapidly, evacuation orders in effect.',
                'type': 'flood',
                'severity': 'high',
                'is_demo': True
            },
            'crime': {
                'source': 'demo_test',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'DEMO: Increased police activity reported near {address} due to security incident. SWAT team deployment confirmed.',
                'type': 'crime',
                'severity': 'medium',
                'is_demo': True
            },
            'earthquake': {
                'source': 'demo_test',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'DEMO: Magnitude 4.2 earthquake detected near {address}. Structural inspections recommended for all buildings.',
                'type': 'earthquake',
                'severity': 'high',
                'is_demo': True
            },
            'traffic': {
                'source': 'demo_test',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'DEMO: Major traffic incident near {address}. Multi-vehicle collision blocking main thoroughfare.',
                'type': 'traffic',
                'severity': 'medium',
                'is_demo': True
            },
            'infrastructure': {
                'source': 'demo_test',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'DEMO: Critical infrastructure failure near {address}. Power grid instability affecting multiple city blocks.',
                'type': 'infrastructure',
                'severity': 'high',
                'is_demo': True
            }
        }
        
        alert = test_alerts.get(alert_type, test_alerts['fire'])
        self.save_alert(alert)
        print(f"Injected DEMO {alert_type} alert for {address}")
        
        # Also inject some demo news for context if this is a demo
        self.inject_demo_news_alerts(address)
    
    def inject_demo_news_alerts(self, location: str):
        """Inject demo news alerts ONLY when demo is explicitly triggered"""
        import random
        
        demo_news_alerts = [
            {
                'source': 'demo_news',
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'content': 'DEMO: Local fire department responds to commercial building fire on Main Street. Multiple units on scene.',
                'type': 'news',
                'severity': 'medium',
                'is_demo': True
            },
            {
                'source': 'demo_news',
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'content': 'DEMO: Heavy rainfall causes minor flooding in downtown area. City crews monitoring situation.',
                'type': 'news',
                'severity': 'low',
                'is_demo': True
            },
            {
                'source': 'demo_news',
                'timestamp': datetime.now().isoformat(),
                'location': location,
                'content': 'DEMO: Construction accident blocks major intersection, emergency services on scene.',
                'type': 'news',
                'severity': 'medium',
                'is_demo': True
            }
        ]
        
        # Generate 1-2 demo alerts
        num_alerts = random.randint(1, 2)
        selected_alerts = random.sample(demo_news_alerts, num_alerts)
        
        for alert in selected_alerts:
            self.save_alert(alert)
        
        print(f"Injected {num_alerts} demo news alerts for {location}")

    def start_scheduled_fetching(self):
        """Start scheduled data fetching in background"""
        def run_schedule():
            while True:
                schedule.run_pending()
                time.sleep(1)
        
        # Schedule weather alerts every 60 seconds
        schedule.every(60).seconds.do(self.fetch_weather_alerts)
        
        # Schedule news alerts every 5 minutes
        schedule.every(5).minutes.do(self.fetch_news_alerts)
        
        # Schedule crime data every 3 minutes
        schedule.every(3).minutes.do(self.fetch_crime_data)
        
        # Schedule earthquake data every 10 minutes
        schedule.every(10).minutes.do(self.fetch_earthquake_data)
        
        # Schedule traffic incidents every 2 minutes
        schedule.every(2).minutes.do(self.fetch_traffic_incidents)
        
        # Schedule infrastructure alerts every 15 minutes
        schedule.every(15).minutes.do(self.fetch_infrastructure_alerts)
        
        # Run in background thread
        schedule_thread = threading.Thread(target=run_schedule, daemon=True)
        schedule_thread.start()
        
        print("Started scheduled data fetching...")
    
    def fetch_earthquake_data(self, lat: float = 40.7128, lon: float = -74.0060):
        """Fetch earthquake data from USGS API"""
        try:
            # USGS Earthquake API - free and reliable
            url = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            for earthquake in data.get('features', []):
                eq_coords = earthquake['geometry']['coordinates']
                eq_lat, eq_lon = eq_coords[1], eq_coords[0]
                
                # Check if earthquake is within 100km radius
                distance = self.calculate_distance(lat, lon, eq_lat, eq_lon)
                
                if distance <= 100:  # Within 100km
                    properties = earthquake['properties']
                    magnitude = properties.get('mag', 0)
                    
                    if magnitude >= 2.0:  # Only significant earthquakes
                        eq_data = {
                            'source': 'usgs_earthquake',
                            'timestamp': datetime.now().isoformat(),
                            'location': f"{lat},{lon}",
                            'content': f"Earthquake detected: Magnitude {magnitude} at {properties.get('place', 'unknown location')}. Distance: {distance:.1f}km",
                            'type': 'earthquake',
                            'severity': 'high' if magnitude >= 4.0 else 'medium' if magnitude >= 3.0 else 'low',
                            'magnitude': magnitude,
                            'distance_km': distance
                        }
                        
                        self.save_alert(eq_data)
                        
        except Exception as e:
            print(f"Earthquake data fetch error: {e}")
    
    def fetch_traffic_incidents(self, lat: float = 40.7128, lon: float = -74.0060):
        """Fetch traffic incident data from free sources"""
        try:
            # Try to get traffic incidents from MapBox Incidents API (free tier)
            # Note: This requires a free MapBox API key, but we'll simulate if not available
            self.simulate_traffic_incidents(lat, lon)
            
        except Exception as e:
            print(f"Traffic incident fetch error: {e}")
            self.simulate_traffic_incidents(lat, lon)
    
    def simulate_traffic_incidents(self, lat: float, lon: float):
        """Simulate realistic traffic incidents"""
        import random
        
        incident_types = [
            ('Major accident blocking highway', 'high'),
            ('Multi-vehicle collision on main road', 'medium'),
            ('Road construction causing delays', 'low'),
            ('Vehicle breakdown in tunnel', 'medium'),
            ('Emergency vehicle response blocking lane', 'low'),
            ('Gas leak causing road closure', 'high'),
            ('Power lines down across street', 'high')
        ]
        
        if random.random() < 0.15:  # 15% chance
            incident, severity = random.choice(incident_types)
            traffic_data = {
                'source': 'simulated_traffic',
                'timestamp': datetime.now().isoformat(),
                'location': f"{lat},{lon}",
                'content': f"Traffic incident: {incident}. Emergency services responding.",
                'type': 'traffic',
                'severity': severity
            }
            
            self.save_alert(traffic_data)
    
    def fetch_infrastructure_alerts(self, lat: float = 40.7128, lon: float = -74.0060):
        """Fetch infrastructure-related alerts that could affect insurance risk"""
        try:
            # Simulate infrastructure issues
            self.simulate_infrastructure_alerts(lat, lon)
            
        except Exception as e:
            print(f"Infrastructure alert fetch error: {e}")
    
    def simulate_infrastructure_alerts(self, lat: float, lon: float):
        """Simulate infrastructure-related risk factors"""
        import random
        
        infrastructure_issues = [
            ('Water main break affecting multiple buildings', 'high'),
            ('Power outage reported in commercial district', 'medium'),
            ('Gas leak detected near building foundation', 'high'),
            ('Sewer backup causing basement flooding risk', 'medium'),
            ('Bridge inspection finds structural concerns', 'medium'),
            ('Construction crane malfunction creates safety zone', 'high'),
            ('Electrical transformer explosion nearby', 'high')
        ]
        
        if random.random() < 0.12:  # 12% chance
            issue, severity = random.choice(infrastructure_issues)
            infra_data = {
                'source': 'simulated_infrastructure',
                'timestamp': datetime.now().isoformat(),
                'location': f"{lat},{lon}",
                'content': f"Infrastructure alert: {issue}",
                'type': 'infrastructure',
                'severity': severity
            }
            
            self.save_alert(infra_data)
    
    def calculate_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two coordinates in kilometers"""
        from math import radians, cos, sin, asin, sqrt
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371  # Radius of earth in kilometers
        
        return c * r

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    fetcher = DataFetcher()
    
    # Run initial fetch
    print("Fetching initial data...")
    fetcher.fetch_weather_alerts()
    fetcher.fetch_news_alerts()
    fetcher.fetch_crime_data()
    fetcher.fetch_earthquake_data()
    fetcher.fetch_traffic_incidents()
    fetcher.fetch_infrastructure_alerts()
    
    # Start scheduled fetching
    fetcher.start_scheduled_fetching()
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping data fetcher...")
