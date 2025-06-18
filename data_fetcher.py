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
                # Simulate news data if no API key
                self.simulate_news_alerts(location)
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
            self.simulate_news_alerts(location)
    
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
        """Simulate crime data (many crime APIs require special access)"""
        import random
        
        crime_types = ['theft', 'vandalism', 'break-in', 'assault', 'robbery']
        
        # Generate simulated crime data occasionally
        if random.random() < 0.2:  # 20% chance
            crime_data = {
                'source': 'simulated_crime',
                'timestamp': datetime.now().isoformat(),
                'location': f"{lat},{lon}",
                'content': f"Police report: {random.choice(crime_types)} incident reported in the area",
                'type': 'crime',
                'severity': random.choice(['low', 'medium', 'high'])
            }
            
            self.save_alert(crime_data)
    
    def inject_test_alert(self, address: str, alert_type: str = "fire"):
        """Inject a test alert for demo purposes"""
        test_alerts = {
            'fire': {
                'source': 'manual_injection',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'4-alarm fire reported near {address}. Emergency services on scene.',
                'type': 'fire',
                'severity': 'critical'
            },
            'flood': {
                'source': 'manual_injection',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'Flash flood warning issued for area near {address}. Water levels rising.',
                'type': 'flood',
                'severity': 'high'
            },
            'crime': {
                'source': 'manual_injection',
                'timestamp': datetime.now().isoformat(),
                'location': address,
                'content': f'Increased police activity reported near {address} due to security incident.',
                'type': 'crime',
                'severity': 'medium'
            }
        }
        
        alert = test_alerts.get(alert_type, test_alerts['fire'])
        self.save_alert(alert)
        print(f"Injected {alert_type} alert for {address}")
    
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
        
        # Run in background thread
        schedule_thread = threading.Thread(target=run_schedule, daemon=True)
        schedule_thread.start()
        
        print("Started scheduled data fetching...")

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    fetcher = DataFetcher()
    
    # Run initial fetch
    print("Fetching initial data...")
    fetcher.fetch_weather_alerts()
    fetcher.fetch_news_alerts()
    fetcher.fetch_crime_data()
    
    # Start scheduled fetching
    fetcher.start_scheduled_fetching()
    
    # Keep alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping data fetcher...")
