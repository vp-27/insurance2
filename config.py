"""
Configuration settings for Live Insurance Risk & Quote Co-Pilot
"""

import os
from typing import Dict, Any

class Config:
    """Application configuration"""
    
    # API Configuration
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "your_openrouter_api_key_here")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "your_newsdata_io_api_key_here")
    
    # Model Configuration
    LLM_MODEL = os.getenv("MODEL_NAME", "google/gemma-3-4b-it:free")
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Application Settings
    DATA_DIRECTORY = "./live_data_feed"
    REFRESH_INTERVAL = int(os.getenv("REFRESH_INTERVAL", "30"))
    WEATHER_FETCH_INTERVAL = int(os.getenv("WEATHER_FETCH_INTERVAL", "60"))
    NEWS_FETCH_INTERVAL = int(os.getenv("NEWS_FETCH_INTERVAL", "300"))
    
    # Insurance Calculation
    BASE_INSURANCE_COST = float(os.getenv("BASE_INSURANCE_COST", "500"))
    RISK_MULTIPLIER = float(os.getenv("RISK_MULTIPLIER", "0.1"))
    COVERAGE_AMOUNT = 1000000  # $1M
    
    # Geographic Defaults
    DEFAULT_LATITUDE = 40.7128  # New York City
    DEFAULT_LONGITUDE = -74.0060
    DEFAULT_LOCATION = "New York"
    
    # System Settings
    MAX_SEARCH_RESULTS = 5
    MIN_SIMILARITY_THRESHOLD = 0.1
    AUTO_REFRESH_ENABLED = True
    
    # Demo Configuration
    DEMO_ADDRESSES = [
        "25 Columbus Dr, Jersey City, NJ",
        "123 Main St, New York, NY",
        "456 Oak Ave, Brooklyn, NY"
    ]
    
    # Risk Score Mappings
    RISK_CATEGORIES = {
        "low": {"min": 1, "max": 3, "label": "Low Risk", "color": "green"},
        "medium": {"min": 4, "max": 6, "label": "Medium Risk", "color": "yellow"},
        "high": {"min": 7, "max": 10, "label": "High Risk", "color": "red"}
    }
    
    # Alert Type Configurations
    ALERT_TYPES = {
        "fire": {
            "default_score": 9,
            "description": "4-alarm fire reported near the address",
            "severity": "critical"
        },
        "flood": {
            "default_score": 7,
            "description": "Flash flood warning issued for the area",
            "severity": "high"
        },
        "crime": {
            "default_score": 5,
            "description": "Increased police activity reported in the area",
            "severity": "medium"
        },
        "weather": {
            "default_score": 4,
            "description": "Severe weather alert issued",
            "severity": "medium"
        }
    }
    
    @classmethod
    def get_risk_category(cls, score: int) -> Dict[str, Any]:
        """Get risk category information for a given score"""
        for category, info in cls.RISK_CATEGORIES.items():
            if info["min"] <= score <= info["max"]:
                return {"category": category, **info}
        return {"category": "unknown", "label": "Unknown Risk", "color": "gray"}
    
    @classmethod
    def calculate_quote(cls, risk_score: int) -> float:
        """Calculate insurance quote based on risk score"""
        return cls.BASE_INSURANCE_COST * (1 + cls.RISK_MULTIPLIER * risk_score)
    
    @classmethod
    def is_simulation_mode(cls) -> bool:
        """Check if running in simulation mode (no API keys)"""
        return (cls.OPENROUTER_API_KEY == "your_openrouter_api_key_here" or
                cls.NEWS_API_KEY == "your_newsdata_io_api_key_here")

# Export configuration instance
config = Config()
