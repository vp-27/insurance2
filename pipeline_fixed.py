import pathway as pw
import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
from location_analyzer import LocationAnalyzer
from openai import OpenAI
from dotenv import load_dotenv

class LiveRAGPipeline:
    def __init__(self, data_dir: str = "./live_data_feed"):
        # Force reload environment variables
        load_dotenv(override=True)
        
        self.data_dir = data_dir
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("MODEL_NAME", "google/gemma-3n-e4b-it:free")
        self.base_cost = float(os.getenv("BASE_INSURANCE_COST", "500"))
        self.risk_multiplier = float(os.getenv("RISK_MULTIPLIER", "0.1"))
        
        # Initialize location analyzer
        self.location_analyzer = LocationAnalyzer()
        
        # Initialize OpenAI client for OpenRouter with robust validation
        self.openai_client = None
        if (self.openrouter_api_key and 
            self.openrouter_api_key != "your_openrouter_api_key_here" and
            len(self.openrouter_api_key.strip()) > 10):  # Basic key validation
            try:
                self.openai_client = OpenAI(
                    base_url=self.openrouter_base_url,
                    api_key=self.openrouter_api_key
                )
                print(f"✅ OpenRouter client initialized with model: {self.model_name}")
            except Exception as e:
                print(f"⚠️ Failed to initialize OpenRouter client: {e}")
                self.openai_client = None
        else:
            print("⚠️ OpenRouter API key not found or invalid - using simulation mode")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Store for vector index
        self.documents = []
        self.embeddings = []
        self.document_metadata = []
        
        # Initialize Pathway pipeline
        self.setup_pathway_pipeline()

    def simulate_llm_response(self, prompt: str) -> str:
        """Simulate LLM response when API is not available - context-aware simulation"""
        # Extract address from prompt for personalized response
        import re
        address_match = re.search(r'PROPERTY: ([^\n]+)', prompt)
        address = address_match.group(1) if address_match else "the specified location"
        
        # Analyze prompt content for real vs demo incidents
        prompt_lower = prompt.lower()
        
        # Check if the prompt indicates demo/test alerts should be ignored
        has_demo_warning = "demo/test alerts" in prompt_lower and "ignore" in prompt_lower
        
        # Check for real incidents in the prompt
        has_real_incidents = "real incidents:" in prompt_lower
        real_incident_section = ""
        if has_real_incidents:
            # Extract the real incidents section
            real_start = prompt.find("REAL INCIDENTS")
            demo_start = prompt.find("DEMO/TEST ALERTS")
            if real_start != -1:
                if demo_start != -1 and demo_start > real_start:
                    real_incident_section = prompt[real_start:demo_start]
                else:
                    # Find the end of the real incidents section
                    location_start = prompt.find("LOCATION ANALYSIS")
                    if location_start != -1 and location_start < real_start:
                        # Find next section after real incidents
                        next_section = prompt.find("\n\n", real_start + 100)
                        if next_section != -1:
                            real_incident_section = prompt[real_start:next_section]
                        else:
                            real_incident_section = prompt[real_start:]
        
        # Check what kind of real incidents exist
        no_real_incidents = (
            "no recent real incidents detected" in real_incident_section.lower() or
            "location appears stable" in real_incident_section.lower() or
            real_incident_section.strip() == ""
        )
        
        # Determine risk level based on actual content
        if has_demo_warning and no_real_incidents:
            # This is a demo scenario with no real incidents - return low risk
            return """**✅ STANDARD RISK ASSESSMENT for the specified location**

**Risk Summary:** Current analysis indicates NORMAL risk levels for the area. No significant incidents, weather alerts, or emergency conditions detected. Demo alerts detected but properly excluded from risk calculation.

**Risk Score:** 2/10

**Insurance Quote:** $550/month for $1M coverage (standard rate)

**Assessment Details:**
- No active real-world incidents detected in the area
- Demo/test alerts properly filtered out of assessment
- Standard area risk profile maintained
- Normal emergency service activity levels
- Routine coverage requirements apply

**Location Status:** All systems normal, area appears stable with standard risk levels."""

        elif "harrassment" in real_incident_section.lower() or "petit larceny" in real_incident_section.lower():
            # Low-level crime incidents
            return """**⚠️ MODERATE RISK ASSESSMENT for the specified location**

**Risk Summary:** Minor criminal incidents detected in the area including harassment and petit larceny reports. These represent typical urban activity that requires monitoring but does not constitute critical risk.

**Risk Score:** 4/10

**Insurance Quote:** $700/month for $1M coverage (moderate increase)

**Assessment Details:**
- Minor criminal activity reported in vicinity
- No major incidents or emergency situations
- Standard urban risk profile
- Normal law enforcement response
- Routine monitoring recommended"""

        else:
            # Default to moderate risk for unknown situations
            return """**📊 STANDARD RISK ASSESSMENT for the specified location**

**Risk Summary:** Current analysis indicates NORMAL to MODERATE risk levels for the area. Standard location-based risk factors apply with no major incidents detected.

**Risk Score:** 3/10

**Insurance Quote:** $650/month for $1M coverage (standard rate)

**Assessment Details:**
- No critical incidents detected
- Standard area risk profile
- Normal emergency service coverage
- Routine insurance coverage recommended
- No elevated threat indicators present"""

# Add minimal stubs for other methods so it doesn't break
# ... (rest of the methods would go here)
