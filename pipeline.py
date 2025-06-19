try:
    import pathway as pw
except ImportError:
    # Mock pathway for development
    class MockPathway:
        def __init__(self):
            pass
        def run(self):
            print("Mock Pathway pipeline running...")
            return "Mock pipeline completed"
    pw = MockPathway()

import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests

# Handle optional ML dependencies
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    HAS_ML_DEPS = True
except ImportError:
    print("Warning: ML dependencies not available. Some features will be limited.")
    SentenceTransformer = None
    np = None
    HAS_ML_DEPS = False
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
        if HAS_ML_DEPS and SentenceTransformer:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            self.embedding_model = None
            print("Warning: Embedding model not available. Vector search disabled.")
        
        # Store for vector index
        self.documents = []
        self.embeddings = []
        self.document_metadata = []
        
        # Initialize Pathway pipeline
        self.setup_pathway_pipeline()
    
    def setup_pathway_pipeline(self):
        """Setup Pathway streaming pipeline"""
        try:
            # Use Pathway's filesystem connector to watch JSON files
            # First, ensure we have some initial data
            os.makedirs(self.data_dir, exist_ok=True)
            
            # Create an initial test file if directory is empty
            files_in_dir = [f for f in os.listdir(self.data_dir) if f.endswith('.json')]
            if not files_in_dir:
                initial_data = {
                    'source': 'system_init',
                    'timestamp': datetime.now().isoformat(),
                    'location': 'system',
                    'content': 'System initialized - monitoring for live data feeds',
                    'type': 'system'
                }
                with open(os.path.join(self.data_dir, 'init.json'), 'w') as f:
                    json.dump(initial_data, f, indent=2)
            
            # Define schema for our data
            class InputSchema(pw.Schema):
                source: str
                timestamp: str
                location: str
                content: str
                type: str
            
            # Use Pathway's filesystem connector for streaming JSON files
            self.input_table = pw.io.fs.read(
                path=self.data_dir,
                format="json",
                schema=InputSchema,
                mode="streaming"
            )
            
            # Process the data stream
            if self.input_table is not None:
                # Apply real-time transformations
                self.processed_table = self.input_table.select(
                    source=pw.this.source,
                    timestamp=pw.this.timestamp,
                    location=pw.this.location,
                    content=pw.this.content,
                    type=pw.this.type,
                    processed_at=pw.now()
                )
                
                print("Pathway streaming pipeline initialized successfully")
                return
            
        except Exception as e:
            print(f"Pathway setup error: {e}")
        
        print("Using manual file monitoring as primary method")
        # Use manual file monitoring as the primary method for this demo
        self.input_table = None
        self.processed_table = None
    
    def embed_text(self, text: str) -> Any:
        """Generate embeddings for text"""
        if self.embedding_model:
            return self.embedding_model.encode(text)
        else:
            # Return a simple hash-based representation as fallback
            return [hash(text) % 1000 for _ in range(384)]  # Mock 384-dim vector
    
    def add_document(self, doc_data: Dict[str, Any]):
        """Add a document to the vector store"""
        content = doc_data.get('content', '')
        if content:
            embedding = self.embed_text(content)
            
            self.documents.append(content)
            self.embeddings.append(embedding)
            self.document_metadata.append(doc_data)
    
    def similarity_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search on the vector store"""
        if not self.embeddings:
            return []
        
        query_embedding = self.embed_text(query)
        embeddings_array = np.array(self.embeddings)
        
        # Calculate cosine similarity
        similarities = np.dot(embeddings_array, query_embedding) / (
            np.linalg.norm(embeddings_array, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Minimum similarity threshold
                results.append({
                    'content': self.documents[idx],
                    'metadata': self.document_metadata[idx],
                    'similarity': float(similarities[idx])
                })
        
        return results
    
    def load_existing_data(self):
        """Load existing data from the data directory"""
        if not os.path.exists(self.data_dir):
            return
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self.add_document(data)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def monitor_new_files(self):
        """Monitor for new files and add them to the vector store"""
        import time
        processed_files = set()
        
        while True:
            try:
                if os.path.exists(self.data_dir):
                    current_files = set(os.listdir(self.data_dir))
                    new_files = current_files - processed_files
                    
                    for filename in new_files:
                        if filename.endswith('.json'):
                            filepath = os.path.join(self.data_dir, filename)
                            try:
                                with open(filepath, 'r') as f:
                                    data = json.load(f)
                                    self.add_document(data)
                                    print(f"Added new document: {filename}")
                            except Exception as e:
                                print(f"Error processing new file {filename}: {e}")
                    
                    processed_files = current_files
                
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error monitoring files: {e}")
                time.sleep(10)
    
    async def call_llm(self, prompt: str) -> str:
        """Call LLM via OpenRouter API using OpenAI client"""
        try:
            if not self.openai_client:
                # Use simulation instead of returning error
                print("⚠️ No API client available, using simulation mode")
                return self.simulate_llm_response(prompt)
            
            # Verify API key is still valid before making request
            if not self.openrouter_api_key or self.openrouter_api_key == "your_openrouter_api_key_here":
                print("⚠️ Invalid API key detected, using simulation mode")
                return self.simulate_llm_response(prompt)
            
            print(f"🤖 Calling OpenRouter API with model: {self.model_name}")
            completion = self.openai_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://localhost:8000",  # Update with actual domain
                    "X-Title": "Live Insurance Risk Assessment",     # Site title for rankings
                },
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert insurance underwriter. Analyze the data and distinguish between real news incidents and test/simulated alerts. Provide accurate risk assessments based on actual events only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent results
                max_tokens=800    # Increased for more detailed responses
            )
            
            response = completion.choices[0].message.content
            print(f"✅ OpenRouter API response received ({len(response)} chars)")
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"🚨 LLM API error: {e}")
            
            # Check if it's a 401 authentication error
            if "401" in error_msg or "auth" in error_msg.lower() or "unauthorized" in error_msg.lower():
                print("🔄 API authentication failed, falling back to simulation mode")
                return self.simulate_llm_response(prompt)
            
            # For other errors, still fall back to simulation
            print("🔄 API error occurred, falling back to simulation mode")
            return self.simulate_llm_response(prompt)
    
    def simulate_llm_response(self, prompt: str) -> str:
        """Simulate LLM response when API is not available"""
        # Extract address from prompt for personalized response
        import re
        address_match = re.search(r'at ([^?]+)\?', prompt)
        address = address_match.group(1) if address_match else "the specified location"
        
        # Analyze prompt content for specific risk factors
        prompt_lower = prompt.lower()
        
        if "fire" in prompt_lower or "4-alarm" in prompt_lower:
            return f"""**🔥 CRITICAL RISK ASSESSMENT for {address}**

**Risk Summary:** CRITICAL FIRE INCIDENT detected in immediate vicinity. Active 4-alarm fire with emergency services deployment creates severe operational risk affecting property values, safety protocols, and insurance liability.

**Risk Score:** 9/10

**Insurance Quote:** $950/month for $1M coverage (90% increase from base rate)

**Critical Factors:**
• Active fire emergency with multi-unit response
• Extreme risk of smoke and water damage from firefighting efforts
• Potential structural damage from heat exposure
• Emergency service accessibility restrictions
• Temporary evacuation protocols may be required

**Immediate Recommendations:**
• Activate emergency response plan
• Document all safety measures
• Coordinate with emergency services"""

        elif "flood" in prompt_lower or "water" in prompt_lower:
            return f"""**🌊 HIGH RISK ASSESSMENT for {address}**

**Risk Summary:** FLOOD WARNING detected with rising water levels. Significant risk to property foundations, electrical systems, and structural integrity requiring immediate attention.

**Risk Score:** 7/10

**Insurance Quote:** $850/month for $1M coverage (70% increase from base rate)

**Key Risk Factors:**
• Flash flood conditions with rapidly rising water
• Foundation and basement flooding potential
• Electrical system vulnerability
• HVAC equipment damage risk
• Business interruption likelihood

**Mitigation Actions:**
• Implement flood protection measures
• Elevate critical equipment
• Monitor drainage systems"""

        elif "earthquake" in prompt_lower or "magnitude" in prompt_lower:
            return f"""**🏗️ HIGH RISK ASSESSMENT for {address}**

**Risk Summary:** SEISMIC ACTIVITY detected. Magnitude 4.2 earthquake requires structural assessment and elevated monitoring for potential aftershocks and building integrity issues.

**Risk Score:** 8/10

**Insurance Quote:** $900/month for $1M coverage (80% increase from base rate)

**Seismic Risk Factors:**
• Structural integrity concerns requiring inspection
• Potential aftershock activity
• Building code compliance verification needed
• Foundation stability assessment required
• Utility line disruption possible

**Required Actions:**
• Schedule immediate structural inspection
• Review seismic safety protocols
• Check utility connections"""

        elif "crime" in prompt_lower or "police" in prompt_lower:
            return f"""**🚨 ELEVATED RISK ASSESSMENT for {address}**

**Risk Summary:** SECURITY INCIDENT with increased law enforcement presence. Enhanced security measures required due to elevated crime activity and potential ongoing threat assessment.

**Risk Score:** 6/10

**Insurance Quote:** $800/month for $1M coverage (60% increase from base rate)

**Security Risk Factors:**
• Active police investigation in vicinity
• Potential for continued criminal activity
• Security system effectiveness concerns
• Staff safety protocols activation needed
• Property access restrictions possible

**Security Enhancements:**
• Increase surveillance monitoring
• Review access control systems
• Coordinate with local law enforcement"""

        elif "traffic" in prompt_lower or "collision" in prompt_lower:
            return f"""**🚗 MODERATE RISK ASSESSMENT for {address}**

**Risk Summary:** TRAFFIC INCIDENT affecting area accessibility. Multi-vehicle collision creating temporary operational disruptions and emergency service response delays.

**Risk Score:** 4/10

**Insurance Quote:** $700/month for $1M coverage (40% increase from base rate)

**Traffic Impact Factors:**
• Emergency service response delays
• Customer/client access limitations
• Delivery and logistics disruptions
• Potential secondary incident risks

**Operational Adjustments:**
• Plan alternative access routes
• Notify stakeholders of delays
• Monitor traffic conditions"""

        elif "infrastructure" in prompt_lower or "power" in prompt_lower:
            return f"""**⚡ HIGH RISK ASSESSMENT for {address}**

**Risk Summary:** CRITICAL INFRASTRUCTURE FAILURE affecting power grid stability. Multi-block power instability creates significant operational and safety risks requiring emergency protocols.

**Risk Score:** 7/10

**Insurance Quote:** $850/month for $1M coverage (70% increase from base rate)

**Infrastructure Risk Factors:**
• Power grid instability affecting operations
• HVAC system failure potential
• Security system vulnerability
• Data system backup requirements
• Emergency lighting activation needed

**Emergency Measures:**
• Activate backup power systems
• Implement data protection protocols
• Monitor utility restoration efforts"""

        else:
            return f"""**✅ STANDARD RISK ASSESSMENT for {address}**

**Risk Summary:** Current analysis indicates NORMAL risk levels for the area. No significant incidents, weather alerts, or emergency conditions detected that would materially impact the property's risk profile.

**Risk Score:** 2/10

**Insurance Quote:** $600/month for $1M coverage (standard rate)

**Baseline Assessment:**
• No active weather or emergency alerts detected
• Standard area risk profile maintained
• Normal emergency service activity levels
• Routine coverage requirements apply
• No elevated threat indicators present

**Regional Factors:**
• Base location risk assessment: Low to moderate
• Standard building and fire safety protocols in effect
• Adequate emergency services coverage
• No unusual environmental hazards detected

**Status:** All systems normal, continue standard monitoring protocols."""
    
    async def query_rag(self, address: str, query: str) -> Dict[str, Any]:
        """Main RAG query function with location-specific analysis"""
        
        # Get location-specific risk factors
        location_factors = self.location_analyzer.analyze_location_risk_factors(address)
        
        # Search for relevant documents
        search_query = f"{address} {query}"
        relevant_docs = self.similarity_search(search_query, top_k=5)
        
        # Build context from relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}\n")
        
        context = "\n---\n".join(context_parts) if context_parts else "No relevant incidents found in recent data."
        
        # Build enhanced prompt with location context
        location_context = self.build_location_context(location_factors)
        
        # Analyze the data sources to distinguish between real and test alerts
        real_incidents = []
        test_incidents = []
        
        for doc in relevant_docs:
            source = doc['metadata'].get('source', '')
            content = doc['content']
            
            # Identify test/demo/simulated data
            if any(indicator in source.lower() for indicator in ['manual_injection', 'test', 'simulated', 'demo_test', 'demo_news']):
                test_incidents.append(f"DEMO/TEST ALERT: {content}")
            elif source in ['newsdata_io', 'nyc_open_data', 'chicago_open_data', 'usgs_earthquake', 'nws_weather']:
                real_incidents.append(f"REAL INCIDENT: {content}")
            else:
                # For unknown source, check if content has DEMO prefix
                if content.startswith('DEMO:'):
                    test_incidents.append(f"DEMO ALERT: {content}")
                else:
                    # Unknown source, treat as real but flag
                    real_incidents.append(f"INCIDENT: {content}")
        
        # Build separate context sections
        real_context = "\n".join(real_incidents) if real_incidents else "No recent real incidents detected in the area. Location appears stable."
        test_context = "\n".join(test_incidents) if test_incidents else "No demo/test alerts active."
        
        # Only include test context if there are actual test incidents
        if test_incidents:
            prompt = f"""You are an AI Insurance Underwriting Assistant analyzing property risk.

PROPERTY: {address}

LOCATION ANALYSIS:
{location_context}

REAL INCIDENTS (use for risk calculation):
{real_context}

DEMO/TEST ALERTS (for reference only - DO NOT include in risk calculation):
{test_context}

CRITICAL: Base your risk assessment ONLY on REAL incidents and location factors. Demo/test alerts are simulated and should be COMPLETELY IGNORED in risk scoring. If only demo/test alerts are present and no real incidents, consider the area SAFE.

Provide:
1. Risk summary focusing ONLY on verified real incidents and location factors
2. Risk score from 1 (safe) to 10 (critical) - based ONLY on REAL incidents
3. Monthly insurance quote for $1M coverage

Base calculation: ${self.base_cost} × (1 + {self.risk_multiplier} × risk_score)
Factor in location-specific risks but IGNORE ALL demo/test scenarios."""
        else:
            # No test alerts - normal assessment
            prompt = f"""You are an AI Insurance Underwriting Assistant analyzing property risk.

PROPERTY: {address}

LOCATION ANALYSIS:
{location_context}

CURRENT REAL INCIDENTS:
{real_context}

Assessment Instructions:
- If no real incidents are present, consider this a SAFE area with normal risk levels
- Base your assessment only on verified real-world data and location factors
- Do not invent or simulate incidents - if the area is quiet, that's a positive indicator

Provide a comprehensive risk assessment including:
1. Risk summary considering location factors and any current verified incidents
2. Risk score from 1 (safe) to 10 (critical) based on actual conditions
3. Monthly insurance quote for $1M coverage

Base calculation: ${self.base_cost} × (1 + {self.risk_multiplier} × risk_score)"""
        
        # Get LLM response
        llm_response = await self.call_llm(prompt)
        
        # Extract risk score from response with location consideration
        risk_score = self.extract_risk_score_enhanced(llm_response, location_factors)
        
        # Calculate quote with location factors
        quote = self.calculate_enhanced_quote(risk_score, location_factors)
        
        return {
            'risk_summary': llm_response,
            'risk_score': risk_score,
            'insurance_quote': round(quote, 2),
            'relevant_documents': len(relevant_docs),
            'location_factors': location_factors,
            'timestamp': datetime.now().isoformat()
        }
    
    def extract_risk_score(self, response: str) -> int:
        """Extract risk score from LLM response"""
        import re
        
        # Look for patterns like "Risk Score: 7/10", "7 out of 10", or "8.5"
        patterns = [
            r'Risk Score:?\s*(\d+(?:\.\d+)?)(?:/10)?',
            r'risk score:?\s*(\d+(?:\.\d+)?)(?:/10)?',
            r'Score:?\s*(\d+(?:\.\d+)?)(?:/10)?',
            r'(\d+(?:\.\d+)?)\s*(?:out of|/)\s*10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = float(match.group(1))
                return min(max(int(round(score)), 1), 10)  # Round and clamp between 1-10
        
        # Default to moderate risk if no score found
        return 5
    
    def build_location_context(self, location_factors: Dict[str, Any]) -> str:
        """Build location context string for LLM prompt"""
        context_parts = []
        
        if location_factors['latitude'] and location_factors['longitude']:
            context_parts.append(f"Coordinates: {location_factors['latitude']:.2f}, {location_factors['longitude']:.2f}")
        
        context_parts.append(f"Area Type: {location_factors['location_description']}")
        context_parts.append(f"Base Risk Level: {location_factors['base_risk_score']:.1f}/10")
        
        if location_factors['primary_risks']:
            context_parts.append(f"Primary Regional Risks: {', '.join(location_factors['primary_risks'])}")
        
        if location_factors['risk_multipliers']:
            multiplier_info = []
            for risk_type, multiplier in location_factors['risk_multipliers'].items():
                multiplier_info.append(f"{risk_type} ({multiplier:.1f}x)")
            context_parts.append(f"Risk Multipliers: {', '.join(multiplier_info)}")
        
        return "\n".join(context_parts)
    
    def extract_risk_score_enhanced(self, llm_response: str, location_factors: Dict[str, Any]) -> int:
        """Enhanced risk score extraction with location consideration"""
        # Check if this is an error message
        if "❌" in llm_response or "Error" in llm_response or "API Error" in llm_response:
            # Return base location risk for errors, not a default high score
            return max(1, int(location_factors.get('base_risk_score', 3)))
        
        # Try to extract from LLM response first
        base_score = self.extract_risk_score(llm_response)
        
        # If using default score, consider location factors
        if base_score == 5:  # Default fallback score
            base_score = max(2, location_factors['base_risk_score'])
        
        # Ensure score is within bounds
        return max(1, min(10, int(base_score)))
    
    def calculate_enhanced_quote(self, risk_score: int, location_factors: Dict[str, Any]) -> float:
        """Calculate quote with location-specific adjustments"""
        base_quote = self.base_cost * (1 + self.risk_multiplier * risk_score)
        
        # Apply location-specific multipliers (dampened to avoid extreme quotes)
        total_multiplier = 1.0
        for risk_type, multiplier in location_factors.get('risk_multipliers', {}).items():
            # Apply a dampened version of the multiplier to avoid extreme quotes
            dampened_multiplier = 1 + (multiplier - 1) * 0.25  # 25% of the full multiplier effect
            total_multiplier *= dampened_multiplier
        
        return base_quote * total_multiplier

if __name__ == "__main__":
    from dotenv import load_dotenv
    import threading
    
    load_dotenv()
    
    # Initialize pipeline
    pipeline = LiveRAGPipeline()
    
    # Load existing data
    pipeline.load_existing_data()
    
    # Start file monitoring in background
    monitor_thread = threading.Thread(target=pipeline.monitor_new_files, daemon=True)
    monitor_thread.start()
    
    print("RAG Pipeline initialized and monitoring for new data...")
    
    # Keep alive
    try:
        while True:
            import time
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping RAG pipeline...")
