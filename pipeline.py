try:
    import pathway as pw
except ImportError:
    # Mock pathway for development
    class MockSchema:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    class MockPathway:
        def __init__(self):
            self.Schema = MockSchema
            self.io = MockIO()
        
        def run(self):
            print("Mock Pathway pipeline running...")
            return "Mock pipeline completed"
    
    class MockIO:
        def __init__(self):
            self.fs = MockFS()
    
    class MockFS:
        def read(self, path, format, schema, mode):
            print(f"Mock Pathway: Would monitor {path} for {format} files")
            return None
    
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
        
        # If numpy is not available, do simple keyword matching
        if not np:
            return self.keyword_search(query, top_k)
        
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
    
    def keyword_search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Fallback keyword-based search when numpy is not available"""
        query_words = set(query.lower().split())
        results = []
        
        for i, (doc, metadata) in enumerate(zip(self.documents, self.document_metadata)):
            doc_words = set(doc.lower().split())
            # Simple word overlap score
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                results.append({
                    'content': doc,
                    'metadata': metadata,
                    'similarity': overlap / len(query_words)  # Normalized overlap
                })
        
        # Sort by similarity and return top-k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
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
        """Context-aware simulation that respects demo vs real alert separation"""
        # Extract address from prompt for personalized response
        import re
        address_match = re.search(r'PROPERTY: ([^\n]+)', prompt)
        address = address_match.group(1) if address_match else "the specified location"
        
        # Analyze prompt structure to distinguish demo scenarios
        prompt_lower = prompt.lower()
        
        # Check if this prompt has the demo warning structure
        has_demo_warning = "demo/test alerts" in prompt_lower and "ignore" in prompt_lower
        
        # Extract the real incidents section
        real_incident_section = ""
        if "real incidents:" in prompt_lower:
            real_start = prompt.find("REAL INCIDENTS")
            if real_start != -1:
                # Find the end of the real incidents section
                demo_start = prompt.find("DEMO/TEST ALERTS")
                if demo_start != -1 and demo_start > real_start:
                    real_incident_section = prompt[real_start:demo_start]
                else:
                    # Find next major section
                    next_section_patterns = ["CRITICAL INSTRUCTIONS:", "Assessment Instructions:", "Assessment Requirements:"]
                    end_pos = len(prompt)
                    for pattern in next_section_patterns:
                        pattern_pos = prompt.find(pattern, real_start)
                        if pattern_pos != -1:
                            end_pos = min(end_pos, pattern_pos)
                    real_incident_section = prompt[real_start:end_pos]
        
        # Check what kind of real incidents exist
        no_real_incidents = (
            "no recent real incidents detected" in real_incident_section.lower() or
            "location appears stable" in real_incident_section.lower() or
            real_incident_section.strip() == "" or
            "REAL INCIDENTS" not in real_incident_section
        )
        
        print(f"🤖 Simulation mode analysis:")
        print(f"   - Has demo warning: {has_demo_warning}")
        print(f"   - No real incidents: {no_real_incidents}")
        print(f"   - Real section: {real_incident_section[:100]}...")
        
        # CRITICAL: If this is a demo scenario with instructions to ignore demo alerts,
        # and there are no real incidents, return LOW RISK
        if has_demo_warning and no_real_incidents:
            print("🔄 Demo scenario detected - returning LOW RISK assessment")
            return f"""**✅ SAFE AREA ASSESSMENT for {address}**

**Risk Summary:** Area analysis indicates NORMAL/LOW risk levels. No verified real incidents detected in the vicinity. Demo alerts were properly excluded from risk assessment as instructed.

**Risk Score:** 2/10

**Insurance Quote:** $550/month for $1M coverage (standard low-risk rate)

**Assessment Details:**
- No active real-world incidents detected in the area
- Demo/test alerts properly filtered out as instructed
- Standard area risk profile maintained
- Normal emergency service activity levels
- Safe area with routine coverage requirements

**Location Status:** All systems normal, area classified as stable with standard risk levels."""
        
        # Check for specific real incident types in the real incidents section
        elif "harrassment" in real_incident_section.lower() or "petit larceny" in real_incident_section.lower():
            print("🚨 Low-level crime incidents detected")
            return f"""**⚠️ MODERATE RISK ASSESSMENT for {address}**

**Risk Summary:** Minor criminal incidents detected in the area including harassment and petit larceny reports. These represent typical urban activity requiring monitoring but not critical risk.

**Risk Score:** 4/10

**Insurance Quote:** $700/month for $1M coverage (moderate increase)

**Assessment Details:**
- Minor criminal activity reported in vicinity
- No major incidents or emergency situations
- Standard urban risk profile
- Normal law enforcement response
- Routine monitoring recommended"""
        
        # For other scenarios, return standard risk
        else:
            print("📊 Standard risk assessment - no specific risks detected")
            return f"""**📊 STANDARD RISK ASSESSMENT for {address}**

**Risk Summary:** Current analysis indicates NORMAL risk levels for the area. Standard location-based risk factors apply with no major incidents detected.

**Risk Score:** 3/10

**Insurance Quote:** $650/month for $1M coverage (standard rate)

**Assessment Details:**
- No critical incidents detected
- Standard area risk profile  
- Normal emergency service coverage
- Routine insurance coverage recommended
- No elevated threat indicators present"""
    
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
            is_demo = doc['metadata'].get('is_demo', False)
            
            # More robust demo/test detection
            demo_indicators = ['manual_injection', 'test', 'simulated', 'demo_test', 'demo_news', 'demo']
            content_demo_indicators = ['DEMO:', 'TEST:', 'SIMULATED:']
            
            # Check multiple indicators for demo content
            is_demo_content = (
                is_demo or  # Explicit demo flag
                any(indicator in source.lower() for indicator in demo_indicators) or
                any(indicator in content for indicator in content_demo_indicators) or
                source == 'demo_test'  # Explicit demo source
            )
            
            if is_demo_content:
                test_incidents.append(f"DEMO/TEST ALERT: {content}")
                print(f"📋 Classified as DEMO: {source} - {content[:50]}...")
            elif source in ['newsdata_io', 'nyc_open_data', 'chicago_open_data', 'usgs_earthquake', 'nws_weather']:
                real_incidents.append(f"REAL INCIDENT: {content}")
                print(f"🚨 Classified as REAL: {source} - {content[:50]}...")
            else:
                # For unknown source, be more conservative - default to real unless clearly demo
                if any(indicator in content.lower() for indicator in ['demo', 'test', 'simulated']):
                    test_incidents.append(f"DEMO ALERT: {content}")
                    print(f"📋 Classified as DEMO (content-based): {content[:50]}...")
                else:
                    real_incidents.append(f"INCIDENT: {content}")
                    print(f"⚠️ Classified as REAL (unknown source): {source} - {content[:50]}...")
        
        # Build separate context sections
        real_context = "\n".join(real_incidents) if real_incidents else "No recent real incidents detected in the area. Location appears stable."
        test_context = "\n".join(test_incidents) if test_incidents else "No demo/test alerts active."
        
        print(f"📊 Analysis summary: {len(real_incidents)} real incidents, {len(test_incidents)} demo alerts")
        
        # Only include test context if there are actual test incidents
        if test_incidents:
            prompt = f"""You are an AI Insurance Underwriting Assistant analyzing property risk.

PROPERTY: {address}

LOCATION ANALYSIS:
{location_context}

REAL INCIDENTS (use for risk calculation):
{real_context}

DEMO/TEST ALERTS (for reference only - COMPLETELY IGNORE in risk calculation):
{test_context}

CRITICAL INSTRUCTIONS: 
- Base your risk assessment ONLY on REAL incidents and location factors
- Demo/test alerts are simulated training data and MUST BE COMPLETELY IGNORED in scoring
- If only demo/test alerts are present with no real incidents, treat the area as SAFE with low risk
- Risk score should reflect ONLY actual verified incidents, not simulated scenarios

Assessment Requirements:
1. Risk summary focusing ONLY on verified real incidents and location factors
2. Risk score from 1 (safe) to 10 (critical) - based ONLY on REAL incidents
3. Monthly insurance quote for $1M coverage

Base calculation: ${self.base_cost} × (1 + {self.risk_multiplier} × risk_score)
Factor in location-specific risks but COMPLETELY IGNORE all demo/test scenarios."""
        else:
            # No test alerts - normal assessment
            prompt = f"""You are an AI Insurance Underwriting Assistant analyzing property risk.

PROPERTY: {address}

LOCATION ANALYSIS:
{location_context}

CURRENT REAL INCIDENTS:
{real_context}

Assessment Instructions:
- Base your assessment only on verified real-world data and location factors
- If no real incidents are present, consider this a SAFE area with normal risk levels
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
