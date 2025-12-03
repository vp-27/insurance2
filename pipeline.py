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
        
        # Track processed files to avoid duplicates
        self.processed_files = set()
        
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
        import re
        # Improved tokenization to handle punctuation
        query_words = set(re.findall(r'\w+', query.lower()))
        results = []
        
        for i, (doc, metadata) in enumerate(zip(self.documents, self.document_metadata)):
            doc_words = set(re.findall(r'\w+', doc.lower()))
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
                # Skip if already processed
                if filename in self.processed_files:
                    continue
                    
                filepath = os.path.join(self.data_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        self.add_document(data)
                        self.processed_files.add(filename)
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
    
    def process_new_files(self):
        """Check for and process new files immediately"""
        try:
            if os.path.exists(self.data_dir):
                current_files = set(os.listdir(self.data_dir))
                new_files = current_files - self.processed_files
                
                count = 0
                for filename in new_files:
                    if filename.endswith('.json'):
                        filepath = os.path.join(self.data_dir, filename)
                        try:
                            with open(filepath, 'r') as f:
                                data = json.load(f)
                                self.add_document(data)
                                self.processed_files.add(filename)
                                print(f"Added new document: {filename}")
                                count += 1
                        except Exception as e:
                            print(f"Error processing new file {filename}: {e}")
                
                if count > 0:
                    print(f"Processed {count} new files")
                    return True
            return False
            
        except Exception as e:
            print(f"Error processing files: {e}")
            return False

    def monitor_new_files(self):
        """Monitor for new files and add them to the vector store"""
        import time
        
        while True:
            try:
                self.process_new_files()
                time.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error monitoring files: {e}")
                time.sleep(10)
    
    async def call_llm(self, prompt: str, location_factors: Dict[str, Any] = None) -> str:
        """Call LLM via OpenRouter API using OpenAI client"""
        try:
            if not self.openai_client:
                print("⚠️ No API client available, using intelligent fallback")
                return self.generate_intelligent_assessment(prompt, location_factors)
            
            # Verify API key is still valid before making request
            if not self.openrouter_api_key or self.openrouter_api_key == "your_openrouter_api_key_here":
                print("⚠️ Invalid API key detected, using intelligent fallback")
                return self.generate_intelligent_assessment(prompt, location_factors)
            
            print(f"🤖 Calling OpenRouter API with model: {self.model_name}")
            completion = self.openai_client.chat.completions.create(
                extra_headers={
                    "HTTP-Referer": "https://localhost:8000",
                    "X-Title": "Sunny Risk Studio",
                },
                model=self.model_name,
                messages=[
                    {"role": "system", "content": """You are Sunny, an expert AI insurance underwriter and risk analyst. 
Provide professional, actionable risk assessments that are:
- Data-driven and specific to the property location
- Clear about the distinction between baseline location risks and active incidents
- Professional in tone with specific recommendations
- Concise but comprehensive

Format your response with clear sections:
**Risk Summary**: Brief overview
**Key Risk Factors**: Bullet points of main risks
**Recommendations**: Actionable advice
**Risk Score**: X/10 with justification"""},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=1000
            )
            
            response = completion.choices[0].message.content
            print(f"✅ OpenRouter API response received ({len(response)} chars)")
            return response
            
        except Exception as e:
            error_msg = str(e)
            print(f"🚨 LLM API error: {e}")
            
            # For any API error, use intelligent fallback
            print("🔄 Using intelligent fallback assessment")
            return self.generate_intelligent_assessment(prompt, location_factors)
            
    async def call_llm_with_context(self, prompt: str, location_factors: Dict[str, Any]) -> str:
        """Wrapper for call_llm that handles fallback with context"""
        try:
            return await self.call_llm(prompt, location_factors)
        except Exception:
            return self.generate_intelligent_assessment(prompt, location_factors)
    
    def generate_intelligent_assessment(self, prompt: str, location_factors: Dict[str, Any] = None) -> str:
        """Generate intelligent risk assessment based on location factors and detected incidents"""
        import re
        
        # Extract address from prompt
        address_match = re.search(r'PROPERTY: ([^\n]+)', prompt)
        address = address_match.group(1) if address_match else "the specified property"
        
        # Parse incidents from prompt
        incidents_section = ""
        if "CURRENT INCIDENTS" in prompt:
            start = prompt.find("CURRENT INCIDENTS")
            end_patterns = ["Assessment Instructions:", "Assessment Requirements:", "Provide a comprehensive"]
            end_pos = len(prompt)
            for pattern in end_patterns:
                pos = prompt.find(pattern, start)
                if pos != -1:
                    end_pos = min(end_pos, pos)
            incidents_section = prompt[start:end_pos].lower()
        
        # Count and categorize incidents
        active_incidents = []
        incident_score_impact = 0
        
        if "fire" in incidents_section:
            count = incidents_section.count("fire")
            active_incidents.append(f"🔥 Fire incident{'s' if count > 1 else ''} detected ({count})")
            incident_score_impact += 2.5 * count
            
        if "flood" in incidents_section:
            count = incidents_section.count("flood")
            active_incidents.append(f"🌊 Flood warning{'s' if count > 1 else ''} active ({count})")
            incident_score_impact += 2.0 * count
            
        if "earthquake" in incidents_section:
            count = incidents_section.count("earthquake")
            active_incidents.append(f"🏚️ Seismic activity detected ({count})")
            incident_score_impact += 3.0 * count
            
        if "crime" in incidents_section or "theft" in incidents_section:
            count = incidents_section.count("crime") + incidents_section.count("theft")
            active_incidents.append(f"⚠️ Security incident{'s' if count > 1 else ''} reported ({count})")
            incident_score_impact += 1.5 * count
        
        # Get base risk from location factors
        base_risk = location_factors.get('base_risk_score', 3) if location_factors else 3
        location_desc = location_factors.get('location_description', 'standard area') if location_factors else 'standard area'
        primary_risks = location_factors.get('primary_risks', []) if location_factors else []
        
        # Calculate final risk score
        final_risk = min(10, max(1, round(base_risk + incident_score_impact)))
        
        # Calculate premium
        base_premium = 500
        premium = base_premium * (1 + 0.1 * final_risk)
        
        # Determine risk level and status
        if final_risk <= 3:
            risk_level = "LOW"
            risk_emoji = "✅"
            recommendation = "Standard coverage recommended. Property is in a relatively low-risk area."
        elif final_risk <= 5:
            risk_level = "MODERATE"
            risk_emoji = "⚠️"
            recommendation = "Standard coverage with consideration for specific regional risks. Monitor conditions."
        elif final_risk <= 7:
            risk_level = "ELEVATED"
            risk_emoji = "🟠"
            recommendation = "Enhanced coverage recommended. Consider additional riders for identified risk factors."
        else:
            risk_level = "HIGH"
            risk_emoji = "🚨"
            recommendation = "Comprehensive coverage essential. Immediate risk mitigation measures advised."
        
        # Build the assessment response
        response_parts = [
            f"**{risk_emoji} RISK ASSESSMENT FOR {address}**",
            "",
            f"**Risk Summary**",
            f"{'Active incidents detected requiring immediate attention.' if active_incidents else 'No active incidents. Assessment based on location risk profile.'}",
            f"Property is located in a {location_desc}.",
            "",
        ]
        
        # Add active incidents section
        if active_incidents:
            response_parts.append("**Active Incidents**")
            for incident in active_incidents:
                response_parts.append(f"• {incident}")
            response_parts.append("")
        
        # Add location risk factors
        response_parts.append("**Location Risk Factors**")
        if primary_risks:
            for risk in primary_risks:
                response_parts.append(f"• {risk.title()}")
        else:
            response_parts.append("• Standard property risks for area type")
        response_parts.append(f"• Base location risk: {base_risk:.1f}/10")
        response_parts.append("")
        
        # Add risk score and quote
        response_parts.extend([
            f"**Risk Score: {final_risk}/10** ({risk_level})",
            "",
            f"**Estimated Monthly Premium: ${premium:.0f}**",
            f"(For $1M coverage)",
            "",
            f"**Recommendation**",
            recommendation
        ])
        
        return "\n".join(response_parts)
    
    async def query_rag(self, address: str, query: str, lat: float = None, lon: float = None) -> Dict[str, Any]:
        """Main RAG query function with location-specific analysis"""
        
        # Get location-specific risk factors
        location_factors = self.location_analyzer.analyze_location_risk_factors(address, lat, lon)
        
        # If validation failed (should be caught by app.py, but double check)
        if not location_factors:
            raise ValueError("Invalid address or location data")
        
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
        
        # Extract key location identifiers from the address for filtering
        address_lower = address.lower()
        address_parts = set()
        
        # Extract city, state, street names, etc.
        import re
        # Common patterns: "Street Name, City, State" or "Number Street Name, City, State"
        if ',' in address:
            parts = [part.strip() for part in address.split(',')]
            # Use more specific parts (avoid generic words)
            for part in parts:
                if len(part.strip()) > 4 and part.strip().lower() not in ['street', 'avenue', 'lane', 'road', 'dr', 'ave']:
                    address_parts.add(part.strip().lower())
        
        # Extract street name and number more precisely
        street_match = re.match(r'(\d+)\s+(.+?)(?:\s+(?:st|street|ave|avenue|lane|ln|road|rd|dr|drive))?(?:\s*,|$)', address, re.IGNORECASE)
        if street_match:
            number, street_name = street_match.groups()
            if len(street_name.strip()) > 3:
                address_parts.add(street_name.strip().lower())
            # Only use street number if it's unique enough (3+ digits)
            if len(number) >= 3:
                address_parts.add(number)
        
        # Remove generic words that cause false matches
        generic_words = {'the', 'and', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'street', 'avenue', 'lane', 'road', 'drive'}
        address_parts = {part for part in address_parts if part not in generic_words and len(part) > 2}
        
        print(f"🔍 Address filtering for: {address}")
        print(f"    Looking for specific parts: {address_parts}")
        
        for doc in relevant_docs:
            source = doc['metadata'].get('source', '')
            content = doc['content']
            is_demo = doc['metadata'].get('is_demo', False)
            doc_location = doc['metadata'].get('location', '').lower()
            
            # Check if this incident is relevant to the specific address
            is_location_relevant = False
            
            if is_demo:
                # For demo alerts, check if the content mentions the specific address
                content_lower = content.lower()
                if any(part in content_lower for part in address_parts if len(part) > 3):
                    is_location_relevant = True
                    print(f"📋 Demo incident matches address: {content[:50]}...")
                else:
                    print(f"🚫 Demo incident filtered out (different location): {content[:50]}...")
                    continue
            else:
                # For real incidents, relax filtering for regional types
                doc_type = doc['metadata'].get('type', '').lower()
                is_regional = doc_type in ['weather', 'earthquake', 'infrastructure']
                
                if is_regional:
                    # Include regional alerts if they are recent (fetched by our localized fetcher)
                    is_location_relevant = True
                    print(f"🌍 Regional incident included: {doc_type} - {content[:50]}...")
                elif any(part in doc_location for part in address_parts if len(part) > 3):
                    is_location_relevant = True
                elif any(part in content.lower() for part in address_parts if len(part) > 4):
                    is_location_relevant = True
                else:
                    print(f"🚫 Real incident filtered out (different location): {content[:50]}...")
                    continue
            
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
        
        # Combine real and test incidents for a unified risk assessment
        all_incidents = real_incidents + test_incidents
        
        # Clean up prefixes for a unified view, making them all "ALERT"
        all_incidents_cleaned = []
        for incident in all_incidents:
            # Remove existing prefixes and add a standard one
            content = incident.split(": ", 1)[-1]
            all_incidents_cleaned.append(f"ALERT: {content}")

        incident_context = "\n".join(all_incidents_cleaned) if all_incidents_cleaned else "No recent incidents detected in the area. Location appears stable."

        print(f"📊 Analysis summary: {len(real_incidents)} real incidents, {len(test_incidents)} demo alerts. Combined for assessment.")

        # A single, unified prompt that treats all incidents (real or demo) as valid for risk calculation
        prompt = f"""You are an AI Insurance Underwriting Assistant analyzing property risk.

PROPERTY: {address}

LOCATION ANALYSIS:
{location_context}

CURRENT INCIDENTS (includes real-time data and active demo events):
{incident_context}

Assessment Instructions:
- Base your assessment on ALL available data: real-world incidents, active demo events, and location factors.
- Treat active demo events (e.g., fire, flood) as if they are real events for this risk assessment. Their presence signifies an elevated risk scenario.
- The presence of multiple incidents (real or demo) should cumulatively increase the perceived risk and the final risk score.
- If no incidents are present, assess risk based on location factors alone.

Provide a comprehensive risk assessment including:
1. A risk summary that clearly mentions the active demo events (if any) and their impact on the assessment.
2. A risk score from 1 (safe) to 10 (critical) based on the combined impact of all real and demo incidents.
3. A monthly insurance quote for $1M coverage, calculated from the final risk score.

Base calculation: ${self.base_cost} × (1 + {self.risk_multiplier} × risk_score)"""
        
        # Get LLM response
        # Use the new context-aware call
        llm_response = await self.call_llm_with_context(prompt, location_factors)
        
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

    def clear_document_store(self):
        """Clear the in-memory document store"""
        self.documents = []
        self.embeddings = []
        self.document_metadata = []
        self.processed_files = set()
        print("🧹 Cleared in-memory document store")

    def rebuild_index(self):
        """Rebuild the document index from scratch by reloading all files"""
        print("🔄 Rebuilding document index...")
        self.clear_document_store()
        self.load_existing_data()
        print(f"✅ Index rebuilt with {len(self.documents)} documents")

if __name__ == "__main__":
    from dotenv import load_dotenv
    import threading
    
    load_dotenv()
    
    print("="*60)
    print("RAG Pipeline Test Mode")
    print("="*60)
    print("Note: This is a test mode. For production use, run: python app.py")
    print("="*60)
    print()
    
    # Initialize pipeline
    pipeline = LiveRAGPipeline()
    
    # Load existing data
    pipeline.load_existing_data()
    print(f"✅ Loaded {len(pipeline.documents)} documents")
    
    # Test a simple query
    print("\n🔍 Testing basic functionality...")
    if pipeline.documents:
        print(f"   Document store: {len(pipeline.documents)} documents")
        print(f"   Embeddings: {len(pipeline.embeddings)} vectors")
        print("   ✅ Pipeline is functional")
    else:
        print("   ⚠️  No documents loaded yet")
    
    print("\n✅ Test complete. Pipeline is ready for use.")
    print("   To start the full application, run: python app.py")
