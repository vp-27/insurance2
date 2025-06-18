import pathway as pw
import asyncio
import json
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
import requests
from sentence_transformers import SentenceTransformer
import numpy as np

class LiveRAGPipeline:
    def __init__(self, data_dir: str = "./live_data_feed"):
        self.data_dir = data_dir
        self.openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
        self.openrouter_base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self.model_name = os.getenv("MODEL_NAME", "mistralai/mixtral-8x7b-instruct")
        self.base_cost = float(os.getenv("BASE_INSURANCE_COST", "500"))
        self.risk_multiplier = float(os.getenv("RISK_MULTIPLIER", "0.1"))
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Store for vector index
        self.documents = []
        self.embeddings = []
        self.document_metadata = []
        
        # Initialize Pathway pipeline
        self.setup_pathway_pipeline()
    
    def setup_pathway_pipeline(self):
        """Setup Pathway streaming pipeline"""
        try:
            # Create input table from JSON files in streaming mode
            self.input_table = pw.io.json.read(
                self.data_dir,
                schema=pw.schema_from_types(
                    source=str,
                    timestamp=str,
                    location=str,
                    content=str,
                    type=str
                ),
                mode="streaming",
                with_metadata=True
            )
            
            # Transform and process the data
            self.processed_table = self.input_table.select(
                source=self.input_table.source,
                timestamp=self.input_table.timestamp,
                location=self.input_table.location,
                content=self.input_table.content,
                type=self.input_table.type,
                processed_time=pw.this.timestamp
            )
            
            print("Pathway pipeline initialized successfully")
            
        except Exception as e:
            print(f"Error setting up Pathway pipeline: {e}")
            # Fallback to manual file monitoring
            self.input_table = None
            self.processed_table = None
    
    def embed_text(self, text: str) -> np.ndarray:
        """Generate embeddings for text"""
        return self.embedding_model.encode(text)
    
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
        """Call LLM via OpenRouter API"""
        try:
            if not self.openrouter_api_key or self.openrouter_api_key == "your_openrouter_api_key_here":
                # Return a simulated response if no API key
                return self.simulate_llm_response(prompt)
            
            headers = {
                "Authorization": f"Bearer {self.openrouter_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 500
            }
            
            response = requests.post(
                f"{self.openrouter_base_url}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except Exception as e:
            print(f"LLM API error: {e}")
            return self.simulate_llm_response(prompt)
    
    def simulate_llm_response(self, prompt: str) -> str:
        """Simulate LLM response when API is not available"""
        # Extract address from prompt for personalized response
        import re
        address_match = re.search(r'at ([^?]+)\?', prompt)
        address = address_match.group(1) if address_match else "the specified location"
        
        # Simple rule-based response generation
        if "fire" in prompt.lower():
            return f"""**Risk Assessment for {address}**

**Risk Summary:** Critical fire incident detected in the immediate vicinity. This represents a significant operational risk that could affect property values, safety protocols, and insurance liability.

**Risk Score:** 9/10

**Insurance Quote:** Based on the elevated risk profile, the estimated monthly premium for $1M coverage is $950.

**Factors Considered:**
- Active fire emergency in proximity
- Potential for smoke and water damage
- Increased emergency service activity
- Temporary accessibility restrictions"""

        elif "flood" in prompt.lower():
            return f"""**Risk Assessment for {address}**

**Risk Summary:** Flood warning or water-related incident detected. This poses significant risk to property foundations, electrical systems, and overall structural integrity.

**Risk Score:** 7/10

**Insurance Quote:** The estimated monthly premium for $1M coverage is $850, reflecting increased flood risk exposure.

**Factors Considered:**
- Water damage potential
- Infrastructure vulnerability
- Emergency response requirements
- Potential business interruption"""

        elif "crime" in prompt.lower():
            return f"""**Risk Assessment for {address}**

**Risk Summary:** Increased security incidents reported in the area. This may indicate higher theft risk and potential safety concerns for the property.

**Risk Score:** 5/10

**Insurance Quote:** The estimated monthly premium for $1M coverage is $750, accounting for elevated security risks.

**Factors Considered:**
- Local crime activity
- Security system requirements
- Law enforcement response times
- Property protection measures needed"""

        else:
            return f"""**Risk Assessment for {address}**

**Risk Summary:** Current analysis shows normal risk levels for the area. No significant incidents or weather alerts detected that would materially impact the property's risk profile.

**Risk Score:** 2/10

**Insurance Quote:** The estimated monthly premium for $1M coverage is $600, reflecting standard risk exposure.

**Factors Considered:**
- No active weather alerts
- Normal emergency service activity
- Standard area risk profile
- Baseline coverage requirements"""
    
    async def query_rag(self, address: str, query: str) -> Dict[str, Any]:
        """Main RAG query function"""
        # Search for relevant documents
        search_query = f"{address} {query}"
        relevant_docs = self.similarity_search(search_query, top_k=5)
        
        # Build context from relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"Document {i}:\n{doc['content']}\n")
        
        context = "\n---\n".join(context_parts) if context_parts else "No relevant incidents found in recent data."
        
        # Build prompt
        prompt = f"""You are an AI Insurance Underwriting Assistant. Using only the live context below:

1. Summarize any current risks at {address}
2. Assign a risk score from 1 (safe) to 10 (critical)
3. Estimate a monthly insurance quote for $1M coverage, using:
   base = ${self.base_cost}, quote = base × (1 + {self.risk_multiplier} × risk_score)

Context:
---
{context}
---

Provide a structured response with clear risk summary, numerical risk score, and calculated insurance quote."""
        
        # Get LLM response
        llm_response = await self.call_llm(prompt)
        
        # Extract risk score from response (simple parsing)
        risk_score = self.extract_risk_score(llm_response)
        
        # Calculate quote
        quote = self.base_cost * (1 + self.risk_multiplier * risk_score)
        
        return {
            'risk_summary': llm_response,
            'risk_score': risk_score,
            'insurance_quote': round(quote, 2),
            'relevant_documents': len(relevant_docs),
            'timestamp': datetime.now().isoformat()
        }
    
    def extract_risk_score(self, response: str) -> int:
        """Extract risk score from LLM response"""
        import re
        
        # Look for patterns like "Risk Score: 7/10" or "7 out of 10"
        patterns = [
            r'Risk Score:?\s*(\d+)(?:/10)?',
            r'risk score:?\s*(\d+)(?:/10)?',
            r'Score:?\s*(\d+)(?:/10)?',
            r'(\d+)\s*(?:out of|/)\s*10'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                score = int(match.group(1))
                return min(max(score, 1), 10)  # Clamp between 1-10
        
        # Default to moderate risk if no score found
        return 5

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
