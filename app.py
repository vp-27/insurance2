from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import asyncio
import threading
import os
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Import our custom modules
from pipeline import LiveRAGPipeline
from data_fetcher import DataFetcher

# Load environment variables
load_dotenv()

app = FastAPI(title="Live Insurance Risk & Quote Co-Pilot", version="1.0.0")

# Enable CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
rag_pipeline = None
data_fetcher = None

class AssessmentRequest(BaseModel):
    address: str
    query: str = "What are the current risks at this address?"

class TestAlertRequest(BaseModel):
    address: str
    alert_type: str = "fire"  # fire, flood, crime

@app.on_event("startup")
async def startup_event():
    """Initialize the application components"""
    global rag_pipeline, data_fetcher
    
    print("Initializing Live Insurance Risk Assessment System...")
    
    # Initialize RAG pipeline
    rag_pipeline = LiveRAGPipeline()
    rag_pipeline.load_existing_data()
    
    # Start file monitoring in background
    monitor_thread = threading.Thread(target=rag_pipeline.monitor_new_files, daemon=True)
    monitor_thread.start()
    
    # Initialize data fetcher
    data_fetcher = DataFetcher()
    
    # Start scheduled data fetching
    data_fetcher.start_scheduled_fetching()
    
    print("System initialized successfully!")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the React frontend"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Live Insurance Risk & Quote Co-Pilot</title>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <style>
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .card-shadow {
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            }
            .pulse-animation {
                animation: pulse 2s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            .risk-low { color: #10b981; }
            .risk-medium { color: #f59e0b; }
            .risk-high { color: #ef4444; }
        </style>
    </head>
    <body class="bg-gray-100">
        <div id="root"></div>
        
        <script type="text/babel">
            const { useState, useEffect } = React;

            function App() {
                const [address, setAddress] = useState('');
                const [query, setQuery] = useState('What are the current risks at this address?');
                const [assessment, setAssessment] = useState(null);
                const [loading, setLoading] = useState(false);
                const [autoRefresh, setAutoRefresh] = useState(false);
                const [lastUpdate, setLastUpdate] = useState(null);

                const getRiskColor = (score) => {
                    if (score <= 3) return 'risk-low';
                    if (score <= 6) return 'risk-medium';
                    return 'risk-high';
                };

                const getRiskLabel = (score) => {
                    if (score <= 3) return 'Low Risk';
                    if (score <= 6) return 'Medium Risk';
                    return 'High Risk';
                };

                const fetchAssessment = async () => {
                    if (!address.trim()) return;
                    
                    setLoading(true);
                    try {
                        const response = await fetch('/get_assessment', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ address, query }),
                        });
                        
                        if (response.ok) {
                            const data = await response.json();
                            setAssessment(data);
                            setLastUpdate(new Date().toLocaleTimeString());
                        } else {
                            console.error('Assessment request failed');
                        }
                    } catch (error) {
                        console.error('Error fetching assessment:', error);
                    } finally {
                        setLoading(false);
                    }
                };

                const injectTestAlert = async (alertType) => {
                    if (!address.trim()) {
                        alert('Please enter an address first');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/inject_test_alert', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ address, alert_type: alertType }),
                        });
                        
                        if (response.ok) {
                            alert(`${alertType} alert injected! Wait a few seconds and refresh to see the updated risk assessment.`);
                            // Auto-refresh after 3 seconds
                            setTimeout(() => {
                                fetchAssessment();
                            }, 3000);
                        }
                    } catch (error) {
                        console.error('Error injecting alert:', error);
                    }
                };

                // Auto-refresh functionality
                useEffect(() => {
                    let interval;
                    if (autoRefresh && address.trim()) {
                        interval = setInterval(() => {
                            fetchAssessment();
                        }, 30000); // Refresh every 30 seconds
                    }
                    return () => {
                        if (interval) clearInterval(interval);
                    };
                }, [autoRefresh, address]);

                return (
                    <div className="min-h-screen gradient-bg">
                        <div className="container mx-auto px-4 py-8">
                            {/* Header */}
                            <div className="text-center mb-8">
                                <h1 className="text-4xl font-bold text-white mb-2">
                                    🏢 Live Insurance Risk & Quote Co-Pilot
                                </h1>
                                <p className="text-white opacity-90">
                                    Real-time risk assessment powered by live data and AI
                                </p>
                            </div>

                            {/* Main Card */}
                            <div className="bg-white rounded-lg card-shadow p-6 mb-6">
                                <div className="grid md:grid-cols-2 gap-6">
                                    {/* Input Section */}
                                    <div>
                                        <h2 className="text-xl font-semibold mb-4">Property Assessment</h2>
                                        
                                        <div className="mb-4">
                                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                                Property Address
                                            </label>
                                            <input
                                                type="text"
                                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                                placeholder="e.g., 25 Columbus Dr, Jersey City, NJ"
                                                value={address}
                                                onChange={(e) => setAddress(e.target.value)}
                                            />
                                        </div>

                                        <div className="mb-4">
                                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                                Assessment Query
                                            </label>
                                            <textarea
                                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                                                rows="3"
                                                value={query}
                                                onChange={(e) => setQuery(e.target.value)}
                                            />
                                        </div>

                                        <div className="mb-4">
                                            <label className="flex items-center">
                                                <input
                                                    type="checkbox"
                                                    checked={autoRefresh}
                                                    onChange={(e) => setAutoRefresh(e.target.checked)}
                                                    className="mr-2"
                                                />
                                                <span className="text-sm text-gray-700">Auto-refresh every 30 seconds</span>
                                            </label>
                                        </div>

                                        <button
                                            onClick={fetchAssessment}
                                            disabled={loading || !address.trim()}
                                            className="w-full bg-blue-600 text-white py-2 px-4 rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
                                        >
                                            {loading ? (
                                                <span className="flex items-center justify-center">
                                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                    </svg>
                                                    Analyzing...
                                                </span>
                                            ) : (
                                                'Get Live Assessment'
                                            )}
                                        </button>
                                    </div>

                                    {/* Results Section */}
                                    <div>
                                        <div className="flex justify-between items-center mb-4">
                                            <h2 className="text-xl font-semibold">Risk Assessment</h2>
                                            {lastUpdate && (
                                                <span className="text-sm text-gray-500">
                                                    Last updated: {lastUpdate}
                                                </span>
                                            )}
                                        </div>

                                        {assessment ? (
                                            <div className="space-y-4">
                                                {/* Risk Score */}
                                                <div className="bg-gray-50 p-4 rounded-lg">
                                                    <h3 className="font-semibold mb-2">Risk Score</h3>
                                                    <div className={`text-2xl font-bold ${getRiskColor(assessment.risk_score)}`}>
                                                        {assessment.risk_score}/10 - {getRiskLabel(assessment.risk_score)}
                                                    </div>
                                                </div>

                                                {/* Insurance Quote */}
                                                <div className="bg-gray-50 p-4 rounded-lg">
                                                    <h3 className="font-semibold mb-2">Monthly Premium Estimate</h3>
                                                    <div className="text-2xl font-bold text-green-600">
                                                        ${assessment.insurance_quote}/month
                                                    </div>
                                                    <p className="text-sm text-gray-600">For $1M coverage</p>
                                                </div>

                                                {/* Risk Summary */}
                                                <div className="bg-gray-50 p-4 rounded-lg">
                                                    <h3 className="font-semibold mb-2">Risk Summary</h3>
                                                    <div className="text-sm text-gray-700 whitespace-pre-line">
                                                        {assessment.risk_summary}
                                                    </div>
                                                </div>

                                                {/* Metadata */}
                                                <div className="text-xs text-gray-500">
                                                    Based on {assessment.relevant_documents} recent data points
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="text-center py-8 text-gray-500">
                                                Enter a property address and click "Get Live Assessment" to see risk analysis
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Demo Controls */}
                            <div className="bg-white rounded-lg card-shadow p-6">
                                <h2 className="text-xl font-semibold mb-4">Demo Controls</h2>
                                <p className="text-gray-600 mb-4">
                                    Inject test alerts to see how the system responds to real-time events:
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    <button
                                        onClick={() => injectTestAlert('fire')}
                                        className="bg-red-600 text-white px-4 py-2 rounded-md hover:bg-red-700"
                                    >
                                        🔥 Inject Fire Alert
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('flood')}
                                        className="bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700"
                                    >
                                        🌊 Inject Flood Alert
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('crime')}
                                        className="bg-yellow-600 text-white px-4 py-2 rounded-md hover:bg-yellow-700"
                                    >
                                        🚨 Inject Crime Alert
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                );
            }

            ReactDOM.render(<App />, document.getElementById('root'));
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/get_assessment")
async def get_assessment(request: AssessmentRequest) -> Dict[str, Any]:
    """Get live risk assessment for a property"""
    try:
        if not rag_pipeline:
            raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
        
        # Perform RAG query
        result = await rag_pipeline.query_rag(request.address, request.query)
        
        return {
            "success": True,
            "data": result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment error: {str(e)}")

@app.post("/inject_test_alert")
async def inject_test_alert(request: TestAlertRequest):
    """Inject a test alert for demo purposes"""
    try:
        if not data_fetcher:
            raise HTTPException(status_code=500, detail="Data fetcher not initialized")
        
        # Inject the test alert
        data_fetcher.inject_test_alert(request.address, request.alert_type)
        
        return {
            "success": True,
            "message": f"{request.alert_type} alert injected for {request.address}"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alert injection error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "rag_pipeline": rag_pipeline is not None,
        "data_fetcher": data_fetcher is not None,
        "documents_indexed": len(rag_pipeline.documents) if rag_pipeline else 0
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    return {
        "documents_indexed": len(rag_pipeline.documents),
        "vector_store_size": len(rag_pipeline.embeddings),
        "data_directory": rag_pipeline.data_dir,
        "model_info": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": rag_pipeline.model_name
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
