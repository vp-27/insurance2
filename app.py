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
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            .gradient-bg {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            .card-shadow {
                box-shadow: 0 10px 25px rgba(0,0,0,0.1);
                border: 1px solid rgba(255,255,255,0.1);
            }
            .pulse-animation {
                animation: pulse 2s infinite;
            }
            .live-indicator {
                animation: blink 1.5s infinite;
            }
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.5; }
            }
            @keyframes blink {
                0%, 50% { opacity: 1; }
                51%, 100% { opacity: 0.3; }
            }
            .risk-low { color: #10b981; }
            .risk-medium { color: #f59e0b; }
            .risk-high { color: #ef4444; }
            .risk-critical { color: #dc2626; }
            .glass-card {
                background: rgba(255, 255, 255, 0.95);
                backdrop-filter: blur(10px);
            }
            .metric-card {
                transition: all 0.3s ease;
            }
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 15px 35px rgba(0,0,0,0.15);
            }
        </style>
    </head>
    <body class="bg-gray-100">
        <div id="root"></div>
        
        <script type="text/babel">
            const { useState, useEffect } = React;

            function App() {
                const [address, setAddress] = useState('25 Columbus Dr, Jersey City, NJ');
                const [query, setQuery] = useState('What are the current risks at this address?');
                const [assessment, setAssessment] = useState(null);
                const [loading, setLoading] = useState(false);
                const [autoRefresh, setAutoRefresh] = useState(false);
                const [lastUpdate, setLastUpdate] = useState(null);
                const [systemStats, setSystemStats] = useState(null);
                const [isLive, setIsLive] = useState(false);

                const getRiskColor = (score) => {
                    if (score <= 2) return 'risk-low';
                    if (score <= 4) return 'risk-medium';
                    if (score <= 7) return 'risk-high';
                    return 'risk-critical';
                };

                const getRiskLabel = (score) => {
                    if (score <= 2) return 'Low Risk';
                    if (score <= 4) return 'Medium Risk';
                    if (score <= 7) return 'High Risk';
                    return 'Critical Risk';
                };

                const getRiskIcon = (score) => {
                    if (score <= 2) return '🟢';
                    if (score <= 4) return '🟡';
                    if (score <= 7) return '🟠';
                    return '🔴';
                };

                const fetchSystemStats = async () => {
                    try {
                        const response = await fetch('/stats');
                        if (response.ok) {
                            const data = await response.json();
                            setSystemStats(data);
                        }
                    } catch (error) {
                        console.error('Error fetching system stats:', error);
                    }
                };

                const fetchAssessment = async () => {
                    if (!address.trim()) return;
                    
                    setLoading(true);
                    setIsLive(true);
                    try {
                        const response = await fetch('/get_assessment', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ address, query }),
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            setAssessment(result.data);
                            setLastUpdate(new Date().toLocaleTimeString());
                            await fetchSystemStats();
                        } else {
                            console.error('Assessment request failed');
                        }
                    } catch (error) {
                        console.error('Error fetching assessment:', error);
                    } finally {
                        setLoading(false);
                        setTimeout(() => setIsLive(false), 3000);
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
                            // Show success animation
                            setIsLive(true);
                            setTimeout(() => setIsLive(false), 2000);
                            
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

                // Initial stats fetch
                useEffect(() => {
                    fetchSystemStats();
                }, []);

                return (
                    <div className="min-h-screen gradient-bg">
                        <div className="container mx-auto px-4 py-8">
                            {/* Header with Live Indicator */}
                            <div className="text-center mb-8">
                                <div className="flex items-center justify-center mb-4">
                                    <h1 className="text-4xl font-bold text-white mr-4">
                                        🏢 Live Insurance Risk & Quote Co-Pilot
                                    </h1>
                                    {isLive && (
                                        <div className="live-indicator bg-red-500 text-white px-3 py-1 rounded-full text-sm font-semibold">
                                            <i className="fas fa-circle text-xs mr-1"></i>LIVE
                                        </div>
                                    )}
                                </div>
                                <p className="text-white opacity-90 text-lg">
                                    Real-time risk assessment powered by live data streams and AI
                                </p>
                                {systemStats && (
                                    <div className="mt-4 flex justify-center space-x-6 text-white opacity-80 text-sm">
                                        <span><i className="fas fa-database mr-1"></i>{systemStats.documents_indexed} documents</span>
                                        <span><i className="fas fa-vector-square mr-1"></i>{systemStats.vector_store_size} embeddings</span>
                                        <span><i className="fas fa-brain mr-1"></i>AI-Powered Analysis</span>
                                    </div>
                                )}
                            </div>

                            {/* Main Dashboard */}
                            <div className="grid lg:grid-cols-3 gap-6 mb-6">
                                {/* Input Section */}
                                <div className="lg:col-span-1">
                                    <div className="glass-card rounded-lg card-shadow p-6">
                                        <h2 className="text-xl font-semibold mb-4 flex items-center">
                                            <i className="fas fa-map-marker-alt mr-2 text-blue-600"></i>
                                            Property Assessment
                                        </h2>
                                        
                                        <div className="mb-4">
                                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                                Property Address
                                            </label>
                                            <input
                                                type="text"
                                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
                                                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
                                                    className="mr-2 text-blue-600"
                                                />
                                                <span className="text-sm text-gray-700">
                                                    <i className="fas fa-sync-alt mr-1"></i>
                                                    Auto-refresh every 30 seconds
                                                </span>
                                            </label>
                                        </div>

                                        <button
                                            onClick={fetchAssessment}
                                            disabled={loading || !address.trim()}
                                            className="w-full bg-gradient-to-r from-blue-600 to-purple-600 text-white py-3 px-4 rounded-md hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 font-semibold"
                                        >
                                            {loading ? (
                                                <span className="flex items-center justify-center">
                                                    <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                    </svg>
                                                    <i className="fas fa-brain mr-2"></i>
                                                    Analyzing...
                                                </span>
                                            ) : (
                                                <>
                                                    <i className="fas fa-search mr-2"></i>
                                                    Get Live Assessment
                                                </>
                                            )}
                                        </button>
                                    </div>
                                </div>

                                {/* Risk Metrics */}
                                <div className="lg:col-span-2">
                                    <div className="grid md:grid-cols-2 gap-4 mb-4">
                                        {/* Risk Score Card */}
                                        <div className="glass-card rounded-lg card-shadow p-6 metric-card">
                                            <div className="flex items-center justify-between mb-3">
                                                <h3 className="font-semibold text-gray-700">Risk Score</h3>
                                                {assessment && (
                                                    <span className="text-2xl">{getRiskIcon(assessment.risk_score)}</span>
                                                )}
                                            </div>
                                            {assessment ? (
                                                <div>
                                                    <div className={`text-3xl font-bold ${getRiskColor(assessment.risk_score)} mb-1`}>
                                                        {assessment.risk_score}/10
                                                    </div>
                                                    <div className={`text-sm font-medium ${getRiskColor(assessment.risk_score)}`}>
                                                        {getRiskLabel(assessment.risk_score)}
                                                    </div>
                                                    {/* Risk Bar */}
                                                    <div className="mt-3 bg-gray-200 rounded-full h-2">
                                                        <div 
                                                            className={`h-2 rounded-full transition-all duration-500 ${
                                                                assessment.risk_score <= 2 ? 'bg-green-500' :
                                                                assessment.risk_score <= 4 ? 'bg-yellow-500' :
                                                                assessment.risk_score <= 7 ? 'bg-orange-500' : 'bg-red-500'
                                                            }`}
                                                            style={{width: `${(assessment.risk_score / 10) * 100}%`}}
                                                        ></div>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="text-gray-400 text-center py-4">
                                                    <i className="fas fa-chart-line text-3xl mb-2"></i>
                                                    <p>No assessment yet</p>
                                                </div>
                                            )}
                                        </div>

                                        {/* Insurance Quote Card */}
                                        <div className="glass-card rounded-lg card-shadow p-6 metric-card">
                                            <div className="flex items-center justify-between mb-3">
                                                <h3 className="font-semibold text-gray-700">Monthly Premium</h3>
                                                <i className="fas fa-dollar-sign text-green-600 text-xl"></i>
                                            </div>
                                            {assessment ? (
                                                <div>
                                                    <div className="text-3xl font-bold text-green-600 mb-1">
                                                        ${assessment.insurance_quote}
                                                    </div>
                                                    <div className="text-sm text-gray-600">
                                                        For $1M coverage
                                                    </div>
                                                    <div className="text-xs text-gray-500 mt-2">
                                                        Base: $500 + Risk Premium
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="text-gray-400 text-center py-4">
                                                    <i className="fas fa-calculator text-3xl mb-2"></i>
                                                    <p>Enter address to calculate</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Risk Summary */}
                                    <div className="glass-card rounded-lg card-shadow p-6">
                                        <div className="flex justify-between items-center mb-4">
                                            <h3 className="font-semibold text-gray-700 flex items-center">
                                                <i className="fas fa-file-alt mr-2"></i>
                                                Risk Analysis
                                            </h3>
                                            {lastUpdate && (
                                                <span className="text-sm text-gray-500 flex items-center">
                                                    <i className="fas fa-clock mr-1"></i>
                                                    Updated: {lastUpdate}
                                                </span>
                                            )}
                                        </div>

                                        {assessment ? (
                                            <div>
                                                <div className="text-sm text-gray-700 whitespace-pre-line mb-4">
                                                    {assessment.risk_summary}
                                                </div>
                                                <div className="flex items-center justify-between text-xs text-gray-500 pt-3 border-t border-gray-200">
                                                    <span>
                                                        <i className="fas fa-database mr-1"></i>
                                                        Based on {assessment.relevant_documents} data points
                                                    </span>
                                                    <span>
                                                        <i className="fas fa-clock mr-1"></i>
                                                        {new Date(assessment.timestamp).toLocaleString()}
                                                    </span>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="text-center py-8 text-gray-500">
                                                <i className="fas fa-search text-4xl mb-4 opacity-50"></i>
                                                <p className="text-lg">Enter a property address and click "Get Live Assessment"</p>
                                                <p>to see AI-powered risk analysis with real-time data</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Demo Controls */}
                            <div className="glass-card rounded-lg card-shadow p-6">
                                <h2 className="text-xl font-semibold mb-4 flex items-center">
                                    <i className="fas fa-flask mr-2 text-purple-600"></i>
                                    Demo Controls & Test Scenarios
                                </h2>
                                <p className="text-gray-600 mb-6">
                                    Inject test alerts to see how the system responds to real-time events and updates risk assessments instantly:
                                </p>
                                
                                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-3">
                                    <button
                                        onClick={() => injectTestAlert('fire')}
                                        className="bg-gradient-to-r from-red-500 to-red-600 text-white px-4 py-3 rounded-md hover:from-red-600 hover:to-red-700 transition-all duration-300 font-semibold shadow-lg"
                                    >
                                        <i className="fas fa-fire mr-2"></i>
                                        Fire Alert
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('flood')}
                                        className="bg-gradient-to-r from-blue-500 to-blue-600 text-white px-4 py-3 rounded-md hover:from-blue-600 hover:to-blue-700 transition-all duration-300 font-semibold shadow-lg"
                                    >
                                        <i className="fas fa-water mr-2"></i>
                                        Flood Alert
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('crime')}
                                        className="bg-gradient-to-r from-yellow-500 to-orange-500 text-white px-4 py-3 rounded-md hover:from-yellow-600 hover:to-orange-600 transition-all duration-300 font-semibold shadow-lg"
                                    >
                                        <i className="fas fa-exclamation-triangle mr-2"></i>
                                        Crime Alert
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('earthquake')}
                                        className="bg-gradient-to-r from-purple-500 to-purple-600 text-white px-4 py-3 rounded-md hover:from-purple-600 hover:to-purple-700 transition-all duration-300 font-semibold shadow-lg"
                                    >
                                        <i className="fas fa-mountain mr-2"></i>
                                        Earthquake Alert
                                    </button>
                                </div>
                                
                                <div className="mt-6 p-4 bg-blue-50 rounded-lg border border-blue-200">
                                    <div className="flex items-start">
                                        <i className="fas fa-info-circle text-blue-500 mr-2 mt-1"></i>
                                        <div className="text-sm text-blue-700">
                                            <strong>How it works:</strong> Test alerts are injected into the live data feed and processed by the Pathway streaming engine. 
                                            The AI will analyze the new data and update the risk score and insurance quote in real-time.
                                        </div>
                                    </div>
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
