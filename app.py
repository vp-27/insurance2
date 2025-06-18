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
from data_manager import DataManager

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
data_manager = None

class AssessmentRequest(BaseModel):
    address: str
    query: str = "What are the current risks at this address?"

class TestAlertRequest(BaseModel):
    address: str
    alert_type: str = "fire"  # fire, flood, crime

class DemoModeRequest(BaseModel):
    address: str
    include_news: bool = True

@app.on_event("startup")
async def startup_event():
    """Initialize the application components"""
    global rag_pipeline, data_fetcher, data_manager
    
    print("Initializing Live Insurance Risk Assessment System...")
    
    # Initialize data manager first
    data_manager = DataManager()
    data_manager.start_data_management()
    
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
    print(f"🗂️ Data management active - maintaining {data_manager.max_active_files} active files")

@app.get("/", response_class=HTMLResponse)
async def get_frontend():
    """Serve the React frontend"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Sunny</title>
        <script src="https://unpkg.com/react@18/umd/react.production.min.js"></script>
        <script src="https://unpkg.com/react-dom@18/umd/react-dom.production.min.js"></script>
        <script src="https://unpkg.com/@babel/standalone/babel.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
            
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            :root {
                /* Modern Fintech Colors */
                --primary-brand: #2563eb;
                --primary-dark: #1d4ed8;
                --primary-light: #3b82f6;
                --accent-purple: #7c3aed;
                --accent-emerald: #059669;
                --accent-orange: #ea580c;
                
                /* Neutral Colors */
                --gray-50: #f9fafb;
                --gray-100: #f3f4f6;
                --gray-200: #e5e7eb;
                --gray-300: #d1d5db;
                --gray-400: #9ca3af;
                --gray-500: #6b7280;
                --gray-600: #4b5563;
                --gray-700: #374151;
                --gray-800: #1f2937;
                --gray-900: #111827;
                
                /* Status Colors */
                --success: #10b981;
                --warning: #f59e0b;
                --error: #ef4444;
                --info: #3b82f6;
                
                /* Backgrounds */
                --bg-primary: #ffffff;
                --bg-secondary: #f9fafb;
                --bg-tertiary: #f3f4f6;
                
                /* Shadows */
                --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
                --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
                --shadow-lg: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
                --shadow-xl: 0 20px 25px -5px rgba(0, 0, 0, 0.1), 0 10px 10px -5px rgba(0, 0, 0, 0.04);
                --shadow-2xl: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
                
                /* Gradients */
                --gradient-primary: linear-gradient(135deg, #2563eb 0%, #3b82f6 100%);
                --gradient-purple: linear-gradient(135deg, #7c3aed 0%, #a855f7 100%);
                --gradient-success: linear-gradient(135deg, #059669 0%, #10b981 100%);
                --gradient-warm: linear-gradient(135deg, #ea580c 0%, #f97316 100%);
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
                line-height: 1.6;
                color: var(--gray-800);
                background: var(--bg-secondary);
                font-weight: 400;
                -webkit-font-smoothing: antialiased;
                -moz-osx-font-smoothing: grayscale;
            }
            
            /* Clean Professional Background */
            .fintech-bg {
                background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
                min-height: 100vh;
                position: relative;
            }
            
            .fintech-bg::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: 
                    radial-gradient(circle at 25% 25%, rgba(59, 130, 246, 0.03) 0%, transparent 50%),
                    radial-gradient(circle at 75% 75%, rgba(124, 58, 237, 0.03) 0%, transparent 50%);
                pointer-events: none;
            }
            
            /* Professional Island Card Styling */
            .fintech-card {
                background: white;
                border-radius: 24px;
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.8);
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
            }
            
            .fintech-card::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 1px;
                background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.3), transparent);
            }
            
            .fintech-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.12);
                border-color: rgba(59, 130, 246, 0.2);
            }
            
            /* Professional Header */
            .fintech-header {
                background: white;
                border-radius: 28px;
                padding: 3rem;
                margin-bottom: 2rem;
                box-shadow: 0 12px 40px rgba(0, 0, 0, 0.08);
                border: 1px solid rgba(255, 255, 255, 0.8);
                position: relative;
                overflow: hidden;
                backdrop-filter: blur(10px);
            }
            
            .fintech-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, transparent, var(--primary-brand), transparent);
            }
            
            .brand-title {
                color: var(--gray-900);
                font-weight: 800;
                letter-spacing: -0.025em;
                margin-bottom: 0.5rem;
            }
            
            .brand-subtitle {
                color: var(--gray-600);
                font-weight: 500;
                letter-spacing: 0.01em;
            }
            
            .status-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 1rem;
                margin-top: 1.5rem;
            }
            
            .status-item {
                background: rgba(248, 250, 252, 0.8);
                border-radius: 20px;
                padding: 1.5rem;
                text-align: center;
                border: 1px solid rgba(226, 232, 240, 0.6);
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            
            .status-item:hover {
                background: white;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.08);
                transform: translateY(-4px);
                border-color: rgba(59, 130, 246, 0.2);
            }
            
            .status-value {
                font-weight: 700;
                font-size: 1.1rem;
                color: var(--primary-brand);
                margin-bottom: 0.25rem;
            }
            
            .status-label {
                font-size: 0.875rem;
                color: var(--gray-600);
                font-weight: 500;
            }
            
            /* Modern Input Styling */
            .fintech-input {
                background: rgba(255, 255, 255, 0.9);
                border: 2px solid rgba(226, 232, 240, 0.8);
                border-radius: 18px;
                padding: 1rem 1.25rem;
                font-size: 0.95rem;
                transition: all 0.3s ease;
                color: var(--gray-800);
                font-weight: 500;
                width: 100%;
                backdrop-filter: blur(10px);
            }
            
            .fintech-input:focus {
                outline: none;
                border-color: var(--primary-brand);
                box-shadow: 0 0 0 6px rgba(37, 99, 235, 0.08);
                background: white;
            }
            
            .fintech-input::placeholder {
                color: var(--gray-500);
                font-weight: 400;
            }
            
            /* Modern Button Styling */
            .fintech-button {
                background: var(--gradient-primary);
                border: none;
                border-radius: 18px;
                color: white;
                font-weight: 600;
                padding: 1rem 1.75rem;
                transition: all 0.3s ease;
                box-shadow: 0 6px 20px rgba(37, 99, 235, 0.25);
                cursor: pointer;
                font-size: 0.95rem;
            }
            
            .fintech-button:hover {
                transform: translateY(-2px);
                box-shadow: 0 12px 30px rgba(37, 99, 235, 0.35);
                background: var(--gradient-primary);
                filter: brightness(1.05);
            }
            
            .fintech-button:disabled {
                opacity: 0.5;
                transform: none;
                cursor: not-allowed;
                filter: none;
            }
            
            /* Metric Cards */
            .metric-card {
                transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
                overflow: hidden;
            }
            
            .metric-card:hover {
                transform: translateY(-8px);
                box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
            }
            
            /* Professional Risk Colors */
            .risk-low { 
                color: var(--success);
                font-weight: 600;
            }
            .risk-medium { 
                color: var(--warning);
                font-weight: 600;
            }
            .risk-high { 
                color: var(--accent-orange);
                font-weight: 600;
            }
            .risk-critical { 
                color: var(--error);
                font-weight: 600;
            }
            
            /* Loading States */
            .loading-skeleton {
                background: linear-gradient(90deg, var(--gray-100) 25%, var(--gray-200) 50%, var(--gray-100) 75%);
                background-size: 200% 100%;
                animation: skeletonLoading 1.5s infinite;
                border-radius: 8px;
            }
            
            @keyframes skeletonLoading {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            
            /* Animations */
            .fade-in {
                animation: fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1);
            }
            
            @keyframes fadeInUp {
                0% { opacity: 0; transform: translateY(20px); }
                100% { opacity: 1; transform: translateY(0); }
            }
            
            .live-indicator {
                animation: pulse 2s infinite;
                background: var(--gradient-success);
                color: white;
                font-weight: 600;
                padding: 0.5rem 1rem;
                border-radius: 20px;
                font-size: 0.875rem;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }
            
            /* Risk Score Visualization */
            .risk-score-circle {
                width: 120px;
                height: 120px;
                border-radius: 50%;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 2rem;
                font-weight: 800;
                color: white;
                position: relative;
                margin: 0 auto 1rem;
                box-shadow: var(--shadow-lg);
            }
            
            .risk-score-inner {
                width: 100px;
                height: 100px;
                border-radius: 50%;
                background: white;
                display: flex;
                align-items: center;
                justify-content: center;
                flex-direction: column;
                box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.1);
            }
            
            /* Progress Bar */
            .progress-container {
                width: 100%;
                height: 8px;
                background: var(--gray-200);
                border-radius: 4px;
                overflow: hidden;
                margin-top: 1rem;
            }
            
            .progress-fill {
                height: 100%;
                border-radius: 4px;
                transition: all 1s cubic-bezier(0.4, 0, 0.2, 1);
                position: relative;
            }
            
            /* Notifications */
            .fintech-notification {
                background: rgba(255, 255, 255, 0.95);
                border-radius: 20px;
                border: 1px solid rgba(226, 232, 240, 0.8);
                box-shadow: 0 20px 40px rgba(0, 0, 0, 0.15);
                animation: slideInRight 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                backdrop-filter: blur(15px);
            }
            
            @keyframes slideInRight {
                0% { transform: translateX(100%); opacity: 0; }
                100% { transform: translateX(0); opacity: 1; }
            }
            
            /* Demo Controls */
            .demo-button {
                border: none;
                border-radius: 20px;
                padding: 1.25rem;
                font-weight: 600;
                transition: all 0.3s ease;
                cursor: pointer;
                text-align: center;
                min-height: 110px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                align-items: center;
                gap: 0.75rem;
                box-shadow: 0 8px 25px rgba(0, 0, 0, 0.12);
            }
            
            .demo-button:hover {
                transform: translateY(-4px);
                box-shadow: 0 15px 35px rgba(0, 0, 0, 0.18);
            }
            
            .demo-button:disabled {
                opacity: 0.5;
                transform: none;
                cursor: not-allowed;
            }
            
            /* Utility Classes */
            .text-primary { color: var(--gray-900); }
            .text-secondary { color: var(--gray-600); }
            .text-muted { color: var(--gray-500); }
            .text-brand { color: var(--primary-brand); }
            
            .bg-gradient-primary { background: var(--gradient-primary); }
            .bg-gradient-purple { background: var(--gradient-purple); }
            .bg-gradient-success { background: var(--gradient-success); }
            .bg-gradient-warm { background: var(--gradient-warm); }
            
            /* Info Panels */
            .info-panel {
                background: rgba(248, 250, 252, 0.6);
                border: 1px solid rgba(226, 232, 240, 0.8);
                border-radius: 20px;
                padding: 1.5rem;
                backdrop-filter: blur(10px);
            }
            
            .info-panel-success {
                background: rgba(16, 185, 129, 0.03);
                border-color: rgba(16, 185, 129, 0.15);
            }
            
            .info-panel-warning {
                background: rgba(245, 158, 11, 0.03);
                border-color: rgba(245, 158, 11, 0.15);
            }
            
            .info-panel-error {
                background: rgba(239, 68, 68, 0.03);
                border-color: rgba(239, 68, 68, 0.15);
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
                const [loadingAnalysis, setLoadingAnalysis] = useState(false);
                const [autoRefresh, setAutoRefresh] = useState(false);
                const [lastUpdate, setLastUpdate] = useState(null);
                const [systemStats, setSystemStats] = useState(null);
                const [isLive, setIsLive] = useState(false);
                const [alertAnimating, setAlertAnimating] = useState(false);
                const [notification, setNotification] = useState(null);

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
                    setLoadingAnalysis(true);
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
                        setTimeout(() => {
                            setLoadingAnalysis(false);
                            setIsLive(false);
                        }, 1000);
                    }
                };

                const injectTestAlert = async (alertType) => {
                    if (!address.trim()) {
                        alert('Please enter an address first');
                        return;
                    }
                    
                    setAlertAnimating(true);
                    setLoadingAnalysis(true);
                    
                    // Show notification
                    const alertIcons = {
                        fire: '🔥',
                        flood: '🌊', 
                        crime: '🚨',
                        earthquake: '🏗️'
                    };
                    
                    setNotification({
                        message: `${alertIcons[alertType]} ${alertType.charAt(0).toUpperCase() + alertType.slice(1)} alert injected! Processing...`,
                        type: 'info'
                    });
                    
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
                            
                            setNotification({
                                message: `✅ Alert processed successfully! Updating risk assessment...`,
                                type: 'success'
                            });
                            
                            // Auto-refresh after 3 seconds
                            setTimeout(() => {
                                fetchAssessment();
                                setAlertAnimating(false);
                                setNotification(null);
                            }, 3000);
                        }
                    } catch (error) {
                        console.error('Error injecting alert:', error);
                        setAlertAnimating(false);
                        setLoadingAnalysis(false);
                        setNotification({
                            message: '❌ Error processing alert. Please try again.',
                            type: 'error'
                        });
                        setTimeout(() => setNotification(null), 3000);
                    }
                };

                const activateDemoMode = async (includeNews) => {
                    if (!address.trim()) {
                        alert('Please enter an address first');
                        return;
                    }
                    
                    setAlertAnimating(true);
                    setLoadingAnalysis(true);
                    
                    try {
                        const response = await fetch('/activate_demo_mode', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({ address, include_news: includeNews }),
                        });
                        
                        if (response.ok) {
                            const result = await response.json();
                            setIsLive(true);
                            
                            setNotification({
                                message: `✅ Demo mode activated! ${result.message}`,
                                type: 'success'
                            });
                            
                            // Auto-refresh after 3 seconds
                            setTimeout(() => {
                                fetchAssessment();
                                setAlertAnimating(false);
                                setNotification(null);
                            }, 3000);
                        }
                    } catch (error) {
                        console.error('Error activating demo mode:', error);
                        setAlertAnimating(false);
                        setLoadingAnalysis(false);
                        setNotification({
                            message: '❌ Error activating demo mode. Please try again.',
                            type: 'error'
                        });
                        setTimeout(() => setNotification(null), 3000);
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

                // Reset assessment when address changes
                useEffect(() => {
                    if (assessment && address.trim()) {
                        // Clear current assessment to show that new data is needed
                        setAssessment(null);
                        setLoadingAnalysis(true);
                        // Auto-clear loading state after a delay if no new assessment is fetched
                        const timeout = setTimeout(() => {
                            setLoadingAnalysis(false);
                        }, 5000);
                        return () => clearTimeout(timeout);
                    }
                }, [address]);

                // Initial stats fetch
                useEffect(() => {
                    fetchSystemStats();
                }, []);

                return (
                    <div className="min-h-screen fintech-bg">
                        {/* Professional Notification Toast */}
                        {notification && (
                            <div className={`fixed top-6 right-6 z-50 p-6 rounded-2xl shadow-2xl transition-all duration-300 fintech-notification ${
                                notification.type === 'success' ? 'border-l-4 border-green-500 bg-green-50' :
                                notification.type === 'error' ? 'border-l-4 border-red-500 bg-red-50' :
                                'border-l-4 border-blue-500 bg-blue-50'
                            } fade-in`}>
                                <div className="flex items-center">
                                    <div className={`p-2 rounded-full mr-3 ${
                                        notification.type === 'success' ? 'bg-green-100 text-green-600' :
                                        notification.type === 'error' ? 'bg-red-100 text-red-600' : 'bg-blue-100 text-blue-600'
                                    }`}>
                                        <i className={`${
                                            notification.type === 'success' ? 'fas fa-check' :
                                            notification.type === 'error' ? 'fas fa-times' : 'fas fa-info'
                                        } text-lg`}></i>
                                    </div>
                                    <span className="font-medium text-sm flex-1">{notification.message}</span>
                                    <button 
                                        onClick={() => setNotification(null)}
                                        className="ml-4 text-2xl font-bold opacity-70 hover:opacity-100 transition-opacity duration-200"
                                    >
                                        ×
                                    </button>
                                </div>
                            </div>
                        )}
                        
                        <div className="container mx-auto px-4 py-8 content-wrapper">
                            {/* Professional Fintech Header */}
                            <div className="fintech-header text-center">
                                <div className="flex items-center justify-center mb-6">
                                    <div>
                                        <h1 className="text-5xl font-bold brand-title mb-2">
                                            Sunny
                                        </h1>
                                        <div className="brand-subtitle text-xl">
                                            Modern Risk Intelligence Platform
                                        </div>
                                    </div>
                                    {isLive && (
                                        <div className="live-indicator ml-6">
                                            <i className="fas fa-circle text-xs mr-2"></i>LIVE
                                        </div>
                                    )}
                                </div>
                                <p className="text-lg text-secondary mb-6 max-w-4xl mx-auto">
                                    AI-powered risk assessment with real-time data analysis and comprehensive geographic intelligence. 
                                    Transforming insurance decisions through advanced technology.
                                </p>
                                {systemStats && (
                                    <div className="status-grid">
                                        <div className="status-item">
                                            <div className="status-value">
                                                <i className="fas fa-database mr-2 text-primary-brand"></i>
                                                {systemStats.documents_indexed}
                                            </div>
                                            <div className="status-label">Documents Indexed</div>
                                        </div>
                                        <div className="status-item">
                                            <div className="status-value">
                                                <i className="fas fa-vector-square mr-2 text-accent-purple"></i>
                                                {systemStats.vector_store_size}
                                            </div>
                                            <div className="status-label">Vector Embeddings</div>
                                        </div>
                                        <div className="status-item">
                                            <div className="status-value">
                                                <i className="fas fa-brain mr-2 text-accent-emerald"></i>
                                                AI
                                            </div>
                                            <div className="status-label">Powered Analysis</div>
                                        </div>
                                        <div className="status-item">
                                            <div className="status-value">
                                                <i className="fas fa-clock mr-2 text-accent-orange"></i>
                                                24/7
                                            </div>
                                            <div className="status-label">Monitoring</div>
                                        </div>
                                    </div>
                                )}
                            </div>

                            {/* Main Dashboard */}
                            <div className="grid lg:grid-cols-3 gap-6 mb-6">
                                {/* Professional Input Section */}
                                <div className="lg:col-span-1">
                                    <div className="fintech-card p-8">                                            <div className="flex items-center mb-6">
                                                <div className="bg-gradient-primary p-4 rounded-2xl mr-4 shadow-lg">
                                                    <i className="fas fa-search text-white text-xl"></i>
                                                </div>
                                                <div>
                                                    <h2 className="text-xl font-bold text-primary">Risk Assessment</h2>
                                                    <p className="text-secondary text-sm">Analyze property risk factors</p>
                                                </div>
                                            </div>
                                        
                                        <div className="space-y-6">
                                            <div>
                                                <label className="block text-sm font-semibold text-gray-700 mb-3">
                                                    <i className="fas fa-map-marker-alt mr-2 text-primary-brand"></i>
                                                    Property Address
                                                </label>
                                                <input
                                                    type="text"
                                                    className="fintech-input"
                                                    placeholder="Enter full property address..."
                                                    value={address}
                                                    onChange={(e) => setAddress(e.target.value)}
                                                />
                                            </div>

                                            <div>
                                                <label className="block text-sm font-semibold text-gray-700 mb-3">
                                                    <i className="fas fa-edit mr-2 text-primary-brand"></i>
                                                    Assessment Query
                                                </label>
                                                <textarea
                                                    className="fintech-input"
                                                    rows="4"
                                                    placeholder="Describe specific risk factors to analyze..."
                                                    value={query}
                                                    onChange={(e) => setQuery(e.target.value)}
                                                />
                                            </div>

                                            <div className="info-panel rounded-xl p-4">
                                                <label className="flex items-center cursor-pointer">
                                                    <input
                                                        type="checkbox"
                                                        checked={autoRefresh}
                                                        onChange={(e) => setAutoRefresh(e.target.checked)}
                                                        className="mr-3 w-5 h-5 text-blue-600 rounded focus:ring-blue-500"
                                                    />
                                                    <div>
                                                        <span className="text-sm font-semibold text-gray-700">
                                                            <i className="fas fa-sync-alt mr-2 text-blue-600"></i>
                                                            Auto-refresh data
                                                        </span>
                                                        <p className="text-xs text-muted mt-1">Update assessment every 30 seconds</p>
                                                    </div>
                                                </label>
                                            </div>

                                            <button
                                                onClick={fetchAssessment}
                                                disabled={loading || !address.trim()}
                                                className="fintech-button w-full font-semibold text-base py-4"
                                            >
                                                {loading ? (
                                                    <span className="flex items-center justify-center">
                                                        <svg className="animate-spin -ml-1 mr-3 h-6 w-6 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                        </svg>
                                                        <i className="fas fa-brain mr-2"></i>
                                                        Analyzing with AI...
                                                    </span>
                                                ) : (
                                                    <>
                                                        <i className="fas fa-search mr-3"></i>
                                                        Generate Risk Assessment
                                                    </>
                                                )}
                                            </button>
                                        </div>
                                    </div>
                                </div>

                                {/* Professional Risk Metrics */}
                                <div className="lg:col-span-2">
                                    <div className="grid md:grid-cols-2 gap-6 mb-6">
                                        {/* Risk Score Card */}
                                        <div className={`fintech-card p-8 metric-card ${alertAnimating ? 'alert-pulse' : ''}`}>
                                            <div className="flex items-center mb-6">
                                                <div className="bg-gradient-primary p-4 rounded-2xl mr-4 shadow-lg">
                                                    <i className="fas fa-chart-line text-white text-xl"></i>
                                                </div>
                                                <div>
                                                    <h3 className="text-xl font-bold text-primary">Risk Score</h3>
                                                    <p className="text-secondary text-sm">Comprehensive assessment</p>
                                                </div>
                                            </div>
                                            {loading ? (
                                                <div className="text-center py-8">
                                                    <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mb-4"></div>
                                                    <div className="space-y-2">
                                                        <div className="loading-skeleton h-4 w-24 mx-auto rounded"></div>
                                                        <div className="loading-skeleton h-3 w-16 mx-auto rounded"></div>
                                                    </div>
                                                </div>
                                            ) : assessment ? (
                                                <div className="fade-in text-center">
                                                    <div className={`risk-score-circle mx-auto mb-4 ${
                                                        assessment.risk_score <= 2 ? 'bg-gradient-success' :
                                                        assessment.risk_score <= 4 ? 'bg-gradient-to-r from-yellow-400 to-yellow-500' :
                                                        assessment.risk_score <= 7 ? 'bg-gradient-warm' : 'bg-gradient-to-r from-red-500 to-red-600'
                                                    }`}>
                                                        <div className="risk-score-inner">
                                                            <div className={`text-3xl font-black ${getRiskColor(assessment.risk_score)}`}>
                                                                {assessment.risk_score}
                                                            </div>
                                                            <div className="text-sm text-gray-600 font-medium">/ 10</div>
                                                        </div>
                                                    </div>
                                                    <div className={`text-lg font-bold ${getRiskColor(assessment.risk_score)} mb-2`}>
                                                        {getRiskLabel(assessment.risk_score)}
                                                    </div>
                                                    {/* Enhanced Risk Bar */}
                                                    <div className="progress-container">
                                                        <div 
                                                            className={`progress-fill ${
                                                                assessment.risk_score <= 2 ? 'bg-gradient-success' :
                                                                assessment.risk_score <= 4 ? 'bg-gradient-to-r from-yellow-400 to-yellow-500' :
                                                                assessment.risk_score <= 7 ? 'bg-gradient-warm' : 'bg-gradient-to-r from-red-500 to-red-600'
                                                            }`}
                                                            style={{width: `${(assessment.risk_score / 10) * 100}%`}}
                                                        ></div>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="text-center py-8">
                                                    <div className="bg-gray-100 rounded-full p-6 w-24 h-24 mx-auto mb-4 flex items-center justify-center">
                                                        <i className="fas fa-chart-line text-3xl text-gray-400"></i>
                                                    </div>
                                                    <p className="font-semibold text-gray-600">No assessment available</p>
                                                    <p className="text-sm text-muted">Enter an address to begin analysis</p>
                                                </div>
                                            )}
                                        </div>

                                        {/* Insurance Quote Card */}
                                        <div className={`fintech-card p-8 metric-card ${alertAnimating ? 'alert-pulse' : ''}`}>
                                            <div className="flex items-center mb-6">
                                                <div className="bg-gradient-success p-4 rounded-2xl mr-4 shadow-lg">
                                                    <i className="fas fa-dollar-sign text-white text-xl"></i>
                                                </div>
                                                <div>
                                                    <h3 className="text-xl font-bold text-primary">Monthly Premium</h3>
                                                    <p className="text-secondary text-sm">Estimated insurance cost</p>
                                                </div>
                                            </div>
                                            {loading ? (
                                                <div className="text-center py-8">
                                                    <div className="inline-block animate-spin rounded-full h-16 w-16 border-4 border-blue-500 border-t-transparent mb-4"></div>
                                                    <div className="space-y-2">
                                                        <div className="loading-skeleton h-4 w-28 mx-auto rounded"></div>
                                                        <div className="loading-skeleton h-3 w-20 mx-auto rounded"></div>
                                                    </div>
                                                </div>
                                            ) : assessment ? (
                                                <div className="fade-in text-center">
                                                    <div className="text-4xl font-black text-brand mb-2">
                                                        ${assessment.insurance_quote}
                                                    </div>
                                                    <div className="text-secondary mb-4">
                                                        <span className="font-medium">For $1M coverage per month</span>
                                                    </div>
                                                    <div className="info-panel rounded-xl p-4">
                                                        <div className="text-sm text-gray-700">
                                                            <div className="flex justify-between mb-2">
                                                                <span>Base Premium:</span>
                                                                <span className="font-semibold">$500</span>
                                                            </div>
                                                            <div className="flex justify-between">
                                                                <span>Risk Adjustment:</span>
                                                                <span className={`font-semibold ${(assessment.insurance_quote - 500) > 0 ? 'text-red-600' : 'text-green-600'}`}>
                                                                    {(assessment.insurance_quote - 500) > 0 ? '+' : ''}${assessment.insurance_quote - 500}
                                                                </span>
                                                            </div>
                                                        </div>
                                                    </div>
                                                </div>
                                            ) : (
                                                <div className="text-center py-8">
                                                    <div className="bg-gray-100 rounded-full p-6 w-24 h-24 mx-auto mb-4 flex items-center justify-center">
                                                        <i className="fas fa-calculator text-3xl text-gray-400"></i>
                                                    </div>
                                                    <p className="font-semibold text-gray-600">Quote unavailable</p>
                                                    <p className="text-sm text-muted">Complete assessment to calculate</p>
                                                </div>
                                            )}
                                        </div>
                                    </div>

                                    {/* Risk Analysis Summary */}
                                    <div className={`fintech-card p-8 ${alertAnimating ? 'alert-pulse' : ''}`}>
                                        <div className="flex justify-between items-center mb-6">                                        <div className="flex items-center">
                                            <div className="bg-gradient-purple p-4 rounded-2xl mr-4 shadow-lg">
                                                <i className="fas fa-chart-bar text-white text-xl"></i>
                                            </div>
                                            <div>
                                                <h3 className="text-xl font-bold text-primary">Risk Analysis Report</h3>
                                                <p className="text-secondary text-sm">AI-powered comprehensive assessment</p>
                                            </div>
                                        </div>
                                            {lastUpdate && !loadingAnalysis && (
                                                <div className="text-right">
                                                    <div className="text-sm text-gray-600 flex items-center justify-end font-medium">
                                                        <i className="fas fa-clock mr-2 text-blue-600"></i>
                                                        Updated: {lastUpdate}
                                                    </div>
                                                    <div className="text-xs font-semibold mt-1 text-green-600">
                                                        <i className="fas fa-check-circle mr-1"></i>
                                                        Real-time data
                                                    </div>
                                                </div>
                                            )}
                                        </div>

                                        {loadingAnalysis ? (
                                            <div className="space-y-6">
                                                <div className="flex items-center justify-center py-8">
                                                    <div className="relative">
                                                        <div className="animate-spin rounded-full h-20 w-20 border-4 border-gray-200"></div>
                                                        <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-600 border-t-transparent absolute top-0 left-0"></div>
                                                        <div className="absolute inset-0 flex items-center justify-center">
                                                            <i className="fas fa-brain text-blue-600 text-xl"></i>
                                                        </div>
                                                    </div>
                                                </div>
                                                <div className="text-center">
                                                    <div className="text-lg font-semibold text-blue-600 mb-2">
                                                        AI analyzing risk factors...
                                                    </div>
                                                    <div className="text-sm text-secondary mb-6">
                                                        Processing live data streams and geographic information
                                                    </div>
                                                </div>
                                                <div className="space-y-3">
                                                    <div className="loading-skeleton h-4 w-full rounded"></div>
                                                    <div className="loading-skeleton h-4 w-5/6 rounded"></div>
                                                    <div className="loading-skeleton h-4 w-4/5 rounded"></div>
                                                    <div className="loading-skeleton h-4 w-full rounded"></div>
                                                    <div className="loading-skeleton h-4 w-3/4 rounded"></div>
                                                </div>
                                            </div>
                                        ) : assessment ? (
                                            <div className="fade-in">
                                                <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-2xl p-6 mb-6 border-l-4 border-blue-500">
                                                    <div 
                                                        className="text-gray-800 leading-relaxed whitespace-pre-wrap"
                                                        style={{
                                                            maxHeight: 'none',
                                                            overflow: 'visible',
                                                            wordWrap: 'break-word',
                                                            lineHeight: '1.7',
                                                            fontSize: '0.95rem'
                                                        }}
                                                    >
                                                        {assessment.risk_summary}
                                                    </div>
                                                </div>
                                                <div className="grid md:grid-cols-3 gap-4 pt-4 border-t border-gray-200">
                                                    <div className="text-center p-4 bg-blue-50 rounded-xl">
                                                        <div className="text-blue-600 font-bold text-lg">{assessment.relevant_documents}</div>
                                                        <div className="text-xs text-blue-700 font-medium">Data Points</div>
                                                    </div>
                                                    <div className="text-center p-4 bg-orange-50 rounded-xl">
                                                        <div className="text-orange-600 font-bold text-lg">
                                                            {new Date(assessment.timestamp).toLocaleDateString()}
                                                        </div>
                                                        <div className="text-xs text-orange-700 font-medium">Assessment Date</div>
                                                    </div>
                                                    <div className="text-center p-4 bg-purple-50 rounded-xl">
                                                        <div className="text-purple-600 font-bold text-lg">AI-Powered</div>
                                                        <div className="text-xs text-purple-700 font-medium">Analysis Type</div>
                                                    </div>
                                                </div>
                                            </div>
                                        ) : (
                                            <div className="text-center py-16 text-gray-500">
                                                <div className="bg-gray-100 rounded-full p-8 w-32 h-32 mx-auto mb-6 flex items-center justify-center">
                                                    <i className="fas fa-search text-5xl text-gray-400"></i>
                                                </div>
                                                <h4 className="text-xl font-semibold mb-2">Ready for Analysis</h4>
                                                <p className="text-lg mb-2">Enter a property address and click "Generate Risk Assessment"</p>
                                                <p className="text-sm">to see AI-powered risk analysis with real-time data</p>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            </div>

                            {/* Professional Demo Controls */}
                            <div className="fintech-card p-8">
                                <div className="flex items-center justify-between mb-6">                                <div className="flex items-center">
                                    <div className="bg-gradient-purple p-5 rounded-2xl mr-4 shadow-lg">
                                        <i className="fas fa-flask text-white text-2xl"></i>
                                    </div>
                                    <div>
                                        <h2 className="text-2xl font-bold text-primary">Demo Controls & Test Scenarios</h2>
                                        <p className="text-secondary mt-1">Simulate real-world events and observe risk assessment updates</p>
                                    </div>
                                </div>
                                    {(alertAnimating || isLive) && (
                                        <div className="live-indicator">
                                            <i className="fas fa-satellite-dish text-xs mr-2"></i>PROCESSING
                                        </div>
                                    )}
                                </div>
                                
                                <div className="info-panel info-panel-success rounded-2xl p-6 mb-8">
                                    <div className="flex items-start">
                                        <div className="bg-blue-100 p-2 rounded-lg mr-4 mt-1">
                                            <i className="fas fa-info-circle text-blue-600 text-lg"></i>
                                        </div>
                                        <div>
                                            <h4 className="font-semibold text-lg mb-2 text-gray-800">How Demo Mode Works</h4>
                                            <p className="text-gray-700 leading-relaxed">
                                                Test alerts are injected into the live data feed and processed by the Pathway streaming engine. 
                                                The AI analyzes the new data and updates the risk score and insurance quote in real-time, 
                                                demonstrating how the system responds to actual emergency situations.
                                            </p>
                                        </div>
                                    </div>
                                </div>
                                
                                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
                                    <button
                                        onClick={() => injectTestAlert('fire')}
                                        disabled={alertAnimating}
                                        className={`demo-button bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 ${
                                            alertAnimating ? 'opacity-50 cursor-not-allowed' : ''
                                        }`}
                                    >
                                        {alertAnimating ? (
                                            <span className="flex items-center justify-center">
                                                <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                </svg>
                                                Processing...
                                            </span>
                                        ) : (
                                            <>
                                                <i className="fas fa-fire text-2xl mb-2"></i>
                                                <div className="font-bold">Fire Alert</div>
                                                <div className="text-xs opacity-90">Emergency Response</div>
                                            </>
                                        )}
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('flood')}
                                        disabled={alertAnimating}
                                        className={`demo-button bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 ${
                                            alertAnimating ? 'opacity-50 cursor-not-allowed' : ''
                                        }`}
                                    >
                                        {alertAnimating ? (
                                            <span className="flex items-center justify-center">
                                                <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                </svg>
                                                Processing...
                                            </span>
                                        ) : (
                                            <>
                                                <i className="fas fa-water text-2xl mb-2"></i>
                                                <div className="font-bold">Flood Alert</div>
                                                <div className="text-xs opacity-90">Weather Emergency</div>
                                            </>
                                        )}
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('crime')}
                                        disabled={alertAnimating}
                                        className={`demo-button bg-gradient-to-r from-yellow-500 to-orange-500 hover:from-yellow-600 hover:to-orange-600 ${
                                            alertAnimating ? 'opacity-50 cursor-not-allowed' : ''
                                        }`}
                                    >
                                        {alertAnimating ? (
                                            <span className="flex items-center justify-center">
                                                <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                </svg>
                                                Processing...
                                            </span>
                                        ) : (
                                            <>
                                                <i className="fas fa-exclamation-triangle text-2xl mb-2"></i>
                                                <div className="font-bold">Crime Alert</div>
                                                <div className="text-xs opacity-90">Security Incident</div>
                                            </>
                                        )}
                                    </button>
                                    <button
                                        onClick={() => injectTestAlert('earthquake')}
                                        disabled={alertAnimating}
                                        className={`demo-button bg-gradient-to-r from-purple-500 to-purple-600 hover:from-purple-600 hover:to-purple-700 ${
                                            alertAnimating ? 'opacity-50 cursor-not-allowed' : ''
                                        }`}
                                    >
                                        {alertAnimating ? (
                                            <span className="flex items-center justify-center">
                                                <svg className="animate-spin h-4 w-4 mr-2" fill="none" viewBox="0 0 24 24">
                                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                </svg>
                                                Processing...
                                            </span>
                                        ) : (
                                            <>
                                                <i className="fas fa-mountain text-2xl mb-2"></i>
                                                <div className="font-bold">Earthquake Alert</div>
                                                <div className="text-xs opacity-90">Seismic Activity</div>
                                            </>
                                        )}
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

@app.post("/activate_demo_mode")
async def activate_demo_mode(request: DemoModeRequest):
    """Activate full demo mode with simulated news and test alerts"""
    try:
        if not data_fetcher:
            raise HTTPException(status_code=500, detail="Data fetcher not initialized")
        
        # Inject demo news if requested
        if request.include_news:
            data_fetcher.inject_demo_news_alerts(request.address)
        
        # Inject a random test alert
        import random
        alert_types = ['fire', 'flood', 'crime', 'earthquake', 'traffic']
        random_alert = random.choice(alert_types)
        data_fetcher.inject_test_alert(request.address, random_alert)
        
        return {
            "success": True,
            "message": f"Demo mode activated for {request.address} with news and {random_alert} alert",
            "alert_type": random_alert,
            "news_included": request.include_news
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Demo mode activation error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    data_stats = data_manager.get_system_stats() if data_manager else {}
    
    return {
        "status": "healthy",
        "rag_pipeline": rag_pipeline is not None,
        "data_fetcher": data_fetcher is not None,
        "data_manager": data_manager is not None,
        "documents_indexed": len(rag_pipeline.documents) if rag_pipeline else 0,
        "active_files": data_stats.get('active_files', 0),
        "data_management_active": data_stats.get('management_active', False)
    }

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not rag_pipeline:
        raise HTTPException(status_code=500, detail="RAG pipeline not initialized")
    
    # Get data management stats
    data_stats = data_manager.get_system_stats() if data_manager else {}
    
    return {
        "documents_indexed": len(rag_pipeline.documents),
        "vector_store_size": len(rag_pipeline.embeddings),
        "data_directory": rag_pipeline.data_dir,
        "model_info": {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "llm_model": rag_pipeline.model_name,
            "api_client_active": rag_pipeline.openai_client is not None
        },
        "data_management": data_stats
    }

@app.get("/data/recent")
async def get_recent_data():
    """Get recent data summary"""
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    
    return data_manager.get_recent_data_summary(hours=2)

@app.post("/data/cleanup")
async def force_data_cleanup():
    """Force immediate data cleanup"""
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    
    result = data_manager.force_cleanup()
    return {
        "success": True,
        "message": "Data cleanup completed",
        "details": result
    }

@app.get("/data/stats")
async def get_data_stats():
    """Get detailed data management statistics"""
    if not data_manager:
        raise HTTPException(status_code=500, detail="Data manager not initialized")
    
    return data_manager.get_system_stats()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
