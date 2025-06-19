#!/usr/bin/env python3
"""
Start script for Render deployment
Render requires binding to 0.0.0.0 and using the PORT environment variable
"""
import uvicorn
import os

if __name__ == "__main__":
    # Render provides PORT environment variable
    port = int(os.environ.get("PORT", 8000))
    
    # For Render deployment, we need to bind to 0.0.0.0
    uvicorn.run(
        "app:app",
        host="0.0.0.0", 
        port=port,
        log_level="info",
        reload=False  # Disable reload in production
    )
