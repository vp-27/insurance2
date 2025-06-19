#!/bin/bash
# Start command for Render deployment
python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1
