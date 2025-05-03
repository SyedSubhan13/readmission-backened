"""
Main entry point for the backend Vercel serverless functions.
"""

import uvicorn
import os
from dotenv import load_dotenv

# Load environment variables if .env file exists
load_dotenv()

# Import the app from app.py
from app import app

# This is for running locally
if __name__ == "__main__":
    # Get port from environment variable (for Vercel or local development)
    port = int(os.getenv("PORT", 8000))
    
    # Run the app with uvicorn
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=port,
        reload=True  # Enable auto-reload for development
    ) 