"""
FastAPI Application Entry Point

This module defines the main FastAPI application that serves both the web interface
and the REST API for COVID-19 and vessel movement explanations.

The application structure:
- Main app: Serves the Vue.js static frontend
- API app: RESTful API mounted at /api with COVID and vessel endpoints
- CORS: Configured to allow cross-origin requests for development

Features:
- Serves compiled Vue.js frontend from cvd_vue/dist
- Mounts REST API at /api endpoint
- Includes routers for COVID-19 and vessel explanations
- CORS middleware for web client access

Endpoints:
- GET /: Root endpoint (redirects to static frontend)
- /api/covid/*: COVID-19 explanation endpoints
- /api/vessels/*: Vessel movement explanation endpoints
- /api/docs: Auto-generated API documentation (Swagger UI)
- /api/redoc: Alternative API documentation (ReDoc)

Usage:
    Run with uvicorn:
    $ uvicorn vessels_api:app --host 0.0.0.0 --port 8000 --reload
"""


from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from starlette.staticfiles import StaticFiles

from vessels_router import vessels_router
from covid_router import covid_router

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

metadata_tags = [
    {
        "name": "Vessels",
        "description": "API for vessels data"
    }
]
app = FastAPI(title="Frontend")

api_app = FastAPI(title="LORE API", description="API for vessels data and covid data",
              version="0.1")
api_app.logger = logger

app.mount("/api", api_app)
app.mount("/", StaticFiles(directory="cvd_vue/dist", html=True), name="static")

app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"]
                   )

api_app.include_router(vessels_router)
api_app.include_router(covid_router)

@app.get("/")
async def root():
    """
    Root endpoint returning a welcome message.
    
    Returns:
        dict: Welcome message
    """
    return {"message": "Welcome to the Vessels API"}

