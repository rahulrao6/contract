"""
FastAPI server for Contract Analyzer API.
"""

import logging
import uvicorn
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.config import Config
from src.api.routes import router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

def create_app(config: Config = None) -> FastAPI:
    """
    Create and configure FastAPI application.
    
    Args:
        config: Configuration settings
        
    Returns:
        Configured FastAPI app
    """
    if config is None:
        config = Config()
    
    app = FastAPI(
        title=config.API_TITLE,
        description=config.API_DESCRIPTION,
        version=config.API_VERSION,
        docs_url="/docs",
        redoc_url="/redoc",
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify allowed origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add routes
    app.include_router(router)
    
    # Add startup and shutdown events
    @app.on_event("startup")
    async def startup_event():
        logger.info("Starting Contract Analyzer API")
    
    @app.on_event("shutdown")
    async def shutdown_event():
        logger.info("Shutting down Contract Analyzer API")
    
    return app

def run_server():
    """Run the API server."""
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    run_server()