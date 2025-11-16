import time
from contextlib import asynccontextmanager
from typing import Any, Dict
import structlog
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.status import HTTP_404_NOT_FOUND, HTTP_405_METHOD_NOT_ALLOWED

from app.config import settings
from app.utils.logging import setup_logger
from app.utils.exceptions import ImageQuarryException
from app.database import init_db_with_retry
from app.utils.model_bootstrap import ensure_model_ready
from app.routes import health, segment, jobs
from app.routes import storage

# Configure logging
logger = setup_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    logger.info(f"Starting Image Quarry API version {settings.APP_VERSION}")
    
    # Initialize database
    try:
        await init_db_with_retry()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise

    # Ensure model and weights are ready on first run
    try:
        await ensure_model_ready()
        logger.info("Model bootstrap ensured")
    except Exception as e:
        logger.error(f"Model bootstrap failed: {str(e)}")
        raise

    yield
    
    # Cleanup
    logger.info("Shutting down Image Quarry API")


# Create FastAPI app
app = FastAPI(
    title="Image Quarry API",
    description="Production-ready FastAPI service for SAM image segmentation",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(segment.router)
app.include_router(jobs.router)
app.include_router(storage.router)

# Add middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)


# Exception handlers
@app.exception_handler(ImageQuarryException)
async def image_quarry_exception_handler(request: Request, exc: ImageQuarryException):
    """Handle custom ImageQuarry exceptions."""
    logger.error(
        f"ImageQuarry exception: {str(exc)}"
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.message,
            "error_code": exc.error_code,
            "details": exc.details,
            "timestamp": time.time(),
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle Pydantic validation errors."""
    logger.warning(f"Validation error errors={exc.errors()} url={str(request.url)}")
    
    return JSONResponse(
        status_code=422,
        content={
            "success": False,
            "message": "Validation error",
            "error_code": "VALIDATION_ERROR",
            "details": {"errors": exc.errors()},
            "timestamp": time.time(),
        },
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Handle HTTP exceptions."""
    logger.warning(f"HTTP exception status_code={exc.status_code} detail={exc.detail} url={str(request.url)}")
    
    # Custom error codes for common HTTP errors
    error_code = "HTTP_ERROR"
    if exc.status_code == HTTP_404_NOT_FOUND:
        error_code = "NOT_FOUND"
    elif exc.status_code == HTTP_405_METHOD_NOT_ALLOWED:
        error_code = "METHOD_NOT_ALLOWED"
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "message": exc.detail,
            "error_code": error_code,
            "details": {},
            "timestamp": time.time(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    logger.error(
        f"Unhandled exception: {str(exc)}"
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "Internal server error",
            "error_code": "INTERNAL_ERROR",
            "details": {},
            "timestamp": time.time(),
        },
    )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Welcome to Image Quarry API",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "info": "/info"
    }


# API info endpoint
@app.get("/info")
async def api_info():
    """Get API information."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Production-ready FastAPI service for SAM image segmentation",
        "endpoints": [
            "/health/",
            "/segment/",
            "/jobs/",
            "/docs",
            "/redoc",
            "/openapi.json",
        ],
    }
