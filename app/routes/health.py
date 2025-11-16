from app.utils.logging import setup_logger
from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.database import get_async_db
from app.api_schema.base import HealthResponse
from app.services.model_manager import ModelManager
from app.config import settings
import redis.asyncio as redis
import asyncio

logger = setup_logger(__name__)
router = APIRouter(prefix="/health", tags=["health"])


@router.get("/", response_model=HealthResponse)
async def health_check(db: AsyncSession = Depends(get_async_db)):
    """Health check endpoint."""
    
    status_info = {
        "status": "healthy",
        "service": "image-quarry-api",
        "version": settings.APP_VERSION,
    }
    
    # Check database connection
    try:
        await db.execute(text("SELECT 1"))
        status_info["database"] = "healthy"
    except Exception as e:
        logger.error(f"Database health check failed: {str(e)}")
        status_info["database"] = "unhealthy"
        status_info["status"] = "unhealthy"
    
    # Check Redis connection
    try:
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        status_info["redis"] = "healthy"
    except Exception as e:
        logger.error(f"Redis health check failed: {str(e)}")
        status_info["redis"] = "unhealthy"
        status_info["status"] = "unhealthy"
    
    # Check model availability
    try:
        model_manager = ModelManager()
        available_models = await model_manager.list_available_models()
        status_info["models"] = {
            "available": len(available_models),
            "cached": len([m for m in available_models if m.get("cached", False)]),
            "loaded": len([m for m in available_models if m.get("loaded", False)]),
        }
    except Exception as e:
        logger.error(f"Model health check failed {str(e)}")
        status_info["models"] = "unhealthy"
        status_info["status"] = "unhealthy"
    
    # Determine overall status
    if status_info["status"] == "unhealthy":
        return HealthResponse(
            status="unhealthy",
            service=status_info["service"],
            version=status_info["version"],
            details=status_info,
        )
    
    return HealthResponse(
        status="healthy",
        service=status_info["service"],
        version=status_info["version"],
        details=status_info,
    )


@router.get("/ready", response_model=HealthResponse)
async def readiness_check(db: AsyncSession = Depends(get_async_db)):
    """Readiness check for Kubernetes."""
    
    # Check if service is ready to handle requests
    try:
        await db.execute(text("SELECT 1"))
        
        # Check Redis
        redis_client = redis.from_url(settings.REDIS_URL)
        await redis_client.ping()
        await redis_client.close()
        
        return HealthResponse(
            status="ready",
            service="image-quarry-api",
            version=settings.APP_VERSION,
            details={"ready": True},
        )
    except Exception as e:
        logger.error(f"Readiness check failed {str(e)}")
        return HealthResponse(
            status="not_ready",
            service="image-quarry-api",
            version=settings.APP_VERSION,
            details={"ready": False, "error": str(e)},
        )


@router.get("/live", response_model=HealthResponse)
async def liveness_check():
    """Liveness check for Kubernetes."""
    
    return HealthResponse(
        status="alive",
        service="image-quarry-api",
        version=settings.APP_VERSION,
        details={"alive": True},
    )