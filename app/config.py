import os
from functools import lru_cache
from typing import Optional, List
from pydantic import Field
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings."""
    
    # Application
    APP_NAME: str = "image-quarry-api"
    APP_VERSION: str = "0.1.0"
    DEBUG: bool = Field(default=False, description="Enable debug mode")
    
    # Server
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of worker processes")
    
    # Database
    DATABASE_URL: str = Field(
        default="postgresql+asyncpg://user:password@localhost:5432/image_quarry",
        description="PostgreSQL database URL"
    )
    
    # Redis
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL"
    )
    
    # Celery
    CELERY_BROKER_URL: str = Field(
        default="redis://localhost:6379/1",
        description="Celery broker URL"
    )
    CELERY_RESULT_BACKEND: str = Field(
        default="redis://localhost:6379/2",
        description="Celery result backend URL"
    )
    CELERY_TASK_TIME_LIMIT: int = Field(
        default=600,
        description="Hard time limit (seconds) for Celery tasks"
    )
    CELERY_TASK_SOFT_TIME_LIMIT: int = Field(
        default=540,
        description="Soft time limit (seconds) for Celery tasks"
    )
    
    # Model Management
    MODEL_CACHE_DIR: str = Field(
        default="./models",
        description="Directory to cache downloaded models"
    )
    DEFAULT_MODEL: str = Field(
        default="vit_h",
        description="Default SAM model to use"
    )
    MODEL_CACHE_SIZE_GB: int = Field(
        default=10,
        description="Maximum model cache size in GB"
    )
    
    # SAM Model URLs
    SAM_MODEL_URLS: dict = Field(
        default={
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        },
        description="SAM model download URLs"
    )
    
    # Image Processing
    MAX_IMAGE_SIZE_MB: int = Field(
        default=50,
        description="Maximum image file size in MB"
    )
    MAX_IMAGE_DIMENSION: int = Field(
        default=4096,
        description="Maximum image dimension in pixels"
    )
    SUPPORTED_IMAGE_FORMATS: List[str] = Field(
        default=["jpeg", "jpg", "png", "webp", "bmp", "tiff"],
        description="Supported image formats"
    )
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(
        default=60,
        description="Requests per minute limit"
    )
    RATE_LIMIT_PER_HOUR: int = Field(
        default=1000,
        description="Requests per hour limit"
    )
    
    # Authentication (optional)
    API_KEY_HEADER: str = Field(
        default="X-API-Key",
        description="API key header name"
    )
    API_KEYS: List[str] = Field(
        default=[],
        description="Valid API keys (if authentication is enabled)"
    )
    
    # CORS
    CORS_ORIGINS: List[str] = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    CORS_METHODS: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="CORS allowed methods"
    )
    CORS_HEADERS: List[str] = Field(
        default=["*"],
        description="CORS allowed headers"
    )
    
    # Storage
    UPLOAD_DIR: str = Field(
        default="./uploads",
        description="Directory for uploaded images"
    )
    RESULTS_DIR: str = Field(
        default="./results",
        description="Directory for segmentation results"
    )
    GOOGLE_DRIVE_ENABLED: bool = Field(
        default=False,
        description="Enable Google Drive integration"
    )
    GOOGLE_SERVICE_ACCOUNT_JSON: Optional[str] = Field(
        default=None,
        description="Path to Google service account JSON"
    )
    DRIVE_FOLDER_ID: Optional[str] = Field(
        default=None,
        description="Default Google Drive folder ID"
    )
    
    # Background Tasks
    JOB_TIMEOUT_MINUTES: int = Field(
        default=30,
        description="Maximum job execution time in minutes"
    )
    CLEANUP_OLDER_THAN_DAYS: int = Field(
        default=7,
        description="Delete jobs older than this many days"
    )
    
    # Monitoring
    ENABLE_METRICS: bool = Field(
        default=True,
        description="Enable Prometheus metrics"
    )
    METRICS_PORT: int = Field(
        default=9090,
        description="Prometheus metrics port"
    )
    
    # Security
    SECRET_KEY: str = Field(
        default="your-secret-key-change-this",
        description="Secret key for JWT tokens"
    )
    
    # Logging
    LOG_LEVEL: str = Field(
        default="INFO",
        description="Logging level"
    )
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format (json or text)"
    )
    
    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=True)

    # Validators to robustly parse list values from .env
    @field_validator("CORS_ORIGINS", "CORS_METHODS", "CORS_HEADERS", mode="before")
    def _parse_env_list(cls, v):
        """Allow JSON array or comma-separated strings in .env for list fields."""
        if isinstance(v, list):
            return v
        if isinstance(v, str):
            s = v.strip()
            # Try JSON first
            if (s.startswith("[") and s.endswith("]")) or (s.startswith("\n[") and s.endswith("]\n")):
                import json
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, list):
                        return parsed
                except Exception:
                    pass
            # Fallback to comma-separated parsing
            return [item.strip() for item in s.split(",") if item.strip()]
        return v


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Export settings instance
settings = get_settings()
