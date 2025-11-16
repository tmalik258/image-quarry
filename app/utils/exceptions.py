from typing import Optional, Dict, Any
from fastapi import status


class ImageQuarryException(Exception):
    """Base exception for Image Quarry API."""
    
    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


class ImageValidationError(ImageQuarryException):
    """Raised when image validation fails."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="IMAGE_VALIDATION_ERROR",
            details=details
        )


class ModelError(ImageQuarryException):
    """Raised when model operations fail."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="MODEL_ERROR",
            details=details
        )


class JobNotFoundError(ImageQuarryException):
    """Raised when a job is not found."""
    
    def __init__(self, message: str = "Job not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="JOB_NOT_FOUND",
            details=details
        )


class BatchJobNotFoundError(ImageQuarryException):
    """Raised when a batch job is not found."""
    
    def __init__(self, message: str = "Batch job not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="BATCH_JOB_NOT_FOUND",
            details=details
        )


class RateLimitError(ImageQuarryException):
    """Raised when rate limit is exceeded."""
    
    def __init__(self, message: str = "Rate limit exceeded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details
        )


class AuthenticationError(ImageQuarryException):
    """Raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(ImageQuarryException):
    """Raised when authorization fails."""
    
    def __init__(self, message: str = "Authorization failed", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            details=details
        )


class ModelNotFoundError(ImageQuarryException):
    """Raised when a model is not found."""
    
    def __init__(self, message: str = "Model not found", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code="MODEL_NOT_FOUND",
            details=details
        )


class ModelNotDownloadedError(ImageQuarryException):
    """Raised when a model is not downloaded."""
    
    def __init__(self, message: str = "Model not downloaded", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="MODEL_NOT_DOWNLOADED",
            details=details
        )


class InvalidModelTypeError(ImageQuarryException):
    """Raised when an invalid model type is provided."""
    
    def __init__(self, message: str = "Invalid model type", details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="INVALID_MODEL_TYPE",
            details=details
        )