from typing import Optional, List, Dict, Any, Union
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from app.api_schema.base import BaseResponse, PaginatedResponse


class Point(BaseModel):
    """Point coordinates for segmentation."""
    
    x: float = Field(..., description="X coordinate (0-1 normalized)")
    y: float = Field(..., description="Y coordinate (0-1 normalized)")
    
    @field_validator('x', 'y')
    def validate_coordinates(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Coordinates must be between 0 and 1')
        return v


class Box(BaseModel):
    """Bounding box for segmentation."""
    
    x1: float = Field(..., description="Top-left X coordinate (0-1 normalized)")
    y1: float = Field(..., description="Top-left Y coordinate (0-1 normalized)")
    x2: float = Field(..., description="Bottom-right X coordinate (0-1 normalized)")
    y2: float = Field(..., description="Bottom-right Y coordinate (0-1 normalized)")
    
    @field_validator('x1', 'y1', 'x2', 'y2')
    def validate_coordinates(cls, v):
        if not 0 <= v <= 1:
            raise ValueError('Coordinates must be between 0 and 1')
        return v
    
    @field_validator('x2')
    def validate_x2(cls, v, values):
        if 'x1' in values and v <= values['x1']:
            raise ValueError('x2 must be greater than x1')
        return v
    
    @field_validator('y2')
    def validate_y2(cls, v, values):
        if 'y1' in values and v <= values['y1']:
            raise ValueError('y2 must be greater than y1')
        return v


class SegmentationRequest(BaseModel):
    """Single image segmentation request."""
    
    model_type: Optional[str] = Field(None, description="Model type to use (defaults to configured default)")
    points: Optional[List[Point]] = Field(None, description="List of points for point-based segmentation")
    boxes: Optional[List[Box]] = Field(None, description="List of bounding boxes for box-based segmentation")
    
    @field_validator('points', 'boxes')
    def validate_inputs(cls, v, values):
        if v is not None and len(v) == 0:
            return None
        return v


class MaskResponse(BaseModel):
    """Mask details for a segmentation result."""
    
    mask_index: int = Field(..., description="Index of mask in result")
    mask_data: str = Field(..., description="Base64 encoded mask data")
    confidence: float = Field(1.0, description="Confidence score")
    bbox: Optional[Dict[str, Any]] = Field(None, description="Bounding box metadata")
    area: Optional[int] = Field(None, description="Pixel area of the mask")


class SegmentationResponse(BaseModel):
    """Single image segmentation response matching route output."""
    
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    model_type: str = Field(..., description="Model type used")
    total_masks: int = Field(..., description="Total number of masks")
    masks: List[MaskResponse] = Field(..., description="List of masks")
    created_at: Optional[datetime] = Field(None, description="Job creation time")
    updated_at: Optional[datetime] = Field(None, description="Last update time")


class BatchSegmentationRequest(BaseModel):
    """Batch segmentation request."""
    
    model_type: Optional[str] = Field(None, description="Model type to use")
    images: List[str] = Field(..., description="List of base64 encoded images")
    points: Optional[List[List[Point]]] = Field(None, description="Points for each image")
    boxes: Optional[List[List[Box]]] = Field(None, description="Boxes for each image")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Additional parameters for processing")
    
    @field_validator('images')
    def validate_images(cls, v):
        if not v:
            raise ValueError('At least one image is required')
        if len(v) > 100:
            raise ValueError('Maximum 100 images allowed per batch')
        return v
    
    @field_validator('points', 'boxes')
    def validate_inputs(cls, v, values):
        if v is not None:
            images = values.get('images', [])
            if len(v) != len(images):
                raise ValueError('Number of input lists must match number of images')
        return v


class BatchSegmentationResponse(BaseResponse):
    """Batch segmentation response."""
    
    data: Dict[str, Any] = Field(..., description="Batch job information")


class JobStatus(BaseModel):
    """Job status information."""
    
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    created_at: datetime = Field(..., description="Job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    progress: float = Field(0.0, description="Progress (0-1)")
    total_images: int = Field(..., description="Total number of images")
    processed_images: int = Field(0, description="Number of processed images")


class JobStatusResponse(BaseModel):
    """Response schema used by jobs routes for listing and details."""
    
    job_id: str = Field(..., description="Job ID")
    status: str = Field(..., description="Job status")
    model_type: str = Field(..., description="Model type")
    total_masks: int = Field(..., description="Total masks")
    created_at: datetime = Field(..., description="Creation time")
    updated_at: Optional[datetime] = Field(None, description="Update time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    batch_job_id: Optional[str] = Field(None, description="Parent batch job ID")


class JobResult(BaseModel):
    """Job result information."""
    
    job_id: str = Field(..., description="Job ID")
    image_index: int = Field(..., description="Image index in batch")
    masks: List[List[List[int]]] = Field(..., description="Segmentation masks")
    scores: Optional[List[float]] = Field(None, description="Confidence scores")
    visualization: Optional[str] = Field(None, description="Base64 encoded visualization")


class JobResponse(BaseResponse):
    """Job status response."""
    
    data: JobStatus = Field(..., description="Job status information")


class JobResultResponse(BaseResponse):
    """Job result response."""
    
    data: JobResult = Field(..., description="Job result information")


class BatchJobStatus(BaseModel):
    """Batch job status information."""
    
    batch_job_id: str = Field(..., description="Batch job ID")
    status: str = Field(..., description="Batch job status")
    created_at: datetime = Field(..., description="Batch job creation time")
    updated_at: datetime = Field(..., description="Last update time")
    completed_at: Optional[datetime] = Field(None, description="Completion time")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    progress: float = Field(0.0, description="Progress (0-1)")
    total_jobs: int = Field(..., description="Total number of individual jobs")
    completed_jobs: int = Field(0, description="Number of completed jobs")
    failed_jobs: int = Field(0, description="Number of failed jobs")


class BatchJobResponse(BaseModel):
    """Batch job submission response matching route output."""
    
    batch_job_id: str = Field(..., description="Batch job ID")
    task_id: str = Field(..., description="Background task ID")
    status: str = Field(..., description="Batch job status")
    total_images: int = Field(..., description="Total images in batch")
    created_at: datetime = Field(..., description="Batch job creation time")


class JobListResponse(PaginatedResponse):
    """Job list response."""
    
    data: List[JobStatus] = Field(..., description="List of jobs")


class BatchJobListResponse(PaginatedResponse):
    """Batch job list response."""
    
    data: List[BatchJobStatus] = Field(..., description="List of batch jobs")