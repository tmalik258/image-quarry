# Image Quarry API Documentation

## Overview

Image Quarry is a production-ready FastAPI service that provides SAM (Segment Anything Model) image segmentation capabilities. This service offers both synchronous and asynchronous processing options, with support for automatic mask generation, bounding-box prompted segmentation, and RGBA object extraction.

## Features

- **SAM Model Integration**: Support for multiple SAM model variants (vit_h, vit_l, vit_b)
- **Automatic Segmentation**: Generate masks automatically with configurable parameters
- **Box-Prompted Segmentation**: Segment specific regions using bounding boxes
- **Object Extraction**: Extract individual objects as RGBA images with transparency
- **Batch Processing**: Process multiple images in parallel
- **Background Tasks**: Asynchronous processing via Celery
- **Google Drive Integration**: Save results directly to Google Drive
- **Comprehensive Health Checks**: Monitor service and dependency status

## Base URL

```
http://localhost:8000
```

## Authentication

API authentication is optional and can be configured via environment variables:

- **Header**: `X-API-Key: your-api-key`
- **Configuration**: Set `API_KEYS` in environment variables

## Response Format

All responses follow a consistent format:

```json
{
  "success": true,
  "message": "Operation completed successfully",
  "timestamp": "2024-01-01T12:00:00Z",
  "data": { ... }
}
```

Error responses include additional error details:

```json
{
  "success": false,
  "message": "Validation error",
  "error_code": "VALIDATION_ERROR",
  "details": { "errors": [...] },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## API Endpoints

### Health & Status

#### GET /health/

**Description**: Comprehensive health check including database, Redis, and model status.

**Response**:
```json
{
  "status": "healthy",
  "service": "image-quarry-api",
  "version": "0.1.0",
  "details": {
    "database": "healthy",
    "redis": "healthy",
    "models": {
      "available": 3,
      "cached": 2,
      "loaded": 1
    }
  }
}
```

**Status Codes**: 200 (Healthy), 503 (Unhealthy)

#### GET /health/ready

**Description**: Readiness check for Kubernetes deployments.

**Response**:
```json
{
  "status": "ready",
  "service": "image-quarry-api",
  "version": "0.1.0",
  "details": {"ready": true}
}
```

#### GET /health/live

**Description**: Liveness check for Kubernetes deployments.

**Response**:
```json
{
  "status": "alive",
  "service": "image-quarry-api",
  "version": "0.1.0",
  "details": {"alive": true}
}
```

### Segmentation

#### POST /segment/single

**Description**: Segment a single image with points or boxes.

**Parameters**:
- `image` (file, required): Image file (JPEG, PNG, WebP, BMP, TIFF)
- `model_type` (string, optional): Model variant (`vit_h`, `vit_l`, `vit_b`). Default: `vit_h`
- `points` (string, optional): JSON array of point coordinates `[{"x": 0.5, "y": 0.5}]`
- `boxes` (string, optional): JSON array of bounding boxes `[{"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}]`

**Request Example**:
```bash
curl -X POST "http://localhost:8000/segment/single" \
  -F "image=@your_image.jpg" \
  -F "model_type=vit_h" \
  -F 'points=[{"x": 0.5, "y": 0.5}]'
```

**Response**:
```json
{
  "job_id": "550e8400-e29b-41d4-a716-446655440000",
  "status": "completed",
  "model_type": "vit_h",
  "total_masks": 5,
  "masks": [
    {
      "mask_index": 0,
      "mask_data": "base64_encoded_mask_data...",
      "confidence": 0.95,
      "bbox": {"x": 100, "y": 150, "width": 200, "height": 300},
      "area": 60000
    }
  ],
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:00:01Z"
}
```

#### POST /segment/auto

**Description**: Automatic mask generation with configurable parameters.

**Parameters**:
- `image` (file, required): Image file
- `model_type` (string, optional): Model variant. Default: `vit_h`
- `points_per_side` (integer, optional): Points per side. Default: `64`
- `pred_iou_thresh` (float, optional): IoU threshold. Default: `0.90`
- `stability_score_thresh` (float, optional): Stability threshold. Default: `0.92`
- `crop_n_layers` (integer, optional): Crop layers. Default: `1`
- `crop_n_points_downscale_factor` (integer, optional): Downscale factor. Default: `2`
- `min_mask_region_area` (integer, optional): Minimum mask area. Default: `1000`

**Request Example**:
```bash
curl -X POST "http://localhost:8000/segment/auto" \
  -F "image=@your_image.jpg" \
  -F "model_type=vit_h" \
  -F "points_per_side=64" \
  -F "pred_iou_thresh=0.90"
```

**Response**:
```json
{
  "total_masks": 8,
  "masks": [
    {
      "segmentation": "base64_encoded_mask...",
      "area": 12543,
      "bbox": [100, 150, 200, 300],
      "predicted_iou": 0.95,
      "point_coords": [[150, 250]],
      "stability_score": 0.93,
      "crop_box": [0, 0, 800, 600]
    }
  ]
}
```

#### POST /segment/auto/async

**Description**: Asynchronous automatic mask generation.

**Parameters**: Same as `/segment/auto`

**Response**:
```json
{
  "task_id": "task-12345-abcde"
}
```

#### POST /segment/box

**Description**: Segment specific regions using bounding boxes.

**Parameters**:
- `image` (file, required): Image file
- `model_type` (string, optional): Model variant. Default: `vit_h`
- `box` (string, optional): JSON bounding box. Default: `{"x": 68, "y": 247, "width": 555, "height": 678}`
- `multimask_output` (boolean, optional): Return multiple masks. Default: `true`

**Request Example**:
```bash
curl -X POST "http://localhost:8000/segment/box" \
  -F "image=@your_image.jpg" \
  -F 'box={"x": 100, "y": 100, "width": 200, "height": 200}'
```

**Response**: Same format as `/segment/auto`

#### POST /segment/box/async

**Description**: Asynchronous box-prompted segmentation.

**Parameters**: Same as `/segment/box`

**Response**:
```json
{
  "task_id": "task-67890-fghij"
}
```

#### POST /segment/objects/extract

**Description**: Extract individual objects as RGBA images with transparency.

**Parameters**:
- `image` (file, required): Image file
- `model_type` (string, optional): Model variant. Default: `vit_h`
- `min_pixels` (integer, optional): Minimum object size in pixels. Default: `5000`
- `save_dir` (string, optional): Local directory to save objects. Default: `./results`
- `save_to_drive` (boolean, optional): Upload to Google Drive. Default: `false`
- `drive_folder_id` (string, optional): Google Drive folder ID

**Request Example**:
```bash
curl -X POST "http://localhost:8000/segment/objects/extract" \
  -F "image=@your_image.jpg" \
  -F "min_pixels=5000" \
  -F "save_to_drive=true"
```

**Response**:
```json
{
  "saved": [
    "./results/object_0.png",
    "./results/object_1.png",
    "./results/object_2.png"
  ],
  "skipped": 2,
  "drive_files": [
    {
      "file_id": "1a2b3c4d5e6f",
      "web_view": "https://drive.google.com/file/d/1a2b3c4d5e6f/view"
    }
  ]
}
```

#### POST /segment/objects/extract/async

**Description**: Asynchronous object extraction.

**Parameters**: Same as `/segment/objects/extract` with additional segmentation parameters

**Response**:
```json
{
  "task_id": "task-13579-klmno"
}
```

#### POST /segment/batch

**Description**: Submit a batch segmentation job for multiple images.

**Request Body**:
```json
{
  "model_type": "vit_h",
  "images": ["base64_encoded_image_1...", "base64_encoded_image_2..."],
  "points": [[{"x": 0.5, "y": 0.5}], [{"x": 0.3, "y": 0.7}]],
  "boxes": [[{"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}]],
  "parameters": {"custom_param": "value"}
}
```

**Response**:
```json
{
  "batch_job_id": "batch-12345",
  "task_id": "task-24680-pqrst",
  "status": "submitted",
  "total_images": 2,
  "created_at": "2024-01-01T12:00:00Z"
}
```

#### GET /segment/job/{job_id}

**Description**: Get segmentation job results by job ID.

**Response**: Same format as `/segment/single`

#### GET /segment/batch/{batch_job_id}

**Description**: Get batch job status and results.

**Response**:
```json
{
  "batch_job_id": "batch-12345",
  "status": "completed",
  "total_images": 2,
  "completed_images": 2,
  "failed_images": 0,
  "created_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:05:00Z",
  "jobs": [
    {
      "job_id": "job-1",
      "status": "completed",
      "model_type": "vit_h",
      "total_masks": 5
    }
  ]
}
```

#### GET /segment/job/{job_id}/visualization

**Description**: Get visualization of segmentation results.

**Response**: SVG image showing segmentation overlay

### Storage

#### GET /storage/drive/status

**Description**: Check Google Drive integration status.

**Response**:
```json
{
  "enabled": true,
  "authenticated": true,
  "folder_access": true
}
```

#### POST /storage/drive/upload

**Description**: Upload a file directly to Google Drive.

**Parameters**:
- `file` (file, required): File to upload
- `folder_id` (string, optional): Google Drive folder ID

**Response**:
```json
{
  "file_id": "1a2b3c4d5e6f",
  "web_view": "https://drive.google.com/file/d/1a2b3c4d5e6f/view"
}
```

### Jobs Management

#### GET /jobs/

**Description**: List all jobs with pagination.

**Parameters**:
- `page` (integer, optional): Page number. Default: `1`
- `page_size` (integer, optional): Items per page. Default: `20`
- `status` (string, optional): Filter by status (`pending`, `running`, `completed`, `failed`)
- `model_type` (string, optional): Filter by model type

**Response**:
```json
{
  "success": true,
  "message": "Jobs retrieved successfully",
  "data": [
    {
      "job_id": "job-12345",
      "status": "completed",
      "model_type": "vit_h",
      "total_masks": 5,
      "created_at": "2024-01-01T12:00:00Z",
      "updated_at": "2024-01-01T12:01:00Z"
    }
  ],
  "total": 42,
  "page": 1,
  "page_size": 20,
  "total_pages": 3,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /jobs/{job_id}

**Description**: Get detailed job information.

**Response**: Same as individual job response

#### GET /jobs/{job_id}/status

**Description**: Get job status only.

**Response**:
```json
{
  "success": true,
  "message": "Job status retrieved successfully",
  "data": {
    "job_id": "job-12345",
    "status": "completed",
    "created_at": "2024-01-01T12:00:00Z",
    "updated_at": "2024-01-01T12:01:00Z",
    "progress": 1.0,
    "total_images": 1,
    "processed_images": 1
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /jobs/{job_id}/result

**Description**: Get job results.

**Response**: Same as segmentation response

#### DELETE /jobs/{job_id}

**Description**: Delete a job and its results.

**Response**:
```json
{
  "success": true,
  "message": "Job deleted successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

### Models

#### GET /models/

**Description**: List available SAM models.

**Response**:
```json
{
  "success": true,
  "message": "Models retrieved successfully",
  "data": [
    {
      "name": "vit_h",
      "description": "SAM ViT-Huge model",
      "size_mb": 2560,
      "cached": true,
      "loaded": true
    }
  ],
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### GET /models/{model_name}

**Description**: Get specific model information.

**Response**: Single model information

#### POST /models/{model_name}/load

**Description**: Load a model into memory.

**Response**:
```json
{
  "success": true,
  "message": "Model loaded successfully",
  "data": {
    "name": "vit_h",
    "loaded": true,
    "memory_usage_mb": 2560
  },
  "timestamp": "2024-01-01T12:00:00Z"
}
```

#### DELETE /models/{model_name}/unload

**Description**: Unload a model from memory.

**Response**:
```json
{
  "success": true,
  "message": "Model unloaded successfully",
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## Error Codes

| Error Code | Description |
|------------|-------------|
| `VALIDATION_ERROR` | Request validation failed |
| `IMAGE_VALIDATION_ERROR` | Invalid image file |
| `MODEL_ERROR` | Model loading/processing error |
| `JOB_NOT_FOUND` | Requested job not found |
| `NOT_FOUND` | Resource not found |
| `METHOD_NOT_ALLOWED` | HTTP method not allowed |
| `INTERNAL_ERROR` | Internal server error |
| `HTTP_ERROR` | General HTTP error |

## Rate Limiting

Default rate limits (configurable via environment variables):
- **Per minute**: 60 requests
- **Per hour**: 1000 requests

## File Size Limits

- **Maximum image size**: 50 MB
- **Maximum image dimension**: 4096 pixels
- **Supported formats**: JPEG, PNG, WebP, BMP, TIFF

## Google Drive Integration

To enable Google Drive integration:

1. Create a Google Cloud project
2. Enable Google Drive API
3. Create a service account
4. Download service account JSON
5. Set environment variables:
   ```bash
   GOOGLE_DRIVE_ENABLED=true
   GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/service-account.json
   DRIVE_FOLDER_ID=your-folder-id
   ```

## Background Tasks (Celery)

Asynchronous tasks are processed via Celery with Redis:

- **Broker URL**: `redis://localhost:6379/1`
- **Result Backend**: `redis://localhost:6379/2`
- **Task Timeout**: 600 seconds (hard), 540 seconds (soft)

### Checking Task Status

Use the task ID returned from async endpoints:

```bash
# Check task status
curl "http://localhost:8000/jobs/task-{task-id}/status"

# Get task result
curl "http://localhost:8000/jobs/task-{task-id}/result"
```

## Configuration

Key environment variables:

```bash
# Server
HOST=0.0.0.0
PORT=8000

# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/image_quarry

# Redis
REDIS_URL=redis://localhost:6379/0

# Models
DEFAULT_MODEL=vit_h
MODEL_CACHE_DIR=./models

# Storage
UPLOAD_DIR=./uploads
RESULTS_DIR=./results

# Google Drive
GOOGLE_DRIVE_ENABLED=false
GOOGLE_SERVICE_ACCOUNT_JSON=
DRIVE_FOLDER_ID=
```

## Docker Deployment

The service includes Docker support with multi-stage builds:

```bash
# Build image
docker build -t image-quarry-api .

# Run with Docker Compose
docker-compose up -d
```

## Monitoring

- **Health endpoints**: `/health/`, `/health/ready`, `/health/live`
- **Metrics**: Prometheus metrics available at `:9090/metrics` (if enabled)
- **Logging**: Structured JSON logging with configurable levels

## Examples

### Basic Segmentation
```python
import requests

# Upload and segment image
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    data = {'model_type': 'vit_h'}
    response = requests.post('http://localhost:8000/segment/auto', files=files, data=data)
    
result = response.json()
print(f"Found {result['total_masks']} masks")
```

### Async Processing
```python
import requests
import time

# Submit async job
with open('image.jpg', 'rb') as f:
    files = {'image': f}
    response = requests.post('http://localhost:8000/segment/auto/async', files=files)
    
task_id = response.json()['task_id']

# Check status
while True:
    status = requests.get(f'http://localhost:8000/jobs/{task_id}/status').json()
    if status['data']['status'] in ['completed', 'failed']:
        break
    time.sleep(1)

# Get results
results = requests.get(f'http://localhost:8000/jobs/{task_id}/result').json()
```

### Batch Processing
```python
import requests
import base64

# Prepare images
images = []
with open('image1.jpg', 'rb') as f:
    images.append(base64.b64encode(f.read()).decode())
with open('image2.jpg', 'rb') as f:
    images.append(base64.b64encode(f.read()).decode())

# Submit batch job
payload = {
    'model_type': 'vit_h',
    'images': images
}
response = requests.post('http://localhost:8000/segment/batch', json=payload)
batch_id = response.json()['batch_job_id']

# Check batch status
batch_status = requests.get(f'http://localhost:8000/segment/batch/{batch_id}').json()
```

## Support

For issues and questions:
- Check health endpoints for service status
- Review logs for error details
- Verify model files are downloaded and cached
- Ensure Redis and database connections are healthy