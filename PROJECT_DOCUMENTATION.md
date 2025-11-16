# Image Quarry - Complete Project Documentation

## 🎯 Project Overview

Image Quarry is a production-ready FastAPI service that migrates the functionality of the original Jupyter notebook `image_quarry.ipynb` to a scalable web service. The service provides SAM (Segment Anything Model) image segmentation with automatic mask generation, bounding-box prompted segmentation, and RGBA object extraction.

## 🏗️ Architecture

### Technology Stack
- **Framework**: FastAPI (async Python web framework)
- **Model**: SAM (Segment Anything Model) - Meta AI
- **Background Tasks**: Celery with Redis
- **Database**: PostgreSQL with SQLAlchemy
- **Storage**: Local filesystem + Google Drive API
- **Monitoring**: Prometheus metrics + structured logging
- **Deployment**: Docker + Docker Compose

### Project Structure
```
app/
├── api_schema/          # Pydantic models for request/response validation
├── config.py           # Configuration management
├── database.py         # Database connection and models
├── main.py            # FastAPI application entry point
├── routes/            # API route handlers
│   ├── health.py      # Health check endpoints
│   ├── models.py      # Model management endpoints
│   ├── segment.py     # Segmentation endpoints
│   ├── jobs.py        # Job management endpoints
│   └── storage.py     # Storage integration endpoints
├── services/          # Business logic and external integrations
│   ├── background_tasks.py  # Celery task definitions
│   ├── image_processor.py   # SAM model processing
│   ├── model_manager.py     # Model lifecycle management
│   ├── object_extractor.py  # RGBA object extraction
│   └── storage/             # Storage services
│       └── google_drive.py  # Google Drive integration
└── utils/            # Utility functions and helpers
```

## 🚀 Key Features

### 1. SAM Model Integration
- **Multiple Model Variants**: Support for vit_h, vit_l, vit_b
- **Automatic Model Download**: Models downloaded on first use
- **Memory Management**: Efficient model caching and loading
- **GPU Support**: Automatic GPU detection and utilization

### 2. Segmentation Modes

#### Automatic Segmentation
- Generates masks automatically across the entire image
- Configurable parameters:
  - `points_per_side`: Grid resolution (default: 64)
  - `pred_iou_thresh`: IoU threshold (default: 0.90)
  - `stability_score_thresh`: Stability threshold (default: 0.92)
  - `min_mask_region_area`: Minimum mask size (default: 1000)

#### Box-Prompted Segmentation
- Segments specific regions using bounding boxes
- Supports multiple boxes per image
- Fallback to default box if none provided

#### Object Extraction
- Extracts individual objects as RGBA images
- Applies transparency to background
- Filters by minimum pixel count
- Saves locally or uploads to Google Drive

### 3. Processing Options

#### Synchronous Processing
- Immediate response with results
- Suitable for small images and quick operations
- Direct model inference

#### Asynchronous Processing
- Background task execution via Celery
- Returns task ID for status tracking
- Suitable for large images and batch operations
- Automatic retry on failure

#### Batch Processing
- Process multiple images in parallel
- Configurable concurrency limits
- Individual job tracking per image
- Aggregate results and statistics

### 4. Storage Integration

#### Local Storage
- Organized directory structure
- Configurable upload and results directories
- Automatic cleanup of old files
- File size and format validation

#### Google Drive Integration
- Service account authentication
- Automatic folder creation
- Web view links for uploaded files
- Configurable folder permissions

## 📋 API Endpoints

### Health & Monitoring
- `GET /health/` - Comprehensive health check
- `GET /health/ready` - Kubernetes readiness probe
- `GET /health/live` - Kubernetes liveness probe

### Segmentation
- `POST /segment/single` - Single image segmentation with points/boxes
- `POST /segment/auto` - Automatic mask generation
- `POST /segment/auto/async` - Async automatic segmentation
- `POST /segment/box` - Box-prompted segmentation
- `POST /segment/box/async` - Async box-prompted segmentation
- `POST /segment/objects/extract` - Extract objects as RGBA images
- `POST /segment/objects/extract/async` - Async object extraction
- `POST /segment/batch` - Batch processing for multiple images

### Job Management
- `GET /jobs/` - List all jobs with pagination
- `GET /jobs/{job_id}` - Get job details
- `GET /jobs/{job_id}/status` - Get job status
- `GET /jobs/{job_id}/result` - Get job results
- `DELETE /jobs/{job_id}` - Delete job and results

### Model Management
- `GET /models/` - List available models
- `GET /models/{model_name}` - Get model details
- `POST /models/{model_name}/load` - Load model into memory
- `DELETE /models/{model_name}/unload` - Unload model

### Storage
- `GET /storage/drive/status` - Google Drive integration status
- `POST /storage/drive/upload` - Upload file to Google Drive

## ⚙️ Configuration

### Environment Variables

#### Core Settings
```bash
# Application
APP_NAME=image-quarry-api
APP_VERSION=0.1.0
DEBUG=false

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=1
```

#### Database & Cache
```bash
# PostgreSQL
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/image_quarry

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2
```

#### Model Configuration
```bash
# Model Management
DEFAULT_MODEL=vit_h
MODEL_CACHE_DIR=./models
MODEL_CACHE_SIZE_GB=10

# SAM Model URLs
SAM_MODEL_URLS={
  "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
  "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
  "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
}
```

#### Processing Limits
```bash
# Image Processing
MAX_IMAGE_SIZE_MB=50
MAX_IMAGE_DIMENSION=4096
SUPPORTED_IMAGE_FORMATS=["jpeg", "jpg", "png", "webp", "bmp", "tiff"]

# Rate Limiting
RATE_LIMIT_PER_MINUTE=60
RATE_LIMIT_PER_HOUR=1000
```

#### Storage Configuration
```bash
# Local Storage
UPLOAD_DIR=./uploads
RESULTS_DIR=./results

# Google Drive
GOOGLE_DRIVE_ENABLED=false
GOOGLE_SERVICE_ACCOUNT_JSON=/path/to/service-account.json
DRIVE_FOLDER_ID=your-folder-id
```

#### Background Tasks
```bash
# Celery
CELERY_TASK_TIME_LIMIT=600
CELERY_TASK_SOFT_TIME_LIMIT=540
JOB_TIMEOUT_MINUTES=30
CLEANUP_OLDER_THAN_DAYS=7
```

#### Security & Monitoring
```bash
# Authentication
API_KEY_HEADER=X-API-Key
API_KEYS=["your-api-key-1", "your-api-key-2"]

# CORS
CORS_ORIGINS=["*"]
CORS_METHODS=["GET", "POST", "PUT", "DELETE"]
CORS_HEADERS=["*"]

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## 🔧 Setup Instructions

### Prerequisites
- Python 3.9+
- PostgreSQL 13+
- Redis 6+
- Docker (optional)
- Google Cloud account (for Drive integration)

### Local Development Setup

1. **Clone and Install Dependencies**
```bash
git clone <repository-url>
cd image-quarry
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Set Up Environment Variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Initialize Database**
```bash
# Create database
createdb image_quarry

# Run migrations
alembic upgrade head
```

4. **Start Redis**
```bash
redis-server
```

5. **Start Celery Worker**
```bash
celery -A app.services.background_tasks worker --loglevel=info --pool=solo
```

6. **Start FastAPI Server**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Setup

1. **Build and Run with Docker Compose**
```bash
docker-compose up -d
```

2. **Check Service Status**
```bash
curl http://localhost:8000/health/
```

3. **View Logs**
```bash
docker-compose logs -f
```

## 🧪 Testing

### Unit Tests
```bash
pytest tests/unit/
```

### Integration Tests
```bash
pytest tests/integration/
```

### Load Testing
```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 📊 Monitoring

### Health Checks
- **Endpoint**: `GET /health/`
- **Checks**: Database, Redis, Models, Overall status
- **Response Time**: < 100ms

### Metrics
- **Prometheus**: Available at `:9090/metrics`
- **Custom Metrics**: Request count, latency, model inference time
- **Grafana**: Pre-configured dashboards (optional)

### Logging
- **Structured JSON**: All logs in JSON format
- **Log Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL
- **Context**: Request ID, user ID, processing time

## 🔒 Security

### Authentication
- **API Key**: Header-based authentication
- **Rate Limiting**: Per-minute and per-hour limits
- **CORS**: Configurable cross-origin policies

### Input Validation
- **File Type**: Image format validation
- **File Size**: Maximum size limits
- **Content Validation**: Malicious content detection
- **Parameter Validation**: Pydantic models for all inputs

### Data Protection
- **Encryption**: TLS/SSL for all communications
- **Storage**: Secure file storage with permissions
- **PII**: No personal information stored
- **GDPR**: Compliant data handling

## 📈 Performance

### Optimization Strategies
- **Model Caching**: Models loaded once and reused
- **Async Processing**: Non-blocking I/O operations
- **Connection Pooling**: Database and Redis connections
- **CDN Integration**: Static asset caching
- **Compression**: Gzip compression for responses

### Benchmarks
- **Single Image**: ~2-5 seconds (depending on size)
- **Batch Processing**: ~1-2 seconds per image (parallel)
- **Memory Usage**: ~2-4GB per model loaded
- **Throughput**: 10-20 images per minute (single worker)

## 🔄 Background Tasks

### Celery Configuration
- **Broker**: Redis for task queue
- **Backend**: Redis for result storage
- **Concurrency**: Configurable worker processes
- **Retry Policy**: Automatic retry with exponential backoff
- **Task Routing**: Priority-based task routing

### Task Types
1. **Auto Segmentation**: `segment_auto_task`
2. **Box Segmentation**: `segment_box_task`
3. **Object Extraction**: `extract_objects_task`
4. **Batch Processing**: `process_batch_job`

### Task Monitoring
- **Flower**: Celery monitoring dashboard
- **Task Status**: Real-time status updates
- **Progress Tracking**: Percentage completion
- **Error Handling**: Detailed error messages and logs

## 📁 File Storage

### Local Storage Structure
```
uploads/
├── 2024/
│   └── 01/
│       └── 15/
│           └── image_12345.jpg

results/
├── segmentations/
│   └── job_12345/
│       ├── mask_0.png
│       ├── mask_1.png
│       └── visualization.png
└── objects/
    └── job_12345/
        ├── object_0.png
        ├── object_1.png
        └── object_2.png
```

### Google Drive Integration
- **Folder Structure**: Organized by date and job ID
- **File Naming**: Consistent naming convention
- **Access Control**: Configurable permissions
- **Backup**: Automatic backup of important results

## 🎯 Use Cases

### E-commerce Product Photography
- **Background Removal**: Extract products with transparency
- **Object Detection**: Identify multiple products in image
- **Batch Processing**: Process entire product catalogs

### Medical Imaging
- **Tissue Segmentation**: Identify different tissue types
- **Organ Detection**: Locate specific organs in scans
- **Research Analysis**: Quantitative analysis of medical images

### Autonomous Vehicles
- **Object Detection**: Identify vehicles, pedestrians, road signs
- **Scene Understanding**: Segment different elements in driving scenes
- **Training Data**: Generate labeled datasets for ML training

### Content Creation
- **Image Editing**: Extract objects for compositing
- **Social Media**: Create engaging visual content
- **Marketing Materials**: Generate product visuals

## 🔧 Troubleshooting

### Common Issues

#### Model Loading Failures
- **Check**: GPU availability and CUDA installation
- **Solution**: Verify model download and file integrity
- **Logs**: Check `/logs/model_loading.log`

#### Memory Issues
- **Symptoms**: Out of memory errors, slow processing
- **Solution**: Reduce batch size, unload unused models
- **Config**: Adjust `MODEL_CACHE_SIZE_GB` setting

#### Database Connection Errors
- **Check**: PostgreSQL service status
- **Solution**: Verify connection string and credentials
- **Test**: Use `psql` to test connection

#### Redis Connection Issues
- **Check**: Redis service status and port
- **Solution**: Restart Redis service
- **Config**: Verify `REDIS_URL` setting

#### Google Drive Authentication
- **Check**: Service account JSON file path
- **Solution**: Verify Google Cloud project setup
- **Test**: Use Google Drive API test tool

### Performance Issues
- **Slow Processing**: Check GPU utilization
- **High Memory**: Monitor model loading and caching
- **Database Slowdowns**: Check query performance and indexes
- **Network Issues**: Verify file upload/download speeds

## 📚 Additional Resources

### API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`
- **OpenAPI**: `http://localhost:8000/openapi.json`

### External Links
- [SAM Model Paper](https://arxiv.org/abs/2304.02643)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Celery Documentation](https://docs.celeryproject.org/)
- [Google Drive API](https://developers.google.com/drive/api/v3)

### Development Tools
- **VS Code**: Recommended IDE with Python extensions
- **Postman**: API testing and documentation
- **Docker Desktop**: Container management
- **pgAdmin**: PostgreSQL administration

## 🤝 Contributing

### Development Workflow
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Code review and merge

### Code Standards
- **Type Hints**: All functions must have type annotations
- **Docstrings**: Comprehensive documentation for all functions
- **Testing**: Minimum 80% code coverage
- **Linting**: Follow PEP 8 standards

### Testing Requirements
- **Unit Tests**: Test individual components
- **Integration Tests**: Test API endpoints
- **Performance Tests**: Load and stress testing
- **Security Tests**: Vulnerability scanning

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Check the troubleshooting section
- Review API documentation
- Check service health endpoints
- Examine application logs
- Contact development team

---

**Note**: This documentation reflects the current state of the Image Quarry API. For the most up-to-date information, please refer to the interactive API documentation at `/docs` when the service is running.