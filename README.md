# Image Quarry

A production-ready FastAPI service for SAM (Segment Anything Model) image segmentation with background job processing, PostgreSQL database, Redis caching, and comprehensive monitoring.

## 🚀 Features

- **SAM Image Segmentation**: Advanced AI-powered image segmentation using Meta's Segment Anything Model
- **FastAPI Backend**: High-performance async API with automatic documentation
- **Background Job Processing**: Celery-based task queue for handling long-running segmentation jobs
- **PostgreSQL Database**: Robust data persistence with SQLAlchemy ORM
- **Redis Caching**: High-performance caching for improved response times
- **Docker Support**: Full containerization with Docker Compose
- **Comprehensive Monitoring**: Prometheus metrics and structured logging
- **Rate Limiting**: API protection with SlowAPI rate limiting
- **Authentication**: JWT-based authentication with secure password hashing

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Installation](#installation)
- [Development Setup](#development-setup)
- [Docker Setup](#docker-setup)
- [API Documentation](#api-documentation)
- [Configuration](#configuration)
- [Testing](#testing)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🏃‍♂️ Quick Start

### Using Docker (Recommended)

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd image-quarry
   ```

2. **Start services with Docker Compose**
   ```bash
   docker compose up -d
   ```

3. **Verify the service is running**
   ```bash
   curl http://localhost:8000/health/
   ```

4. **Access API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

### Local Development

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # For development
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Start the development server**
   ```bash
   python -m uvicorn app.main:app --reload
   ```

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- PostgreSQL 15+
- Redis 7+
- Docker and Docker Compose (for containerized setup)
- CUDA-compatible GPU (optional, for enhanced performance)

### Python Environment Setup

1. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies using pip**
   ```bash
   # Upgrade pip to latest version
   python -m pip install --upgrade pip
   
   # Install production dependencies
   pip install -r requirements.txt
   
   # Install development dependencies (optional)
   pip install -r requirements-dev.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"
   ```

### Database Setup

1. **Create PostgreSQL database**
   ```bash
   createdb image_quarry
   ```

2. **Run database migrations**
   ```bash
   alembic upgrade head
   ```

3. **Seed initial data** (optional)
   ```bash
   python scripts/init_db.py
   ```

## 🔧 Development Setup

### Local Development

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd image-quarry
   ```

2. **Install development dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Set up pre-commit hooks**
   ```bash
   pre-commit install
   ```

4. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your local configuration
   ```

5. **Start development services**
   ```bash
   # Start PostgreSQL and Redis (if using Docker)
   docker compose up -d db redis
   
   # Run database migrations
   alembic upgrade head
   
   # Start the API server
   python -m uvicorn app.main:app --reload
   
   # Start Celery worker (in another terminal)
   python -m celery -A app.services.background_tasks worker --loglevel=info
   
   # Start Celery beat scheduler (optional)
   python -m celery -A app.services.background_tasks beat --loglevel=info
   ```

### Code Quality Tools

The project includes several code quality tools configured for development:

- **Black**: Code formatting
- **isort**: Import sorting
- **Flake8**: Linting
- **MyPy**: Type checking
- **Pre-commit**: Automated checks before commits

Run code quality checks:
```bash
# Format code
black app/ tests/

# Sort imports
isort app/ tests/

# Run linting
flake8 app/ tests/

# Type checking
mypy app/

# Run all pre-commit hooks
pre-commit run --all-files
```

## 🐳 Docker Setup

### Using Docker Compose (Recommended)

1. **Build and start all services**
   ```bash
   docker compose up -d
   ```

2. **View logs**
   ```bash
   docker compose logs -f app
   docker compose logs -f worker
   ```

3. **Scale workers**
   ```bash
   docker compose up -d --scale worker=3
   ```

4. **Stop services**
   ```bash
   docker compose down
   ```

### Docker Services

- **app**: FastAPI application server
- **worker**: Celery worker for background tasks
- **scheduler**: Celery beat scheduler for periodic tasks
- **db**: PostgreSQL database
- **redis**: Redis cache and message broker

### Docker Configuration

The Docker setup includes:
- Health checks for all services
- Resource limits for workers
- Persistent volumes for data
- Network isolation
- Non-root user for security

## 📚 API Documentation

### Authentication

Most endpoints require authentication via JWT tokens:

1. **Register a new user**
   ```bash
   curl -X POST "http://localhost:8000/auth/register" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "securepassword"}'
   ```

2. **Login to get token**
   ```bash
   curl -X POST "http://localhost:8000/auth/login" \
     -H "Content-Type: application/json" \
     -d '{"email": "user@example.com", "password": "securepassword"}'
   ```

3. **Use token in requests**
   ```bash
   curl -X POST "http://localhost:8000/segment/" \
     -H "Authorization: Bearer YOUR_TOKEN" \
     -F "file=@your_image.jpg"
   ```

### Image Segmentation

**Synchronous segmentation**
```bash
curl -X POST "http://localhost:8000/segment/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@image.jpg" \
  -F "model_type=vit_h"
```

**Asynchronous segmentation (background job)**
```bash
curl -X POST "http://localhost:8000/segment/async/" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@image.jpg" \
  -F "model_type=vit_h"
```

**Check job status**
```bash
curl -X GET "http://localhost:8000/jobs/JOB_ID/status" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Get job results**
```bash
curl -X GET "http://localhost:8000/jobs/JOB_ID/result" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Health Checks

**API health**
```bash
curl http://localhost:8000/health/
```

**Database health**
```bash
curl http://localhost:8000/health/db
```

**Redis health**
```bash
curl http://localhost:8000/health/redis
```

## ⚙️ Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@localhost:5432/image_quarry

# Redis
REDIS_URL=redis://localhost:6379/0
CELERY_BROKER_URL=redis://localhost:6379/1
CELERY_RESULT_BACKEND=redis://localhost:6379/2

# File Storage
MODEL_CACHE_DIR=./models
UPLOAD_DIR=./uploads

# Security
SECRET_KEY=your-secret-key-here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Monitoring
LOG_LEVEL=INFO
LOG_FORMAT=json
PROMETHEUS_METRICS_ENABLED=true

# Model Configuration
DEFAULT_MODEL_TYPE=vit_h
MODEL_DEVICE=cuda  # or cpu
MAX_FILE_SIZE=10485760  # 10MB
```

### Model Configuration

The service supports different SAM model variants:

- **vit_h**: ViT-H model (best quality, slowest)
- **vit_l**: ViT-L model (balanced quality/speed)
- **vit_b**: ViT-B model (fastest, lower quality)

Configure default model in environment variables or specify per request.

## 🧪 Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_segmentation.py

# Run with coverage
python -m pytest --cov=app --cov-report=html

# Run async tests only
python -m pytest -k "async"

# Run with verbose output
python -m pytest -v
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: API endpoint testing
- **Async Tests**: Background task testing
- **Model Tests**: SAM model functionality testing

### Performance Testing

```bash
# Load testing with locust (install first: pip install locust)
locust -f tests/performance/locustfile.py --host=http://localhost:8000
```

## 🚀 Deployment

### Production Deployment

1. **Environment Setup**
   ```bash
   # Set production environment variables
   export ENVIRONMENT=production
   export LOG_LEVEL=WARNING
   export PROMETHEUS_METRICS_ENABLED=true
   ```

2. **Database Migration**
   ```bash
   alembic upgrade head
   ```

3. **Start Services**
   ```bash
   # Start API server
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   
   # Or with Docker Compose
   docker compose -f docker-compose.prod.yml up -d
   ```

### Scaling Considerations

- **Horizontal Scaling**: Run multiple API server instances behind a load balancer
- **Worker Scaling**: Scale Celery workers based on queue length
- **Database**: Use connection pooling and read replicas
- **Cache**: Implement Redis cluster for high availability

### Security Best Practices

- Use strong secret keys
- Enable HTTPS in production
- Implement proper CORS configuration
- Regularly update dependencies
- Use security scanning tools
- Monitor for vulnerabilities

## 📊 Monitoring

### Prometheus Metrics

The service exposes metrics at `/metrics` endpoint:

- Request count and duration
- Error rates
- Model inference time
- Queue length and processing time
- Database connection pool status

### Grafana Dashboard

Import the provided Grafana dashboard for comprehensive monitoring:

```bash
# Dashboard JSON file
grafana/dashboards/image-quarry-dashboard.json
```

### Logging

Structured logging with correlation IDs:

```bash
# View logs in JSON format
docker compose logs -f app | jq '.'

# Filter by log level
docker compose logs -f app | jq 'select(.level == "ERROR")'
```

### Alerting

Configure alerts for:
- High error rates
- Long response times
- Queue backlog
- Service downtime
- Resource usage

## 🔧 Troubleshooting

### Common Issues

**1. PyTorch CUDA Issues**
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check CUDA version
python -c "import torch; print(torch.version.cuda)"
```

**2. Database Connection Issues**
```bash
# Test database connection
python -c "import asyncio; from app.database import get_async_db; print('Database connected')"
```

**3. Redis Connection Issues**
```bash
# Test Redis connection
python -c "import redis; r = redis.from_url('redis://localhost:6379'); print(r.ping())"
```

**4. Model Loading Issues**
```bash
# Check model cache directory
ls -la models/

# Verify model files
curl http://localhost:8000/models/
```

**5. Celery Worker Issues**
```bash
# Check worker status
curl http://localhost:8000/health/celery

# View worker logs
docker compose logs -f worker
```

### Performance Optimization

**1. Model Loading Optimization**
- Pre-download models during build
- Use model caching effectively
- Implement model warming

**2. Database Optimization**
- Add proper indexes
- Use connection pooling
- Optimize queries

**3. Image Processing Optimization**
- Resize large images before processing
- Use appropriate image formats
- Implement client-side compression

### Getting Help

- Check the [API documentation](http://localhost:8000/docs)
- Review service logs: `docker compose logs -f`
- Check system resources: `docker stats`
- Monitor metrics: `curl http://localhost:8000/metrics`

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for detailed contribution guidelines.

## 📞 Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation
- Review the troubleshooting section

---

**Note**: This project has been migrated from UV to pip package management for simplified development setup. All functionality has been preserved while reducing complexity.