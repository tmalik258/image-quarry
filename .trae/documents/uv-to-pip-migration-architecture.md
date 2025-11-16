# UV to Pip Migration Technical Architecture

## 1. Current State Analysis

### 1.1 UV Usage Overview

The Image Quarry project currently uses UV as the primary package manager with the following components:

* **pyproject.toml**: Contains project metadata, dependencies, and UV-specific configuration

* **uv.lock**: Lock file ensuring reproducible builds with exact dependency versions

* **.python-version**: Specifies Python 3.13 for the project

* **Dockerfile**: Uses UV for dependency installation and virtual environment management

* **Docker Compose**: References UV-based image and commands

### 1.2 Current UV Configuration

```dockerfile
# Environment variables for UV
ENV UV_PYTHON=3.12 \
    UV_HTTP_TIMEOUT=600 \
    UV_HTTP_RETRIES=5 \
    UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
    UV_INDEX=https://pypi.org/simple \
    UV_INDEX_STRATEGY=unsafe-best-match

# UV installation
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# UV virtual environment setup
RUN uv venv "$VIRTUAL_ENV"
RUN uv sync --frozen --python "$VIRTUAL_ENV/bin/python"
```

### 1.3 Dependencies Analysis

Current dependencies in pyproject.toml:

* **Core Framework**: FastAPI, Uvicorn, Pydantic

* **Database**: SQLAlchemy, AsyncPG, Alembic

* **ML/AI**: PyTorch, TorchVision, Segment-Anything, OpenCV, Pillow, NumPy

* **Async/Background**: Redis, Celery, Aiofiles, HTTPX

* **Auth/Security**: Python-JOSE, Passlib, SlowAPI

* **Monitoring**: Prometheus Client, Structlog, Tenacity

* **Dev Tools**: Pytest, Black, Isort, Flake8, MyPy, Pre-commit

## 2. Migration Strategy to Pip

### 2.1 Requirements.txt Creation

Convert pyproject.toml dependencies to requirements.txt format:

```txt
# Core dependencies
fastapi>=0.104.1
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
pydantic-settings>=2.1.0

# Database
sqlalchemy>=2.0.23
asyncpg>=0.29.0
alembic>=1.13.0

# ML/AI
torch>=2.1.0
torchvision>=0.16.0
segment-anything>=1.0
opencv-python>=4.8.1
pillow>=10.1.0
numpy>=1.24.0
supervision>=0.18.0

# Async/Background
redis>=5.0.1
celery>=5.3.4
aiofiles>=23.2.1
httpx>=0.25.2

# Auth/Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
slowapi>=0.1.9

# Monitoring
prometheus-client>=0.19.0
structlog>=23.2.0
tenacity>=8.2.3
aioredis>=2.0.1

# Form handling
python-multipart>=0.0.6
```

### 2.2 Development Dependencies

Create separate requirements-dev.txt:

```txt
# Testing
pytest>=7.4.3
pytest-asyncio>=0.21.1
pytest-cov>=4.1.0

# Code Quality
black>=23.11.0
isort>=5.12.0
flake8>=6.1.0
mypy>=1.7.1
pre-commit>=3.6.0
```

### 2.3 Version Pinning Strategy

* **Critical ML libraries** (torch, torchvision, segment-anything): Pin to exact versions

* **Core framework** (fastapi, pydantic): Use compatible release operator (\~=)

* **Database libraries**: Pin minor versions for stability

* **Development tools**: Allow flexible versions for latest features

## 3. Dockerfile Updates

### 3.1 New Dockerfile Configuration

```dockerfile
# Single-stage local development image with CUDA and PyTorch (Python 3.12)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Install system libraries required by OpenCV and scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN python -m pip install --upgrade pip

# Create non-root user for better security
RUN groupadd -r appuser && useradd -m -r -g appuser appuser
ENV HOME=/home/appuser

# Set work directory
WORKDIR /app
RUN chown appuser:appuser /app

# Copy requirements files first for better caching
COPY --chown=appuser:appuser requirements.txt requirements-dev.txt ./

# Install Python dependencies as root first, then switch to appuser
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Default command: run FastAPI app via Uvicorn
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## 4. Docker Compose Updates

### 4.1 Service Configuration Changes

```yaml
services:
  app:
    build:
      context: .
      dockerfile: app/Dockerfile
    image: image-quarry-app:pip-latest
    ports:
      - "8000:8000"
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/image_quarry
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/2
      - MODEL_CACHE_DIR=/app/models
      - UPLOAD_DIR=/app/uploads
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      - PYTHONPATH=/app
    volumes:
      - ./app:/app
      - model_cache:/app/models
      - upload_data:/app/uploads
      - app_logs:/app/logs
    depends_on:
      - db
      - redis
    networks:
      - app_network
    restart: unless-stopped
    command: ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

  worker:
    image: image-quarry-app:pip-latest
    command: ["python", "-m", "celery", "-A", "app.services.background_tasks", "worker", "--loglevel=info", "--concurrency=2"]
    environment:
      - DATABASE_URL=postgresql+asyncpg://postgres:password@db:5432/image_quarry
      - REDIS_URL=redis://redis:6379/0
      - CELERY_BROKER_URL=redis://redis:6379/1
      - CELERY_RESULT_BACKEND=redis://redis:6379/2
      - MODEL_CACHE_DIR=/app/models
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      - PYTHONPATH=/app
    volumes:
      - ./app:/app
      - model_cache:/app/models
      - worker_logs:/app/logs
    depends_on:
      - db
      - redis
    networks:
      - app_network
    restart: unless-stopped
```

### 4.2 Volume and Network Configuration

* Remove UV cache volumes

* Maintain existing data volumes for PostgreSQL, Redis, models, and uploads

* Keep network configuration unchanged

## 5. Testing and Validation Procedures

### 5.1 Pre-Migration Testing

1. **Current State Validation**

   * Run existing test suite: `docker compose exec app python -m pytest`

   * Verify all endpoints work: `docker compose exec app python -m pytest tests/`

   * Check Celery workers: `docker compose logs worker`

   * Validate model loading and inference

2. **Dependency Analysis**

   * Export current UV lock file state

   * Document exact versions of critical ML libraries

   * Verify CUDA compatibility with PyTorch versions

### 5.2 Migration Testing

1. **Requirements Validation**

   ```bash
   # Test requirements installation
   pip install -r requirements.txt
   pip install -r requirements-dev.txt

   # Verify no conflicts
   pip check
   ```

2. **Docker Build Testing**

   ```bash
   # Build new image
   docker build -t image-quarry-app:pip-test -f app/Dockerfile .

   # Test container startup
   docker run --rm image-quarry-app:pip-test python -c "import torch; print(torch.__version__)"
   ```

3. **Full Stack Testing**

   ```bash
   # Start services
   docker compose -f docker-compose.yml up -d

   # Run health checks
   curl http://localhost:8000/health/

   # Test image segmentation endpoint
   curl -X POST -F "file=@sample.jpg" http://localhost:8000/segment/
   ```

### 5.3 Post-Migration Validation

1. **Functional Testing**

   * Image upload and processing

   * SAM model inference

   * Background job processing

   * Database operations

   * Redis caching

2. **Performance Testing**

   * Compare inference times before/after migration

   * Memory usage analysis

   * Container startup time

3. **Security Validation**

   * Verify no additional vulnerabilities introduced

   * Check dependency security with `pip-audit`

   * Validate container security scanning

## 6. Rollback Plan

### 6.1 Rollback Triggers

* Critical functionality failures

* Performance degradation >20%

* Security vulnerabilities introduced

* Dependency conflicts that cannot be resolved

### 6.2 Rollback Procedure

1. **Immediate Actions**

   ```bash
   # Stop pip-based services
   docker compose -f docker-compose.yml down

   # Revert to UV-based services
   git checkout HEAD~1  # or specific commit hash
   docker compose up -d
   ```

2. **Data Preservation**

   * Backup database state before migration

   * Preserve uploaded images and models

   * Export logs for analysis

3. **Communication Plan**

   * Notify team of rollback

   * Document issues encountered

   * Plan remediation strategy

## 7. Documentation Updates

### 7.1 README.md Updates

````markdown
## Installation

### Using pip (Recommended)
```bash
# Install dependencies
pip install -r requirements.txt
````

### Using Docker

```bash
# Build and run with Docker Compose
docker compose up -d
```

````

### 7.2 Development Setup Guide
```markdown
## Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd image-quarry
````

1. **Install Python dependencies**

   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

2. **Set up environment variables**

   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

3. **Run database migrations**

   ```bash
   alembic upgrade head
   ```

4. **Start development server**

   ```bash
   python -m uvicorn app.main:app --reload
   ```

```

### 7.3 API Documentation Updates
- Update any references to UV in API docs
- Ensure all endpoints are documented
- Add migration notes for developers

## 8. Implementation Timeline

### Phase 1: Preparation (Day 1)
- Create requirements.txt files
- Update Dockerfile
- Prepare test environment

### Phase 2: Testing (Day 2)
- Local testing with pip
- Docker build testing
- Integration testing

### Phase 3: Migration (Day 3)
- Deploy to staging environment
- Performance validation
- Security scanning

- Monitoring and validation
- Documentation updates

### Phase 5: Cleanup (Day 5)
- Remove UV-related files
- Update CI/CD pipelines
- Archive old configurations

## 9. Risk Assessment

### 9.1 High-Risk Areas
- **ML Library Compatibility**: PyTorch and CUDA version alignment
- **Performance Impact**: Dependency resolution and import times
- **Security**: New dependency versions may introduce vulnerabilities

### 9.2 Mitigation Strategies
### Phase 4: Release (Day 4)
- Deployment validation

- **Staged Rollout**: Deploy to staging
- **Comprehensive Testing**: Full regression testing before deployment
- **Monitoring**: Enhanced logging and monitoring during migration
- **Rollback Capability**: Maintain ability to quickly revert changes

## 10. Success Criteria

### 10.1 Functional Criteria
- ✅ All existing endpoints work correctly
- ✅ Image segmentation functionality preserved
- ✅ Background job processing operational
- ✅ Database operations function normally

### 10.2 Performance Criteria
- ✅ Container startup time < 30 seconds
- ✅ API response times within 10% of baseline
- ✅ Memory usage comparable to UV setup
- ✅ No significant CPU overhead

### 10.3 Operational Criteria
- ✅ Simplified development setup
- ✅ Reduced complexity in Docker images
- ✅ Easier dependency management
- ✅ Improved documentation clarity

This migration will significantly simplify the development environment while maintaining all required functionality and ensuring backward compatibility.
```

