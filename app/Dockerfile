# Single-stage local development image with CUDA and PyTorch (Python 3.12)
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Environment configuration
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

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
    git \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip to latest version
RUN python -m pip install --upgrade pip

# Create non-root user for better security
RUN groupadd -r appuser && useradd -m -r -g appuser appuser
ENV HOME=/home/appuser

# Set work directory
WORKDIR /code
RUN chown appuser:appuser /code

# Create logs directory with proper permissions for non-root user
RUN mkdir -p /code/logs && chown appuser:appuser /code/logs

# Copy requirements files first for better Docker layer caching
COPY --chown=appuser:appuser requirements.txt requirements-dev.txt ./

# Install Python dependencies as root for system-wide availability
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files into workdir (avoid nested /code/code)
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Default command: run FastAPI app via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]