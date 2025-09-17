# Use slim Python image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only torch + torchvision pinned versions
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.8.0+cpu torchvision==0.23.0+cpu --no-cache-dir

# Install other Python dependencies
RUN pip install -r /app/requirements.txt --no-cache-dir

# Copy application code
COPY . /app

# Ensure models folder exists and copy model
RUN mkdir -p /app/models
COPY models/ /app/models/

# Copy embeddings & labels if they exist (won’t fail if missing)
RUN [ -f embeddings.npy ] && cp embeddings.npy /app/ || true
RUN [ -f labels.npy ] && cp labels.npy /app/ || true

# Expose port (Render dynamically maps $PORT)
EXPOSE 5000

# Run Gunicorn using the Render-provided PORT
CMD gunicorn app:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 300 --preload
