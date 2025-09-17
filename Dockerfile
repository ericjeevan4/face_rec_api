# Use slim Python image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt

# Upgrade pip
RUN pip install --upgrade pip

# Install CPU-only torch + torchvision
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --no-cache-dir

# Install other Python dependencies
RUN pip install -r /app/requirements.txt --no-cache-dir

# Copy your application code
COPY . /app

# ✅ Copy the models folder explicitly (ensure vggface2_resnet.pth is inside)
COPY models /app/models

# ✅ Copy embeddings & labels
COPY embeddings.npy labels.npy /app/

# Expose default dev port (Render remaps dynamically)
EXPOSE 5000

# Run Gunicorn with 1 worker
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
