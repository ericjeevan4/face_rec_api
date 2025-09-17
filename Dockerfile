# Use Python 3.9 base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for git-lfs, Pillow, etc.)
RUN apt-get update && apt-get install -y \
    git git-lfs build-essential libgl1 libglib2.0-0 \
    && git lfs install

# Copy requirement file first
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Ensure LFS files (model weights) are pulled
RUN git lfs pull

# Expose the Render port
EXPOSE 10000

# Command to start Gunicorn with Flask app
CMD exec gunicorn --bind :$PORT --workers 1 --threads 8 --timeout 0 app:app
