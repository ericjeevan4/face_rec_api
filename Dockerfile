FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 libgl1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip

# Install CPU-only torch + torchvision
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision --no-cache-dir

# Install other Python deps
RUN pip install -r /app/requirements.txt --no-cache-dir

COPY . /app

# Expose default dev port (Render remaps dynamically)
EXPOSE 5000

# Run Gunicorn with 1 worker (reduce memory usage)
CMD gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --timeout 120
