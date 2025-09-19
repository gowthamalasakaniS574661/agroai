# Use Python base image
FROM python:3.11-slim

WORKDIR /app

# System packages for pillow, numpy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libglib2.0-0 libsm6 libxrender1 libxext6 git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy everything
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
