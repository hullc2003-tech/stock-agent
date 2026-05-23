# AI Stock Agent v2.5 - Production Docker image for Google Cloud Run
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8080

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies (includes torch + transformers)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# App code
COPY . .

ENV PYTHONPATH=/app

# Cloud Run expects the container to listen on $PORT
EXPOSE 8080

# Run as web service using your FastAPI app
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]
