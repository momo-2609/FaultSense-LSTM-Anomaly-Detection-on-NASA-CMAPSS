FROM python:3.11-slim

WORKDIR /app

# Install system deps (needed for numpy/scipy builds on slim)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Create checkpoints dir (mounted at runtime via volume in compose)
RUN mkdir -p checkpoints

# Expose API port
EXPOSE 8000

# Default: launch API. Override CMD for training.
WORKDIR /app/api
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
