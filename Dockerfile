# ✅ Use official PyTorch base image with CUDA 12.8 and Python preinstalled
FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime

# Environment setup
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENSSL_FIPS=0

# Install any additional system dependencies
RUN apt-get update && apt-get install -y \
    git gcc libgl1-mesa-glx libglib2.0-0 ffmpeg curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code first (so model_loader.py is available)
COPY . .

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Expose FastAPI port
EXPOSE 8000

# Start the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]