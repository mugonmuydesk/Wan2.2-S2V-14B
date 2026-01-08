# Dockerfile for Wan2.2-S2V-14B on RunPod Serverless
# Requires 80GB+ VRAM (H100, A100-80GB)

FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Clone official Wan2.2 repository
RUN git clone https://github.com/Wan-Video/Wan2.2.git . && \
    git checkout main

# Install Python dependencies (flash_attn needs torch first, so install separately)
RUN pip install --no-cache-dir --upgrade pip && \
    grep -v flash_attn requirements.txt > requirements_no_flash.txt && \
    pip install --no-cache-dir -r requirements_no_flash.txt && \
    pip install --no-cache-dir runpod huggingface_hub[cli]

# Install Flash Attention 2 (requires torch to be installed first)
RUN pip install --no-cache-dir flash-attn --no-build-isolation

# Create model directory
RUN mkdir -p /models

# Download model weights (this makes the image large but faster cold starts)
# Comment out if you prefer to download at runtime via network volume
RUN huggingface-cli download Wan-AI/Wan2.2-S2V-14B \
    --local-dir /models/Wan2.2-S2V-14B \
    --local-dir-use-symlinks False

# Copy handler
COPY handler.py /app/handler.py

# Set environment variables
ENV MODEL_DIR=/models/Wan2.2-S2V-14B
ENV DEFAULT_SIZE=832*480
ENV DEFAULT_STEPS=30
ENV OFFLOAD_MODEL=True
ENV HF_HOME=/models/hf_cache

# Expose port (optional, for health checks)
EXPOSE 8000

# Start handler
CMD ["python", "-u", "handler.py"]
