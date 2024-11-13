# Base image with CUDA 12.2 and CUDNN 8
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables for CUDA
ENV CUDA_VERSION=12.1.0
ENV CUDNN_VERSION=8
ENV PATH=/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Create a working directory
WORKDIR /app

# Install Python dependencies (including WhisperX, PyTorch, etc.)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install git+https://github.com/m-bain/whisperx.git pydub pandas ctranslate2==4.4.0

# Copy the script into the container
COPY create_dataset.py .

# Set entrypoint to handle arguments passed to the container
ENTRYPOINT ["python3", "create_dataset.py"]
