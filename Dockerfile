FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0 \
    cuda-drivers \
    && rm -rf /var/lib/apt/lists/*
# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt

# Set the working directory
# Set up environment variables
ENV CUDA_HOME=/usr/local/cuda \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    PATH=/usr/local/cuda/bin:$PATH

# Copy the Python application files into the container
COPY . /app
WORKDIR /app

# Command to run the Python application with CUDA
#CMD ["python3", "-m", "src.inference", "-d"]

# Set the entrypoint
ENTRYPOINT [ "python3" ]