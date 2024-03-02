FROM nvidia/cuda:11.0.3-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive


# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    git \
    python3-pip \
    python3-dev \
    python3-opencv \
    libglib2.0-0
# Install any python packages you need
COPY requirements.txt requirements.txt

RUN python3 -m pip install -r requirements.txt


# Set the working directory
WORKDIR /app

# Set the entrypoint
ENTRYPOINT [ "python3" ]