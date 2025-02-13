# Use NVIDIA's CUDA image with cuDNN 8.9 and Ubuntu 22.04 (Debian-based)
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Set environment variables for CUDA
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
ENV CUDA_HOME=/usr/local/cuda

# Install necessary system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libreadline-dev \
    libffi-dev \
    libsqlite3-dev \
    curl \
    libbz2-dev \
    git \
    python3-pip \
    openssh-client \
    rsync \
    # Remove apt cache
    && rm -rf /var/lib/apt/lists/*

# Copy the global.env file
COPY global.env /tmp/global.env

# Install Python Version (if different from system Python)
RUN . /tmp/global.env \
    && echo "Using Python version: ${TUSTU_PYTHON_VERSION}" \
    && if [ "${TUSTU_PYTHON_VERSION}" != "3.11" ]; then \
        echo "Downloading Python version: ${TUSTU_PYTHON_VERSION}" \
        && wget --no-check-certificate https://www.python.org/ftp/python/${TUSTU_PYTHON_VERSION}/Python-${TUSTU_PYTHON_VERSION}.tgz \
        && tar -xf Python-${TUSTU_PYTHON_VERSION}.tgz \
        && cd Python-${TUSTU_PYTHON_VERSION} \
        && ./configure --enable-optimizations \
        && make -j$(nproc) \
        && make altinstall \
        && cd .. \
        && rm -rf Python-${TUSTU_PYTHON_VERSION} Python-${TUSTU_PYTHON_VERSION}.tgz \
        && ln -s /usr/local/bin/python${TUSTU_PYTHON_VERSION%.*} /usr/local/bin/python3 \
        && ln -s /usr/local/bin/python${TUSTU_PYTHON_VERSION%.*} /usr/local/bin/python; \
    fi

# Set the working directory
WORKDIR /home/app

# Copy the python requirements list
COPY requirements.txt .

# Install dependencies inside a virtual environment
RUN python3 -m pip install -r requirements.txt \
    && rm requirements.txt
