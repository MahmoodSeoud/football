FROM ubuntu:20.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    libgl1-mesa-dev \
    libsdl2-dev \
    libsdl2-image-dev \
    libsdl2-ttf-dev \
    libsdl2-gfx-dev \
    libboost-all-dev \
    libdirectfb-dev \
    libst-dev \
    mesa-utils \
    xvfb \
    x11vnc \
    python3.8 \
    python3-pip \
    python3.8-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install GRF and dependencies
RUN pip install gfootball \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    tqdm \
    h5py

# Create working directory
WORKDIR /app

# Set display for headless rendering
ENV DISPLAY=:99

# Default command: start Xvfb and run Python
ENTRYPOINT ["bash", "-c", "Xvfb :99 -screen 0 1024x768x24 &> /dev/null & exec \"$@\"", "--"]
CMD ["python"]
