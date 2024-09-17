# Use an ARM-compatible Ubuntu image
FROM arm64v8/ubuntu:22.04

# Set environment variables to prevent prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Update package lists and install necessary tools and libraries
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    wget \
    vim \
    zsh \
    curl \
    libboost-all-dev \
    libtbb-dev \
    libeigen3-dev \
    libxrender1 \
    libxext6 \
    libsm6 \
    libx11-dev \
    xvfb \
    x11-apps \
    graphviz \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies including matplotlib for plotting
RUN pip3 install numpy pyparsing==2.4.7 matplotlib graphviz astropy

# Build GTSAM from source with Python bindings
WORKDIR /opt
RUN git clone https://github.com/borglab/gtsam.git && \
    cd gtsam && \
    mkdir build && \
    cd build && \
    cmake -DGTSAM_USE_SYSTEM_EIGEN=ON -DGTSAM_BUILD_PYTHON=ON .. && \
    make -j$(nproc) && \
    make install

# Install GTSAM Python bindings using pip
RUN pip3 install /opt/gtsam/build/python

# Set Python3 as the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1

# Install oh-my-zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" || true

# Set zsh as the default shell
RUN chsh -s /bin/zsh

# Create a working directory for the examples
WORKDIR /ion_gnss_2024

# Copy the current directory content (where your Python script is) to /examples in the container
COPY . /ion_gnss_2024/

# Start the Xvfb virtual framebuffer when the container starts
CMD ["zsh"]
