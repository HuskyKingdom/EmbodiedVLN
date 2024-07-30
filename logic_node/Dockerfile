FROM osrf/ros:melodic-desktop-full
# Use osrf/ros instead of ros:melodic for pre-installed development packages
# Ref: https://github.com/osrf/docker_images#osrf-profile

# Ref: https://robotics.stackexchange.com/a/105553
# Ref: https://github.com/osrf/docker_images/issues/697#issuecomment-1819626877
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 4B63CF8FDE49746E98FA01DDAD19BAB3CBF125EA
# Setup dependencies for install script
RUN apt-get update && \
    apt-get install -y \
        wget \
        curl \
        rfkill \
        software-properties-common \
        systemd \
    && rm -rf /var/lib/apt/lists/*
# Ref: https://discourse.ros.org/t/ros-gpg-key-expiration-incident/20669
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -





# conda install
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh

RUN conda --version
