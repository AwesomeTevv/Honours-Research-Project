# Base image
FROM ubuntu:22.04.3

# Setting environment variables
ENV DEBIAN_FRONTEND=noninteractive

# Installing the necessary system packages
RUN apt-get update && apt-get install -y \
python3 \
python3-pip \
python3-dev \
wget \
git \
curl \
build-essential \
libglib2.0-0 \
libsm6 \
libxext6 \
libxrender-dev \
libgl1-mesa-glx \
&& rm -f /var/lib/apt/lists/*

# Setting the working directory
WORKDIR /workspace

# Copying the project files
COPY . /workspace

# Installing the python dependencies
COPY requirements.txt /workspace/requirements.txt
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

# Making sure the scripts are executable
RUN chmod +x /workspace/*.sh /workspace/*.py

# Entry point
ENTRYPOINT [ "python3", "/workspace/ppo_drone.py" ]