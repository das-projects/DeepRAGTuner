# Use the Ubuntu base image
FROM ubuntu:22.04 as base

# Set up some metadata
LABEL authors="Arijit Das"
LABEL maintainer="arijit.das@selfsupervised.de"
LABEL version="0.1"
LABEL description="Docker image for End to End RAG fine tuning"

# Avoid prompts with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set up environment variables for Conda
ENV PATH /opt/conda/bin:$PATH
ENV CONDA_AUTO_UPDATE_CONDA=false

# Install Linux dependencies and Conda (Miniconda in this case for a smaller footprint)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    curl \
    ca-certificates \
    sudo \
    less \
    htop \
    git \
    tzdata \
    wget \
    tmux \
    zip \
    bzip2 \
    unzip \
    zsh stow subversion fasd \
    && rm -rf /var/lib/apt/lists/* \
     # openmpi-bin \ # openmpi-bin for MPI (multi-node training)
    && apt-get clean

RUN apt-get update && \
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
    && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
    && apt-get update && apt-get install -y nvidia-container-toolkit \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
ENV HOME=/home/user/
RUN mkdir -p /home/user && chmod 777 /home/user
WORKDIR $HOME

# Copy the code into the container at $HOME
COPY . $HOME/

# Add Conda environment file and install dependencies
# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1
RUN conda env update --name base --file environment.yml
RUN pip install .

# Allow running runmpi as root
# ENV OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

# Specify the default command to run on container start (optional)
CMD ["python", "src/train.py"]



