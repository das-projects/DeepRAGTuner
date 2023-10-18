# Use the latest NVIDIA PyTorch base image
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel as base

# Set up some metadata
LABEL authors="Arijit Das"
LABEL maintainer="arijit.das@selfsupervised.de"
LABEL version="0.1"
LABEL description="Docker image for End to End RAG fine tuning"

# Avoid prompts with apt-get
ENV DEBIAN_FRONTEND=noninteractive

# Set the working directory in the container
ENV HOME=/workspace
RUN mkdir -p $HOME && chmod 777 $HOME
WORKDIR $HOME

# Copy the code into the container at $HOME
COPY . $HOME

# Compile requirements based on dependencies essential to the project
# Disable pip cache: https://stackoverflow.com/questions/45594707/what-is-pips-no-cache-dir-good-for
ENV PIP_NO_CACHE_DIR=1
RUN pip install --upgrade pip pip-tools
RUN pip-compile --resolver=backtracking requirements.in

# Create virtual environment and install only required dependencies
RUN pip-sync requirements.txt
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation
RUN pip install .

RUN git clone https://github.com/Dao-AILab/flash-attention.git \
    && cd flash-attention \
    && cd csrc/layer_norm && pip install . && cd ../../ \
    && cd csrc/fused_dense_lib && pip install . && cd ../../ \
    && cd .. && rm -rf flash-attention

# Specify the default command to run on container start (optional)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
