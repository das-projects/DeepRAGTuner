# Use the latest NVIDIA PyTorch base image
FROM nvcr.io/nvidia/pytorch:23.09-py3 as base

# Set up some metadata
LABEL authors="Arijit Das"
LABEL maintainer="arijit.das@selfsupervised.de"
LABEL version="0.1"
LABEL description="Docker image for End to End RAG fine tuning"

# Set the working directory in the container
ENV HOME=/home/user
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
RUN python -m venv $HOME/venv
RUN . $HOME/venv/bin/activate
RUN pip-sync requirements.txt
RUN MAX_JOBS=4 pip install flash-attn --no-build-isolation
RUN pip install .

# Start a new stage to create smaller runtime images
FROM nvcr.io/nvidia/pytorch:23.09-py3 as runtime

COPY --from=base $HOME $HOME

ENV PATH = "$HOME/venv/bin:$PATH"
RUN . $HOME/venv/bin/activate

# Specify the default command to run on container start (optional)
CMD ["python", "src/train.py"]



