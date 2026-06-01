FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/opt/conda/bin:$PATH"

RUN apt-get update && apt-get install -y \
    wget git curl vim && \
    rm -rf /var/lib/apt/lists/*

# Install Miniconda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p /opt/conda && \
    rm /tmp/miniconda.sh

WORKDIR /app

# Setup DivRL environment
RUN conda create -n DivRL python=3.10 -y && \
    conda run -n DivRL conda install -y -c nvidia cuda-toolkit=12.4 && \
    conda clean -afy
COPY DivRL/ /app/DivRL/
RUN conda run -n DivRL pip install -e /app/DivRL

# Setup reward MTG environment
RUN conda create -n mtg python=3.11 -y && conda clean -afy
COPY mind-the-glitch/requirements.txt /app/mind-the-glitch/
RUN conda run -n mtg pip install -r /app/mind-the-glitch/requirements.txt

# Aliases
RUN echo 'source /opt/conda/etc/profile.d/conda.sh' >> /root/.bashrc && \
    echo 'alias rl="conda activate DivRL"' >> /root/.bashrc && \
    echo 'alias reward="conda activate mtg"' >> /root/.bashrc && \
    echo 'echo "Envs ready. Use: rl | reward"' >> /root/.bashrc