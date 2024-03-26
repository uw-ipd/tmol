# syntax=docker/dockerfile:1
FROM mambaorg/micromamba:jammy-cuda-12.1.1

USER root

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    unzip \
    git \
    vim \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

ENV PATH "$MAMBA_ROOT_PREFIX/bin:$PATH"

WORKDIR /code

COPY --chown=$MAMBA_USER:$MAMBA_USER environments/linux-cuda /code/environments/linux-cuda
RUN micromamba install -y -n base -f /code/environments/linux-cuda/env.yml && micromamba clean --all --yes
RUN pip install -r /code/environments/linux-cuda/requirements-dev-linux-cuda.txt
COPY --chown=$MAMBA_USER:$MAMBA_USER . /code
RUN pip install .[dev]
