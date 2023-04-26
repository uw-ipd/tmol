# syntax=docker/dockerfile:1
FROM nvidia/cuda:11.8.0-devel-ubuntu20.04 as base
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

# general environment for docker
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends sudo wget curl bzip2 ca-certificates git vim rsync jq software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH ${CONDA_DIR}/bin:$PATH

RUN curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh \
  && bash Mambaforge-Linux-x86_64.sh -b -p $CONDA_DIR \
  && rm Mambaforge-Linux-x86_64.sh \
  && conda clean -afy \
  && printf "source ${CONDA_DIR}/etc/profile.d/conda.sh\nsource ${CONDA_DIR}/etc/profile.d/mamba.sh\nmamba activate base" >> /etc/skel/.bashrc \
  && printf "source ${CONDA_DIR}/etc/profile.d/conda.sh\nsource ${CONDA_DIR}/etc/profile.d/mamba.sh\nmamba activate base" >> ~/.bashrc

# create group for conda install
RUN groupadd conda \
    && chgrp -R conda ${CONDA_DIR} \
    && chmod 770 -R ${CONDA_DIR}

# create docker user
RUN useradd -m -s /bin/bash docker && echo "docker:docker" | chpasswd && adduser docker sudo && adduser docker conda

# enable passwordless sudo
RUN echo "docker ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/docker


# Inlined tini version in fetch url
# Question: why are we writing directly to /usr/bin?
ENV TINI_VERSION v0.16.1
RUN wget --quiet -O /usr/bin/tini https://github.com/krallin/tini/releases/download/v0.16.1/tini
RUN chmod +x /usr/bin/tini

# create the entry point now before we lose sudo strength
RUN echo "#!/bin/bash" > /docker-entrypoint.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /docker-entrypoint.sh && \
    echo "conda activate tmol" >> /docker-entrypoint.sh && \
    echo "exec \"\$@\"" >> /docker-entrypoint.sh  && \
    chmod a+rx /docker-entrypoint.sh

USER    docker
WORKDIR /home/docker

# FIXME needed for newer versions of sparse but should be replaced by explicit casts to dense where needed
ENV SPARSE_AUTO_DENSIFY 1

FROM base as builder


# NOTE we copy requirements over first so layers are cached and we don't have
# to reinstall dependencies if only source has changed

### NOTE Dependencies can be installed in two different ways

### 1. install a fresh environment by re-resolving dependencies based on specification in env.yml and requirements{-dev}.in ###

# COPY --chown=docker requirements.in env.yml /home/docker/tmol/
# RUN mamba env update -n base -f /home/docker/tmol/env.yml
# COPY --chown=docker requirements-dev.in /home/docker/tmol/
# RUN pip install -r /home/docker/tmol/requirements-dev.in

### 2. install a frozen environment with exact versions specified from env-dev.yml ###
# For the testing server, we choose (2) here for reproducibility

COPY --chown=docker env-dev.yml /home/docker/tmol/
RUN mamba create -n tmol && \
    mamba env update -n tmol -f /home/docker/tmol/env-dev.yml && \
    mamba clean -afy


ENTRYPOINT [ "/usr/bin/tini", "--", "/docker-entrypoint.sh" ]
CMD [ "/bin/bash" ]

