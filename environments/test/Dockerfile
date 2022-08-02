FROM nvidia/cuda:11.3.1-devel-ubuntu20.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

### Install basic packages
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git jq software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### Install Conda
# Installation steps from https://hub.docker.com/r/continuumio/miniconda3/~/dockerfile/
ENV PATH /opt/conda/bin:$PATH

# Updated to miniconda 4.5.1
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    echo "conda =4.8.3" >> /opt/conda/conda-meta/pinned && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Configure a docker-entrypoint.sh to enable conda
RUN echo "#!/bin/bash" >> /docker-entrypoint.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /docker-entrypoint.sh && \
    echo "exec \"\$@\"" >> /docker-entrypoint.sh  && \
    chmod a+rx /docker-entrypoint.sh

# Inlined tini version in fetch url
ENV TINI_VERSION v0.16.1
RUN wget --quiet -O /usr/bin/tini https://github.com/krallin/tini/releases/download/v0.16.1/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--", "/docker-entrypoint.sh" ]
CMD [ "/bin/bash" ]

### Packages needed for graphviz png output
# libxrender: missing package dependency in graphviz conda package
# gsfonts: fonts for png text rendering
#RUN apt-get update --fix-missing && \
#    apt-get install -y libxrender1 gsfonts && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

### Setup test environment

COPY env.yml /environments/env.yml

# Install tmol environment
# Configure a docker-entrypoint.sh to activate the tmol conda environment
RUN conda create -n tmol && \
    conda env update -n tmol --file=/environments/env.yml && \
    conda clean -tipsy

RUN echo "#!/bin/bash" > /docker-entrypoint.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /docker-entrypoint.sh && \
    echo "conda activate tmol" >> /docker-entrypoint.sh && \
    echo "exec \"\$@\"" >> /docker-entrypoint.sh  && \
    chmod a+rx /docker-entrypoint.sh

