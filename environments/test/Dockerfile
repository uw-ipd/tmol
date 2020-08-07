FROM nvidia/cuda:10.0-devel-ubuntu18.04
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8

### Install basic packages
RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git jq software-properties-common && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

### Install Conda
# Installation steps from https://hub.docker.com/r/continuumio/miniconda3/~/dockerfile/
ENV PATH /opt/conda/bin:$PATH

# Updated to miniconda 4.5.1
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.6.14-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    echo "conda =4.6" >> /opt/conda/conda-meta/pinned && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Configure a docker-entrypoint.sh to activate the base conda environment
RUN echo "#!/bin/bash" >> /docker-entrypoint.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /docker-entrypoint.sh && \
    echo "conda activate base" >> /docker-entrypoint.sh && \
    echo "exec \"\$@\"" >> /docker-entrypoint.sh  && \
    chmod a+rx /docker-entrypoint.sh

# Inlined tini version in fetch url
ENV TINI_VERSION v0.16.1
RUN wget --quiet -O /usr/bin/tini https://github.com/krallin/tini/releases/download/v0.16.1/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--", "/docker-entrypoint.sh" ]
CMD [ "/bin/bash" ]

### Pre-install pyrosetta in "support" namespace.
#COPY environments/support.dependencies.txt /environments/
#COPY environments/support.channels.txt /environments/
#RUN conda install -n base -c `cat /environments/support.channels.txt` --file /environments/support.dependencies.txt && conda clean -tipsy

### Packages needed for graphviz png output
# libxrender: missing package dependency in graphviz conda package
# gsfonts: fonts for png text rendering
#RUN apt-get update --fix-missing && \
#    apt-get install -y libxrender1 gsfonts && \
#    apt-get clean && \
#    rm -rf /var/lib/apt/lists/*

### Setup test environment

RUN conda update -n base conda && conda clean -tipsy

COPY environments /environments
RUN conda update -n base --file /environments/locks/test-linux-64.lock && \
    conda clean -tipsy
