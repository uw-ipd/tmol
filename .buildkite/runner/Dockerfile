FROM nvidia/cuda:9.1-runtime-ubuntu16.04

### Install Conda
# Installation steps from https://hub.docker.com/r/continuumio/miniconda3/~/dockerfile/
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 ca-certificates curl git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Updated to miniconda 4.5.1
RUN wget --quiet https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc

# Inlined tini version in fetch url
ENV TINI_VERSION v0.16.1
RUN wget --quiet -O /usr/bin/tini https://github.com/krallin/tini/releases/download/v0.16.1/tini
RUN chmod +x /usr/bin/tini

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]

### Install Buildkite Agent
# Installs the buildkite agent binary into /usr/local/bin
# This allows use of buildkite-agent in docker steps if the source agent
# is running within a docker container, as the agent binary is *not* present in the host.
# 
# Requires subsequent environment vars in job steps
#   --env BUILDKITE_JOB_ID
#   --env BUILDKITE_BUILD_ID
#   --env BUILDKITE_AGENT_ACCESS_TOKEN
#
# Add to /usr/local/bin so the above env vars can be set via the docker buildkite plugin
# without masking the binary via the /usr/bin/buildkite-agent bindmount
RUN curl -s https://api.github.com/repos/buildkite/agent/releases/latest | \
    grep browser_download_url | grep buildkite-agent-linux-amd64 | sed 's/"browser_download_url"://' | \
    xargs curl -L | tar -xz -C /usr/local/bin ./buildkite-agent

### Setup test environment
# Layering setup to allow partial caching of conda packages

RUN conda update -n base conda && conda clean -tipsy

COPY environments/core.dependencies.txt /environments/
RUN conda install -n base --file /environments/core.dependencies.txt && conda clean -tipsy

COPY environments/linux.cuda.dependencies.txt /environments/
RUN conda install -n base --file /environments/linux.cuda.dependencies.txt && conda clean -tipsy

COPY environments/test.dependencies.txt /environments/
RUN conda install -n base --file /environments/test.dependencies.txt && conda clean -tipsy

COPY environments /environments
RUN /environments/render linux.cuda test > /environment.yml && \
    conda env update -n base -f /environment.yml && \
    conda clean -tipsy
