# tmol Development Container - Dependencies only (tmol itself is bind-mounted for development)
#
# To build this Docker image, from the project root, run:
#   docker build -t tmol-dev -f containers/docker/tmol-dev.Dockerfile .
#
# To run interactively with your local code:
#   docker run --gpus all -it -v $(pwd):/tmol_host -w /tmol_host tmol-dev bash
#   pip install -e .  # inside container

FROM nvcr.io/nvidia/pytorch:25.12-py3

LABEL author="Institute for Protein Design"
LABEL version="1.0-dev"
LABEL description="tmol - Dependencies only (for development and scoring)"

# Create directories for bind mounts
RUN mkdir -p /tmol_host /projects /net /squash

# Install X11 libraries needed by OpenBabel
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml to extract dependencies
COPY pyproject.toml /opt/tmol_pyproject.toml

# Python dependency installation (separate layer for better caching)
# Note: Starting from PyTorch 25.03, the container includes /etc/pip/constraint.txt
# which specifies versions of all packages used during container creation.
# We use --constraint to respect these versions and avoid conflicts.
RUN \
    # Upgrade pip
    python -m pip install --upgrade pip && \
    \
    # Install uv for fast dependency resolution
    pip install uv && \
    \
    # Compile dependency list from pyproject.toml (with all extras)
    mv /opt/tmol_pyproject.toml /opt/pyproject.toml && \
    uv pip compile /opt/pyproject.toml --output-file /opt/tmol_requirements.txt --all-extras --constraint /etc/pip/constraint.txt && \
    rm /opt/pyproject.toml && \
    \
    # Filter out packages already provided by the NGC base image to avoid conflicts
    # (torch, numpy, nvidia-*, triton are pre-installed in the container)
    cat /opt/tmol_requirements.txt | \
        grep -vE "^(torch(|vision|audio)|numpy|nvidia-.*|pynvml|packaging|triton)==" | \
        awk -F'==' '!seen[$1]++' > /opt/combined_requirements.txt && \
    \
    # Print requirements for debugging
    echo "=== tmol requirements to install ===" && \
    cat /opt/combined_requirements.txt && \
    echo "====================================" && \
    \
    # Install the cleaned requirements
    uv pip install --system --break-system-packages -r /opt/combined_requirements.txt --constraint /etc/pip/constraint.txt && \
    \
    # Clean up to minimize image size
    uv cache clean && \
    pip cache purge && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* 2>/dev/null || true && \
    rm -rf /root/.cache 2>/dev/null || true && \
    rm -rf /opt/*.txt 2>/dev/null || true

# Default working directory (bind-mount your tmol checkout here)
WORKDIR /tmol_host

# Default command
ENTRYPOINT ["python"]
