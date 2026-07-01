FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.12
ENV PYTHON_VERSION=${PYTHON_VERSION}

# Make pip/uv behave well in a container
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential provides the C toolchain used to compile the optional native
# accelerator (src/sofab/_speedups.pyx → C via Cython, pulled in as a PEP 517
# build dependency). Without it the package still installs — it just falls back
# to the pure-Python engine — but the devcontainer builds the fast path.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    gnupg \
    && rm -rf /var/lib/apt/lists/*

# Python (deadsnakes) + pip/venv + common build headers
RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
       python${PYTHON_VERSION} \
       python${PYTHON_VERSION}-venv \
       python${PYTHON_VERSION}-dev \
       python3-pip \
       pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python  python  /usr/bin/python${PYTHON_VERSION} 1

# pipx + uv (fast Python package/venv manager)
ENV PIPX_HOME=/usr/local/pipx \
    PIPX_BIN_DIR=/usr/local/bin
RUN python${PYTHON_VERSION} -m ensurepip --upgrade 2>/dev/null || true \
    && python3 -m pip install --no-cache-dir --break-system-packages pipx \
    && pipx install uv \
    && pipx install poetry \
    && pipx install hatch

# Node.js (LTS) for Claude Code
RUN curl -fsSL https://deb.nodesource.com/setup_lts.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Claude Code
RUN npm install -g @anthropic-ai/claude-code

# GitHub CLI (gh) — installed from GitHub's official apt repository
RUN mkdir -p -m 755 /etc/apt/keyrings \
    && curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
       -o /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
       > /etc/apt/sources.list.d/github-cli.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends gh \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["/bin/bash"]

