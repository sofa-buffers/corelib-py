FROM ubuntu:26.04

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
#
# valgrind is what bench/run_callgrind.sh runs under: instructions/op is the
# benchmark number that means anything on a shared or virtualised host, since it
# is independent of the clock and the scheduler, and it is one of the three
# benchmark tools every corelib ships (CORELIB_PLAN §10). Without it that script
# is the one shipped tool the container cannot run.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    valgrind \
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

# A second interpreter, headers included, so the native accelerator can be built
# and tested against more than one CPython locally — CI runs 3.9–3.14, and the
# accelerator's fast paths depend on CPython-internal layouts that they verify at
# import (see __sofab_digits_selftest / __sofab_field_slot_attrs in
# _speedups.pyx). Those self-tests are only meaningful if someone actually runs
# them on a second version:
#
#   python3.14 -m venv /tmp/v314 \
#     && /tmp/v314/bin/pip install -e . pytest \
#     && /tmp/v314/bin/python -m pytest
#
# It is not the default interpreter and nothing depends on it, so dropping this
# layer only costs that cross-check.
ARG EXTRA_PYTHON_VERSION=3.14
ENV EXTRA_PYTHON_VERSION=${EXTRA_PYTHON_VERSION}
RUN apt-get update && apt-get install -y --no-install-recommends \
       python${EXTRA_PYTHON_VERSION} \
       python${EXTRA_PYTHON_VERSION}-venv \
       python${EXTRA_PYTHON_VERSION}-dev \
    && rm -rf /var/lib/apt/lists/*

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

