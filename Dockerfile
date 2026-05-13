# syntax=docker/dockerfile:1.7

# Evaluation job image for surogate-eval.  Designed to run as a
# short-lived job (like training): receives a YAML config, executes
# benchmarks / metrics / red-teaming, writes results to eval_results/.

ARG PYTHON_VERSION=3.12
ARG VERSION=0.0.0

# ── builder ────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS builder

ARG VERSION

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project --extra security

COPY surogate_eval ./surogate_eval
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra security

# Install packages with quarantined deps (mistralai) — must use --no-deps
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps \
      'bfcl-eval==2025.10.27.1' \
      'git+https://github.com/sierra-research/tau-bench'

# ── runtime ────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      ca-certificates \
      curl \
      uidmap \
      dbus-user-session \
      fuse-overlayfs \
      slirp4netns \
      # weasyprint needs these for PDF report generation
      libpango-1.0-0 \
      libpangocairo-1.0-0 \
      libgdk-pixbuf-2.0-0 \
      libffi-dev \
      libcairo2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1001 surogate

# Install rootless Docker — runs as unprivileged user, no --privileged needed.
# Download the static binaries directly (the install script needs systemd).
RUN DOCKER_VERSION=27.5.1 \
    && curl -fsSL "https://download.docker.com/linux/static/stable/$(uname -m)/docker-${DOCKER_VERSION}.tgz" \
       | tar xz --strip-components=1 -C /usr/local/bin docker/docker \
    && curl -fsSL "https://download.docker.com/linux/static/stable/$(uname -m)/docker-rootless-extras-${DOCKER_VERSION}.tgz" \
       | tar xz --strip-components=1 -C /usr/local/bin docker-rootless-extras/dockerd-rootless.sh docker-rootless-extras/rootlesskit \
    && mkdir -p /home/surogate/.local/share/docker \
    && chown -R 1001:1001 /home/surogate/.local

RUN mkdir -p /app && chown 1001:1001 /app
WORKDIR /app

COPY --from=builder --chown=1001:1001 /app/.venv /app/.venv
COPY --from=builder --chown=1001:1001 /app/surogate_eval /app/surogate_eval

# Bundle example configs and datasets so jobs can reference them
COPY --chown=1001:1001 examples ./examples

ENV PATH="/app/.venv/bin:/usr/local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Telemetry opt-outs
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    DEEPEVAL_FILE_SYSTEM=READ_ONLY \
    # Trust remote code for HF datasets / models
    HF_DATASETS_TRUST_REMOTE_CODE=1 \
    HF_ALLOW_CODE_EVAL=1 \
    TRUST_REMOTE_CODE=1 \
    # Rootless Docker config
    DOCKER_HOST=unix:///run/user/1001/docker.sock \
    XDG_RUNTIME_DIR=/run/user/1001

# Results are written here; mount a volume to persist them
RUN mkdir -p /app/eval_results && chown 1001:1001 /app/eval_results
VOLUME ["/app/eval_results"]

USER surogate

# No ENTRYPOINT — dstack needs to run shell commands (base64 decode config, then surogate-eval).
CMD ["surogate-eval", "eval", "--help"]
