# syntax=docker/dockerfile:1.7

# Evaluation job image for surogate-eval.  Designed to run as a
# short-lived job (like training): receives a YAML config, executes
# benchmarks / metrics / red-teaming, writes results to eval_results/.

ARG PYTHON_VERSION=3.12
ARG VERSION=0.0.0

# Extras installed into the venv, and the only difference between the two
# images this file builds. The default is the CPU image every ordinary eval
# runs on. The GPU variant is built with `--extra security --extra judge`,
# which adds vLLM so a judge model can be served inside the eval pod.
#
# Two images rather than one, because this layer is a ~8GB copy and the pull
# sits on the critical path of every run: a colocated judge is the exception,
# and making every CPU eval pull a serving stack it never loads has already
# cost us runs killed while waiting for an instance.
# Fails the build if a stage uses it without redeclaring `ARG EXTRAS`: an
# unset value expands to nothing, and `uv sync` with no extras succeeds while
# quietly producing an image missing the packages every eval needs.
ARG EXTRAS="--extra security"

# ── deps: install dependencies only (rarely changes) ──────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS deps

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ARG EXTRAS

# Only copy lock files — this layer is cached as long as deps don't change
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project ${EXTRAS:?must be set}

# ── builder: install project code on top of cached deps ───────────
FROM deps AS builder

ARG VERSION
ARG EXTRAS

COPY surogate_eval ./surogate_eval
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev ${EXTRAS:?must be set}

# Install quarantined packages (after sync so they don't get overwritten)
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --no-deps \
      'bfcl-eval==2025.10.27.1' \
      'git+https://github.com/sierra-research/tau-bench' \
    && uv pip install tree-sitter-java tree-sitter-javascript rank-bm25 \
      'mistralai<2' \
      anthropic sentence-transformers faiss-cpu cohere \
      google-genai 'mistralai>=1.0.0' boto3 overrides tenacity \
      qwen-agent writerai mpmath html2text google-search-results \
      soundfile librosa opencv-python-headless \
      datamodel-code-generator

# ── runtime ───────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim-bookworm AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      ca-certificates \
      curl \
      # soundfile / librosa need libsndfile
      libsndfile1 \
      # opencv needs libgl
      libgl1 libglib2.0-0 \
      # weasyprint needs these for PDF report generation
      libpango-1.0-0 \
      libpangocairo-1.0-0 \
      libgdk-pixbuf-2.0-0 \
      libffi-dev \
      libcairo2 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1001 surogate

# Docker CLI + daemon for code benchmarks (SWE-bench, terminal-bench).
# Only starts when the pod runs in privileged mode.
RUN curl -fsSL https://get.docker.com | sh \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p /app && chown 1001:1001 /app
WORKDIR /app

# Copy full venv from builder (one layer, ~8GB).
# The deps stage caches the build so rebuilds are fast even if this
# layer re-pushes on source changes.
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/surogate_eval /app/surogate_eval

# Bundle example configs and datasets so jobs can reference them
COPY examples ./examples

ENV PATH="/app/.venv/bin:/usr/local/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
# Telemetry opt-outs
    DEEPEVAL_TELEMETRY_OPT_OUT=1 \
    DEEPEVAL_FILE_SYSTEM=READ_ONLY \
    # Trust remote code for HF datasets / models
    HF_DATASETS_TRUST_REMOTE_CODE=1 \
    HF_ALLOW_CODE_EVAL=1 \
    TRUST_REMOTE_CODE=1

# Pre-download gpt2 tokenizer so API-only models can use lm-eval
# without a network fetch at runtime.
RUN /app/.venv/bin/python -c "from transformers import AutoTokenizer; AutoTokenizer.from_pretrained('gpt2')"

# Results are written here; mount a volume to persist them
RUN mkdir -p /app/eval_results && chown 1001:1001 /app/eval_results
VOLUME ["/app/eval_results"]

# No USER directive — dstack runs as root (uid 0) and needs root for dockerd.
# No ENTRYPOINT — dstack needs to run shell commands (base64 decode config, then surogate-eval).
CMD ["surogate-eval", "eval", "--help"]
