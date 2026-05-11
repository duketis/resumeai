FROM python:3.12-slim

# System deps: tectonic for LaTeX, Node 20 for the claude CLI, curl + ca-certs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        gnupg \
        wget \
        xz-utils \
    && rm -rf /var/lib/apt/lists/*

# Tectonic (single-binary LaTeX engine). Pinned + downloaded directly from
# the upstream release tarball so the build is deterministic and offline-safe
# (no shell pipe).
ARG TECTONIC_VERSION=0.15.0
RUN ARCH="$(uname -m)" \
    && case "${ARCH}" in \
         x86_64)  TECTONIC_ARCH=x86_64-unknown-linux-musl ;; \
         aarch64) TECTONIC_ARCH=aarch64-unknown-linux-musl ;; \
         *)       echo "unsupported arch ${ARCH}"; exit 1 ;; \
       esac \
    && curl -fsSL "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@${TECTONIC_VERSION}/tectonic-${TECTONIC_VERSION}-${TECTONIC_ARCH}.tar.gz" \
       -o /tmp/tectonic.tar.gz \
    && tar -xzf /tmp/tectonic.tar.gz -C /usr/local/bin tectonic \
    && rm /tmp/tectonic.tar.gz \
    && tectonic --version

# Node 20 + claude CLI. Auth is mounted at runtime via /root/.claude.
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && npm install -g @anthropic-ai/claude-code

WORKDIR /app

# Install Python deps first so the layer is cacheable independently of source.
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

# Source last so a code change rebuilds only the final layer.
COPY resumeai ./resumeai

ENV PYTHONUNBUFFERED=1 \
    RESUMEAI_RUNS_ROOT=/data/runs \
    RESUMEAI_CONTEXT_ROOT=/data/UserContext

EXPOSE 8765

CMD ["uvicorn", "resumeai.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8765"]
