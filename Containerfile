# syntax=docker/dockerfile:1

# ------------------------------------------------------------------------------
# Stage 1: Build the wheel
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

RUN pip install --no-cache-dir uv==0.12.17

ARG ITS_HUB_VERSION=0.0.0
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${ITS_HUB_VERSION}

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY its_hub ./its_hub

RUN uv build --wheel --out-dir /dist

# ------------------------------------------------------------------------------
# Stage 2: Runtime
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

RUN pip install --no-cache-dir uv==0.12.17

# Non-root runtime user.
RUN useradd --create-home --uid 1001 its

COPY --from=builder /dist/*.whl /tmp/
# Install the wheel with the iaas extra, then drop the wheel.
RUN uv pip install --system --no-cache "$(ls /tmp/its_hub-*.whl)[iaas]" \
    && rm -rf /tmp/*.whl

ENV ITS_IAAS_HOST=0.0.0.0 \
    ITS_IAAS_PORT=8109 \
    ITS_LOG_LEVEL=INFO

EXPOSE 8109
USER its

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8109/health')" || exit 1

CMD ["its-iaas"]
