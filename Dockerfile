# ---------------------------------------------------------------------------
# Build stage — produces a wheel from the source tree
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tooling only (not kept in the final image)
RUN pip install --no-cache-dir build==1.2.2

# Copy the minimum needed to resolve + build the package
COPY pyproject.toml .
COPY src/ src/

# Build a self-contained wheel; all transitive deps are declared in pyproject.toml
RUN python -m build --wheel --outdir /dist

# ---------------------------------------------------------------------------
# Runtime stage — lean image with only what the app needs to run
# ---------------------------------------------------------------------------
FROM python:3.11-slim

# curl is used by the HEALTHCHECK and is useful for debugging in production
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python runtime hygiene
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Cache directory for sentence-transformers / HuggingFace models.
# Mount a Fly.io volume at /app/.cache to avoid re-downloading on every deploy.
ENV TRANSFORMERS_CACHE=/app/.cache \
    HF_HOME=/app/.cache

# Copy and install the pre-built wheel (pulls all declared dependencies)
COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Alembic migration files live outside the Python package — copy them explicitly
COPY alembic.ini .
COPY alembic/ alembic/

# Drop root: run as an unprivileged user
RUN useradd --no-create-home --shell /bin/false appuser \
    && mkdir -p /app/.cache \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Liveness probe: fail fast if the process hangs
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:8000/health || exit 1

# Two workers saturate the 2 vCPUs allocated in fly.toml.
# Increase via FLY_PROCESS_GROUP or by scaling vertically.
CMD ["uvicorn", "llmgateway.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "warning"]
