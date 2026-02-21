FROM python:3.11-slim

# System deps for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.8.15 /uv /usr/local/bin/uv

WORKDIR /app

# Copy source and project metadata
COPY pyproject.toml .
COPY src/ src/

# Install package and dependencies
RUN uv pip install --system .

# Non-root user with writable output directory
RUN useradd --create-home appuser && mkdir -p /output && chown appuser:appuser /output
USER appuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

CMD ["python", "-m", "src.server", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8080"]
