# CPU-only reproducible image for papa-vision.
# Build:  docker build -t papa-vision .
# Test:   docker run --rm papa-vision uv run pytest
#
# Note: training inside Docker uses CPU (no MPS passthrough on macOS). For real
# training on Apple Silicon, run natively with `make`. This image exists to
# guarantee byte-for-byte reproducibility of the test suite and CPU experiments.
FROM python:3.12-slim

# uv for fast, locked dependency installation.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Install dependencies first (cached layer) from the lockfile.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --extra dev

# Copy the rest of the project.
COPY . .

CMD ["uv", "run", "pytest"]
